"""Re-record two randomly selected successes and two failures from a completed run."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import shutil
import signal
from datetime import datetime, timezone
from pathlib import Path

from lightnav.habitat.launch_fm_stop_eval import REPOSITORY, write_json
from lightnav.habitat.stop_report import check_coverage, episode_key, load_expected_episodes

DEFAULT_SOURCE = REPOSITORY / "outputs/betanav_eval/20260928T012647_241235Z_fm_stop_r2r_unseen"


def record_key(row):
    return episode_key(row["scene_id"], row.get("habitat_episode_id", row.get("episode_id")))


def select_samples(source_run: Path, seed: int) -> dict:
    source_run = Path(source_run).resolve()
    if json.loads((source_run / "status.json").read_text()).get("status") != "completed":
        raise ValueError("source evaluation must be completed before sampling videos")
    job_bytes = (source_run / "job.json").read_bytes()
    results_bytes = (source_run / "results.jsonl").read_bytes()
    job = json.loads(job_bytes)
    records = [json.loads(line) for line in results_bytes.splitlines() if line.strip()]
    if not check_coverage(job["episodes"], records)["complete"]:
        raise ValueError("source results do not have exact episode coverage")
    ordered = sorted(job["episodes"], key=lambda e: e["scene_id"])
    indices = {record_key(row): i for i, row in enumerate(ordered)}
    generator = random.Random(seed)
    samples = []
    for label, success in (("success", True), ("failure", False)):
        group = [r for r in records if bool(r["success"]) == success]
        if len(group) < 2:
            raise ValueError(f"need at least two {label} episodes")
        for number, row in enumerate(generator.sample(group, 2), 1):
            samples.append(dict(label=f"{label}_{number}", dataset_index=indices[record_key(row)], original=row))
    return dict(source_run=str(source_run), seed=seed, samples=samples,
                source_results_sha256=hashlib.sha256(results_bytes).hexdigest(),
                source_job_sha256=hashlib.sha256(job_bytes).hexdigest())


def verify_dataset_order(job, actual_episodes):
    expected = [record_key(row) for row in sorted(job["episodes"], key=lambda e: e["scene_id"])]
    actual = [record_key(row) for row in sorted(actual_episodes, key=lambda e: e["scene_id"])]
    if expected != actual:
        raise ValueError("current dataset episode order differs from the completed evaluation")


def compare_replay(original, replayed):
    fields = {}
    for key in ("success", "oracle_success", "steps", "stop_source", "final_distance",
                "min_distance", "spl", "ndtw", "stop_score"):
        left, right = original.get(key), replayed.get(key)
        if key == "stop_score" and key not in original and key not in replayed:
            # A budget STOP skips policy inference and has no head score.
            match = True
        elif key in {"final_distance", "min_distance", "spl", "ndtw", "stop_score"}:
            match = (isinstance(left, (int, float)) and isinstance(right, (int, float))
                     and math.isfinite(left) and math.isfinite(right)
                     and math.isclose(left, right, rel_tol=1e-5, abs_tol=1e-4))
        else:
            match = left is not None and right is not None and left == right
        fields[key] = dict(original=left, replayed=right, match=bool(match))
    identity = record_key(original) == record_key(replayed)
    instruction = original.get("instruction") == replayed.get("instruction")
    return dict(identity_match=identity, instruction_match=instruction, fields=fields,
                matches_original=identity and instruction and all(f["match"] for f in fields.values()))


def inspect_video(path, expected_frames, expected_fps):
    """Decode every frame; the general runner permits failed optional videos."""
    import cv2

    capture = cv2.VideoCapture(str(path))
    frames, shape = 0, None
    fps = capture.get(cv2.CAP_PROP_FPS)
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            if shape is not None and frame.shape != shape:
                raise ValueError(f"video dimensions changed: {path}")
            shape = frame.shape
            frames += 1
    finally:
        capture.release()
    if frames != expected_frames or not math.isclose(fps, expected_fps, abs_tol=0.02):
        raise ValueError(f"incomplete video {path}: frames={frames}, fps={fps}")
    return dict(frames=frames, fps=fps, width=shape[1], height=shape[0], duration_seconds=frames / fps)


def record_samples(source_job, selection, output: Path, *, port: int, fps: int):
    from lightnav.habitat import runner
    from lightnav.habitat.eval_preflight import (
        build_habitat_env,
        check_habitat_python,
        validate_resources,
    )
    from lightnav.habitat.fm_stop_worker import (
        WorkerInterrupted,
        build_eval_config,
        configure_device,
        run_shard,
    )

    job = {**source_job, "output_dir": str(output), "repository": str(REPOSITORY),
           "num_shards": len(source_job["episodes"])}
    episodes = load_expected_episodes(job["dataset_root"], job["split"])
    verify_dataset_order(source_job, episodes)
    ordered = sorted(episodes, key=lambda e: e["scene_id"])
    resources = validate_resources(job["dataset_root"], job["scenes_dir"],
                                   [ordered[s["dataset_index"]] for s in selection["samples"]], job["split"])
    original_gt_hash = source_job.get("resources", {}).get("ndtw", {}).get("sha256")
    if original_gt_hash and resources["ndtw"]["sha256"] != original_gt_hash:
        raise ValueError("NDTW ground truth changed since the original evaluation")
    metadata_hash = hashlib.sha256((Path(job["checkpoint"]) / "metadata.json").read_bytes()).hexdigest()
    if metadata_hash != source_job["checkpoint_metadata_sha256"]:
        raise ValueError("checkpoint metadata changed since the original evaluation")
    environment = check_habitat_python(job["habitat_python"], build_habitat_env(REPOSITORY))
    runner._require_video_deps()
    write_json(output / "replay_job.json", dict(source_job=source_job, resources=resources,
               habitat_environment=environment, server_port=port, video_fps=fps,
               note="New recordings of the selected episodes; original evaluation did not retain frames."))
    configure_device(int(job["cpu_threads"]))
    worker = dict(worker_id=0, port=port)
    cached = runner._default_engine_factory(build_eval_config(job, worker, output))
    videos = output / "videos"
    videos.mkdir()
    previous_handlers, comparisons = {}, []

    def interrupted(signum, _frame):
        for sig in (signal.SIGINT, signal.SIGTERM):
            signal.signal(sig, signal.SIG_IGN)
        raise WorkerInterrupted(f"video recording interrupted by signal {signum}")

    try:
        for sig in (signal.SIGINT, signal.SIGTERM):
            previous_handlers[sig] = signal.signal(sig, interrupted)
        for sample_number, sample in enumerate(selection["samples"], 1):
            original, index = sample["original"], sample["dataset_index"]
            manifest = output / f"episode_{sample_number}_manifest.json"
            # Exact absolute scene spelling retains the original FM noise seed.
            write_json(manifest, [dict(scene_id=original["scene_id"],
                                       habitat_episode_id=original["habitat_episode_id"])])

            def config_factory(job, worker, folder):
                cfg = build_eval_config(job, worker, folder)
                cfg.episodes = 1
                cfg.episode_manifest = str(manifest)
                cfg.save_video = True
                cfg.video_fps = fps
                cfg.record_dir = str(folder / "records")
                return cfg

            print(f"Recording {sample['label']}: {record_key(original)}", flush=True)
            run_shard(job, worker, index, cached, config_factory=config_factory)
            folder = output / f"shard_{index:04d}"
            rows = [json.loads(line) for line in (folder / "results.jsonl").read_text().splitlines() if line.strip()]
            if len(rows) != 1:
                raise ValueError(f"expected exactly one replayed episode for {sample['label']}")
            replayed = rows[0]
            comparison = compare_replay(original, replayed)
            if not comparison["identity_match"] or not comparison["instruction_match"]:
                raise ValueError(f"replay identity or instruction mismatch for {sample['label']}")
            video = folder / replayed.get("video", "missing.mp4")
            video_info = inspect_video(video, int(replayed["steps"]) + 1, fps)
            scan, episode = record_key(original)
            outcome = "success" if replayed["success"] else "failure"
            # Names reflect the new recording's measured outcome, even if it drifted.
            destination = videos / f"sample_{sample_number:02d}_{scan}_episode_{episode}_{outcome}.mp4"
            shutil.copyfile(video, destination)
            comparisons.append(dict(sample=sample, replayed=replayed, comparison=comparison,
                                    video=str(destination), video_info=video_info))
            write_json(output / "comparison.json", comparisons)
            print(json.dumps(dict(video=str(destination), matches_original=comparison["matches_original"])), flush=True)
    finally:
        for sig, previous in previous_handlers.items():
            signal.signal(sig, previous)
    return comparisons


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-run", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--selection-seed", type=int, default=20260928)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--port", type=int, default=28880)
    parser.add_argument("--video-fps", type=int, default=5)
    parser.add_argument("--prepare-only", action="store_true", help="freeze selection without accessing models, scenes or GPUs")
    args = parser.parse_args(argv)
    if args.selection_seed < 0 or args.video_fps < 1 or not 1024 <= args.port <= 65535:
        parser.error("seed must be nonnegative, fps positive, and port between 1024 and 65535")
    selection = select_samples(args.source_run, args.selection_seed)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    output = (args.output_dir or REPOSITORY / "outputs/betanav_videos" / f"{stamp}_four_samples").resolve()
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "selection.json", selection)
    write_json(output / "status.json", dict(status="selected" if args.prepare_only else "running"))
    print(f"Selection: {output / 'selection.json'}", flush=True)
    if args.prepare_only:
        return 0
    source_job = json.loads((args.source_run / "job.json").read_text())
    try:
        comparisons = record_samples(source_job, selection, output, port=args.port, fps=args.video_fps)
        matched = all(c["comparison"]["matches_original"] for c in comparisons)
        write_json(output / "status.json", dict(status="completed" if matched else "completed_with_differences",
                   videos=len(comparisons), original_metrics_matched=matched,
                   comparison=str(output / "comparison.json")))
        print(f"Videos: {output / 'videos'}; originals matched={matched}", flush=True)
        return 0 if matched else 2
    except BaseException as exc:
        write_json(output / "status.json", dict(status="failed", error=repr(exc)))
        raise


if __name__ == "__main__":
    raise SystemExit(main())
