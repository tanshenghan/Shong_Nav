#!/usr/bin/env python3
"""Plan, render and export a complete R2R/RxR source selection in resumable shards."""

from __future__ import annotations

import argparse
import concurrent.futures
import fcntl
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from data_pipeline.collect_r2r import sha256, write_json  # noqa: E402
from data_pipeline.pointing import PointingConfig  # noqa: E402
from data_pipeline.vlnce_source import DATA_ROOTS, LANGUAGES, SPLITS, load_source  # noqa: E402

DATA_ROOT = Path("/mnt/cfs/data/habitat/datasets/R2R_VLNCE_v1-3_preprocessed")
SCENES_ROOT = Path("/mnt/cfs/data/habitat/scene_datasets")
RVQ_ROOT = Path("/mnt/cfs/Models/Light/action_tokenizer")


def digest_json(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def plan_routes(episodes, gt, routes_per_shard=8):
    """Reuse only byte-equivalent physical inputs; never deduplicate instructions."""
    if routes_per_shard < 1:
        raise ValueError("routes_per_shard must be positive")
    groups, seen = defaultdict(list), set()
    for episode in episodes:
        episode_id = str(episode["episode_id"])
        if episode_id in seen:
            raise ValueError(f"duplicate source episode ID: {episode_id}")
        seen.add(episode_id)
        actions = gt[episode_id]["actions"]
        if not actions or actions[-1] != 0 or any(a not in (1, 2, 3) for a in actions[:-1]):
            raise ValueError(f"invalid expert action/STOP sequence: {episode_id}")
        physical = {k: episode[k] for k in ("start_position", "start_rotation", "goals")}
        physical["gt"] = gt[episode_id]
        key = (episode["scene_id"], str(episode.get("trajectory_id", episode_id)), digest_json(physical))
        # The original instruction tokens are not used: retain the exact text and original ID.
        record = {k: v for k, v in episode.items() if k != "instruction"}
        record["instruction"] = {"instruction_text": episode["instruction"]["instruction_text"]}
        groups[key].append(record)
    by_scene = defaultdict(list)
    for (scene, trajectory, digest), records in sorted(groups.items()):
        records.sort(key=lambda e: str(e["episode_id"]))
        by_scene[scene].append(dict(scene_id=scene, trajectory_id=trajectory, physical_sha256=digest,
            representative_id=str(records[0]["episode_id"]), frames=len(gt[str(records[0]["episode_id"])]["actions"]),
            episodes=records))
    shards = []
    for scene, routes in sorted(by_scene.items()):
        for start in range(0, len(routes), routes_per_shard):
            batch = routes[start:start + routes_per_shard]
            shards.append(dict(id=len(shards), scene_id=scene, routes=batch,
                episodes=sum(len(r["episodes"]) for r in batch),
                frames=sum(r["frames"] * len(r["episodes"]) for r in batch)))
    return dict(source_episodes=len(episodes), physical_routes=len(groups), scenes=len(by_scene),
        frames=sum(s["frames"] for s in shards), rendered_frames=sum(r["frames"] for rs in by_scene.values() for r in rs),
        routes_per_shard=routes_per_shard, shards=shards)


def create_plan(args):
    episodes, gt, source_selection = load_source(
        args.data_root, args.dataset, args.split, args.role, args.languages)
    paths = [Path(source_selection[k]) for k in ("episode_file", "gt_file")]
    plan = plan_routes(episodes, gt, args.routes_per_shard)
    if not episodes:
        raise ValueError("empty training split")
    for scene in {e["scene_id"] for e in episodes}:
        path = args.scenes_dir / scene.removeprefix("data/scene_datasets/")
        for asset in (path, path.with_suffix(".navmesh")):
            if not asset.is_file():
                raise FileNotFoundError(asset)
    config = PointingConfig(**json.loads(args.pointing_config.read_text()))
    config.require_current()
    if config.opos_visibility_window != "last_half":
        raise ValueError("full train requires the agreed last_half AND depth-visible opos policy")
    definition_files = [args.pointing_config, REPO / "habitat_server/configs/vlnce_r2r.yaml",
                        *(REPO / "data_pipeline" / n for n in ("collect_r2r.py", "geometry.py", "pointing.py", "vlnce_source.py")),
                        REPO / "src/lightnav/traj_vocab.py"]
    rvq_files = sorted(p for p in args.rvq_bundle.iterdir() if p.is_file())
    plan.update(format="beta_nav_full_train_plan_v1", split=args.split, dataset=args.dataset,
        source_selection=source_selection, role=args.role, languages=source_selection["languages"],
        width=470, height=280, seed=20260911,
        repo_id=args.repo_id, data_root=str(args.data_root), scenes_dir=str(args.scenes_dir),
        rvq_bundle=str(args.rvq_bundle), pointing_config=str(args.pointing_config),
        output_dir=str(args.output_dir), source_sha256={str(p): sha256(p) for p in paths},
        definition_sha256={str(p): sha256(p) for p in definition_files},
        rvq_sha256={p.name: sha256(p) for p in rvq_files})
    plan["plan_sha256"] = digest_json(plan)
    return plan


def expand_instructions(raw, shard):
    """Create temporary annotation views over identical rendered observations."""
    for route in shard["routes"]:
        base = raw / "episodes" / route["representative_id"]
        original = json.loads((base / "episode.json").read_text())
        if original["frames"] != route["frames"]:
            raise ValueError("rendered frame count differs from source GT")
        for episode in route["episodes"]:
            episode_id = str(episode["episode_id"])
            if episode_id == route["representative_id"]:
                continue
            folder = raw / "episodes" / episode_id
            folder.mkdir()
            for name in ("targets.npz", "samples.jsonl"):
                os.link(base / name, folder / name)
            metadata = dict(original, episode_id=episode_id,
                instruction=episode["instruction"]["instruction_text"], source_episode=str(base),
                source_reference_path=episode.get("reference_path"),
                shared_physical_sha256=route["physical_sha256"],
                source_hashes={name: sha256(base / name) for name in ("frames.jsonl", "targets.npz")})
            if "source_metadata" in episode:
                metadata["source_metadata"] = episode["source_metadata"]
            write_json(folder / "episode.json", metadata)
    manifest = json.loads((raw / "manifest.json").read_text())
    manifest.update(episodes=shard["episodes"], selection="all source instructions; identical physical routes rendered once",
                    episode_ids=[str(e["episode_id"]) for r in shard["routes"] for e in r["episodes"]])
    write_json(raw / "manifest.json", manifest)


def worker(plan_path, shard_id, gpu_id):
    import pyarrow as pa

    from data_pipeline.export_lerobot import export_dataset
    from data_pipeline.validate_lerobot import validate_lerobot

    pa.set_cpu_count(2)
    pa.set_io_thread_count(2)
    plan = json.loads(plan_path.read_text())
    shard = plan["shards"][shard_id]
    work = plan_path.parent
    scratch = work / "scratch" / f"{shard_id:05d}"
    scratch.mkdir(parents=True, exist_ok=False)
    raw, output = scratch / "capture", scratch / "lerobot"
    ids = scratch / "episode_ids.json"
    write_json(ids, [r["representative_id"] for r in shard["routes"]])
    env = dict(os.environ, BETA_GPU_ID=str(gpu_id), BETA_DATA_ROOT=plan["data_root"],
               BETA_SCENES_DIR=plan["scenes_dir"], BETA_RVQ_BUNDLE=plan["rvq_bundle"])
    start = time.monotonic()
    subprocess.run(["bash", str(REPO / "data_pipeline/run.sh"), str(raw), "-1",
                    "--episode-ids-file", str(ids), "--pointing-config", plan["pointing_config"],
                    "--dataset", plan.get("dataset", "r2r"), "--split", plan["split"],
                    "--role", plan.get("role", "guide"), "--languages", *plan.get("languages", ["all"])],
                   env=env, check=True)
    captured = time.monotonic()
    expand_instructions(raw, shard)
    export_dataset(raw, output, repo_id=f"{plan['repo_id']}-shard-{shard_id:05d}")
    exported = time.monotonic()
    # Check all frames after removing only this shard's own temporary captures.
    shutil.rmtree(raw)
    report = validate_lerobot(output)
    if report["frames"] != shard["frames"] or len(report["episodes"]) != shard["episodes"]:
        raise ValueError("validated shard does not cover its plan")
    write_json(output / "meta/validation.json", report)
    write_json(output / "meta/build_shard.json", dict(complete=True, plan_sha256=plan["plan_sha256"],
        shard_id=shard_id, episodes=shard["episodes"], frames=shard["frames"], physical_routes=len(shard["routes"]),
        gpu_id=gpu_id, capture_seconds=captured-start, export_seconds=exported-captured,
        validation_seconds=time.monotonic()-exported, elapsed_seconds=time.monotonic()-start))
    ready = work / "shards" / f"{shard_id:05d}"
    output.rename(ready)
    shutil.rmtree(scratch)


def completed_shards(work, plan):
    result = {}
    for shard in plan["shards"]:
        path = work / "shards" / f"{shard['id']:05d}" / "meta/build_shard.json"
        if path.is_file():
            record = json.loads(path.read_text())
            if not record["complete"] or record["plan_sha256"] != plan["plan_sha256"]:
                raise ValueError(f"shard belongs to a different generation plan: {path}")
            result[shard["id"]] = record
    return result


def run_shard(work, plan, shard_id, gpu_id, retries):
    for attempt in range(1, retries + 2):
        scratch = work / "scratch" / f"{shard_id:05d}"
        if scratch.exists():
            # Retain previous failures for diagnosis, never erase unrelated data.
            scratch.rename(work / "failed" / f"{shard_id:05d}-{time.time_ns()}")
        cache = work / "cache" / f"{shard_id:05d}"
        env = dict(os.environ, HF_HUB_OFFLINE="1", HF_DATASETS_CACHE=str(cache),
                   HF_DATASETS_DISABLE_PROGRESS_BARS="1", PYTHONDONTWRITEBYTECODE="1",
                   OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1", NUMEXPR_NUM_THREADS="1")
        log_path = work / "logs" / f"shard-{shard_id:05d}-attempt-{attempt}-{time.time_ns()}.log"
        with log_path.open("w") as log:
            process = subprocess.run([sys.executable, str(Path(__file__).resolve()), "--worker-plan", str(work / "plan.json"),
                                      "--shard-id", str(shard_id), "--gpu-id", str(gpu_id)],
                                     env=env, stdout=log, stderr=subprocess.STDOUT)
        if process.returncode == 0:
            if cache.exists():
                shutil.rmtree(cache)
            return
        print(json.dumps(dict(shard_failed=shard_id, attempt=attempt, log=str(log_path))), flush=True)
    raise RuntimeError(f"shard {shard_id} failed; inspect {log_path}; resume with the same command")


EMPTY_DEPTH_GUARD = ('    if not depth_valid.any():\n'
                     '        raise ValueError("depth sensor has no valid surface pixels")\n')


def compatible_resume_plan(original, current, work):
    """Accept only the audited empty-depth guard removal; never reinterpret labels.

    The original immutable plan and successful provenance remain untouched. The
    sole allowed extension retains all-zero depth with its already-defined false
    mask. Every input previously accepted produces exactly the same values.
    """
    if original["plan_sha256"] == current["plan_sha256"]:
        return original
    ignored = {"plan_sha256", "definition_sha256"}
    if {k: v for k, v in original.items() if k not in ignored} != {k: v for k, v in current.items() if k not in ignored}:
        raise ValueError("resume plan differs: use the original arguments or a new output")
    changed = {p for p in original["definition_sha256"]
               if original["definition_sha256"][p] != current["definition_sha256"].get(p)}
    collector = str(REPO / "data_pipeline/collect_r2r.py")
    if changed != {collector} or original["definition_sha256"].keys() != current["definition_sha256"].keys():
        raise ValueError("GT definition changed; use a new output")
    updates = work / "definition_updates"
    candidates = [updates / "original_collect_r2r.py", *sorted((work / "shards").glob("*/meta/provenance/collect_r2r.py"))]
    frozen = next((p for p in candidates if p.is_file() and sha256(p) == original["definition_sha256"][collector]), None)
    if frozen is None or frozen.read_text().count(EMPTY_DEPTH_GUARD) != 1:
        raise ValueError("cannot prove the original collector's compatibility")
    if frozen.read_text().replace(EMPTY_DEPTH_GUARD, "", 1) != Path(collector).read_text():
        raise ValueError("only the exact empty-depth guard removal is compatible with this plan")
    updates.mkdir(exist_ok=True)
    if not (updates / "original_collect_r2r.py").exists():
        shutil.copyfile(frozen, updates / "original_collect_r2r.py")
        shutil.copyfile(collector, updates / "empty_depth_collect_r2r.py")
        write_json(updates / "empty_depth.json", dict(
            original_plan_sha256=original["plan_sha256"], original_collector_sha256=original["definition_sha256"][collector],
            current_collector_sha256=current["definition_sha256"][collector],
            change="remove only the no-valid-depth rejection; preserve actual RGB/depth, false masks and all poses/GT",
            compatibility="source text differs only by the two-line rejection; outputs on all formerly accepted inputs are unchanged",
            reason="MP3D outdoor scan gaps can yield an empty initial view before expert rotation reveals valid geometry"))
    print("Resuming the immutable plan with the audited empty-depth acceptance extension.", flush=True)
    return original


def run(args):
    plan = create_plan(args)
    overview = {k: plan[k] for k in ("source_episodes", "physical_routes", "scenes", "frames", "rendered_frames", "plan_sha256")}
    print(json.dumps(dict(overview, source_selection=plan.get("source_selection"),
                         shards=len(plan["shards"]), output=str(args.output_dir)), indent=2), flush=True)
    if args.dry_run:
        return
    work = args.output_dir.with_name(args.output_dir.name + ".work")
    work.mkdir(parents=True, exist_ok=True)
    with (work / "build.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        existing = work / "plan.json"
        if existing.is_file():
            plan = compatible_resume_plan(json.loads(existing.read_text()), plan, work)
            overview["plan_sha256"] = plan["plan_sha256"]
        else:
            if args.output_dir.exists():
                raise FileExistsError(args.output_dir)
            write_json(existing, plan)
        if args.output_dir.exists():
            report = json.loads((args.output_dir / "meta/full_train_validation.json").read_text())
            if report["complete"] and report["plan_sha256"] == plan["plan_sha256"]:
                print("Full train dataset is already complete.", flush=True)
                return
            raise FileExistsError(args.output_dir)
        for name in ("shards", "scratch", "failed", "logs", "cache"):
            (work / name).mkdir(exist_ok=True)
        done = completed_shards(work, plan)
        pending = [s["id"] for s in plan["shards"] if s["id"] not in done]
        if args.max_new_shards is not None:
            pending = pending[:args.max_new_shards]
        errors = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(run_shard, work, plan, shard_id, args.gpus[i % len(args.gpus)], args.retries): shard_id
                       for i, shard_id in enumerate(pending)}
            active = set(futures)
            while active:
                finished, active = concurrent.futures.wait(active, timeout=15,
                    return_when=concurrent.futures.FIRST_COMPLETED)
                for future in finished:
                    try:
                        future.result()
                    except Exception as exc:
                        errors.append(dict(shard_id=futures[future], error=repr(exc)))
                done = completed_shards(work, plan)
                status = dict(state="generating", plan_sha256=plan["plan_sha256"], total_shards=len(plan["shards"]),
                    completed_shards=len(done), episodes=sum(s["episodes"] for s in done.values()),
                    frames=sum(s["frames"] for s in done.values()), expected_episodes=plan["source_episodes"],
                    expected_frames=plan["frames"], errors=errors, updated_unix=time.time())
                write_json(work / "status.json", status)
                if finished:
                    print(json.dumps(status), flush=True)
        if errors:
            done = completed_shards(work, plan)
            write_json(work / "status.json", dict(state="failed", plan_sha256=plan["plan_sha256"],
                completed_shards=len(done), total_shards=len(plan["shards"]), errors=errors,
                episodes=sum(s["episodes"] for s in done.values()), frames=sum(s["frames"] for s in done.values()),
                expected_episodes=plan["source_episodes"], expected_frames=plan["frames"], updated_unix=time.time()))
            raise RuntimeError(f"{len(errors)} shards failed; successful shards are retained, resume to retry")
        done = completed_shards(work, plan)
        if len(done) != len(plan["shards"]):
            print("Pilot limit reached; dataset is incomplete. Resume without --max-new-shards.", flush=True)
            return
        write_json(work / "status.json", dict(state="merging", completed_shards=len(done), plan_sha256=plan["plan_sha256"]))
        from data_pipeline.merge_lerobot import merge_train

        merge_train(work, plan, workers=args.workers)
        write_json(work / "status.json", dict(state="complete", **overview, output_dir=str(args.output_dir)))


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--dataset", choices=["r2r", "rxr"], default="r2r")
    parser.add_argument("--split", choices=SPLITS, default="train")
    parser.add_argument("--role", choices=["guide", "follower"], default="guide")
    parser.add_argument("--languages", nargs="+", choices=["all", *LANGUAGES], default=["all"])
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--scenes-dir", type=Path, default=SCENES_ROOT)
    parser.add_argument("--rvq-bundle", type=Path, default=RVQ_ROOT)
    parser.add_argument("--pointing-config", type=Path, default=REPO / "data_pipeline/configs/pointing.json")
    parser.add_argument("--repo-id", help="Local dataset name; defaults to the selected dataset/split/role")
    parser.add_argument("--routes-per-shard", type=int, default=8)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--gpus", type=int, nargs="+", default=[1, 3])
    parser.add_argument("--retries", type=int, default=1)
    parser.add_argument("--max-new-shards", type=int, help="Pilot limit; keep the same full plan for subsequent resume")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--worker-plan", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--shard-id", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--gpu-id", type=int, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.workers < 1 or args.routes_per_shard < 1 or args.retries < 0:
        parser.error("workers/routes-per-shard must be positive and retries nonnegative")
    if args.max_new_shards is not None and args.max_new_shards < 1:
        parser.error("max-new-shards must be positive")
    if args.worker_plan is None:
        if args.output_dir is None:
            parser.error("--output-dir is required")
        args.data_root = args.data_root or DATA_ROOTS[args.dataset]
        args.repo_id = args.repo_id or ("beta-nav/r2r-train" if args.dataset == "r2r"
                                       else f"beta-nav/rxr-{args.split}-{args.role}")
        for key in ("output_dir", "data_root", "scenes_dir", "rvq_bundle", "pointing_config"):
            setattr(args, key, getattr(args, key).resolve())
    return args


if __name__ == "__main__":
    args = arguments()
    if args.worker_plan is not None:
        worker(args.worker_plan, args.shard_id, args.gpu_id)
    else:
        run(args)
