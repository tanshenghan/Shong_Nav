"""Foreground multi-GPU FM+Stop evaluation with persistent workers and dynamic shards."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import signal
import socket
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from lightnav.habitat.stop_report import build_stop_report, check_coverage, load_expected_episodes

REPOSITORY = Path(__file__).resolve().parents[3]
DEFAULT_CHECKPOINT = "/mnt/cfs/Models/beta_nav/r2r_fm_stop_h30_prefix_kv_all36_beta1_1p5/checkpoints/step-00003699"
DEFAULT_DATA = "/mnt/cfs/data/habitat/datasets/R2R_VLNCE_v1-3_preprocessed"
DEFAULT_SCENES = "/mnt/cfs/data/habitat/scene_datasets"


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def gpu_tokens(mask, visible_count, count):
    """Map CUDA-local slots through the inherited mask, including GPU UUIDs."""
    tokens = [x.strip() for x in mask.split(",")] if mask is not None else [str(i) for i in range(visible_count)]
    if (count < 1 or visible_count < count or len(tokens) < count
            or any(not x or x == "-1" for x in tokens) or len(set(tokens)) != len(tokens)):
        raise ValueError(f"need {count} allocated visible GPUs; got count={visible_count}, mask={mask!r}")
    return tokens[:count]


def partition_episodes(episodes, count):
    """Exactly match Habitat's stable scene-sorted contiguous split semantics."""
    if not 1 <= count <= len(episodes):
        raise ValueError("number of shards must be between 1 and the episode count")
    ordered = sorted(episodes, key=lambda e: e["scene_id"])
    size = len(ordered) // count
    return [ordered[i * size:(i + 1) * size if i < count - 1 else len(ordered)] for i in range(count)]


def worker_specs(tokens, workers_per_gpu, base_port):
    """Spread each replica round over allocated GPUs before adding another round."""
    return [dict(worker_id=i, gpu_token=tokens[i % len(tokens)], port=base_port + i)
            for i in range(len(tokens) * workers_per_gpu)]


def stop_process_group(process, timeout=10):
    """Stop only a worker group created by this launcher, including its simulator."""
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    try:
        process.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        pass
    # A worker can exit before a stuck simulator; ensure no group child survives.
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    process.wait()


def start_worker(job_path, job, worker):
    folder = Path(job["output_dir"]) / "workers" / f"worker_{worker['worker_id']}"
    folder.mkdir(parents=True)
    env = os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES=worker["gpu_token"], PYTHONUNBUFFERED="1",
               TOKENIZERS_PARALLELISM="false", OMP_NUM_THREADS=str(job["cpu_threads"]),
               HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    env["PYTHONPATH"] = str(REPOSITORY / "src") + os.pathsep + env.get("PYTHONPATH", "")
    command = [sys.executable, "-u", "-m", "lightnav.habitat.fm_stop_worker", "--job", str(job_path),
               "--worker-id", str(worker["worker_id"])]
    with (folder / "worker.log").open("x") as log:
        process = subprocess.Popen(command, cwd=REPOSITORY, env=env, stdin=subprocess.PIPE,
                                   stdout=log, stderr=subprocess.STDOUT, text=True, start_new_session=True)
    return dict(process=process, worker=worker, folder=folder, active=None, stopping=False, started=time.monotonic())


def send_task(slot, shard):
    slot["process"].stdin.write(json.dumps(None if shard is None else {"shard": shard}) + "\n")
    slot["process"].stdin.flush()
    slot["active"] = shard
    slot["stopping"] = shard is None
    slot["assigned_at"] = time.monotonic()


def run_workers(job_path, job, *, spawn=start_worker, interval=1.):
    """Load a bounded number of models concurrently; dispatch longest shard first."""
    pending = sorted(range(job["num_shards"]), key=lambda i: len(job["shards"][i]), reverse=True)
    waiting, slots, complete = list(job["workers"]), [], set()
    output, last_report = Path(job["output_dir"]), 0.
    old_handlers = {}

    def interrupted(signum, _frame):
        raise InterruptedError(f"evaluation interrupted by signal {signum}")

    try:
        for signum in (signal.SIGTERM, signal.SIGINT):
            old_handlers[signum] = signal.signal(signum, interrupted)
        while waiting or any(not s["stopping"] or s["process"].poll() is None for s in slots):
            booting = 0
            for slot in slots:
                code = slot["process"].poll()
                state_path = slot["folder"] / "status.json"
                state = json.loads(state_path.read_text()) if state_path.exists() else {"status": "starting"}
                if code is not None:
                    if code or not slot["stopping"]:
                        raise RuntimeError(f"worker {slot['worker']['worker_id']} exited {code}: {state}")
                    continue
                if state["status"] == "failed":
                    raise RuntimeError(f"worker failed: {state}")
                if slot["stopping"]:
                    if time.monotonic() - slot["assigned_at"] > job["shutdown_timeout"]:
                        raise TimeoutError(f"worker shutdown timed out: {slot['folder'] / 'worker.log'}")
                    continue
                if state["status"] == "starting":
                    booting += 1
                    if time.monotonic() - slot["started"] > job["load_timeout"]:
                        raise TimeoutError(f"model load timed out: {slot['folder'] / 'worker.log'}")
                finished = state["status"] == "completed" and state.get("shard") == slot["active"]
                if (slot["active"] is not None and not finished
                        and time.monotonic() - slot["assigned_at"] > job["shard_timeout"]):
                    raise TimeoutError(f"shard {slot['active']} timed out: {slot['folder'] / 'worker.log'}")
                if finished:
                    complete.add(slot["active"])
                    slot["active"] = None
                if slot["active"] is None and (finished or state["status"] == "ready"):
                    send_task(slot, pending.pop(0) if pending else None)
            while waiting and booting < job["load_parallelism"]:
                slots.append(spawn(job_path, job, waiting.pop(0)))
                booting += 1
            if time.monotonic() - last_report >= 30:
                progress = dict(completed_shards=len(complete), total_shards=job["num_shards"],
                                active_workers=sum(s["active"] is not None for s in slots),
                                starting_workers=booting, pending_shards=len(pending))
                write_json(output / "progress.json", progress)
                print(json.dumps(progress), flush=True)
                last_report = time.monotonic()
            time.sleep(interval)
        if complete != set(range(job["num_shards"])):
            raise RuntimeError("worker pool exited with incomplete shards")
        if any(slot["process"].returncode != 0 for slot in slots):
            raise RuntimeError("worker exited with an error during shutdown")
        write_json(output / "progress.json", dict(completed_shards=len(complete), total_shards=len(complete)))
    finally:
        # Broadcast termination first so cleanup has one shared deadline, not N
        # sequential grace periods. Repeated platform signals must not strand GPUs.
        for signum in old_handlers:
            signal.signal(signum, signal.SIG_IGN)
        try:
            for slot in slots:
                try:
                    os.killpg(slot["process"].pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
            deadline = time.monotonic() + 10
            for slot in slots:
                stop_process_group(slot["process"], timeout=max(0, deadline - time.monotonic()))
                if slot["process"].stdin is not None:
                    slot["process"].stdin.close()
        finally:
            for signum, handler in old_handlers.items():
                signal.signal(signum, handler)


def merge_completed(job, elapsed):
    """Reject partial/corrupt/mixed-protocol outputs before publishing global metrics."""
    from lightnav.habitat.merge import merge_results

    output, records, folders = Path(job["output_dir"]), [], []
    expected_protocol = dict(beta_checkpoint=job["checkpoint"], beta_checkpoint_step=job["checkpoint_step"],
                             training_mode="fm_stop", stop_protocol="learned", stop_threshold=job["stop_threshold"],
                             fm_seed=job["fm_seed"], fm_steps=job["fm_steps"], max_steps=job["max_steps"])
    expected_protocol["stop_confirm_steps"] = job.get("stop_confirm_steps", 1)
    for index, expected in enumerate(job["shards"]):
        folder = output / f"shard_{index:04d}"
        summary = json.loads((folder / "summary.json").read_text())
        for key, value in expected_protocol.items():
            actual = summary.get(key, 1 if key == "stop_confirm_steps" else None)
            if actual != value:
                raise ValueError(f"shard {index} protocol mismatch for {key}: {summary.get(key)!r}")
        shard_records = [json.loads(line) for line in (folder / "results.jsonl").read_text().splitlines() if line.strip()]
        if any(row.get("oracle_stop") for row in shard_records):
            raise ValueError("privileged oracle STOP is not permitted in this evaluation")
        coverage = check_coverage(expected, shard_records)
        if not coverage["complete"] or summary.get("num_episodes") != len(shard_records):
            write_json(output / "coverage.json", dict(failed_shard=index, **coverage))
            raise ValueError(f"shard {index} coverage is incomplete or duplicated")
        records.extend(shard_records)
        folders.append(folder)
    coverage = check_coverage(job["episodes"], records)
    write_json(output / "coverage.json", coverage)
    if not coverage["complete"]:
        raise ValueError("global episode coverage is not exact")
    diagnostics = build_stop_report(records, success_distance=job["success_distance"])
    # Dynamic shards run sequentially within workers; max(shard time) is not wall time.
    summary = merge_results(folders, output, extra_info={**expected_protocol, "parallelism": dict(
        num_gpus=job["num_gpus"], workers_per_gpu=job["workers_per_gpu"], num_shards=job["num_shards"]),
        "total_time_sec": round(elapsed, 3), "episodes_per_hour": round(len(records) * 3600 / elapsed, 3)})
    write_json(output / "stop_diagnostics.json", diagnostics)
    return summary


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--model-path", default="/mnt/cfs/Models/Light")
    parser.add_argument("--dataset-root", default=DEFAULT_DATA)
    parser.add_argument("--split", choices=("val_seen", "val_unseen"), default="val_unseen")
    parser.add_argument("--scenes-dir", default=DEFAULT_SCENES)
    parser.add_argument("--habitat-python", default=str(REPOSITORY / ".venv-habitat/bin/python"))
    parser.add_argument("--habitat-config", default=str(REPOSITORY / "habitat_server/configs/vlnce_r2r.yaml"))
    parser.add_argument("--num-gpus", type=int, default=8)
    parser.add_argument("--workers-per-gpu", type=int, choices=range(1, 7), default=6)
    # For 1,839 episodes: 96 shards give 95*19 + 34. With 192 shards,
    # Habitat's remainder rule would leave one long shard of 120 episodes.
    parser.add_argument("--shards-per-worker", type=int, default=2)
    parser.add_argument("--base-port", type=int, default=28600)
    parser.add_argument("--cpu-threads", type=int, default=2)
    parser.add_argument("--ready-timeout", type=int, default=900)
    parser.add_argument("--load-timeout", type=int, default=1200)
    parser.add_argument("--load-parallelism", type=int, default=2)
    parser.add_argument("--shard-timeout", type=int, default=21600, help="maximum seconds for one shard")
    parser.add_argument("--shutdown-timeout", type=int, default=30, help="worker shutdown grace period in seconds")
    parser.add_argument("--stop-threshold", type=float, default=0.5)
    parser.add_argument("--stop-confirm-steps", type=int, default=1)
    parser.add_argument("--fm-steps", type=int, default=10)
    parser.add_argument("--fm-seed", type=int, default=42)
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--check-only", action="store_true", help="CPU resource/import preflight; no GPU/model/simulator launch")
    args = parser.parse_args(argv)
    for name in ("num_gpus", "shards_per_worker", "cpu_threads", "ready_timeout", "load_timeout",
                 "load_parallelism", "fm_steps", "shard_timeout", "shutdown_timeout", "stop_confirm_steps"):
        if getattr(args, name) < 1:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if not 1 <= args.max_steps <= 500 or args.fm_seed < 0:
        parser.error("max-steps must be in [1,500], fm-seed must be nonnegative")
    if not math.isfinite(args.stop_threshold) or not 0 <= args.stop_threshold <= 1:
        parser.error("stop-threshold must be finite and in [0,1]")
    worker_count = args.num_gpus * args.workers_per_gpu
    if args.base_port < 1024 or args.base_port + worker_count > 65536:
        parser.error("worker ports must be between 1024 and 65535")
    return args


def main(argv=None):
    args = parse_args(argv)
    from lightnav.habitat.eval_preflight import (
        build_habitat_env,
        check_habitat_python,
        validate_resources,
    )
    from lightnav.inference.beta_model import resolve_beta_checkpoint

    worker_count = args.num_gpus * args.workers_per_gpu
    checkpoint, metadata = resolve_beta_checkpoint(args.checkpoint)
    beta = metadata["beta_config"]
    if beta["training_mode"] != "fm_stop" or beta.get("stop_head_type") != "last_prefix_mlp_v1":
        raise ValueError("evaluation requires a trained fm_stop checkpoint with learned StopHead")
    for name in ("model_path", "dataset_root", "scenes_dir", "habitat_config"):
        setattr(args, name, str(Path(getattr(args, name)).resolve()))
    # Resolving a venv executable symlink can silently drop its site-packages.
    args.habitat_python = str(Path(args.habitat_python).absolute())
    if not Path(args.habitat_config).is_file() or not Path(args.model_path, "config.json").is_file():
        raise FileNotFoundError("Habitat config or original model config is unavailable")
    try:
        episodes = load_expected_episodes(args.dataset_root, args.split)
        resources = validate_resources(args.dataset_root, args.scenes_dir, episodes, split=args.split)
        development_audit = None
        if args.split == "val_seen":
            from lightnav.habitat.dev_split import verify_dev_split

            development_audit = verify_dev_split(
                args.dataset_root, episodes, split=args.split, checkpoint_metadata=metadata,
            )
        episode_file = Path(args.dataset_root) / args.split / f"{args.split}.json.gz"
        episode_file_sha256 = hashlib.sha256(episode_file.read_bytes()).hexdigest()
    except PermissionError as exc:
        raise PermissionError("The platform job account must be able to read the R2R/MP3D dataset and scenes: "
                              f"{args.dataset_root}; {args.scenes_dir}. Original error: {exc}") from exc
    environment = check_habitat_python(args.habitat_python, build_habitat_env(REPOSITORY))
    shards = partition_episodes(episodes, worker_count * args.shards_per_worker)
    check = dict(checkpoint=str(checkpoint), checkpoint_step=metadata.get("step"), num_episodes=len(episodes),
                 num_gpus=args.num_gpus, workers_per_gpu=args.workers_per_gpu, num_shards=len(shards),
                 split=args.split, stop_confirm_steps=args.stop_confirm_steps,
                 checkpoint_metadata_sha256=hashlib.sha256((checkpoint / "metadata.json").read_bytes()).hexdigest(),
                 episode_file_sha256=episode_file_sha256, development_audit=development_audit,
                 resources=resources, habitat_environment=environment)
    if args.check_only:
        print(json.dumps(check, indent=2), flush=True)
        return check

    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; allocate eight GPUs through the platform before launching")
    tokens = gpu_tokens(os.environ.get("CUDA_VISIBLE_DEVICES"), torch.cuda.device_count(), args.num_gpus)
    memory = [torch.cuda.get_device_properties(i).total_memory / 2**30 for i in range(args.num_gpus)]
    if args.workers_per_gpu >= 2 and min(memory) < 80:
        raise ValueError("multiple model workers per GPU require >=80 GiB/card; use --workers-per-gpu 1. "
                         "This minimum does not guarantee capacity for all requested replicas.")
    for index in range(args.num_gpus):
        with torch.cuda.device(index):
            if not torch.cuda.is_bf16_supported():
                raise RuntimeError(f"visible GPU {index} lacks BF16 support")
    reservations = []
    try:
        for port in range(args.base_port, args.base_port + worker_count):
            sock = socket.socket()
            reservations.append(sock)
            sock.bind(("127.0.0.1", port))
    finally:
        for sock in reservations:
            sock.close()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    output = (args.output_dir or REPOSITORY / "outputs/betanav_eval" / f"{stamp}_fm_stop_r2r_{args.split}").resolve()
    output.mkdir(parents=True, exist_ok=False)
    def minimal(rows):
        return [dict(scene_id=e["scene_id"], episode_id=e["episode_id"]) for e in rows]
    job = {**vars(args), **check, "checkpoint": str(checkpoint), "output_dir": str(output),
           "repository": str(REPOSITORY), "success_distance": 3.0,
           "episodes": minimal(episodes), "shards": [minimal(s) for s in shards],
           "checkpoint_metadata_sha256": hashlib.sha256((checkpoint / "metadata.json").read_bytes()).hexdigest(),
           "visible_gpu_memory_gib": memory,
           "workers": worker_specs(tokens, args.workers_per_gpu, args.base_port)}
    job_path = output / "job.json"
    write_json(job_path, job)
    print(json.dumps({k: v for k, v in job.items() if k not in {"episodes", "shards", "resources"}}, indent=2), flush=True)
    started = time.monotonic()
    write_json(output / "status.json", dict(status="running", job=str(job_path), pid=os.getpid()))
    try:
        run_workers(job_path, job)
        summary = merge_completed(job, time.monotonic() - started)
        write_json(output / "status.json", dict(status="completed", elapsed_seconds=time.monotonic() - started,
                                                episodes=summary["num_episodes"], summary=str(output / "summary.json")))
    except BaseException as exc:
        write_json(output / "status.json", dict(status="failed", error=repr(exc), elapsed_seconds=time.monotonic() - started))
        raise
    print(f"Evaluation complete: {output / 'summary.json'}", flush=True)
    return summary


if __name__ == "__main__":
    main()
