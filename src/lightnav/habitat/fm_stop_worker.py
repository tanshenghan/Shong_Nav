"""Persistent single-GPU FM + StopHead evaluation worker.

The parent assigns one shard at a time through JSON lines on stdin. The VLM and
expert stay loaded while a fresh Habitat server handles each assigned shard.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Callable, TextIO

from lightnav.habitat import runner


class WorkerInterrupted(BaseException):
    """Shutdown that escapes the runner's interactive KeyboardInterrupt handler."""


def write_status(path: Path, *, worker_id: int, state: str, shard: int | None,
                 finished_count: int, error: str | None = None) -> None:
    payload = {
        "status": state,
        "worker_id": worker_id,
        "shard": shard,
        "finished_count": finished_count,
        "updated_at": time.time(),
        "pid": os.getpid(),
    }
    if error is not None:
        payload["error"] = error
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    os.replace(temporary, path)


def configure_device(cpu_threads: int) -> None:
    """Use logical device zero within the parent's single-GPU visibility mask."""
    import torch

    if cpu_threads <= 0:
        raise ValueError("cpu_threads must be positive")
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if not visible or "," in visible:
        raise ValueError("worker requires exactly one CUDA_VISIBLE_DEVICES token")
    torch.set_num_threads(cpu_threads)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable in the FM + StopHead worker")
    torch.cuda.set_device(0)


def build_eval_config(job: dict[str, Any], worker: dict[str, Any], output: Path
                      ) -> runner.HabitatEvalConfig:
    return runner.HabitatEvalConfig(
        model_path=job["model_path"],
        server=f"tcp://127.0.0.1:{worker['port']}",
        backend="hf",
        action_output="fm",
        beta_checkpoint=job["checkpoint"],
        fm_only_stop_mode="learned",
        stop_threshold=float(job["stop_threshold"]),
        stop_confirm_steps=int(job.get("stop_confirm_steps", 1)),
        fm_steps=int(job["fm_steps"]),
        fm_seed=int(job["fm_seed"]),
        fm_stop_mode="baseline",
        episodes=-1,
        oracle_stop_distance=None,
        max_steps=int(job["max_steps"]),
        force_stop_at_max_steps=True,
        output_dir=str(output),
        save_video=False,
    )


def server_command(job: dict[str, Any], worker: dict[str, Any], shard: int,
                   ready_file: Path) -> list[str]:
    return [
        job["habitat_python"], "-m", "lightnav_habitat.serve",
        "--task", "vlnce",
        "--config", job["habitat_config"],
        "--port", str(worker["port"]),
        "--data-path", job["dataset_root"],
        "--scenes-dir", job["scenes_dir"],
        "--split", job["split"],
        "--split-id", str(shard),
        "--split-num", str(job["num_shards"]),
        "--max-steps", str(job["max_steps"]),
        "--success-distance", str(job["success_distance"]),
        "--ready-file", str(ready_file),
    ]


def wait_for_server(process: subprocess.Popen, ready_file: Path, timeout: float) -> None:
    """Wait for simulator readiness, failing immediately if its own process exits."""
    deadline = time.monotonic() + timeout
    while True:
        code = process.poll()
        if code is not None:
            raise RuntimeError(f"Habitat server exited before readiness (exit={code}); see server.log")
        if ready_file.is_file():
            return
        if time.monotonic() >= deadline:
            raise TimeoutError(f"Habitat server was not ready within {timeout:g}s; see server.log")
        time.sleep(min(0.2, max(0.0, deadline - time.monotonic())))


def terminate_server(process: subprocess.Popen, timeout: float = 10.0) -> None:
    """Reap only the server this worker created, escalating after a bounded wait."""
    if process.poll() is None:
        try:
            process.terminate()
        except ProcessLookupError:
            pass
    try:
        process.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        try:
            process.kill()
        except ProcessLookupError:
            pass
        process.wait(timeout=timeout)


def run_shard(job: dict[str, Any], worker: dict[str, Any], shard: int,
              cached_engine: tuple[Any, Any], *,
              config_factory: Callable[
                  [dict[str, Any], dict[str, Any], Path], runner.HabitatEvalConfig
              ] = build_eval_config) -> None:
    from lightnav.habitat.eval_preflight import build_habitat_env

    output = Path(job["output_dir"]) / f"shard_{shard:04d}"
    # Existing output can contain partial results. Never append a duplicate run.
    output.mkdir(parents=True, exist_ok=False)
    ready_file = output / ".ready"
    env = build_habitat_env(Path(job["repository"]))
    env["HABITAT_SIM_GPU_ID"] = "0"
    env["CUDA_VISIBLE_DEVICES"] = os.environ["CUDA_VISIBLE_DEVICES"]
    process = None
    with (output / "server.log").open("w") as log:
        try:
            process = subprocess.Popen(
                server_command(job, worker, shard, ready_file),
                cwd=job["repository"],
                env=env,
                stdin=subprocess.DEVNULL,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=False,
            )
            wait_for_server(process, ready_file, float(job["ready_timeout"]))
            runner.run_habitat_eval(
                config_factory(job, worker, output),
                engine_factory=lambda _cfg: cached_engine,
            )
            code = process.poll()
            # The runner closes its remote env in finally, asking the server to
            # exit normally. It may already have finished by this poll.
            if code not in (None, 0):
                raise RuntimeError(f"Habitat server exited during evaluation (exit={code}); see server.log")
        finally:
            if process is not None:
                terminate_server(process)


def run_worker(job: dict[str, Any], worker_id: int, *, input_stream: TextIO | None = None
               ) -> int:
    stream = sys.stdin if input_stream is None else input_stream
    worker_dir = Path(job["output_dir"]) / "workers" / f"worker_{worker_id}"
    status_path = worker_dir / "status.json"
    shard = None
    finished_count = 0
    seen: set[int] = set()

    def status(state: str, error: str | None = None) -> None:
        write_status(status_path, worker_id=worker_id, state=state, shard=shard,
                     finished_count=finished_count, error=error)

    previous_handlers = {}

    def interrupted(signum, _frame):
        # Repeated termination requests must not interrupt server cleanup.
        for sig in (signal.SIGINT, signal.SIGTERM):
            signal.signal(sig, signal.SIG_IGN)
        raise WorkerInterrupted(f"worker received signal {signum}")

    try:
        for sig in (signal.SIGINT, signal.SIGTERM):
            previous_handlers[sig] = signal.signal(sig, interrupted)
        status("starting")
        matches = [worker for worker in job["workers"] if worker["worker_id"] == worker_id]
        if len(matches) != 1:
            raise ValueError(f"expected exactly one worker entry for worker_id={worker_id}")
        worker = matches[0]
        configure_device(int(job["cpu_threads"]))
        cached_engine = runner._default_engine_factory(build_eval_config(job, worker, worker_dir))
        status("ready")
        for line in stream:
            if not line.strip():
                continue
            assignment = json.loads(line)
            if assignment is None:
                break
            if not isinstance(assignment, dict) or set(assignment) != {"shard"}:
                raise ValueError("worker assignment must contain exactly one 'shard' field")
            candidate = assignment["shard"]
            if type(candidate) is not int or not 0 <= candidate < int(job["num_shards"]):
                raise ValueError(f"invalid shard assignment: {candidate!r}")
            if candidate in seen:
                raise ValueError(f"duplicate shard assignment: {candidate}")
            shard = candidate
            seen.add(shard)
            status("running")
            run_shard(job, worker, shard, cached_engine)
            finished_count += 1
            status("completed")
        status("stopped")
        return 0
    except WorkerInterrupted as exc:
        status("stopped", error=str(exc))
        return 130
    except Exception as exc:
        traceback.print_exc()
        status("failed", error=f"{type(exc).__name__}: {exc}")
        return 1
    finally:
        for sig, handler in previous_handlers.items():
            signal.signal(sig, handler)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job", required=True, type=Path)
    parser.add_argument("--worker-id", required=True, type=int)
    args = parser.parse_args(argv)
    job = json.loads(args.job.read_text())
    return run_worker(job, args.worker_id)


if __name__ == "__main__":
    raise SystemExit(main())
