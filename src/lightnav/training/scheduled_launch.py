"""One-shot server-side launch queue for an already prepared BetaNav supervisor.

This process must stay alive until dispatch. It does not install a system service
or restart a container. A detached training supervisor records the training result.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, value):
    path = Path(path)
    temp = path.with_suffix(path.suffix + ".tmp")
    with temp.open("w") as stream:
        json.dump(value, stream, indent=2)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    temp.replace(path)


def deadline(schedule):
    timestamp = datetime.fromisoformat(schedule["not_before_utc"].replace("Z", "+00:00"))
    if timestamp.tzinfo is None:
        raise ValueError("not_before_utc requires an explicit timezone")
    return timestamp.timestamp()


def training_pids(proc_root=Path("/proc")):
    result = []
    for proc in proc_root.iterdir():
        if not proc.name.isdecimal():
            continue
        try:
            args = (proc / "cmdline").read_bytes().split(b"\0")
        except OSError:
            continue
        if any(arg == b"train_betanav.py" or (arg.startswith(b"/") and arg.endswith(b"/train_betanav.py"))
               or arg == b"lightnav.training.profile_batch" for arg in args):
            result.append(int(proc.name))
    return result


def readiness(schedule, now=None):
    now = time.time() if now is None else now
    if now < deadline(schedule):
        return "waiting_for_time", {"remaining_seconds": deadline(schedule) - now}
    previous = schedule.get("wait_for_launch_dir")
    if previous:
        meta = read_json(Path(previous) / "launch.json")
        if meta.get("status") not in {"completed", "failed", "interrupted", "cancelled"}:
            return "waiting_for_previous_run", {"previous_status": meta.get("status")}
    pids = training_pids()
    if pids:
        return "waiting_for_gpus", {"training_pids": pids}
    try:
        apps = subprocess.check_output(
            ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"],
            text=True, stderr=subprocess.STDOUT, timeout=15).strip()
        if apps:
            return "waiting_for_gpus", {"gpu_compute_pids": apps.splitlines()}
        memory = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index,memory.used", "--format=csv,noheader,nounits"],
            text=True, stderr=subprocess.STDOUT, timeout=15)
        devices = {int(row.split(",")[0]): int(row.split(",")[1]) for row in memory.splitlines() if row.strip()}
    except (OSError, ValueError, subprocess.SubprocessError) as exc:
        return "waiting_for_gpus", {"gpu_check_error": str(exc)}
    wanted = schedule["gpu_indices"]
    if any(index not in devices or devices[index] > 1024 for index in wanted):
        return "waiting_for_gpus", {"memory_used_mib": devices, "required_devices": wanted}
    return "ready", {}


def verify_prepared_files(schedule):
    for name, expected in schedule["prepared_sha256"].items():
        if hashlib.sha256(Path(name).read_bytes()).hexdigest() != expected:
            raise ValueError(f"prepared launch file changed: {name}")


def run(schedule_path):
    schedule_path = Path(schedule_path).resolve()
    schedule = read_json(schedule_path)
    deadline(schedule)
    interval = schedule.get("poll_seconds", 30)
    if not 1 <= interval <= 60:
        raise ValueError("poll_seconds must be between 1 and 60")
    launch = Path(schedule["launch_dir"])
    state_path = schedule_path.with_name("schedule_state.json")
    with schedule_path.with_suffix(".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        state = read_json(state_path) if state_path.exists() else {}
        # A second invocation must never repeat a launch, including an ambiguous
        # crash between recording dispatch intent and creating the subprocess.
        if state.get("status") in {"launching", "launched", "completed", "failed", "cancelled"}:
            return state
        state.update(queue_pid=os.getpid(), not_before_utc=schedule["not_before_utc"],
                     launch_dir=str(launch))

        def record(status, **details):
            state.update(status=status, checked_at_utc=datetime.now(timezone.utc).isoformat(), details=details)
            write_json(state_path, state)

        try:
            verify_prepared_files(schedule)
            if read_json(launch / "launch.json").get("status") != "prepared":
                raise ValueError("launch is no longer in the prepared state")
            while True:
                status, details = readiness(schedule)
                record(status, **details)
                if status == "ready":
                    break
                if status == "waiting_for_time":
                    time.sleep(min(interval, details["remaining_seconds"]))
                else:
                    time.sleep(interval)
            verify_prepared_files(schedule)
            record("launching")
            with (launch / "supervisor.log").open("x") as log:
                child = subprocess.Popen(
                    [sys.executable, "-u", str(launch / "supervisor.py")],
                    cwd=schedule["repository"], stdin=subprocess.DEVNULL,
                    stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            state["supervisor_pid"] = child.pid
            state["dispatched_at_utc"] = datetime.now(timezone.utc).isoformat()
            record("launched")
            code = child.wait()
            outcome = read_json(launch / "launch.json")
            record("completed" if code == 0 and outcome.get("status") == "completed" else "failed",
                   supervisor_exit_code=code, launch_status=outcome.get("status"))
        except Exception as exc:
            record("failed", error=repr(exc))
            raise
        return state


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--schedule", type=Path, required=True, help="prepared schedule JSON")
    parser.add_argument("--check", action="store_true", help="read-only readiness check; never launches")
    args = parser.parse_args()
    if args.check:
        schedule = read_json(args.schedule)
        verify_prepared_files(schedule)
        status, details = readiness(schedule)
        print(json.dumps({"status": status, **details}, indent=2))
    else:
        run(args.schedule)


if __name__ == "__main__":
    main()
