"""The delayed launch must never overlap training or dispatch twice."""

import hashlib
import subprocess
from pathlib import Path

import pytest

from lightnav.training import scheduled_launch as queue


@pytest.fixture
def schedule(tmp_path):
    previous = tmp_path / "previous"
    previous.mkdir()
    queue.write_json(previous / "launch.json", {"status": "completed"})
    return {
        "not_before_utc": "2026-09-17T17:08:44+00:00",
        "wait_for_launch_dir": str(previous),
        "gpu_indices": list(range(8)),
    }


def mock_idle(monkeypatch, apps="", memory=None):
    monkeypatch.setattr(queue, "training_pids", lambda: [])

    def query(command, **kwargs):
        if "--query-compute-apps=pid" in command:
            return apps
        return memory if memory is not None else "\n".join(f"{i}, 0" for i in range(8))

    monkeypatch.setattr(queue.subprocess, "check_output", query)


def test_deadline_requires_timezone(schedule):
    schedule["not_before_utc"] = "2026-09-17T17:08:44"
    with pytest.raises(ValueError, match="timezone"):
        queue.deadline(schedule)


def test_future_deadline_does_not_check_devices_or_previous_run(schedule, monkeypatch):
    schedule["wait_for_launch_dir"] = "/does/not/exist"

    def unexpected():
        pytest.fail("checked processes before the deadline")

    monkeypatch.setattr(queue, "training_pids", unexpected)
    status, details = queue.readiness(schedule, now=queue.deadline(schedule) - 60)
    assert status == "waiting_for_time"
    assert details["remaining_seconds"] == 60


def test_running_predecessor_blocks_launch(schedule):
    queue.write_json(Path(schedule["wait_for_launch_dir"]) / "launch.json", {"status": "running"})
    assert queue.readiness(schedule, now=queue.deadline(schedule))[0] == "waiting_for_previous_run"


@pytest.mark.parametrize("case", ["trainer", "compute", "memory", "missing_gpu", "driver_error"])
def test_busy_or_unavailable_gpu_blocks_launch(schedule, monkeypatch, case):
    mock_idle(monkeypatch)
    if case == "trainer":
        monkeypatch.setattr(queue, "training_pids", lambda: [1234])
    elif case == "compute":
        mock_idle(monkeypatch, apps="1234\n")
    elif case == "memory":
        mock_idle(monkeypatch, memory="\n".join(f"{i}, 2048" for i in range(8)))
    elif case == "missing_gpu":
        mock_idle(monkeypatch, memory="\n".join(f"{i}, 0" for i in range(7)))
    else:
        def failed_query(*args, **kwargs):
            raise subprocess.CalledProcessError(1, "nvidia-smi")

        monkeypatch.setattr(queue.subprocess, "check_output", failed_query)
    assert queue.readiness(schedule, now=queue.deadline(schedule))[0] == "waiting_for_gpus"


def test_completed_predecessor_and_all_devices_idle(schedule, monkeypatch):
    mock_idle(monkeypatch)
    assert queue.readiness(schedule, now=queue.deadline(schedule)) == ("ready", {})


def test_process_scan_matches_arguments_not_embedded_shell_text(tmp_path):
    for pid, args in {
        10: b"python\0/repo/train_betanav.py\0",
        11: b"python\0-m\0lightnav.training.profile_batch\0",
        12: b"bash\0-c\0cat /repo/train_betanav.py\0",
        13: b"python\0unrelated.py\0",
    }.items():
        proc = tmp_path / str(pid)
        proc.mkdir()
        (proc / "cmdline").write_bytes(args)
    assert set(queue.training_pids(tmp_path)) == {10, 11}


def prepare_dummy_launch(tmp_path):
    supervisor = tmp_path / "supervisor.py"
    supervisor.write_text(
        "from pathlib import Path\n"
        "import json\n"
        "root = Path(__file__).parent\n"
        "with (root / 'dispatches').open('a') as f: f.write('launched\\n')\n"
        "(root / 'launch.json').write_text(json.dumps({'status': 'completed'}))\n"
    )
    queue.write_json(tmp_path / "launch.json", {"status": "prepared"})
    path = tmp_path / "schedule.json"
    queue.write_json(path, {
        "not_before_utc": "2026-09-17T17:08:44+00:00",
        "repository": str(tmp_path),
        "launch_dir": str(tmp_path),
        "gpu_indices": list(range(8)),
        "prepared_sha256": {str(supervisor): hashlib.sha256(supervisor.read_bytes()).hexdigest()},
    })
    return path


def test_wait_then_launch_once_with_real_dummy_supervisor(tmp_path, monkeypatch):
    path = prepare_dummy_launch(tmp_path)
    states = iter([
        ("waiting_for_time", {"remaining_seconds": 10}),
        ("waiting_for_previous_run", {}),
        ("waiting_for_gpus", {}),
        ("ready", {}),
    ])
    monkeypatch.setattr(queue, "readiness", lambda _: next(states))
    sleeps = []
    monkeypatch.setattr(queue.time, "sleep", sleeps.append)
    assert queue.run(path)["status"] == "completed"
    assert queue.run(path)["status"] == "completed"
    assert sleeps == [10, 30, 30]
    assert (tmp_path / "dispatches").read_text() == "launched\n"


def test_changed_prepared_file_fails_without_dispatch(tmp_path):
    path = prepare_dummy_launch(tmp_path)
    (tmp_path / "supervisor.py").write_text("raise RuntimeError('changed')\n")
    with pytest.raises(ValueError, match="prepared launch file changed"):
        queue.run(path)
    assert queue.read_json(tmp_path / "schedule_state.json")["status"] == "failed"
    assert not (tmp_path / "dispatches").exists()


@pytest.mark.parametrize("status", ["launching", "launched", "cancelled"])
def test_existing_dispatch_intent_never_launches_again(tmp_path, status):
    path = prepare_dummy_launch(tmp_path)
    queue.write_json(tmp_path / "schedule_state.json", {"status": status})
    assert queue.run(path)["status"] == status
    assert not (tmp_path / "dispatches").exists()
