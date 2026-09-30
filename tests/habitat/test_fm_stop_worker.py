"""Persistent-worker lifecycle and process ownership without CUDA or Habitat."""

from __future__ import annotations

import io
import json
import signal
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from lightnav.habitat import fm_stop_worker as worker
from lightnav.habitat.runner import run_habitat_eval as real_run_habitat_eval


@pytest.fixture
def job(tmp_path):
    return {
        "repository": str(tmp_path / "repo"),
        "output_dir": str(tmp_path / "output"),
        "checkpoint": str(tmp_path / "checkpoint"),
        "model_path": str(tmp_path / "model"),
        "habitat_python": str(tmp_path / "habitat-python"),
        "habitat_config": str(tmp_path / "vlnce.yaml"),
        "dataset_root": str(tmp_path / "dataset"),
        "scenes_dir": str(tmp_path / "scenes"),
        "split": "val_unseen",
        "num_shards": 16,
        "max_steps": 500,
        "stop_threshold": 0.5,
        "fm_steps": 10,
        "fm_seed": 42,
        "success_distance": 3.0,
        "ready_timeout": 900,
        "cpu_threads": 2,
        "workers": [{"worker_id": 0, "gpu_token": "GPU-test", "port": 15555}],
    }


class FakeProcess:
    def __init__(self, code=None):
        self.code = code
        self.terminated = 0
        self.killed = 0
        self.waits = []

    def poll(self):
        return self.code

    def terminate(self):
        self.terminated += 1
        self.code = -15

    def kill(self):
        self.killed += 1
        self.code = -9

    def wait(self, timeout):
        self.waits.append(timeout)
        return self.code


@pytest.fixture
def fake_runtime(monkeypatch):
    calls = SimpleNamespace(loads=[], evaluations=[], processes=[], statuses=[], threads=[])
    cached = (object(), object())

    def make_engine(cfg):
        calls.loads.append(cfg)
        return cached

    def evaluate(cfg, *, engine_factory):
        calls.evaluations.append((cfg, engine_factory(cfg)))
        return []

    def popen(command, **kwargs):
        ready = Path(command[command.index("--ready-file") + 1])
        ready.touch()
        process = FakeProcess()
        calls.processes.append((process, command, kwargs))
        return process

    original_status = worker.write_status

    def status(path, **kwargs):
        calls.statuses.append(dict(kwargs))
        original_status(path, **kwargs)

    monkeypatch.setattr(worker, "configure_device", lambda count: calls.threads.append(count))
    monkeypatch.setattr(worker.runner, "_default_engine_factory", make_engine)
    monkeypatch.setattr(worker.runner, "run_habitat_eval", evaluate)
    monkeypatch.setattr(worker.subprocess, "Popen", popen)
    monkeypatch.setattr(worker, "write_status", status)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "GPU-test")
    monkeypatch.setitem(sys.modules, "lightnav.habitat.eval_preflight", SimpleNamespace(
        build_habitat_env=lambda repo: {"TEST_REPO": str(repo)},
    ))
    calls.cached = cached
    return calls


def read_status(job):
    path = Path(job["output_dir"]) / "workers" / "worker_0" / "status.json"
    return json.loads(path.read_text())


def test_eval_config_uses_learned_stop_without_oracle(job, tmp_path):
    cfg = worker.build_eval_config(job, job["workers"][0], tmp_path)
    assert cfg.model_path == job["model_path"]
    assert cfg.beta_checkpoint == job["checkpoint"]
    assert cfg.backend == "hf" and cfg.action_output == "fm"
    assert cfg.fm_only_stop_mode == "learned" and cfg.fm_stop_mode == "baseline"
    assert cfg.oracle_stop_distance is None and cfg.force_stop_at_max_steps
    assert cfg.max_steps == 500 and cfg.stop_threshold == 0.5
    assert cfg.fm_steps == 10 and cfg.fm_seed == 42
    assert cfg.episodes == -1 and not cfg.save_video
    assert cfg.server == "tcp://127.0.0.1:15555"


def test_two_shards_load_model_once_and_clean_up_servers(job, fake_runtime):
    result = worker.run_worker(job, 0, input_stream=io.StringIO('{"shard": 3}\n{"shard": 7}\nnull\n'))
    assert result == 0
    assert len(fake_runtime.loads) == 1
    assert fake_runtime.threads == [2]
    assert len(fake_runtime.evaluations) == 2
    for index, shard in enumerate((3, 7)):
        cfg, cached = fake_runtime.evaluations[index]
        assert cached is fake_runtime.cached
        assert cfg.output_dir.endswith(f"shard_{shard:04d}")
        process, command, kwargs = fake_runtime.processes[index]
        assert process.terminated == 1 and process.killed == 0
        assert process.waits == [10.0]
        assert command[command.index("--split-id") + 1] == str(shard)
        assert command[command.index("--split-num") + 1] == "16"
        assert command[command.index("--data-path") + 1] == job["dataset_root"]
        assert kwargs["env"]["HABITAT_SIM_GPU_ID"] == "0"
        assert kwargs["env"]["CUDA_VISIBLE_DEVICES"] == "GPU-test"
        assert kwargs["start_new_session"] is False
        assert kwargs["stdin"] == subprocess.DEVNULL
    assert [status["state"] for status in fake_runtime.statuses] == [
        "starting", "ready", "running", "completed", "running", "completed", "stopped",
    ]
    final = read_status(job)
    assert final["finished_count"] == 2 and final["shard"] == 7
    assert not list(Path(job["output_dir"]).rglob("*.tmp"))


def test_worker_carries_confirmation_and_development_split(job, tmp_path):
    job.update(stop_confirm_steps=2, split="val_seen")
    cfg = worker.build_eval_config(job, job["workers"][0], tmp_path)
    assert cfg.stop_confirm_steps == 2
    command = worker.server_command(job, job["workers"][0], 0, tmp_path / "ready")
    assert command[command.index("--split") + 1] == "val_seen"


def test_shard_config_factory_enables_video_without_reloading_model(job, fake_runtime):
    def video_config(job_spec, worker_spec, output):
        cfg = worker.build_eval_config(job_spec, worker_spec, output)
        cfg.episodes = 1
        cfg.save_video = True
        cfg.video_fps = 5
        cfg.record_dir = str(output / "records")
        cfg.episode_manifest = str(output.parent / "selected.json")
        return cfg

    worker.run_shard(job, job["workers"][0], 3, fake_runtime.cached,
                     config_factory=video_config)

    [(cfg, cached)] = fake_runtime.evaluations
    assert cached is fake_runtime.cached and fake_runtime.loads == []
    assert cfg.episodes == 1 and cfg.save_video and cfg.video_fps == 5
    assert cfg.record_dir == str(Path(cfg.output_dir) / "records")
    assert cfg.episode_manifest.endswith("/selected.json")
    assert cfg.fm_only_stop_mode == "learned" and cfg.oracle_stop_distance is None
    assert fake_runtime.processes[0][0].terminated == 1


def test_shard_config_factory_failure_cleans_up_its_server(job, fake_runtime):
    def broken_config(*_args):
        raise ValueError("invalid video config")

    with pytest.raises(ValueError, match="invalid video config"):
        worker.run_shard(job, job["workers"][0], 3, fake_runtime.cached,
                         config_factory=broken_config)
    assert fake_runtime.evaluations == []
    assert fake_runtime.processes[0][0].terminated == 1


def test_duplicate_shard_is_failed_without_second_evaluation(job, fake_runtime):
    result = worker.run_worker(job, 0, input_stream=io.StringIO('{"shard": 3}\n{"shard": 3}\n'))
    assert result == 1
    assert len(fake_runtime.evaluations) == len(fake_runtime.processes) == 1
    final = read_status(job)
    assert final["status"] == "failed" and final["finished_count"] == 1
    assert "duplicate shard" in final["error"]


def test_server_exiting_normally_after_env_close_completes_shard(job, fake_runtime, monkeypatch):
    def evaluate(_cfg, *, engine_factory):
        assert engine_factory(_cfg) is fake_runtime.cached
        # Real run_habitat_eval sends ZMQ close in finally; the server exits 0.
        fake_runtime.processes[-1][0].code = 0
        return []

    monkeypatch.setattr(worker.runner, "run_habitat_eval", evaluate)
    assert worker.run_worker(job, 0, input_stream=io.StringIO('{"shard": 3}\nnull\n')) == 0
    assert read_status(job)["finished_count"] == 1
    assert "completed" in [entry["state"] for entry in fake_runtime.statuses]
    process = fake_runtime.processes[0][0]
    assert process.terminated == process.killed == 0
    assert process.waits == [10.0]


def test_server_nonzero_exit_after_evaluation_fails_shard(job, fake_runtime, monkeypatch):
    def evaluate(_cfg, *, engine_factory):
        fake_runtime.processes[-1][0].code = 7
        return []

    monkeypatch.setattr(worker.runner, "run_habitat_eval", evaluate)
    assert worker.run_worker(job, 0, input_stream=io.StringIO('{"shard": 3}\n')) == 1
    assert read_status(job)["status"] == "failed"
    assert read_status(job)["finished_count"] == 0
    assert "exit=7" in read_status(job)["error"]


@pytest.mark.parametrize("assignment", [{"shard": True}, {"shard": -1}, {"shard": 16}, {"shard": 1, "other": 2}])
def test_invalid_assignments_do_not_start_server(job, fake_runtime, assignment):
    assert worker.run_worker(job, 0, input_stream=io.StringIO(json.dumps(assignment) + "\n")) == 1
    assert fake_runtime.processes == []
    assert read_status(job)["status"] == "failed"


def test_existing_shard_results_are_never_appended(job, fake_runtime):
    output = Path(job["output_dir"]) / "shard_0003"
    output.mkdir(parents=True)
    existing = output / "results.jsonl"
    existing.write_text("preserve me\n")
    assert worker.run_worker(job, 0, input_stream=io.StringIO('{"shard": 3}\n')) == 1
    assert existing.read_text() == "preserve me\n"
    assert fake_runtime.processes == []
    assert "FileExistsError" in read_status(job)["error"]


def test_wait_for_server_checks_exit_before_ready_marker(tmp_path):
    ready = tmp_path / ".ready"
    ready.touch()
    with pytest.raises(RuntimeError, match="exit=7"):
        worker.wait_for_server(FakeProcess(code=7), ready, timeout=900)


def test_wait_for_server_has_deadline(tmp_path, monkeypatch):
    ticks = iter([0.0, 2.0])
    monkeypatch.setattr(worker.time, "monotonic", lambda: next(ticks))
    with pytest.raises(TimeoutError, match="within 1s"):
        worker.wait_for_server(FakeProcess(), tmp_path / ".ready", timeout=1)


def test_readiness_failure_cleans_up_and_reports_failed(job, fake_runtime, monkeypatch):
    def fail(*_args):
        raise RuntimeError("server readiness failed")

    monkeypatch.setattr(worker, "wait_for_server", fail)
    assert worker.run_worker(job, 0, input_stream=io.StringIO('{"shard": 3}\n')) == 1
    assert fake_runtime.evaluations == []
    assert fake_runtime.processes[0][0].terminated == 1
    assert read_status(job)["finished_count"] == 0
    assert "server readiness failed" in read_status(job)["error"]


def test_terminate_server_escalates_only_own_process():
    process = FakeProcess()
    attempts = []

    def wait(timeout):
        attempts.append(timeout)
        if len(attempts) == 1:
            raise subprocess.TimeoutExpired("habitat", timeout)
        return -9

    process.wait = wait
    worker.terminate_server(process, timeout=0.01)
    assert process.terminated == process.killed == 1
    assert attempts == [0.01, 0.01]


def test_terminate_already_exited_server_only_reaps():
    process = FakeProcess(code=0)
    worker.terminate_server(process)
    assert process.terminated == process.killed == 0
    assert process.waits == [10.0]


def test_signal_during_evaluation_cleans_up_and_restores_handler(job, fake_runtime, monkeypatch):
    previous = signal.getsignal(signal.SIGTERM)

    def interrupted(*_args, **_kwargs):
        signal.raise_signal(signal.SIGTERM)

    monkeypatch.setattr(worker.runner, "run_habitat_eval", interrupted)
    assert worker.run_worker(job, 0, input_stream=io.StringIO('{"shard": 3}\n')) == 130
    assert fake_runtime.processes[0][0].terminated == 1
    assert read_status(job)["status"] == "stopped"
    assert read_status(job)["finished_count"] == 0
    assert signal.getsignal(signal.SIGTERM) == previous


def test_signal_escapes_real_runners_keyboard_interrupt_handler(job, fake_runtime, monkeypatch):
    class InterruptingEnv:
        closed = False

        def reset(self):
            signal.raise_signal(signal.SIGTERM)

        def close(self):
            self.closed = True
            fake_runtime.processes[-1][0].code = 0

    env = InterruptingEnv()

    def evaluate(cfg, *, engine_factory):
        return real_run_habitat_eval(
            cfg, engine_factory=engine_factory, env_factory=lambda _cfg: env,
        )

    monkeypatch.setattr(worker.runner, "run_habitat_eval", evaluate)
    assert worker.run_worker(job, 0, input_stream=io.StringIO('{"shard": 3}\nnull\n')) == 130
    assert env.closed
    assert read_status(job)["status"] == "stopped"
    assert read_status(job)["finished_count"] == 0
    assert "completed" not in [entry["state"] for entry in fake_runtime.statuses]


def test_configure_device_uses_single_visible_logical_gpu(monkeypatch):
    calls = []
    fake_torch = SimpleNamespace(
        set_num_threads=lambda value: calls.append(("threads", value)),
        cuda=SimpleNamespace(is_available=lambda: True, set_device=lambda value: calls.append(("device", value))),
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "GPU-abcd")
    worker.configure_device(2)
    assert calls == [("threads", 2), ("device", 0)]
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1")
    with pytest.raises(ValueError, match="exactly one"):
        worker.configure_device(2)


def test_main_reads_job_and_worker_id(job, tmp_path, monkeypatch):
    path = tmp_path / "job.json"
    path.write_text(json.dumps(job))
    calls = []
    monkeypatch.setattr(worker, "run_worker", lambda payload, worker_id: calls.append((payload, worker_id)) or 7)
    assert worker.main(["--job", str(path), "--worker-id", "2"]) == 7
    assert calls == [(job, 2)]
