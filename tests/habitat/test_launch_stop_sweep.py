"""Eight-group experiment isolation and reporting, without CUDA or Habitat."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from lightnav.habitat import launch_stop_sweep as sweep


def test_default_plan_is_full_factorial_and_respects_allocated_gpu_uuids(tmp_path):
    args = sweep.parse_args([])
    tokens = [f"GPU-allocated-{index}" for index in range(8)]
    plan = sweep.make_plan(args, tmp_path, tokens)
    assert plan["split"] == "val_seen"
    assert plan["total_workers"] == 48 and plan["global_load_parallelism"] == 2
    assert {(g["stop_threshold"], g["stop_confirm_steps"]) for g in plan["groups"]} == {
        (threshold, confirm) for threshold in (0.5, 0.6, 0.7, 0.8) for confirm in (1, 2)
    }
    assert [g["gpu_token"] for g in plan["groups"]] == tokens
    ports, common = [], []
    for group in plan["groups"]:
        command = group["command"]
        arguments = dict(zip(command[4::2], command[5::2], strict=True))
        assert arguments["--num-gpus"] == "1"
        assert arguments["--load-parallelism"] == "1"
        assert arguments["--workers-per-gpu"] == "6"
        assert arguments["--fm-seed"] == "42"
        ports.extend(range(group["base_port"], group["base_port"] + 6))
        for varying in ("--base-port", "--stop-threshold", "--stop-confirm-steps", "--output-dir"):
            arguments.pop(varying)
        common.append(arguments)
    assert len(set(ports)) == 48
    assert common == [common[0]] * 8


@pytest.mark.parametrize("arguments", [
    ["--split", "val_unseen"], ["--workers-per-gpu", "7"], ["--load-parallelism", "0"],
    ["--base-port", "65500"], ["--max-steps", "501"], ["--seed", "-1"],
    ["--plan-only", "--check-only"],
])
def test_invalid_sweep_arguments_fail_before_resources_or_gpus(arguments):
    with pytest.raises(SystemExit):
        sweep.parse_args(arguments)


def test_plan_only_needs_neither_checkpoint_nor_dataset_nor_cuda(tmp_path, monkeypatch, capsys):
    def forbidden(*args, **kwargs):
        raise AssertionError("plan-only must not launch child processes")

    monkeypatch.setattr(sweep.subprocess, "Popen", forbidden)
    output = tmp_path / "plan"
    plan = sweep.main(["--plan-only", "--checkpoint", "/missing/checkpoint", "--dataset-root", "/missing/data",
                       "--output-dir", str(output)])
    assert json.loads((output / "sweep_plan.json").read_text()) == plan
    assert plan["gpu_assignment_verified"] is False
    assert json.loads(capsys.readouterr().out)["groups"] == plan["groups"]
    assert not (output / "status.json").exists()


def test_start_group_isolated_cuda_environment_and_process_session(tmp_path, monkeypatch):
    group = sweep.make_plan(sweep.parse_args([]), tmp_path, [f"GPU-{i}" for i in range(8)])["groups"][3]
    calls = []
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "GPU-parent-a,GPU-parent-b")
    monkeypatch.setattr(sweep.subprocess, "Popen", lambda command, **kwargs: calls.append((command, kwargs)) or object())
    sweep.start_group(group, tmp_path)
    command, kwargs = calls[0]
    assert command == group["command"]
    assert kwargs["env"]["CUDA_VISIBLE_DEVICES"] == "GPU-3"
    assert kwargs["start_new_session"] is True
    assert kwargs["env"]["HF_HUB_OFFLINE"] == "1"


def test_loading_slot_held_until_every_worker_has_loaded(tmp_path):
    group = dict(output_dir=str(tmp_path))
    slot = dict(group=group, process=SimpleNamespace(poll=lambda: None))
    assert sweep.group_loading(slot, 2)
    for index, state in enumerate(("ready", "starting")):
        folder = tmp_path / "workers" / f"worker_{index}"
        folder.mkdir(parents=True)
        sweep.write_json(folder / "status.json", dict(status=state))
    assert sweep.group_loading(slot, 2)
    sweep.write_json(tmp_path / "workers/worker_1/status.json", dict(status="running"))
    assert not sweep.group_loading(slot, 2)
    slot["process"] = SimpleNamespace(poll=lambda: 0)
    assert not sweep.group_loading(slot, 99)


class FakeProcess:
    def __init__(self, exit_code=0):
        self.exit_code = exit_code
        self.polls = 0

    def poll(self):
        self.polls += 1
        return self.exit_code if self.polls > 2 else None


def test_eight_groups_finish_with_bounded_loading_and_cleanup(tmp_path):
    plan = sweep.make_plan(sweep.parse_args([]), tmp_path)
    active, cleaned = [], []

    def spawn(group, _output):
        folder = Path(group["output_dir"])
        folder.mkdir()
        sweep.write_json(folder / "status.json", dict(status="completed"))
        slot = dict(group=group, process=FakeProcess())
        active.append(slot)
        return slot

    observed_loading = []

    def loading(slot, _workers):
        value = slot["process"].polls <= 1
        observed_loading.append(value)
        return value

    sweep.run_groups(plan, spawn=spawn, loading=loading, cleanup=lambda slots: cleaned.extend(slots), interval=0)
    assert len(active) == 8 and cleaned == active
    progress = json.loads((tmp_path / "progress.json").read_text())
    assert len(progress["completed_groups"]) == 8
    assert progress["waiting_groups"] == 0
    assert any(observed_loading)


def test_failure_cleans_only_children_created_by_this_sweep(tmp_path):
    plan = sweep.make_plan(sweep.parse_args([]), tmp_path)
    launched, cleaned = [], []

    def spawn(group, _output):
        slot = dict(group=group, process=FakeProcess(exit_code=2))
        launched.append(slot)
        return slot

    with pytest.raises(RuntimeError, match="exited 2"):
        sweep.run_groups(plan, spawn=spawn, loading=lambda *_: True,
                         cleanup=lambda slots: cleaned.extend(slots), interval=0)
    assert len(launched) == 2
    assert cleaned == launched


@pytest.fixture
def completed_sweep(tmp_path):
    args = sweep.parse_args([])
    plan = sweep.make_plan(args, tmp_path)
    preflight = dict(checkpoint=args.checkpoint, checkpoint_step=3699,
                     checkpoint_metadata_sha256="model-digest", episode_file_sha256="episode-digest",
                     split="val_seen", resources=dict(ndtw=dict(sha256="gt-digest")),
                     development_audit=dict(verified=True, split="val_seen", num_development_episodes=3), num_episodes=3)
    episodes = [dict(scene_id="scan", episode_id=str(i)) for i in range(3)]
    for group in plan["groups"]:
        folder = Path(group["output_dir"])
        folder.mkdir()
        sweep.write_json(folder / "status.json", dict(status="completed"))
        job = {field: getattr(args, field) for field in sweep.SHARED_FIELDS if hasattr(args, field)}
        job.update(preflight, success_distance=3.0, num_shards=1, episodes=episodes, shards=[episodes],
                   stop_threshold=group["stop_threshold"], stop_confirm_steps=group["stop_confirm_steps"])
        sweep.write_json(folder / "job.json", job)
        summary = dict(beta_checkpoint=args.checkpoint, beta_checkpoint_step=3699,
                       training_mode="fm_stop", stop_protocol="learned", fm_seed=42, fm_steps=10, max_steps=500,
                       stop_threshold=group["stop_threshold"], stop_confirm_steps=group["stop_confirm_steps"], num_episodes=3)
        sweep.write_json(folder / "summary.json", summary)
        records = [dict(habitat_episode_id=str(i), scene_id="scan", instruction=f"instruction {i}",
                        success=i == 0, oracle_success=i < 2, spl=0.9 if i == 0 else 0.0,
                        ndtw=0.8, final_distance=1.0 if i == 0 else 6.0, min_distance=1.0 if i < 2 else 6.0,
                        steps=20 + i, stop_source="learned_head", oracle_stop=False) for i in range(3)]
        (folder / "results.jsonl").write_text("\n".join(json.dumps(row) for row in records) + "\n")
        sweep.write_json(folder / "coverage.json", sweep.check_coverage(episodes, records))
    return plan, preflight


def test_summary_recomputes_exact_metrics_and_stop_failures(completed_sweep):
    plan, preflight = completed_sweep
    report = sweep.summarize_sweep(plan, preflight)
    assert len(report["rows"]) == 8
    row = report["rows"][0]
    assert row["SR_pct"] == pytest.approx(100 / 3)
    assert row["OS_pct"] == pytest.approx(200 / 3)
    assert row["SPL_pct"] == pytest.approx(30)
    assert row["nDTW_pct"] == pytest.approx(80)
    assert row["NE_m"] == pytest.approx(13 / 3)
    assert row["learned_stop_failures"] == 2
    assert row["failed_stop_never_reached"] == row["failed_stop_reached_then_left"] == 1
    assert row["max_steps_count"] == 0
    output = Path(plan["output_dir"])
    assert (output / "comparison.csv").read_text().count("\n") == 9
    assert "not final val_unseen results" in (output / "comparison.md").read_text()
    assert report["development_selection"]["threshold"] == 0.5


@pytest.mark.parametrize("file,field,value,message", [
    ("job.json", "checkpoint_metadata_sha256", "changed", "preflight mismatch"),
    ("job.json", "episode_file_sha256", "changed", "preflight mismatch"),
    ("job.json", "resources", {}, "preflight mismatch"),
    ("job.json", "development_audit", {}, "preflight mismatch"),
    ("job.json", "fm_steps", 5, "shared protocol"),
    ("job.json", "stop_confirm_steps", 7, "protocol mismatch"),
    ("summary.json", "stop_confirm_steps", 7, "protocol mismatch"),
    ("summary.json", "fm_seed", 7, "summary mismatch"),
    ("coverage.json", "complete", False, "coverage"),
    ("status.json", "status", "running", "not completed"),
])
def test_report_refuses_mixed_or_incomplete_runs(completed_sweep, file, field, value, message):
    plan, preflight = completed_sweep
    path = Path(plan["groups"][-1]["output_dir"]) / file
    content = json.loads(path.read_text())
    content[field] = value
    sweep.write_json(path, content)
    with pytest.raises(ValueError, match=message):
        sweep.summarize_sweep(plan, preflight)
    assert not (Path(plan["output_dir"]) / "comparison.json").exists()


def test_duplicate_results_rejected_even_when_saved_coverage_says_complete(completed_sweep):
    plan, preflight = completed_sweep
    path = Path(plan["groups"][-1]["output_dir"]) / "results.jsonl"
    path.write_text(path.read_text() + path.read_text().splitlines()[0] + "\n")
    with pytest.raises(ValueError, match="coverage"):
        sweep.summarize_sweep(plan, preflight)


def test_instructions_must_match_for_same_episode_keys(completed_sweep):
    plan, preflight = completed_sweep
    path = Path(plan["groups"][-1]["output_dir"]) / "results.jsonl"
    records = [json.loads(line) for line in path.read_text().splitlines()]
    records[0]["instruction"] = "changed instruction"
    path.write_text("\n".join(json.dumps(row) for row in records))
    with pytest.raises(ValueError, match="instructions differ"):
        sweep.summarize_sweep(plan, preflight)


def test_oracle_stop_rejected(completed_sweep):
    plan, preflight = completed_sweep
    path = Path(plan["groups"][0]["output_dir"]) / "results.jsonl"
    records = [json.loads(line) for line in path.read_text().splitlines()]
    records[0]["oracle_stop"] = True
    path.write_text("\n".join(json.dumps(row) for row in records))
    with pytest.raises(ValueError, match="oracle STOP"):
        sweep.summarize_sweep(plan, preflight)


@pytest.mark.parametrize("audit", [None, {}, dict(verified=False, split="val_seen"),
                                    dict(verified=1, split="val_seen"),
                                    dict(verified=True, split="val_unseen"),
                                    dict(verified=True, split="val_seen", num_development_episodes=999)])
def test_unverified_development_data_refused_before_cuda_or_group_launch(completed_sweep, monkeypatch, audit):
    from lightnav.habitat import launch_fm_stop_eval as single

    _, preflight = completed_sweep
    preflight["development_audit"] = audit
    monkeypatch.setattr(single, "main", lambda _argv: preflight)
    with pytest.raises(ValueError, match="independence|episode count"):
        sweep.main(["--check-only"])


def test_check_only_preserves_verified_snapshot_and_never_starts_groups(completed_sweep, tmp_path, monkeypatch):
    from lightnav.habitat import launch_fm_stop_eval as single

    _, preflight = completed_sweep
    calls = []
    monkeypatch.setattr(single, "main", lambda argv: calls.append(argv) or preflight)
    monkeypatch.setattr(sweep, "run_groups", lambda *_args: pytest.fail("check-only started groups"))
    result = sweep.main(["--check-only", "--output-dir", str(tmp_path / "checked")])
    assert len(calls) == 1 and "--check-only" in calls[0]
    assert result["preflight"] == preflight
    assert json.loads((tmp_path / "checked/preflight.json").read_text()) == preflight


@pytest.mark.parametrize("worker_pid_reused", [False, True])
def test_cleanup_escalation_only_kills_our_child_and_verified_worker_session(tmp_path, monkeypatch, worker_pid_reused):
    folder = tmp_path / "workers/worker_0"
    folder.mkdir(parents=True)
    sweep.write_json(folder / "status.json", dict(pid=201))
    other = tmp_path / "workers/worker_1"
    other.mkdir()
    sweep.write_json(other / "status.json", dict(pid=202))
    calls, identity_calls = [], []

    class Process:
        pid = 200
        waited = False

        def poll(self):
            return None

        def terminate(self):
            calls.append(("terminate", self.pid))

        def wait(self, timeout=None):
            if not self.waited:
                self.waited = True
                raise sweep.subprocess.TimeoutExpired("fake", timeout)
            calls.append(("wait", self.pid))

    def identity(pid):
        identity_calls.append(pid)
        if pid == 201:
            start = 999 if worker_pid_reused and identity_calls.count(pid) > 1 else 123
            return 200, 201, start
        return 987, 202, 123  # Same directory is insufficient: this is another parent's process.

    monkeypatch.setattr(sweep, "_proc_identity", identity)
    monkeypatch.setattr(sweep.os, "killpg", lambda pid, sig: calls.append(("killpg", pid)))
    slot = dict(group=dict(output_dir=str(tmp_path)), process=Process())
    sweep.stop_groups([slot], timeout=0)
    assert ("terminate", 200) in calls
    assert ("killpg", 200) in calls
    assert (("killpg", 201) in calls) is not worker_pid_reused
    assert ("killpg", 202) not in calls
