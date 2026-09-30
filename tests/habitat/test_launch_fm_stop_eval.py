"""Exercise queued evaluation orchestration without model weights or simulators."""

from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

import pytest

from lightnav.habitat import launch_fm_stop_eval as launch


@pytest.mark.parametrize("mask,visible,count,expected", [
    (None, 8, 3, ["0", "1", "2"]),
    ("5, 2,7", 3, 2, ["5", "2"]),
    ("GPU-ab,GPU-cd", 2, 2, ["GPU-ab", "GPU-cd"]),
])
def test_gpu_mapping_preserves_platform_allocation(mask, visible, count, expected):
    assert launch.gpu_tokens(mask, visible, count) == expected


@pytest.mark.parametrize("mask,visible,count", [
    ("", 0, 8), ("0,0", 2, 2), ("-1", 1, 1),
    ("0,1", 1, 2), ("0,,2", 3, 2), (None, 8, 0),
])
def test_bad_gpu_allocations_fail_without_remapping(mask, visible, count):
    with pytest.raises(ValueError, match="allocated visible GPUs"):
        launch.gpu_tokens(mask, visible, count)


def test_six_worker_defaults_cover_r2r_without_a_large_remainder_shard():
    args = launch.parse_args([])
    assert args.num_gpus == 8 and args.workers_per_gpu == 6
    assert args.shards_per_worker == 2 and args.load_parallelism == 2
    episodes = [dict(scene_id=f"scan_{i % 11}", episode_id=i) for i in range(1839)]
    shards = launch.partition_episodes(episodes, args.num_gpus * args.workers_per_gpu * args.shards_per_worker)
    assert len(shards) == 96
    assert [len(s) for s in shards] == [19] * 95 + [34]
    assert sorted(e["episode_id"] for shard in shards for e in shard) == list(range(1839))


@pytest.mark.parametrize("replicas", range(1, 7))
def test_worker_counts_map_all_replicas_and_unique_ports_through_gpu_uuids(replicas):
    args = launch.parse_args(["--workers-per-gpu", str(replicas)])
    tokens = [f"GPU-allocated-{i}" for i in range(8)]
    workers = launch.worker_specs(tokens, args.workers_per_gpu, args.base_port)
    assert len(workers) == 8 * replicas
    assert [w["worker_id"] for w in workers] == list(range(8 * replicas))
    assert [w["gpu_token"] for w in workers] == tokens * replicas
    assert [w["port"] for w in workers] == list(range(28600, 28600 + 8 * replicas))


@pytest.mark.parametrize("arguments", [
    ["--workers-per-gpu", "0"], ["--workers-per-gpu", "7"],
    ["--workers-per-gpu", "6", "--base-port", "65500"],
])
def test_invalid_worker_count_or_port_overflow_fails_before_starting_any_worker(arguments):
    with pytest.raises(SystemExit) as error:
        launch.parse_args(arguments)
    assert error.value.code == 2


def test_partitions_match_habitat_stable_scene_sort_and_remainder():
    episodes = [dict(scene_id=scene, episode_id=i) for i, scene in enumerate("babaabb")]
    shards = launch.partition_episodes(episodes, 3)
    assert [[e["episode_id"] for e in shard] for shard in shards] == [[1, 3], [4, 0], [2, 5, 6]]
    with pytest.raises(ValueError):
        launch.partition_episodes(episodes, 8)


def test_development_split_and_confirmation_are_explicit_protocol_settings():
    args = launch.parse_args(["--split", "val_seen", "--stop-confirm-steps", "2"])
    assert args.split == "val_seen" and args.stop_confirm_steps == 2
    assert launch.parse_args([]).split == "val_unseen"
    assert launch.parse_args([]).stop_confirm_steps == 1
    with pytest.raises(SystemExit):
        launch.parse_args(["--stop-confirm-steps", "0"])
    with pytest.raises(SystemExit):
        launch.parse_args(["--split", "train"])


@pytest.fixture
def completed_job(tmp_path):
    episodes = [dict(scene_id="scan_a", episode_id=i) for i in range(2)]
    job = dict(output_dir=str(tmp_path), checkpoint="/model/step-00003699", checkpoint_step=3699,
               stop_threshold=0.5, fm_seed=42, fm_steps=10, max_steps=500, success_distance=3.0,
               num_gpus=8, workers_per_gpu=2, num_shards=2, episodes=episodes,
               shards=[[row] for row in episodes])
    for i in range(2):
        folder = tmp_path / f"shard_{i:04d}"
        folder.mkdir()
        summary = dict(beta_checkpoint=job["checkpoint"], beta_checkpoint_step=3699,
                       training_mode="fm_stop", stop_protocol="learned", stop_threshold=0.5,
                       fm_seed=42, fm_steps=10, max_steps=500, num_episodes=1, total_time_sec=30)
        launch.write_json(folder / "summary.json", summary)
        row = dict(episode_id="episode_000", habitat_episode_id=i, scene_id="scan_a", success=i == 0,
                   oracle_success=True, spl=0.8 if i == 0 else 0.0, ndtw=0.5,
                   final_distance=1.0 if i == 0 else 6.0, steps=20 if i == 0 else 500,
                   stop_source="learned_head" if i == 0 else "max_steps", forced_stop=i == 1, oracle_stop=False)
        launch.write_json(folder / "results.jsonl", row)
        # JSONL has exactly one JSON document per line.
        (folder / "results.jsonl").write_text(json.dumps(row) + "\n")
    return job


def test_merge_reports_nav_stop_coverage_and_actual_wall_time(completed_job):
    summary = launch.merge_completed(completed_job, elapsed=100)
    assert summary["metrics"]["SR_success_rate_pct"] == 50
    assert summary["metrics"]["OS_oracle_success_pct"] == 100
    assert summary["metrics"]["SPL_pct"] == 40
    assert summary["total_time_sec"] == 100 and summary["episodes_per_hour"] == 72
    output = Path(completed_job["output_dir"])
    assert json.loads((output / "coverage.json").read_text())["complete"] is True
    diagnostics = json.loads((output / "stop_diagnostics.json").read_text())
    assert diagnostics["metrics"]["OS_minus_SR_percentage_points"] == 50
    assert diagnostics["by_stop_source"]["learned_head"]["count"] == 1
    assert diagnostics["by_stop_source"]["max_steps"]["count"] == 1


@pytest.mark.parametrize("field,value", [
    ("stop_protocol", "baseline"), ("stop_threshold", 0.7), ("beta_checkpoint_step", 3000),
    ("fm_seed", 9), ("fm_steps", 3), ("training_mode", "fm_only"),
])
def test_merge_refuses_mixed_experiments(completed_job, field, value):
    folder = Path(completed_job["output_dir"])
    path = folder / "shard_0001/summary.json"
    launch.write_json(path, {**json.loads(path.read_text()), field: value})
    with pytest.raises(ValueError, match="protocol mismatch"):
        launch.merge_completed(completed_job, elapsed=10)
    assert not (folder / "summary.json").exists()


def test_merge_requires_matching_confirmation_protocol(completed_job):
    completed_job["stop_confirm_steps"] = 2
    output = Path(completed_job["output_dir"])
    for path in output.glob("shard_*/summary.json"):
        launch.write_json(path, {**json.loads(path.read_text()), "stop_confirm_steps": 2})
    path = output / "shard_0001/summary.json"
    launch.write_json(path, {**json.loads(path.read_text()), "stop_confirm_steps": 1})
    with pytest.raises(ValueError, match="protocol mismatch for stop_confirm_steps"):
        launch.merge_completed(completed_job, elapsed=10)
    launch.write_json(path, {**json.loads(path.read_text()), "stop_confirm_steps": 2})
    assert launch.merge_completed(completed_job, elapsed=10)["stop_confirm_steps"] == 2


def test_legacy_job_means_single_confirmation_and_rejects_double_confirm_shards(completed_job):
    path = Path(completed_job["output_dir"]) / "shard_0001/summary.json"
    launch.write_json(path, {**json.loads(path.read_text()), "stop_confirm_steps": 2})
    with pytest.raises(ValueError, match="protocol mismatch for stop_confirm_steps"):
        launch.merge_completed(completed_job, elapsed=10)


@pytest.mark.parametrize("bad_record", ["duplicate", "missing", "oracle"])
def test_merge_refuses_incomplete_or_privileged_results(completed_job, bad_record):
    folder = Path(completed_job["output_dir"])
    path = folder / "shard_0000/results.jsonl"
    if bad_record == "duplicate":
        path.write_text(path.read_text() * 2)
    elif bad_record == "missing":
        path.write_text("")
    else:
        row = json.loads(path.read_text())
        path.write_text(json.dumps({**row, "oracle_stop": True}) + "\n")
    with pytest.raises(ValueError, match="coverage|oracle"):
        launch.merge_completed(completed_job, elapsed=10)
    assert not (folder / "summary.json").exists()


# Real, tiny Python children exercise pipe dispatch, status races and process
# cleanup. They never import Torch or Habitat and never spawn model processes.
_FAKE_WORKER = '''
import json, pathlib, sys, time
folder, mode = pathlib.Path(sys.argv[1]), sys.argv[2]
def status(state, shard=None):
    p = folder / 'status.tmp'
    p.write_text(json.dumps(dict(status=state, shard=shard)))
    p.replace(folder / 'status.json')
if mode == 'loading':
    time.sleep(30)
status('ready')
for line in sys.stdin:
    task = json.loads(line)
    if task is None:
        if mode == 'shutdown':
            time.sleep(30)
        sys.exit(9 if mode == 'bad_exit' else 0)
    shard = task['shard']
    status('running', shard)
    with (folder / 'assignments.jsonl').open('a') as f:
        f.write(json.dumps(shard) + '\\n')
    if mode == 'failed':
        status('failed', shard)
        sys.exit(7)
    if mode == 'running':
        time.sleep(30)
    time.sleep(0.01)
    status('completed', shard)
'''


def fake_pool(tmp_path, mode="normal"):
    job = dict(output_dir=str(tmp_path), num_shards=7, shards=[[i] for i in range(7)],
               workers=[dict(worker_id=i) for i in range(3)], load_parallelism=1,
               load_timeout=5, shard_timeout=5, shutdown_timeout=5)
    slots = []

    def spawn(_job_path, _job, worker):
        folder = tmp_path / str(worker["worker_id"])
        folder.mkdir()
        process = subprocess.Popen([sys.executable, "-c", _FAKE_WORKER, str(folder), mode],
                                   stdin=subprocess.PIPE, text=True, start_new_session=True)
        slot = dict(process=process, worker=worker, folder=folder, active=None, stopping=False,
                    started=time.monotonic())
        slots.append(slot)
        return slot

    return job, slots, spawn


def test_dynamic_pool_evaluates_every_shard_exactly_once_and_reaps_workers(tmp_path):
    job, slots, spawn = fake_pool(tmp_path)
    launch.run_workers(tmp_path / "job.json", job, spawn=spawn, interval=0.005)
    assignments = [json.loads(line) for path in tmp_path.glob("*/assignments.jsonl")
                   for line in path.read_text().splitlines()]
    assert sorted(assignments) == list(range(7))
    assert len(slots) == 3 and all(slot["process"].poll() == 0 for slot in slots)
    assert json.loads((tmp_path / "progress.json").read_text())["completed_shards"] == 7


@pytest.mark.parametrize("mode", ["failed", "bad_exit", "loading", "running", "shutdown"])
def test_pool_failures_and_hangs_abort_and_reap_all_children(tmp_path, mode):
    job, slots, spawn = fake_pool(tmp_path, mode)
    if mode == "loading":
        job["load_timeout"] = 0.2
    elif mode == "running":
        job["shard_timeout"] = 0.2
    elif mode == "shutdown":
        job["shutdown_timeout"] = 0.2
    with pytest.raises((RuntimeError, TimeoutError)):
        launch.run_workers(tmp_path / "job.json", job, spawn=spawn, interval=0.005)
    assert all(slot["process"].poll() is not None for slot in slots)
    if mode == "loading":
        assert len(slots) == 1  # Loading concurrency never exceeds its bound.
