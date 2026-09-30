"""Merging the per-shard outputs of a parallel Habitat evaluation."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from lightnav.cli import eval_merge
from lightnav.habitat.merge import find_shard_dirs, load_results, merge_results


def _vlnce_record(i: int, success: float, spl: float, ndtw: float, dist: float) -> dict:
    return {
        "episode_id": f"episode_{i:03d}",
        "habitat_episode_id": str(100 + i),
        "scene_id": "scene",
        "rollout_idx": 0,
        "success": success,
        "oracle_success": success > 0,
        "spl": spl,
        "ndtw": ndtw,
        "soft_spl": 0.0,
        "object_category": "",
        "steps": 10 + i,
        "final_distance": dist,
        "min_distance": dist,
        "instruction": "go",
        "termination_reason": "agent_stop",
        "termination_details": {},
    }


def _write_shard(root: Path, name: str, records: list[dict], total_time: float) -> Path:
    shard = root / name
    shard.mkdir(parents=True)
    with open(shard / "results.jsonl", "w") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")
    (shard / "summary.json").write_text(
        json.dumps({"total_time_sec": total_time, "model": "/ckpt", "backend": "vllm_local"})
    )
    return shard


@pytest.fixture
def two_shards(tmp_path):
    a = _write_shard(tmp_path, "shard_0", [_vlnce_record(0, 1.0, 0.8, 0.9, 1.0)], 30.0)
    b = _write_shard(
        tmp_path,
        "shard_1",
        [_vlnce_record(0, 0.0, 0.0, 0.5, 6.0), _vlnce_record(1, 1.0, 0.6, 0.7, 2.0)],
        50.0,
    )
    return tmp_path, a, b


def test_find_shard_dirs_accepts_parent_or_shards(two_shards):
    root, a, b = two_shards
    assert find_shard_dirs([root]) == [a, b]
    assert find_shard_dirs([b, a, a]) == [b, a]  # explicit order kept, deduped
    assert find_shard_dirs([root / "nope"]) == []


def test_load_results_skips_garbage_lines(two_shards):
    _, a, _ = two_shards
    with open(a / "results.jsonl", "a") as f:
        f.write("not json\n\n")
    assert len(load_results(a)) == 1


def test_merge_is_the_summary_of_the_concatenated_episodes(two_shards):
    root, a, b = two_shards
    out = root / "merged"
    summary = merge_results([a, b], out)

    assert summary["num_episodes"] == 3
    m = summary["metrics"]
    assert m["SR_success_rate_pct"] == pytest.approx(200 / 3, abs=0.01)
    assert m["SPL_pct"] == pytest.approx((0.8 + 0.0 + 0.6) / 3 * 100, abs=0.01)
    assert m["NDTW_pct"] == pytest.approx((0.9 + 0.5 + 0.7) / 3 * 100, abs=0.01)
    assert m["NE_navigation_error_m"] == pytest.approx(3.0, abs=1e-3)
    # parallel shards: wall clock is the longest shard, not the sum
    assert summary["total_time_sec"] == 50.0
    assert summary["model"] == "/ckpt" and summary["backend"] == "vllm_local"
    assert [s["episodes"] for s in summary["shards"]] == [1, 2]

    written = json.loads((out / "summary.json").read_text())
    assert written["num_episodes"] == 3
    merged_lines = [json.loads(line) for line in (out / "results.jsonl").read_text().splitlines()]
    assert [r["shard"] for r in merged_lines] == ["shard_0", "shard_1", "shard_1"]


def test_merge_prefixes_relative_video_path_with_shard(tmp_path):
    record = _vlnce_record(0, 1.0, 0.8, 0.9, 1.0)
    record["video"] = "videos/episode_000.mp4"
    shard = _write_shard(tmp_path, "shard_2", [record], 3.0)

    merge_results([shard], tmp_path / "merged")

    merged = json.loads((tmp_path / "merged/results.jsonl").read_text().strip())
    assert merged["video"] == "shard_2/videos/episode_000.mp4"


def test_merge_dispatches_objectnav_summary(tmp_path):
    rec = _vlnce_record(0, 1.0, 0.5, 0.0, 0.05)
    rec["object_category"] = "chair"
    rec["soft_spl"] = 0.4
    shard = _write_shard(tmp_path, "shard_0", [rec], 1.0)
    summary = merge_results([shard], tmp_path / "out")
    assert "SoftSPL_pct" in summary["metrics"]
    assert summary["per_category_sr"] == {"chair": 100.0}


def test_merge_with_no_episodes_returns_none(tmp_path):
    shard = _write_shard(tmp_path, "shard_0", [], 0.0)
    assert merge_results([shard], tmp_path / "out") is None
    assert (tmp_path / "out" / "results.jsonl").read_text() == ""
    assert not (tmp_path / "out" / "summary.json").exists()


def test_merge_requires_shards():
    with pytest.raises(ValueError):
        merge_results([], "x")


def test_merge_preserves_learned_stop_threshold_and_training_provenance(tmp_path):
    shard = _write_shard(tmp_path, "shard_0", [_vlnce_record(0, 1.0, 0.8, 0.9, 1.0)], 1.0)
    metadata = {
        "stop_protocol": "learned", "stop_threshold": 0.7, "stop_confirm_steps": 2,
        "stop_head_type": "last_prefix_mlp_v1", "stop_positive_weight": 12.0,
        "stop_training_data_sha256": "manifest-hash",
        "stop_supervision": {"version": "test_v1", "counts": {"positive": 2, "negative": 24}},
    }
    (shard / "summary.json").write_text(json.dumps(metadata))
    summary = merge_results([shard], tmp_path / "out")
    for key, value in metadata.items():
        assert summary[key] == value


def test_cli_merges_a_parent_directory_in_place(two_shards, capsys):
    root, _, _ = two_shards
    assert eval_merge.main([str(root)]) == 0
    assert json.loads((root / "summary.json").read_text())["num_episodes"] == 3
    assert "Merged 3 episodes from 2 shard(s)" in capsys.readouterr().out


def test_cli_explicit_shards_and_output(two_shards):
    root, a, b = two_shards
    out = root / "custom"
    assert eval_merge.main([str(a), str(b), "--output", str(out)]) == 0
    assert (out / "summary.json").exists()


def test_cli_reports_missing_results(tmp_path, capsys):
    assert eval_merge.main([str(tmp_path)]) == 1
    assert "no results.jsonl" in capsys.readouterr().err
