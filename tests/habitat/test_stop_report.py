"""Coverage and STOP metrics are independent of Habitat, torch and GPU inference."""

from __future__ import annotations

import gzip
import json

import pytest

from lightnav.habitat.stop_report import (
    build_stop_report,
    check_coverage,
    episode_key,
    load_expected_episodes,
)


def _record(source="learned_head", *, success=True, oracle=True, distance=1.0, steps=12):
    return {
        "scene_id": "/datasets/mp3d/scan_a/scan_a.glb",
        "habitat_episode_id": "10",
        "episode_id": "episode_000",
        "stop_source": source,
        "success": success,
        "oracle_success": oracle,
        "spl": 0.8 if success else 0.0,
        "final_distance": distance,
        "steps": steps,
        "forced_stop": source == "max_steps",
        "oracle_stop": source == "oracle_stop",
    }


def _write_dataset(tmp_path, episodes):
    path = tmp_path / "val_unseen" / "val_unseen.json.gz"
    path.parent.mkdir(exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8") as stream:
        json.dump({"episodes": episodes}, stream)
    return tmp_path


@pytest.mark.parametrize("scene", [
    "scan_a", "/datasets/mp3d/scan_a/", "mp3d/scan_a/scan_a.glb",
    "/datasets/mp3d/scan_a/other_filename.navmesh", "mp3d\\scan_a\\scan_a.glb",
])
def test_episode_key_normalizes_scan_directory_not_asset_filename(scene):
    assert episode_key(scene, 10) == ("scan_a", "10")


@pytest.mark.parametrize("scene,identifier", [
    ("", "1"), ("/", "1"), ("..", "1"), ("scan.glb", "1"),
    ("scene", ""), ("scene", None), ("scene", True), ("scene", 1.5),
])
def test_episode_key_rejects_missing_or_ambiguous_identity(scene, identifier):
    with pytest.raises(ValueError):
        episode_key(scene, identifier)


def test_load_expected_preserves_metadata_and_requires_unique_keys(tmp_path):
    episodes = [{"scene_id": "mp3d/scan_a/scan_a.glb", "episode_id": 10, "instruction": "go"}]
    assert load_expected_episodes(_write_dataset(tmp_path, episodes)) == episodes
    duplicate = {"scene_id": "/elsewhere/scan_a/scan_a.glb", "episode_id": "10"}
    _write_dataset(tmp_path, episodes + [duplicate])
    with pytest.raises(ValueError, match="duplicate expected"):
        load_expected_episodes(tmp_path)


@pytest.mark.parametrize("split", ["../val_unseen", ".", "", "/val_unseen"])
def test_load_expected_rejects_non_split_paths(tmp_path, split):
    with pytest.raises(ValueError, match="split"):
        load_expected_episodes(tmp_path, split)


def test_load_expected_rejects_empty_dataset(tmp_path):
    with pytest.raises(ValueError, match="nonempty list"):
        load_expected_episodes(_write_dataset(tmp_path, []))


def test_coverage_matches_source_ids_not_shard_display_names():
    expected = [{"scene_id": "mp3d/scan_a/scan_a.glb", "episode_id": 10}]
    report = check_coverage(expected, [_record()])
    assert report == {
        "expected": 1, "records": 1, "unique": 1,
        "missing": [], "unexpected": [], "duplicates": [],
        "duplicate_records": 0, "complete": True,
    }


def test_coverage_detects_duplicates_missing_and_unexpected_records():
    expected = [
        {"scene_id": "scan_a", "episode_id": 10},
        {"scene_id": "scan_b", "episode_id": 20},
    ]
    report = check_coverage(expected, [_record(), _record(), {**_record(), "habitat_episode_id": 99}])
    assert report["complete"] is False
    assert report["expected"] == 2 and report["records"] == 3 and report["unique"] == 2
    assert report["missing"] == [{"scene_id": "scan_b", "episode_id": "20"}]
    assert report["unexpected"] == [{"scene_id": "scan_a", "episode_id": "99"}]
    assert report["duplicates"] == [{"scene_id": "scan_a", "episode_id": "10", "count": 2}]
    assert report["duplicate_records"] == 1


def test_coverage_does_not_collapse_different_scan_dirs_with_same_asset_name():
    expected = [{"scene_id": "mp3d/scan_a/common.glb", "episode_id": 10}]
    record = {**_record(), "scene_id": "/datasets/scan_b/common.glb"}
    report = check_coverage(expected, [record])
    assert report["missing"] == [{"scene_id": "scan_a", "episode_id": "10"}]
    assert report["unexpected"] == [{"scene_id": "scan_b", "episode_id": "10"}]


def test_coverage_rejects_duplicate_expected_and_explicitly_empty_habitat_id():
    expected = [{"scene_id": "scan_a", "episode_id": 10}]
    with pytest.raises(ValueError, match="duplicate episode"):
        check_coverage(expected * 2, [])
    with pytest.raises(ValueError, match="episode_id"):
        check_coverage(expected, [{**_record(), "habitat_episode_id": ""}])
    assert not check_coverage([], [])["complete"]


def test_stop_report_keeps_learned_zero_and_forced_stop_separate():
    records = [
        _record("learned_head", success=True, oracle=True, distance=1.0, steps=10),
        _record("learned_head", success=False, oracle=False, distance=7.0, steps=20),
        _record("max_steps", success=False, oracle=True, distance=5.0, steps=500),
        _record("fm_zero_command", success=True, oracle=True, distance=2.0, steps=30),
    ]
    report = build_stop_report(records)
    assert report["num_episodes"] == report["source_rate_denominator"] == 4
    metrics = report["metrics"]
    assert metrics["SR_success_rate_pct"] == 50
    assert metrics["OS_oracle_success_pct"] == 75
    assert metrics["SPL_pct"] == 40
    assert metrics["OS_minus_SR_percentage_points"] == 25
    misses = report["oracle_success_without_success"]
    assert misses["count"] == 1
    assert misses["rate_pct_of_all_episodes"] == 25
    assert misses["conditional_rate_pct_of_oracle_success"] == pytest.approx(100 / 3)
    assert misses["conditional_denominator"] == 3
    head = report["by_stop_source"]["learned_head"]
    assert head["count"] == head["success_rate_denominator"] == head["goal_range_rate_denominator"] == 2
    assert head["rate_pct_of_all_episodes"] == head["success_rate_pct"] == head["goal_range_rate_pct"] == 50
    assert head["success_count"] == head["goal_range_count"] == 1
    assert head["final_distance_m"] == {"count": 2, "mean": 4.0, "median": 4.0, "min": 1.0, "max": 7.0}
    assert head["steps"]["mean"] == 15
    assert report["by_stop_source"]["fm_zero_command"]["count"] == 1
    assert report["by_stop_source"]["fm_zero_command"]["rate_pct_of_all_episodes"] == 25
    assert report["forced_stop"] == {"count": 1, "rate_pct_of_all_episodes": 25, "denominator": 4}
    json.dumps(report, allow_nan=False)


def test_goal_range_is_strict_and_independent_of_environment_success():
    records = [
        _record(success=False, distance=2.9),
        _record(success=False, distance=3.0),
    ]
    report = build_stop_report(records)
    head = report["by_stop_source"]["learned_head"]
    assert head["success_rate_pct"] == 0
    assert head["goal_range_rate_pct"] == 50


def test_missing_source_is_unknown_not_learned_and_absent_categories_are_explicit():
    record = _record()
    del record["stop_source"]
    report = build_stop_report([record])
    assert report["by_stop_source"]["unknown"]["count"] == 1
    assert report["by_stop_source"]["learned_head"]["count"] == 0
    assert report["by_stop_source"]["learned_head"]["success_rate_pct"] is None


@pytest.mark.parametrize("field,value", [
    ("stop_source", "learned_stop"), ("stop_source", 4),
    ("success", "false"), ("oracle_success", float("nan")),
    ("final_distance", float("inf")), ("final_distance", -1),
    ("final_distance", None), ("steps", 1.5), ("steps", -1),
    ("spl", float("nan")), ("spl", 1.1), ("spl", True),
    ("forced_stop", True), ("oracle_stop", True),
])
def test_stop_report_rejects_invalid_or_inconsistent_record(field, value):
    with pytest.raises(ValueError):
        build_stop_report([{**_record(), field: value}])


@pytest.mark.parametrize("field", ["success", "oracle_success", "final_distance", "steps", "spl"])
def test_stop_report_rejects_missing_navigation_metric(field):
    record = _record()
    del record[field]
    with pytest.raises(ValueError):
        build_stop_report([record])


@pytest.mark.parametrize("distance", [0, -1, float("nan"), float("inf"), True])
def test_stop_report_requires_positive_finite_success_distance(distance):
    with pytest.raises(ValueError, match="success_distance"):
        build_stop_report([], success_distance=distance)


def test_stop_report_empty_and_no_oracle_success_have_null_conditional_rates():
    empty = build_stop_report([])
    assert empty["metrics"]["SR_success_rate_pct"] is None
    assert empty["by_stop_source"]["learned_head"]["final_distance_m"]["mean"] is None
    report = build_stop_report([_record(success=False, oracle=False, distance=9.0)])
    assert report["oracle_success_without_success"]["conditional_rate_pct_of_oracle_success"] is None
    json.dumps(empty, allow_nan=False)


def test_forced_stop_can_be_inferred_only_from_max_steps_source():
    record = _record("max_steps", success=False)
    del record["forced_stop"]
    assert build_stop_report([record])["forced_stop"]["count"] == 1
    with pytest.raises(ValueError, match="forced_stop"):
        build_stop_report([{**record, "forced_stop": False}])
