"""CPU-only checks for reproducible sampling and honest replay comparisons."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from unittest.mock import Mock

import pytest

from lightnav.habitat.sample_videos import (
    compare_replay,
    select_samples,
    verify_dataset_order,
)
from lightnav.habitat.stop_report import episode_key


def _key(record):
    return episode_key(
        record["scene_id"], record.get("habitat_episode_id", record.get("episode_id")),
    )


@pytest.fixture
def completed_run(tmp_path):
    # Source order intentionally differs from scene order and result order. IDs
    # within each scene are unsorted, as Habitat only stably sorts scene_id.
    identities = [
        ("scan_b", "3"), ("scan_a", "9"), ("scan_b", "1"), ("scan_c", "7"),
        ("scan_a", "1"), ("scan_c", "2"), ("scan_b", "4"), ("scan_a", "3"),
    ]
    episodes = [
        {"scene_id": f"mp3d/{scene}/{scene}.glb", "episode_id": identifier}
        for scene, identifier in identities
    ]
    records = []
    for index in [3, 1, 6, 0, 7, 2, 5, 4]:
        scene, identifier = identities[index]
        success = index % 2 == 1
        records.append({
            "scene_id": f"/datasets/mp3d/{scene}/{scene}.glb",
            "episode_id": "episode_000",  # These are shard-local display names.
            "habitat_episode_id": identifier,
            "instruction": f"Walk to target {scene}/{identifier}.",
            "success": float(success),
            "oracle_success": success,
            "steps": 20 + index,
            "stop_source": "learned_head",
            "final_distance": 1.5 if success else 8.0,
            "min_distance": 1.0 if success else 4.0,
            "spl": 0.8 if success else 0.0,
            "ndtw": 0.7 if success else 0.2,
            "stop_score": 0.75,
            "extra_details": {"must_remain_in_manifest": index},
        })
    (tmp_path / "job.json").write_text(json.dumps({"episodes": episodes}))
    (tmp_path / "status.json").write_text(json.dumps({"status": "completed"}))
    _write_records(tmp_path, records)
    return tmp_path, episodes, records


def _write_records(directory, records):
    (directory / "results.jsonl").write_text(
        "".join(json.dumps(record) + "\n" for record in records),
    )


def test_select_samples_is_seeded_balanced_and_preserves_full_source(completed_run):
    source, _, records = completed_run
    manifest = select_samples(source, seed=42)
    assert manifest == select_samples(source, seed=42)
    assert manifest["seed"] == 42
    assert Path(manifest["source_run"]).resolve() == source.resolve()
    assert manifest["source_results_sha256"] == hashlib.sha256(
        (source / "results.jsonl").read_bytes(),
    ).hexdigest()
    assert manifest["source_job_sha256"] == hashlib.sha256(
        (source / "job.json").read_bytes(),
    ).hexdigest()
    samples = manifest["samples"]
    assert [sample["label"] for sample in samples] == [
        "success_1", "success_2", "failure_1", "failure_2",
    ]
    # random.Random(42): sample successes, then failures, in results.jsonl order.
    assert [_key(sample["original"]) for sample in samples] == [
        ("scan_c", "7"), ("scan_c", "2"), ("scan_b", "1"), ("scan_b", "4"),
    ]
    assert len({_key(sample["original"]) for sample in samples}) == 4
    lookup = {_key(record): record for record in records}
    for sample in samples:
        assert sample["original"] == lookup[_key(sample["original"])]
    assert [bool(sample["original"]["success"]) for sample in samples] == [
        True, True, False, False,
    ]


def test_dataset_index_matches_stable_scene_sort_not_record_or_episode_sort(completed_run):
    source, episodes, _ = completed_run
    sorted_episodes = sorted(episodes, key=lambda episode: episode["scene_id"])
    for sample in select_samples(source, seed=13)["samples"]:
        assert _key(sorted_episodes[sample["dataset_index"]]) == _key(sample["original"])


@pytest.mark.parametrize("mutation", ["missing", "duplicate", "unexpected"])
def test_select_samples_rejects_inexact_coverage(completed_run, mutation):
    source, _, records = completed_run
    if mutation == "missing":
        records.pop()
    elif mutation == "duplicate":
        records.append(copy.deepcopy(records[0]))
    else:
        records[0]["habitat_episode_id"] = "not_in_dataset"
    _write_records(source, records)
    with pytest.raises(ValueError):
        select_samples(source, seed=42)


@pytest.mark.parametrize("status", ["running", "failed", "starting"])
def test_select_samples_rejects_unfinished_run(completed_run, status):
    source, _, _ = completed_run
    (source / "status.json").write_text(json.dumps({"status": status}))
    with pytest.raises(ValueError):
        select_samples(source, seed=42)


@pytest.mark.parametrize("minority_success", [False, True])
def test_select_samples_requires_two_in_each_outcome_group(completed_run, minority_success):
    source, _, records = completed_run
    for index, record in enumerate(records):
        record["success"] = minority_success if index == 0 else not minority_success
    _write_records(source, records)
    with pytest.raises(ValueError):
        select_samples(source, seed=42)


def test_verify_dataset_order_accepts_equivalent_scene_roots_and_identifier_types(completed_run):
    _, episodes, _ = completed_run
    actual = copy.deepcopy(episodes)
    for episode in actual:
        episode["scene_id"] = "/new_data/" + episode["scene_id"]
        episode["episode_id"] = int(episode["episode_id"])
    verify_dataset_order({"episodes": episodes}, actual)


@pytest.mark.parametrize("mutation", ["missing", "replacement", "within_scene_reordering"])
def test_verify_dataset_order_rejects_silent_dataset_changes(completed_run, mutation):
    _, episodes, _ = completed_run
    actual = copy.deepcopy(episodes)
    if mutation == "missing":
        actual.pop()
    elif mutation == "replacement":
        actual[0]["episode_id"] = "changed"
    else:
        actual[0], actual[2] = actual[2], actual[0]
    with pytest.raises(ValueError):
        verify_dataset_order({"episodes": episodes}, actual)


def test_compare_replay_uses_habitat_identity_and_normalized_scene(completed_run):
    _, _, records = completed_run
    original = records[0]
    replayed = copy.deepcopy(original)
    replayed["episode_id"] = "episode_999"
    replayed["scene_id"] = "mp3d/scan_c/different_asset.glb"
    replayed["habitat_episode_id"] = int(original["habitat_episode_id"])
    comparison = compare_replay(original, replayed)
    assert comparison["identity_match"] is True
    assert comparison["instruction_match"] is True
    assert comparison["matches_original"] is True
    assert set(comparison["fields"]) >= {
        "success", "oracle_success", "steps", "stop_source", "final_distance",
        "min_distance", "spl", "ndtw", "stop_score",
    }


@pytest.mark.parametrize("field,value", [
    ("success", 0.0), ("oracle_success", False), ("steps", 24),
    ("stop_source", "max_steps"), ("final_distance", 1.6), ("min_distance", 1.1),
    ("spl", 0.7), ("ndtw", 0.6), ("stop_score", 0.9),
])
def test_compare_replay_reports_drift_even_when_success_label_is_unchanged(
    completed_run, field, value,
):
    _, _, records = completed_run
    original = records[0]
    replayed = {**original, field: value}
    comparison = compare_replay(original, replayed)
    assert comparison["matches_original"] is False
    assert comparison["fields"][field] == {
        "original": original[field], "replayed": value, "match": False,
    }
    assert comparison["identity_match"] is True


@pytest.mark.parametrize("field,value,flag", [
    ("habitat_episode_id", "wrong", "identity_match"),
    ("scene_id", "other_scene", "identity_match"),
    ("instruction", "Follow an entirely different instruction.", "instruction_match"),
])
def test_compare_replay_rejects_other_episode_or_instruction(completed_run, field, value, flag):
    _, _, records = completed_run
    original = records[0]
    comparison = compare_replay(original, {**original, field: value})
    assert comparison[flag] is False
    assert comparison["matches_original"] is False


def test_compare_replay_tolerates_small_float_rounding_but_not_material_drift(completed_run):
    _, _, records = completed_run
    original = records[0]
    replayed = {**original, "final_distance": original["final_distance"] + 0.00005}
    assert compare_replay(original, replayed)["matches_original"] is True
    replayed["final_distance"] = original["final_distance"] + 0.001
    assert compare_replay(original, replayed)["matches_original"] is False


def test_compare_replay_does_not_treat_missing_metric_as_equal(completed_run):
    _, _, records = completed_run
    original = records[0]
    replayed = copy.deepcopy(original)
    replayed.pop("spl")
    comparison = compare_replay(original, replayed)
    assert comparison["matches_original"] is False
    assert comparison["fields"]["spl"]["match"] is False


def test_budget_stop_can_match_without_a_head_score(completed_run):
    _, _, records = completed_run
    original = {**records[0], "stop_source": "max_steps", "steps": 500}
    original.pop("stop_score")
    assert compare_replay(original, dict(original))["matches_original"] is True
    assert compare_replay(original, {**original, "stop_score": 0.6})["matches_original"] is False


def test_record_samples_reuses_engine_and_records_exact_selected_episodes(completed_run, monkeypatch):
    from lightnav.habitat import eval_preflight, fm_stop_worker, runner, sample_videos

    source, episodes, _ = completed_run
    selection = select_samples(source, seed=42)
    checkpoint = source / "checkpoint"
    checkpoint.mkdir()
    metadata = b'{"training_mode": "fm_stop", "step": 3699}\n'
    (checkpoint / "metadata.json").write_bytes(metadata)
    source_job = {
        "episodes": episodes,
        "dataset_root": str(source / "dataset"),
        "scenes_dir": str(source / "scenes"),
        "split": "val_unseen",
        "checkpoint": str(checkpoint),
        "checkpoint_metadata_sha256": hashlib.sha256(metadata).hexdigest(),
        "resources": {"ndtw": {"sha256": "unchanged-ground-truth"}},
        "habitat_python": str(source / "habitat-python"),
        "model_path": str(source / "model"),
        "cpu_threads": 2,
        "stop_threshold": 0.5,
        "fm_steps": 10,
        "fm_seed": 42,
        "max_steps": 500,
        "num_shards": 64,
    }
    unchanged_job = copy.deepcopy(source_job)
    output = source / "recordings"
    output.mkdir()
    ordered = sorted(episodes, key=lambda episode: episode["scene_id"])
    selected_by_index = {sample["dataset_index"]: sample for sample in selection["samples"]}

    load_dataset = Mock(return_value=episodes)
    monkeypatch.setattr(sample_videos, "load_expected_episodes", load_dataset)
    validate = Mock(return_value={"ndtw": {"sha256": "unchanged-ground-truth"}})
    monkeypatch.setattr(eval_preflight, "validate_resources", validate)
    environment = {"CPU_TEST_ONLY": "1"}
    build_environment = Mock(return_value=environment)
    check_python = Mock(return_value={"imports_ok": True})
    monkeypatch.setattr(eval_preflight, "build_habitat_env", build_environment)
    monkeypatch.setattr(eval_preflight, "check_habitat_python", check_python)
    configure_device = Mock()
    monkeypatch.setattr(fm_stop_worker, "configure_device", configure_device)
    video_dependencies = Mock()
    monkeypatch.setattr(runner, "_require_video_deps", video_dependencies)
    cached_engine = (object(), object())
    load_engine = Mock(return_value=cached_engine)
    monkeypatch.setattr(runner, "_default_engine_factory", load_engine)

    shard_calls = []
    inspected = []

    def run_shard(job, worker, index, engine, *, config_factory):
        assert engine is cached_engine
        assert job["num_shards"] == len(episodes)
        assert job["output_dir"] == str(output)
        assert worker == {"worker_id": 0, "port": 28991}
        assert index in selected_by_index
        sample = selected_by_index[index]
        original = sample["original"]
        assert _key(ordered[index]) == _key(original)
        folder = output / f"shard_{index:04d}"
        folder.mkdir()
        cfg = config_factory(job, worker, folder)
        assert cfg.episodes == 1
        assert cfg.save_video is True and cfg.video_fps == 7
        assert cfg.record_dir == str(folder / "records")
        assert cfg.output_dir == str(folder)
        assert cfg.beta_checkpoint == str(checkpoint)
        assert cfg.fm_seed == 42 and cfg.fm_steps == 10
        assert cfg.fm_only_stop_mode == "learned" and cfg.stop_threshold == 0.5
        assert cfg.oracle_stop_distance is None
        manifest = json.loads(Path(cfg.episode_manifest).read_text())
        assert manifest == [{
            "scene_id": original["scene_id"],
            "habitat_episode_id": original["habitat_episode_id"],
        }]
        assert Path(manifest[0]["scene_id"]).is_absolute()
        video_relative = "videos/episode_000.mp4"
        (folder / "videos").mkdir()
        (folder / video_relative).write_bytes(f"mock-video-{index}".encode())
        _write_records(folder, [{**original, "video": video_relative}])
        shard_calls.append(index)

    monkeypatch.setattr(fm_stop_worker, "run_shard", run_shard)

    def inspect_video(path, expected_frames, expected_fps):
        index = int(path.parent.parent.name.removeprefix("shard_"))
        original = selected_by_index[index]["original"]
        assert path.read_bytes() == f"mock-video-{index}".encode()
        assert expected_frames == original["steps"] + 1
        assert expected_fps == 7
        inspected.append(index)
        return {"frames": expected_frames, "fps": 7, "width": 64, "height": 48}

    monkeypatch.setattr(sample_videos, "inspect_video", inspect_video)
    comparisons = sample_videos.record_samples(source_job, selection, output, port=28991, fps=7)

    assert source_job == unchanged_job
    load_dataset.assert_called_once_with(source_job["dataset_root"], "val_unseen")
    validate.assert_called_once_with(
        source_job["dataset_root"], source_job["scenes_dir"],
        [ordered[sample["dataset_index"]] for sample in selection["samples"]], "val_unseen",
    )
    build_environment.assert_called_once_with(sample_videos.REPOSITORY)
    check_python.assert_called_once_with(source_job["habitat_python"], environment)
    configure_device.assert_called_once_with(2)
    video_dependencies.assert_called_once_with()
    load_engine.assert_called_once()
    assert load_engine.call_args.args[0].beta_checkpoint == str(checkpoint)
    assert load_engine.call_args.args[0].output_dir == str(output)
    assert shard_calls == inspected == [sample["dataset_index"] for sample in selection["samples"]]
    assert len(comparisons) == len(list((output / "videos").glob("*.mp4"))) == 4
    assert json.loads((output / "comparison.json").read_text()) == comparisons
    for number, comparison in enumerate(comparisons, 1):
        assert comparison["comparison"]["matches_original"] is True
        video = Path(comparison["video"])
        sample = comparison["sample"]
        scan, episode = _key(sample["original"])
        outcome = "success" if sample["original"]["success"] else "failure"
        assert video.name == f"sample_{number:02d}_{scan}_episode_{episode}_{outcome}.mp4"
        assert video.read_bytes() == f"mock-video-{sample['dataset_index']}".encode()
    replay_job = json.loads((output / "replay_job.json").read_text())
    assert replay_job["source_job"] == source_job
    assert replay_job["server_port"] == 28991 and replay_job["video_fps"] == 7


@pytest.fixture
def short_cpu_video(tmp_path):
    cv2 = pytest.importorskip("cv2")
    np = pytest.importorskip("numpy")
    path = tmp_path / "short.mp4"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 5, (64, 48))
    if not writer.isOpened():
        pytest.skip("OpenCV was installed without an MP4 encoder")
    try:
        for index in range(5):
            writer.write(np.full((48, 64, 3), index * 40, dtype=np.uint8))
    finally:
        writer.release()
    return path


def test_inspect_video_decodes_complete_cpu_video(short_cpu_video):
    from lightnav.habitat.sample_videos import inspect_video

    assert inspect_video(short_cpu_video, expected_frames=5, expected_fps=5) == {
        "frames": 5, "fps": 5.0, "width": 64, "height": 48, "duration_seconds": 1.0,
    }


@pytest.mark.parametrize("frames,fps", [(6, 5), (5, 6)])
def test_inspect_video_rejects_missing_frames_or_wrong_fps(short_cpu_video, frames, fps):
    from lightnav.habitat.sample_videos import inspect_video

    with pytest.raises(ValueError, match="incomplete video"):
        inspect_video(short_cpu_video, expected_frames=frames, expected_fps=fps)
