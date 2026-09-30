"""Development calibration must not reuse FM train samples or local ID semantics."""

from __future__ import annotations

import copy
import gzip
import hashlib
import json
from pathlib import Path

import pytest

from lightnav.habitat.dev_split import verify_dev_split


def _episode(identifier=1, *, scene="house", text="Go left.", start=(0, 0, 0), goal=(1, 0, 0)):
    return {
        "episode_id": identifier, "scene_id": f"mp3d/{scene}/{scene}.glb",
        "start_position": list(start), "start_rotation": [0, 0, 0, 1],
        "goals": [{"position": list(goal)}],
        "instruction": {"instruction_text": text},
        "reference_path": [list(start), list(goal)],
    }


def _write_split(root, split, episodes):
    path = root / split / f"{split}.json.gz"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(gzip.compress(json.dumps({"episodes": episodes}).encode()))
    return path


@pytest.fixture
def audit(tmp_path):
    train = [_episode()]
    dev = [_episode(text="Go right.", start=(0, 1, 0), goal=(2, 1, 0))]
    _write_split(tmp_path, "train", train)
    _write_split(tmp_path, "val_seen", dev)
    manifest = {
        "format": "betanav_training_data_v1", "split_by": "all", "val_episodes": [],
        "dataset_root": "/training/data", "train_episodes": [0],
        "stop_supervision": {
            "train_episode_ids": [0],
            "episodes": {"0": {"source_episode_id": "1", "source_scene_id": "mp3d/house/house.glb"}},
        },
    }
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest))
    metadata = {"contract": {
        "data_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "config": {"training_mode": "fm_stop", "dataset_root": "/training/data",
                   "data_manifest": "/original/location/manifest.json"},
    }}
    return tmp_path, train, dev, path, metadata


def _verify(audit, **kwargs):
    root, _, dev, manifest, metadata = audit
    return verify_dev_split(root, dev, checkpoint_metadata=metadata, training_manifest=manifest, **kwargs)


def test_seen_scenes_and_reused_split_local_ids_are_allowed(audit):
    result = _verify(audit)
    assert result["verified"] is True
    assert result["num_training_episodes"] == result["num_development_episodes"] == 1
    assert result["shared_scenes_allowed"] == ["house"]
    assert result["duplicate_samples"] == result["duplicate_reference_routes"] == 0
    assert result["train_source"]["sha256"] != result["dev_source"]["sha256"]
    # A manifest relocated without changing bytes still binds to its checkpoint.
    assert result["training_manifest_sha256"] == result["checkpoint_manifest_sha256"]


def test_identical_sample_relabelled_with_new_id_is_leakage(audit):
    root, train, dev, _, _ = audit
    dev[0] = copy.deepcopy(train[0])
    dev[0]["episode_id"] = 999
    dev[0]["instruction"]["instruction_text"] = "  GO   LEFT. \n"
    dev[0]["start_rotation"] = [0, 0, 0, -1]
    _write_split(root, "val_seen", dev)
    with pytest.raises(ValueError, match="1 duplicate samples"):
        _verify(audit)


def test_same_route_with_different_instruction_is_leakage(audit):
    root, train, dev, _, _ = audit
    dev[0] = copy.deepcopy(train[0])
    dev[0]["episode_id"] = 999
    dev[0]["instruction"]["instruction_text"] = "Move through the doorway."
    _write_split(root, "val_seen", dev)
    with pytest.raises(ValueError, match="0 duplicate samples, 1 duplicate reference routes"):
        _verify(audit)


def test_scene_name_is_part_of_sample_and_route_identity(audit):
    root, train, dev, _, _ = audit
    dev[0] = copy.deepcopy(train[0])
    dev[0]["scene_id"] = "mp3d/another/another.glb"
    _write_split(root, "val_seen", dev)
    assert _verify(audit)["shared_scenes_allowed"] == []


def test_coordinate_equivalence_and_scene_prefix_do_not_hide_leakage(audit):
    root, train, dev, _, _ = audit
    dev[0] = copy.deepcopy(train[0])
    dev[0]["scene_id"] = "data/scene_datasets/mp3d/house/house.glb"
    dev[0]["start_position"][0] += 1e-8
    dev[0].pop("reference_path")
    _write_split(root, "val_seen", dev)
    with pytest.raises(ValueError, match="1 duplicate samples"):
        _verify(audit)


def test_missing_val_seen_is_not_replaced_with_available_train(audit):
    root, *_ = audit
    (root / "val_seen/val_seen.json.gz").unlink()
    with pytest.raises(FileNotFoundError, match="no split fallback"):
        _verify(audit)


def test_val_seen_permission_failure_preserves_actionable_path(audit, monkeypatch):
    original = Path.read_bytes

    def read(path):
        if path.name == "val_seen.json.gz":
            raise PermissionError("fixture denied")
        return original(path)

    monkeypatch.setattr(Path, "read_bytes", read)
    with pytest.raises(PermissionError, match="val_seen.json.gz.*no split fallback"):
        _verify(audit)


@pytest.mark.parametrize("split", ["train", "val_unseen", "../train", "test"])
def test_only_official_seen_dev_is_accepted(audit, split):
    with pytest.raises(ValueError, match="requires val_seen"):
        _verify(audit, split=split)


def test_changed_manifest_does_not_retain_checkpoint_binding(audit):
    audit[3].write_text(audit[3].read_text() + "\n")
    with pytest.raises(ValueError, match="Training manifest SHA-256"):
        _verify(audit)


def test_missing_checkpoint_contract_is_rejected(audit):
    root, _, dev, manifest, _ = audit
    with pytest.raises(ValueError, match="contract.data_sha256"):
        verify_dev_split(root, dev, training_manifest=manifest)


def test_manifest_training_root_must_match_checkpoint(audit):
    audit[4]["contract"]["config"]["dataset_root"] = "/other/dataset"
    with pytest.raises(ValueError, match="dataset_root differ"):
        _verify(audit)


def test_source_train_membership_must_match_all_trained_examples(audit):
    root, train, *_ = audit
    _write_split(root, "train", [*train, _episode(identifier=2)])
    with pytest.raises(ValueError, match="exactly cover"):
        _verify(audit)


def test_empty_and_incomplete_dev_coverage_rejected(audit):
    root, _, dev, manifest, metadata = audit
    with pytest.raises(ValueError, match="nonempty"):
        verify_dev_split(root, [], checkpoint_metadata=metadata, training_manifest=manifest)
    _write_split(root, "val_seen", [*dev, _episode(identifier=2, scene="another")])
    with pytest.raises(ValueError, match="complete val_seen"):
        _verify(audit)


def test_passed_episode_content_must_match_loaded_source(audit):
    audit[2][0]["instruction"]["instruction_text"] = "Fabricated new instruction"
    with pytest.raises(ValueError, match="differs from original source"):
        _verify(audit)


@pytest.mark.parametrize("field,value,match", [
    ("start_position", [0, float("nan"), 0], "finite start_position"),
    ("start_rotation", [0, 0, 0, 0], "nonzero start_rotation"),
    ("goals", [], "nonempty goals"),
    ("instruction", {"instruction_text": " "}, "nonempty instruction_text"),
    ("reference_path", [], "reference_path must be"),
])
def test_invalid_episode_geometry_or_instruction_fails_closed(audit, field, value, match):
    root, _, dev, _, _ = audit
    dev[0][field] = value
    _write_split(root, "val_seen", dev)
    with pytest.raises(ValueError, match=match):
        _verify(audit)
