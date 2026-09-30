"""Audit an official R2R development split against the frozen FM training set.

Scene overlap is expected for val_seen. Episode IDs are local to each source
split, so leakage is checked using actual observations, goals and instructions,
plus complete reference routes when available. This says nothing about the
released VLM backbone's pretraining data.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import math
import unicodedata
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from lightnav.habitat.stop_report import episode_key

DEFAULT_MANIFEST = Path(__file__).resolve().parents[3] / "configs/betanav/data/r2r_all_train_stop_v1.json"


def _digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _vector(value: Any, size: int, field: str) -> tuple[float, ...]:
    if (not isinstance(value, (list, tuple)) or len(value) != size
            or any(isinstance(x, bool) or not isinstance(x, (int, float))
                   or not math.isfinite(x) for x in value)):
        raise ValueError(f"Development split audit requires finite {field} of length {size}")
    return tuple(float(x) for x in value)


def _point(value: Any, field: str) -> tuple[float, ...]:
    return tuple(round(x, 6) for x in _vector(value, 3, field))


def _sample_identity(episode: Mapping[str, Any]) -> tuple[Any, ...]:
    scene, _ = episode_key(episode.get("scene_id"), episode.get("episode_id"))
    instruction = episode.get("instruction")
    text = instruction.get("instruction_text") if isinstance(instruction, Mapping) else instruction
    if not isinstance(text, str) or not text.strip():
        raise ValueError("Development split audit requires a nonempty instruction_text")
    normalized = " ".join(unicodedata.normalize("NFKC", text).casefold().split())
    position = _point(episode.get("start_position"), "start_position")
    rotation = _vector(episode.get("start_rotation"), 4, "start_rotation quaternion")
    norm = math.sqrt(sum(x * x for x in rotation))
    if norm < 1e-12:
        raise ValueError("Development split audit requires a nonzero start_rotation quaternion")
    rotation = tuple(x / norm for x in rotation)
    # q and -q represent the same orientation.
    sign = next((1 if x > 0 else -1 for x in rotation if abs(x) > 1e-12), 1)
    rotation = tuple(round(sign * x, 6) for x in rotation)
    goals = episode.get("goals")
    if not isinstance(goals, list) or not goals or any(not isinstance(g, Mapping) for g in goals):
        raise ValueError("Development split audit requires nonempty goals")
    goal_positions = tuple(sorted(_point(g.get("position"), "goal.position") for g in goals))
    return scene, position, rotation, goal_positions, normalized


def _route_identity(episode: Mapping[str, Any]) -> tuple[Any, ...] | None:
    route = episode.get("reference_path")
    if route is None:
        return None
    if not isinstance(route, list) or not route:
        raise ValueError("reference_path must be a nonempty list when present")
    scene, _ = episode_key(episode.get("scene_id"), episode.get("episode_id"))
    return scene, tuple(_point(point, "reference_path point") for point in route)


def _read_source(dataset_root: Path, split: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    path = dataset_root / split / f"{split}.json.gz"
    try:
        compressed = path.read_bytes()
    except OSError as exc:
        raise type(exc)(f"Cannot read required {split} source {path}; no split fallback is allowed: {exc}") from exc
    payload = json.loads(gzip.decompress(compressed))
    episodes = payload.get("episodes") if isinstance(payload, Mapping) else None
    if not isinstance(episodes, list) or not episodes:
        raise ValueError(f"{path}: episodes must be a nonempty list")
    keys = [episode_key(e.get("scene_id"), e.get("episode_id")) for e in episodes]
    if len(keys) != len(set(keys)):
        raise ValueError(f"{path}: duplicate scene/episode identities")
    return episodes, {"path": str(path), "sha256": _digest(compressed), "num_episodes": len(episodes)}


def verify_dev_split(
    dataset_root: str | Path,
    episodes: Sequence[Mapping[str, Any]],
    *,
    split: str = "val_seen",
    checkpoint_metadata: Mapping[str, Any] | None = None,
    training_manifest: str | Path = DEFAULT_MANIFEST,
) -> dict[str, Any]:
    """Fail closed unless val_seen is disjoint from this checkpoint's FM train.

    The checkpoint's data_sha256 binds the manifest by content, allowing the
    same frozen manifest to be relocated. Audited source scene/episode pairs
    map LeRobot training indices into the official train source. Cross-split
    comparisons then ignore local episode IDs and compare normalized sample
    content; an identical reference route also counts as leakage even when its
    instruction differs. The complete val_seen source must be supplied.
    """
    if split != "val_seen":
        raise ValueError("Independent STOP calibration requires val_seen; no train/val_unseen fallback")
    if not episodes:
        raise ValueError("Development episodes must be nonempty")
    root = Path(dataset_root).absolute()
    dev, dev_source = _read_source(root, split)
    manifest_path = Path(training_manifest).absolute()
    manifest_bytes = manifest_path.read_bytes()
    manifest_hash = _digest(manifest_bytes)
    manifest = json.loads(manifest_bytes)
    contract = checkpoint_metadata.get("contract", {}) if isinstance(checkpoint_metadata, Mapping) else {}
    if not isinstance(contract, Mapping) or contract.get("data_sha256") != manifest_hash:
        raise ValueError("Training manifest SHA-256 differs from or is missing in checkpoint contract.data_sha256")
    config = contract.get("config", {})
    if not isinstance(config, Mapping) or config.get("training_mode") != "fm_stop":
        raise ValueError("Development split audit requires the fm_stop checkpoint training contract")
    if (manifest.get("format") != "betanav_training_data_v1" or manifest.get("split_by") != "all"
            or manifest.get("val_episodes") != []):
        raise ValueError("Expected the frozen all-train manifest for this FM + STOP checkpoint")
    if Path(str(config.get("dataset_root", ""))) != Path(str(manifest.get("dataset_root", ""))):
        raise ValueError("Checkpoint and training manifest dataset_root differ")
    train_indices = manifest.get("train_episodes")
    supervision = manifest.get("stop_supervision", {})
    if (not isinstance(train_indices, list) or not train_indices
            or any(type(index) is not int for index in train_indices)
            or len(train_indices) != len(set(train_indices))
            or not isinstance(supervision, Mapping)
            or set(supervision.get("train_episode_ids", [])) != set(train_indices)):
        raise ValueError("Invalid audited training episode membership")
    audited = supervision.get("episodes", {})
    if not isinstance(audited, Mapping) or set(audited) != {str(i) for i in train_indices}:
        raise ValueError("Audited source records must cover the all-train manifest exactly")
    source_keys = []
    for index in train_indices:
        entry = audited[str(index)]
        source_keys.append(episode_key(entry.get("source_scene_id"), entry.get("source_episode_id")))
    if len(source_keys) != len(set(source_keys)):
        raise ValueError("Duplicated source identities in training manifest")
    train, train_source = _read_source(root, "train")
    train_by_key = {episode_key(e["scene_id"], e["episode_id"]): e for e in train}
    if set(source_keys) != set(train_by_key):
        raise ValueError("Original train source does not exactly cover the frozen all-train source identities")

    dev_by_key = {episode_key(e["scene_id"], e["episode_id"]): e for e in dev}
    provided_keys = [episode_key(e.get("scene_id"), e.get("episode_id")) for e in episodes]
    if len(provided_keys) != len(set(provided_keys)) or set(provided_keys) != set(dev_by_key):
        raise ValueError("Development episode list must cover the complete val_seen source exactly")
    for key, episode in zip(provided_keys, episodes):
        if (_sample_identity(episode) != _sample_identity(dev_by_key[key])
                or _route_identity(episode) != _route_identity(dev_by_key[key])):
            raise ValueError(f"Development episode content differs from original source: {key}")

    train_samples = {_sample_identity(e) for e in train}
    train_routes = {route for e in train if (route := _route_identity(e)) is not None}
    sample_overlap, route_overlap = [], []
    for episode in dev:
        key = episode_key(episode["scene_id"], episode["episode_id"])
        if _sample_identity(episode) in train_samples:
            sample_overlap.append(key)
        route = _route_identity(episode)
        if route is not None and route in train_routes:
            route_overlap.append(key)
    if sample_overlap or route_overlap:
        raise ValueError(
            "Development/train leakage detected: "
            f"{len(sample_overlap)} duplicate samples, {len(route_overlap)} duplicate reference routes; "
            f"examples={sorted(set(sample_overlap + route_overlap))[:5]}"
        )
    train_scenes = {key[0] for key in train_by_key}
    dev_scenes = {key[0] for key in dev_by_key}
    return {
        "split": split, "verified": True,
        "scope": "FM + STOP training manifest; released VLM pretraining is not audited",
        "training_manifest": str(manifest_path), "training_manifest_sha256": manifest_hash,
        "checkpoint_manifest_sha256": contract["data_sha256"],
        "train_source": train_source, "dev_source": dev_source,
        "num_training_episodes": len(train), "num_development_episodes": len(dev),
        "num_training_scenes": len(train_scenes), "num_development_scenes": len(dev_scenes),
        "shared_scenes_allowed": sorted(train_scenes & dev_scenes),
        "duplicate_samples": 0, "duplicate_reference_routes": 0,
        "identity_rule": "scene/start_pose/goals/NFKC_casefold_whitespace_instruction; coordinates rounded to 1e-6",
        "route_rule": "scene/full_reference_path independent of instruction; coordinates rounded to 1e-6",
    }
