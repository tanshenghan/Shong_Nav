"""Audited STOP supervision for frozen-backbone continuous training.

Preparation reads only explicit label columns and copied provenance, never pixels.
Workers consume the compact manifest contract and recheck each raw row; they do
not infer success from a final frame or load exporter provenance.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

STOP_LABEL_VERSION = "betanav_stop_labels_v1"
STOP_COLUMNS = [
    "index", "episode_index", "frame_index", "source.episode_id", "source.scene_id",
    "action.explicit_stop", "action.stop_loss_valid",
]


def _hash_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _inside(root, relative):
    path = (root / relative).resolve()
    if not path.is_relative_to(root) or not path.is_file():
        raise ValueError(f"invalid STOP supervision source path: {relative}")
    return path


def _boolean(value):
    if isinstance(value, list) and len(value) == 1:
        value = value[0]
    if type(value) is not bool:
        raise ValueError("STOP fields must be booleans, not inferred truth values")
    return value


def _canonical_row(row):
    result = {key: row[key] for key in STOP_COLUMNS}
    for key in ("action.explicit_stop", "action.stop_loss_valid"):
        result[key] = _boolean(result[key])
    return result


def _update_digest(digest, row):
    digest.update(json.dumps(row, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode())
    digest.update(b"\n")


def stop_target_for_row(episode, frame, row):
    """Return independent target/validity; never borrow the positive-only RVQ mask."""
    n = episode["length"]
    if type(frame) is not int or not 0 <= frame < n:
        raise ValueError("STOP frame index outside audited episode")
    if (str(row["source.episode_id"]) != episode["source_episode_id"]
            or row["source.scene_id"] != episode["source_scene_id"]):
        raise ValueError("STOP source episode/scene identity mismatch")
    explicit = _boolean(row["action.explicit_stop"])
    positive_valid = _boolean(row["action.stop_loss_valid"])
    if positive_valid != explicit or (explicit and frame != n - 1):
        raise ValueError("contradictory explicit STOP fields or nonterminal STOP")
    if episode["successful"]:
        if explicit != (frame == n - 1):
            raise ValueError("successful audited episode must have one explicit terminal STOP")
        return float(explicit), True
    if explicit:
        raise ValueError("explicit STOP contradicts unsuccessful/unknown termination provenance")
    return 0.0, False


def _episode_audit(record, metadata):
    if (str(metadata.get("episode_id")) != str(record["episode_id"])
            or metadata.get("scene_id") != record["scene_id"]
            or metadata.get("frames") != record["frames"]):
        raise ValueError("STOP provenance episode/frame identity mismatch")
    for key in ("complete", "terminal_stop_supervised"):
        if key not in metadata or type(metadata[key]) is not bool:
            raise ValueError(f"STOP provenance lacks explicit {key} status")
    success = metadata["complete"] and metadata["terminal_stop_supervised"]
    if metadata["terminal_stop_supervised"] and not metadata["complete"]:
        raise ValueError("terminal STOP provenance claims an incomplete episode")
    if success:
        distance = metadata.get("goal_distance_m")
        deviation = metadata.get("max_gt_route_deviation_m")
        if (not isinstance(distance, (float, int)) or not math.isfinite(distance) or not 0 <= distance < 3
                or not isinstance(deviation, (float, int)) or not math.isfinite(deviation) or not 0 <= deviation <= .5):
            raise ValueError("STOP provenance lacks successful expert replay geometry")
    return dict(length=record["frames"], source_episode_id=str(record["episode_id"]),
                source_scene_id=record["scene_id"], successful=success)


def audit_stop_supervision(root, records, episode_rows, train_ids):
    """Audit the project's beta_nav_lerobot_v1 export, including merged shards."""
    import pyarrow.parquet as pq

    root = Path(root).resolve()
    info = json.loads((root / "meta/info.json").read_text())
    episodes, provenance_hashes, source_labels = {}, {}, {}
    shard_offsets = Counter()
    for record in sorted(records, key=lambda row: row["episode_index"]):
        index = record["episode_index"]
        if "source_shard" in record:
            shard = record["source_shard"]
            if type(shard) is not int or shard < 0:
                raise ValueError("invalid STOP provenance shard identity")
            local = shard_offsets[shard]
            shard_offsets[shard] += 1
            folder = f"meta/provenance/shards/{shard:05d}/episodes/{local:06d}"
        else:
            folder = f"meta/provenance/episodes/{index:06d}"
        saved = {}
        for name in ("episode.json", "samples.jsonl"):
            relative = f"{folder}/{name}"
            path = _inside(root, relative)
            digest = _hash_file(path)
            if record.get("source_hashes", {}).get(name) != digest:
                raise ValueError(f"STOP provenance differs from exporter fingerprint: {relative}")
            provenance_hashes[relative] = digest
            saved[name] = path
        episode = _episode_audit(record, json.loads(saved["episode.json"].read_text()))
        labels = []
        with saved["samples.jsonl"].open() as stream:
            for frame, line in enumerate(stream):
                sample = json.loads(line)
                if sample.get("frame_id") != frame or sample.get("array_row") != frame:
                    raise ValueError("STOP provenance samples are not frame aligned")
                row = {"source.episode_id": episode["source_episode_id"],
                       "source.scene_id": episode["source_scene_id"],
                       "action.explicit_stop": sample["explicit_stop"],
                       "action.stop_loss_valid": sample["stop_loss_valid"]}
                stop_target_for_row(episode, frame, row)
                labels.append((_boolean(sample["explicit_stop"]), _boolean(sample["stop_loss_valid"])))
        if len(labels) != episode["length"]:
            raise ValueError("STOP provenance samples/frame count mismatch")
        episodes[str(index)], source_labels[index] = episode, labels

    files = defaultdict(dict)
    metadata = {row["episode_index"]: row for row in episode_rows}
    if set(metadata) != {int(i) for i in episodes}:
        raise ValueError("STOP audit episode coverage mismatch")
    offset = 0
    for expected_index, row in enumerate(sorted(episode_rows, key=lambda value: value["episode_index"])):
        if (row["episode_index"] != expected_index or row["length"] <= 0
                or row["dataset_from_index"] != offset or row["dataset_to_index"] != offset + row["length"]):
            raise ValueError("STOP metadata episode/global frame ranges must be complete and contiguous")
        offset += row["length"]
    if offset != info["total_frames"] or len(metadata) != info["total_episodes"]:
        raise ValueError("STOP metadata global frame coverage mismatch")
    for index, row in metadata.items():
        relative = info["data_path"].format(chunk_index=row["data/chunk_index"], file_index=row["data/file_index"])
        files[relative][index] = row
    counts = {split: dict(positive=0, negative=0, invalid=0) for split in ("train", "validation")}
    rows_hashes = {}
    for relative, expected in sorted(files.items()):
        digest, seen = hashlib.sha256(), Counter()
        previous_index = -1
        file = pq.ParquetFile(_inside(root, relative))
        for batch in file.iter_batches(columns=STOP_COLUMNS, use_threads=False):
            for saved_row in batch.to_pylist():
                row = _canonical_row(saved_row)
                index, frame = row["episode_index"], row["frame_index"]
                if row["index"] <= previous_index:
                    raise ValueError("STOP data global indices must be strictly increasing")
                previous_index = row["index"]
                if index not in expected or frame != seen[index]:
                    raise ValueError("STOP data episode/frame ordering mismatch")
                meta = expected[index]
                if row["index"] != meta["dataset_from_index"] + frame:
                    raise ValueError("STOP data global frame identity mismatch")
                if frame >= len(source_labels[index]) or source_labels[index][frame] != (
                    row["action.explicit_stop"], row["action.stop_loss_valid"]
                ):
                    raise ValueError("STOP data labels disagree with provenance samples")
                target, valid = stop_target_for_row(episodes[str(index)], frame, row)
                split = "train" if index in train_ids else "validation"
                counts[split]["positive" if valid and target else "negative" if valid else "invalid"] += 1
                seen[index] += 1
                _update_digest(digest, row)
        if any(seen[index] != row["length"] for index, row in expected.items()):
            raise ValueError("STOP data episode length mismatch")
        rows_hashes[relative] = digest.hexdigest()
    train = counts["train"]
    if not train["positive"] or not train["negative"]:
        raise ValueError("STOP training requires both positive and negative examples")
    result = dict(version=STOP_LABEL_VERSION, episodes=episodes, counts=counts,
                  train_episode_ids=sorted(train_ids),
                  validation_episode_ids=sorted(set(metadata) - set(train_ids)),
                  positive_weight=train["negative"] / train["positive"],
                  source_sha256=provenance_hashes, rows_sha256=rows_hashes)
    validate_stop_supervision(result)
    return result


def validate_stop_supervision(supervision):
    """Validate the compact, immutable training/worker contract."""
    if not isinstance(supervision, dict) or supervision.get("version") != STOP_LABEL_VERSION:
        raise ValueError("fm_stop requires an audited manifest prepared with --with-stop-labels")
    train = supervision.get("counts", {}).get("train", {})
    pos, neg = train.get("positive", 0), train.get("negative", 0)
    weight = supervision.get("positive_weight", 0)
    if (type(pos) is not int or type(neg) is not int or pos <= 0 or neg <= 0
            or not isinstance(weight, (float, int)) or not math.isfinite(weight) or weight != neg / pos):
        raise ValueError("invalid frozen STOP class counts/positive weight")
    episodes = supervision.get("episodes")
    if not isinstance(episodes, dict) or not episodes:
        raise ValueError("STOP supervision requires audited episodes")
    for key, episode in episodes.items():
        if (not isinstance(key, str) or not key.isdecimal() or str(int(key)) != key
                or type(episode.get("length")) is not int or episode["length"] < 1
                or type(episode.get("successful")) is not bool
                or not isinstance(episode.get("source_episode_id"), str)
                or not isinstance(episode.get("source_scene_id"), str)):
            raise ValueError("invalid audited STOP episode record")
    observed = set()
    for split, key in (("train", "train_episode_ids"), ("validation", "validation_episode_ids")):
        indices = supervision.get(key)
        if (not isinstance(indices, list) or any(type(index) is not int for index in indices)
                or len(set(indices)) != len(indices) or observed & set(indices)):
            raise ValueError("invalid STOP supervision split identities")
        observed.update(indices)
        counts = dict(positive=0, negative=0, invalid=0)
        for index in indices:
            episode = episodes.get(str(index))
            if episode is None:
                raise ValueError("STOP split contains an unaudited episode")
            if episode["successful"]:
                counts["positive"] += 1
                counts["negative"] += episode["length"] - 1
            else:
                counts["invalid"] += episode["length"]
        if supervision["counts"].get(split) != counts:
            raise ValueError("STOP counts disagree with audited episode/split coverage")
    if observed != {int(key) for key in episodes}:
        raise ValueError("STOP splits do not cover audited episodes")
    for field in ("source_sha256", "rows_sha256"):
        values = supervision.get(field)
        if not isinstance(values, dict) or not values:
            raise ValueError("STOP supervision requires provenance and label fingerprints")
        if any(not isinstance(digest, str) or len(digest) != 64
               or any(c not in "0123456789abcdef" for c in digest) for digest in values.values()):
            raise ValueError("invalid STOP source fingerprint")


def verify_stop_supervision(root, supervision):
    """Rank-zero verification without reading RGB, depth, or training workers."""
    import pyarrow.parquet as pq

    validate_stop_supervision(supervision)
    root = Path(root).resolve()
    for relative, expected in supervision["source_sha256"].items():
        if _hash_file(_inside(root, relative)) != expected:
            raise ValueError(f"STOP provenance changed after preparation: {relative}")
    for relative, expected in supervision["rows_sha256"].items():
        digest = hashlib.sha256()
        file = pq.ParquetFile(_inside(root, relative))
        for batch in file.iter_batches(columns=STOP_COLUMNS, use_threads=False):
            for row in batch.to_pylist():
                _update_digest(digest, _canonical_row(row))
        if digest.hexdigest() != expected:
            raise ValueError(f"STOP row labels changed after preparation: {relative}")
