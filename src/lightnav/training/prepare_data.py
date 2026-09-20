"""Create an all-training or scene-separated manifest and FM scales from saved metadata.

This training utility reads episode statistics, never RGB/depth, and writes one new manifest.
The exporter-provided beta_nav.json is read once here, not in training workers.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def combine_statistics(rows, floors):
    """Parallel-variance combination; zero-count STOP-only episodes contribute nothing."""
    count, mean, m2 = 0, np.zeros(3), np.zeros(3)
    prefix = "stats/action.fm.clean_se2_diff/"
    for row in rows:
        n = int(np.asarray(row[prefix + "count"]).item())
        if n < 0:
            raise ValueError("negative FM statistics count")
        if not n:
            continue
        mu, std = [np.asarray(row[prefix + name], dtype=np.float64) for name in ("mean", "std")]
        if mu.shape != (3,) or std.shape != (3,) or not np.isfinite([mu, std]).all() or (std < 0).any():
            raise ValueError("invalid per-episode FM statistics")
        delta = mu - mean
        m2 += std ** 2 * n + delta ** 2 * count * n / (count + n)
        mean += delta * n / (count + n)
        count += n
    floors = np.asarray(floors, dtype=np.float64)
    if not count or floors.shape != (3,) or not np.isfinite(floors).all() or (floors <= 0).any():
        raise ValueError("training actions and three positive scale floors are required")
    std = np.sqrt(m2 / count)
    return dict(valid_steps=count, mean=mean.tolist(), std=std.tolist(), scale_floor=floors.tolist(),
                action_scale=np.maximum(std, floors).tolist(), transform="clean_se2_diff / action_scale")


def make_manifest(root, *, seed=42, val_fraction=0.1, scale_floor=(0.05, 0.02, 0.05), normalization_from=None):
    import pyarrow.parquet as pq

    root = Path(root).resolve()
    if not 0 <= val_fraction < 1:
        raise ValueError("val_fraction must be in [0,1); 0 uses all episodes for training")
    custom_path = root / "meta/beta_nav.json"
    custom = json.loads(custom_path.read_text())
    records = {int(e["episode_index"]): (e["scene_id"], int(e["frames"])) for e in custom["episodes"]}
    if len(records) != len(custom["episodes"]):
        raise ValueError("duplicate episode identities")
    del custom
    scenes = sorted({s for s, _ in records.values()},
                    key=lambda s: hashlib.sha256(f"{seed}:{s}".encode()).digest())
    if val_fraction and len(scenes) < 2:
        raise ValueError("scene-separated validation requires at least two scenes")
    val_scenes = set(scenes[:max(1, min(len(scenes) - 1, round(len(scenes) * val_fraction)))]) if val_fraction else set()
    train = sorted(i for i, (s, _) in records.items() if s not in val_scenes)
    validation = sorted(set(records) - set(train))
    train_ids, stats_rows, seen, fingerprints = set(train), [], set(), {}
    prefix = "stats/action.fm.clean_se2_diff/"
    columns = ["episode_index", "length", *[prefix + k for k in ("mean", "std", "count")]]
    for path in sorted((root / "meta/episodes").rglob("*.parquet")):
        fingerprints[str(path.relative_to(root))] = sha256(path)
        for row in pq.read_table(path, columns=columns, use_threads=False).to_pylist():
            i = row["episode_index"]
            if i in seen or i not in records or row["length"] != records[i][1]:
                raise ValueError("episode metadata identity/length mismatch")
            seen.add(i)
            if i in train_ids:
                stats_rows.append(row)
    info = json.loads((root / "meta/info.json").read_text())
    if seen != set(records) or len(seen) != info["total_episodes"] or sum(n for _, n in records.values()) != info["total_frames"]:
        raise ValueError("episode metadata coverage mismatch")
    fingerprints["meta/info.json"] = sha256(root / "meta/info.json")
    fingerprints["meta/beta_nav.json"] = sha256(custom_path)
    result = dict(format="betanav_training_data_v1", dataset_root=str(root), seed=seed,
                split_by="scene" if val_fraction else "all", val_fraction=val_fraction, train_episodes=train, val_episodes=validation,
                train_scenes=sorted(set(scenes) - val_scenes), val_scenes=sorted(val_scenes),
                train_frames=sum(records[i][1] for i in train), val_frames=sum(records[i][1] for i in validation),
                normalization=combine_statistics(stats_rows, scale_floor), source_sha256=fingerprints)
    if normalization_from is not None:
        previous = json.loads(Path(normalization_from).read_text())
        if previous["dataset_root"] != str(root) or previous["source_sha256"] != fingerprints:
            raise ValueError("normalization source must describe the same unchanged dataset")
        result["normalization"] = previous["normalization"]
        result["normalization_source"] = dict(path=str(Path(normalization_from).resolve()), sha256=sha256(normalization_from),
                                              reason="Preserve the action scale used by existing trained weights")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--scale-floor", type=float, nargs=3, default=[0.05, 0.02, 0.05])
    parser.add_argument("--normalization-from", type=Path, help="reuse a frozen scale from the same dataset for continued training")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output already exists; use a new manifest path")
    result = make_manifest(args.dataset, seed=args.seed, val_fraction=args.val_fraction, scale_floor=args.scale_floor,
                           normalization_from=args.normalization_from)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2, ensure_ascii=False)
        stream.write("\n")
    print(json.dumps({k: result[k] for k in ("train_frames", "val_frames", "normalization")}, indent=2))


if __name__ == "__main__":
    main()
