"""Merge the per-shard outputs of a parallel Habitat evaluation into one summary.

``scripts/eval_habitat.sh`` runs one env server + one eval client per GPU, each on a
disjoint shard of the split (``--split-id i --split-num N``) and each writing its own
``results.jsonl`` / ``summary.json``. Every metric in a summary is an unweighted
per-episode mean, so the merged numbers are simply the summary of the concatenated
episode records — this module concatenates them and reuses the same summary writers.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from lightnav.habitat.results import make_json_safe, print_objectnav_summary, print_vlnce_summary

RESULTS_FILENAME = "results.jsonl"
SUMMARY_FILENAME = "summary.json"


def find_shard_dirs(roots: list[str | Path]) -> list[Path]:
    """Directories holding a ``results.jsonl``: the roots themselves or their children."""
    found: list[Path] = []
    for root in roots:
        root = Path(root)
        if (root / RESULTS_FILENAME).is_file():
            found.append(root)
            continue
        if root.is_dir():
            found.extend(sorted(p.parent for p in root.glob(f"*/{RESULTS_FILENAME}")))
    seen: set[Path] = set()
    out: list[Path] = []
    for p in found:
        rp = p.resolve()
        if rp not in seen:
            seen.add(rp)
            out.append(p)
    return out


def load_results(shard_dir: Path) -> list[dict[str, Any]]:
    """Episode records of one shard (unparsable lines are skipped)."""
    records: list[dict[str, Any]] = []
    with open(shard_dir / RESULTS_FILENAME, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(rec, dict):
                records.append(rec)
    return records


def _load_summary(shard_dir: Path) -> dict[str, Any]:
    path = shard_dir / SUMMARY_FILENAME
    if not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return data if isinstance(data, dict) else {}


def merge_results(
    shard_dirs: list[Path],
    output_dir: str | Path,
    *,
    extra_info: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    """Concatenate the shards' episode records and write the merged summary.

    Writes ``<output_dir>/results.jsonl`` (all records, shard order, each tagged with
    ``"shard": <dir name>``) and ``<output_dir>/summary.json``. ``total_time_sec`` in the
    merged summary is the LONGEST shard time (the shards ran in parallel, so that is the
    wall-clock duration) and ``shards`` lists the per-shard episode counts. Returns the
    summary dict, or None when no shard holds any episode.
    """
    if not shard_dirs:
        raise ValueError("no shard directories given")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_results: list[dict[str, Any]] = []
    shard_infos: list[dict[str, Any]] = []
    elapsed = 0.0
    info: dict[str, Any] = dict(extra_info or {})
    for shard in shard_dirs:
        records = load_results(shard)
        for rec in records:
            video = rec.get("video")
            if video and not Path(str(video)).is_absolute():
                rec["video"] = str(Path(shard.name) / str(video))
            rec.setdefault("shard", shard.name)
        all_results.extend(records)
        summary = _load_summary(shard)
        shard_infos.append({"dir": str(shard), "episodes": len(records)})
        elapsed = max(elapsed, float(summary.get("total_time_sec") or 0.0))
        for key in (
            "model",
            "backend",
            "action_output",
            "beta_checkpoint",
            "beta_checkpoint_step",
            "fm_seed",
            "fm_steps",
            "fm_stop_mode",
            "training_mode",
            "stop_protocol",
            "stop_threshold",
            "stop_confirm_steps",
            "stop_head_type",
            "stop_positive_weight",
            "stop_training_data_sha256",
            "stop_supervision",
            "max_steps",
            "episode_manifest",
            "oracle_stop_distance",
        ):
            if key not in info and summary.get(key) is not None:
                info[key] = summary[key]

    merged_jsonl = output_dir / RESULTS_FILENAME
    with open(merged_jsonl, "w", encoding="utf-8") as f:
        for rec in all_results:
            f.write(json.dumps(make_json_safe(rec)) + "\n")

    if not all_results:
        print("No episodes found in the given shard directories.")
        return None

    info["shards"] = shard_infos
    if any(r.get("object_category") for r in all_results):
        summary = print_objectnav_summary(all_results, elapsed, str(output_dir), extra_info=info)
    else:
        summary = print_vlnce_summary(all_results, elapsed, str(output_dir), extra_info=info)
    print(f"Merged {len(all_results)} episodes from {len(shard_dirs)} shard(s) -> {output_dir}")
    print(f"  {os.path.relpath(merged_jsonl)} / {SUMMARY_FILENAME}")
    return summary
