"""Strict split coverage and STOP diagnostics for completed Habitat rollouts.

This module uses only the standard library. Rates are percentages and report their
denominators explicitly; a learned STOP, a zero FM command and a budget STOP are
different termination sources. Missing navigation metrics are errors rather than
implicit failures or zeros.
"""

from __future__ import annotations

import gzip
import json
import math
import statistics
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

_STOP_SOURCES = frozenset({
    "learned_head", "max_steps", "fm_zero_command", "oracle_stop",
    "explicit_token_stop", "near_zero_rvq_stop", "rvq_zero_command",
    "decode_failure", "environment_stop", "environment_truncated",
    "fm_zero_suppressed", "fm_only_zero_command_suppressed", "unknown",
})


def episode_key(scene_id: str, episode_id: str | int) -> tuple[str, str]:
    """Match an R2R scene asset path to Habitat's scene name and episode ID.

    Asset paths use the *containing scan directory*, not the asset's filename.
    A bare ``scan.glb`` lacks that identity and is rejected rather than guessed.
    """
    if not isinstance(scene_id, str) or not scene_id.strip():
        raise ValueError("scene_id must be a nonempty string")
    path = Path(scene_id.strip().replace("\\", "/"))
    scene = path.parent.name if path.suffix.lower() in {".glb", ".navmesh"} else path.name
    if not scene or scene in {".", ".."}:
        raise ValueError(f"scene_id has no scan directory/name: {scene_id!r}")
    if isinstance(episode_id, bool) or not isinstance(episode_id, (str, int)):
        raise ValueError("episode_id must be a nonempty string or integer")
    identifier = str(episode_id).strip()
    if not identifier:
        raise ValueError("episode_id must be nonempty")
    return scene, identifier


def _entry_key(entry: Mapping[str, Any], *, result: bool) -> tuple[str, str]:
    if not isinstance(entry, Mapping):
        raise ValueError("episode entries must be objects")
    # Results' episode_id is a shard-local display name (episode_000), whereas
    # habitat_episode_id is the dataset identity. Never fall back from an explicitly
    # empty Habitat ID to a display name; that would hide an invalid evaluation.
    field = "habitat_episode_id" if result and "habitat_episode_id" in entry else "episode_id"
    return episode_key(entry.get("scene_id"), entry.get(field))


def load_expected_episodes(
    dataset_root: str | Path, split: str = "val_unseen",
) -> list[dict[str, Any]]:
    """Read and validate ``<root>/<split>/<split>.json.gz`` without Habitat.

    The returned list preserves source order and full episode metadata, so callers
    may also use it to construct explicit disjoint worker manifests.
    """
    if not isinstance(split, str) or not split or Path(split).name != split or split in {".", ".."}:
        raise ValueError("split must be a single nonempty directory name")
    path = Path(dataset_root) / split / f"{split}.json.gz"
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        payload = json.load(stream)
    entries = payload.get("episodes") if isinstance(payload, dict) else None
    if not isinstance(entries, list) or not entries:
        raise ValueError(f"{path}: episodes must be a nonempty list")
    seen: set[tuple[str, str]] = set()
    for entry in entries:
        key = _entry_key(entry, result=False)
        if key in seen:
            raise ValueError(f"{path}: duplicate expected episode {key!r}")
        seen.add(key)
    return entries


def _key_record(key: tuple[str, str]) -> dict[str, str]:
    return {"scene_id": key[0], "episode_id": key[1]}


def check_coverage(
    expected_entries: Iterable[Mapping[str, Any]], records: Iterable[Mapping[str, Any]],
) -> dict[str, Any]:
    """Check exactly-once coverage, including duplicates across different shards.

    ``duplicates`` gives each repeated key's total record count, while
    ``duplicate_records`` counts only the excess records. Invalid identities or
    duplicate expected entries raise ``ValueError``.
    """
    expected_keys = [_entry_key(entry, result=False) for entry in expected_entries]
    expected = set(expected_keys)
    if len(expected) != len(expected_keys):
        raise ValueError("expected entries contain duplicate episode keys")
    actual = Counter(_entry_key(record, result=True) for record in records)
    missing = [_key_record(key) for key in sorted(expected - actual.keys())]
    unexpected = [_key_record(key) for key in sorted(actual.keys() - expected)]
    duplicates = [
        {**_key_record(key), "count": count}
        for key, count in sorted(actual.items()) if count > 1
    ]
    return {
        "expected": len(expected),
        "records": sum(actual.values()),
        "unique": len(actual),
        "missing": missing,
        "unexpected": unexpected,
        "duplicates": duplicates,
        "duplicate_records": sum(item["count"] - 1 for item in duplicates),
        "complete": bool(expected) and not (missing or unexpected or duplicates),
    }


def _finite_number(record: Mapping[str, Any], field: str) -> float:
    value = record.get(field)
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{field} must be a finite number, got {value!r}")
    return float(value)


def _boolean(value: Any, field: str) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and math.isfinite(value) and value in (0, 1):
        return bool(value)
    raise ValueError(f"{field} must be boolean or numeric 0/1, got {value!r}")


def _pct(numerator: int | float, denominator: int) -> float | None:
    return 100.0 * numerator / denominator if denominator else None


def _stats(values: list[float]) -> dict[str, int | float | None]:
    return {
        "count": len(values),
        "mean": statistics.fmean(values) if values else None,
        "median": statistics.median(values) if values else None,
        "min": min(values) if values else None,
        "max": max(values) if values else None,
    }


def build_stop_report(
    records: Iterable[Mapping[str, Any]], success_distance: float = 3.0,
) -> dict[str, Any]:
    """Summarize STOP sources and OS/SR using unrounded per-episode results.

    Required fields: success, oracle_success, spl, final_distance and steps.
    Missing/empty stop_source is explicitly ``unknown``; unrecognized nonempty
    sources and inconsistent forced/oracle STOP flags are rejected. Source
    frequency and forced-STOP frequency use *all* records as denominator.
    ``goal_range_rate_pct`` uses a strict ``distance < success_distance`` test
    and does not replace the environment's success measurement.
    """
    if (
        isinstance(success_distance, bool)
        or not isinstance(success_distance, (int, float))
        or not math.isfinite(success_distance)
        or success_distance <= 0
    ):
        raise ValueError("success_distance must be finite and positive")
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    rows: list[dict[str, Any]] = []
    for index, record in enumerate(records):
        if not isinstance(record, Mapping):
            raise ValueError(f"record {index} must be an object")
        source = record.get("stop_source")
        source = "unknown" if source is None or source == "" else source
        if not isinstance(source, str) or source not in _STOP_SOURCES:
            raise ValueError(f"record {index}: unrecognized stop_source {source!r}")
        success = _boolean(record.get("success"), "success")
        oracle_success = _boolean(record.get("oracle_success"), "oracle_success")
        distance = _finite_number(record, "final_distance")
        steps = _finite_number(record, "steps")
        spl = _finite_number(record, "spl")
        if distance < 0 or steps < 0 or not steps.is_integer() or not 0 <= spl <= 1:
            raise ValueError(f"record {index}: invalid distance, step count or SPL range")
        forced = _boolean(record.get("forced_stop", source == "max_steps"), "forced_stop")
        oracle = _boolean(record.get("oracle_stop", source == "oracle_stop"), "oracle_stop")
        if forced != (source == "max_steps"):
            raise ValueError(f"record {index}: forced_stop disagrees with stop_source")
        if oracle != (source == "oracle_stop"):
            raise ValueError(f"record {index}: oracle_stop disagrees with stop_source")
        row = {
            "success": success, "oracle_success": oracle_success,
            "final_distance": distance, "steps": steps, "spl": spl,
            "goal_range": distance < success_distance, "forced_stop": forced,
        }
        rows.append(row)
        grouped[source].append(row)

    total = len(rows)
    successes = sum(row["success"] for row in rows)
    oracle_successes = sum(row["oracle_success"] for row in rows)
    oracle_misses = sum(row["oracle_success"] and not row["success"] for row in rows)
    # Include zero-count categories needed by the fm_stop launcher, so absence is
    # distinguishable from a report that forgot to measure a source.
    source_names = sorted(set(grouped) | {"learned_head", "max_steps", "fm_zero_command", "unknown"})
    by_source: dict[str, dict[str, Any]] = {}
    for source in source_names:
        subset = grouped[source]
        count = len(subset)
        source_successes = sum(row["success"] for row in subset)
        near_goal = sum(row["goal_range"] for row in subset)
        by_source[source] = {
            "count": count,
            "rate_pct_of_all_episodes": _pct(count, total),
            "success_count": source_successes,
            "success_rate_pct": _pct(source_successes, count),
            "success_rate_denominator": count,
            "goal_range_count": near_goal,
            "goal_range_rate_pct": _pct(near_goal, count),
            "goal_range_rate_denominator": count,
            "final_distance_m": _stats([row["final_distance"] for row in subset]),
            "steps": _stats([row["steps"] for row in subset]),
        }
    return {
        "schema_version": "lightnav_stop_diagnostics_v1",
        "num_episodes": total,
        "success_distance_m": float(success_distance),
        "goal_range_rule": "final_distance < success_distance_m",
        "rate_unit": "percent",
        "source_rate_denominator": total,
        "metrics": {
            "success_count": successes,
            "oracle_success_count": oracle_successes,
            "SR_success_rate_pct": _pct(successes, total),
            "OS_oracle_success_pct": _pct(oracle_successes, total),
            "SPL_pct": _pct(sum(row["spl"] for row in rows), total),
            "OS_minus_SR_percentage_points": _pct(oracle_successes - successes, total),
        },
        "oracle_success_without_success": {
            "count": oracle_misses,
            "rate_pct_of_all_episodes": _pct(oracle_misses, total),
            "conditional_rate_pct_of_oracle_success": _pct(oracle_misses, oracle_successes),
            "conditional_denominator": oracle_successes,
        },
        "forced_stop": {
            "count": sum(row["forced_stop"] for row in rows),
            "rate_pct_of_all_episodes": _pct(sum(row["forced_stop"] for row in rows), total),
            "denominator": total,
        },
        "by_stop_source": by_source,
        "interpretation": {
            "learned_head": "The learned head requested STOP; excludes FM zero commands and budget stops.",
            "goal_range": "Final pose within goal radius; distinct from environment-measured success.",
            "oracle_success_without_success": "Reached the goal radius during rollout but finished unsuccessfully; does not alone identify the cause.",
            "null_rates": "A null rate has a zero denominator.",
        },
    }
