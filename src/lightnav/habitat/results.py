"""Evaluation summaries for VLN-CE and ObjectNav episode results.

Both printers aggregate a list of per-episode result dicts (as produced by
``lightnav.habitat.runner``), print a human-readable table and write
``<output_dir>/summary.json``.
"""

from __future__ import annotations

import json
import os
from typing import Any

import numpy as np


def make_json_safe(obj: Any) -> Any:
    """Recursively convert numpy scalars/arrays to plain Python for JSON serialization."""
    if isinstance(obj, dict):
        return {k: make_json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [make_json_safe(v) for v in obj]
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (bool, int, float, str, type(None))):
        return obj
    return None


def _write_summary(summary: dict[str, Any], output_dir: str) -> str:
    os.makedirs(output_dir, exist_ok=True)
    summary_path = os.path.join(output_dir, "summary.json")
    with open(summary_path, "w") as f:
        json.dump(make_json_safe(summary), f, indent=2)
    print(f"\nSaved summary: {summary_path}")
    return summary_path


def print_vlnce_summary(
    results: list[dict[str, Any]],
    elapsed: float,
    output_dir: str,
    extra_info: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    """Print and save the VLN-CE summary (SR / OS / SPL / NDTW / NE)."""
    if not results:
        print("No results to summarize.")
        return None

    n = len(results)
    successes = [1.0 if r.get("success", False) else 0.0 for r in results]
    spls = [r.get("spl", 0.0) for r in results]
    ndtws = [r.get("ndtw", 0.0) for r in results]
    oracle_successes = [1.0 if r.get("oracle_success", False) else 0.0 for r in results]
    final_distances = [r.get("final_distance", 0.0) for r in results]
    steps = [r.get("steps", 0) for r in results]

    sr = np.mean(successes) * 100
    os_rate = np.mean(oracle_successes) * 100
    ne = np.mean(final_distances)
    spl = np.mean(spls) * 100
    ndtw = np.mean(ndtws) * 100

    print("\n" + "=" * 60)
    print("VLN-CE Results Summary")
    print("=" * 60)
    if extra_info:
        for k, v in extra_info.items():
            print(f"{k + ':':<25}{v}")
    print(f"{'Episodes:':<25}{n}")
    print()
    print(f"  SR (Success Rate):     {sr:.1f}%")
    print(f"  OS (Oracle Success):   {os_rate:.1f}%")
    print(f"  SPL:                   {spl:.1f}%")
    print(f"  NDTW:                  {ndtw:.1f}%")
    print(f"  NE (Navigation Error): {ne:.2f}m")
    print()
    print(f"  Table format: {sr:.1f} / {os_rate:.1f} / {spl:.1f} / {ndtw:.1f} / {ne:.2f}")
    print()
    print(f"{'Avg Steps:':<25}{np.mean(steps):.1f}")
    print(f"{'Total Time:':<25}{elapsed:.1f}s")
    print(f"{'Time/Episode:':<25}{elapsed / n:.1f}s")
    print(f"{'Output Directory:':<25}{output_dir}")
    print("=" * 60)

    summary: dict[str, Any] = {
        "num_episodes": n,
        "metrics": {
            "SR_success_rate_pct": round(float(sr), 2),
            "OS_oracle_success_pct": round(float(os_rate), 2),
            "SPL_pct": round(float(spl), 2),
            "NDTW_pct": round(float(ndtw), 2),
            "NE_navigation_error_m": round(float(ne), 3),
        },
        "table_format": f"{sr:.1f} / {os_rate:.1f} / {spl:.1f} / {ndtw:.1f} / {ne:.2f}",
        "avg_steps": round(float(np.mean(steps)), 1),
        "total_time_sec": round(elapsed, 1),
        "episodes": [
            {
                "episode_id": r.get("episode_id", f"episode_{i:03d}"),
                "habitat_episode_id": r.get("habitat_episode_id", ""),
                "scene_id": r.get("scene_id", ""),
                "rollout_idx": r.get("rollout_idx", 0),
                "success": float(r.get("success", False)),
                "oracle_success": float(r.get("oracle_success", False)),
                "spl": round(r.get("spl", 0.0), 4),
                "ndtw": round(r.get("ndtw", 0.0), 4),
                "steps": r.get("steps", 0),
                "final_distance": round(r.get("final_distance", 0.0), 3),
                "min_distance": round(r.get("min_distance", 0.0), 3),
                "instruction": r.get("instruction", ""),
                "stop_source": r.get("stop_source", "unknown"),
                "oracle_stop": bool(r.get("oracle_stop", False)),
                "policy_event_counts": r.get("policy_event_counts", {}),
            }
            for i, r in enumerate(results)
        ],
    }
    if extra_info:
        summary.update(extra_info)

    _write_summary(summary, output_dir)
    return summary


def print_objectnav_summary(
    results: list[dict[str, Any]],
    elapsed: float,
    output_dir: str,
    extra_info: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    """Print and save the ObjectNav summary (SR / SPL / SoftSPL / NE + per-category SR)."""
    if not results:
        print("No results to summarize.")
        return None

    n = len(results)
    successes = [1.0 if r.get("success", False) else 0.0 for r in results]
    spls = [float(r.get("spl", 0.0) or 0.0) for r in results]
    soft_spls = [float(r.get("soft_spl", 0.0) or 0.0) for r in results]
    final_distances = [float(r.get("final_distance", 0.0) or 0.0) for r in results]
    steps = [r.get("steps", 0) for r in results]

    sr = np.mean(successes) * 100
    spl = np.mean(spls) * 100
    soft_spl = np.mean(soft_spls) * 100
    ne = np.mean(final_distances)

    cat_data: dict[str, list[float]] = {}
    for r in results:
        cat = r.get("object_category", "") or "unknown"
        cat_data.setdefault(cat, []).append(1.0 if r.get("success", False) else 0.0)

    print("\n" + "=" * 60)
    print("ObjectNav Results Summary")
    print("=" * 60)
    if extra_info:
        for k, v in extra_info.items():
            print(f"{k + ':':<25}{v}")
    print(f"{'Episodes:':<25}{n}")
    print()
    print(f"  SR (Success Rate):     {sr:.1f}%")
    print(f"  SPL:                   {spl:.1f}%")
    print(f"  SoftSPL:               {soft_spl:.1f}%")
    print(f"  NE (Navigation Error): {ne:.2f}m")
    print()
    print(f"  Table format: {sr:.1f} / {spl:.1f} / {soft_spl:.1f} / {ne:.2f}")
    print()
    print("  Per-category SR:")
    for cat in sorted(cat_data):
        slist = cat_data[cat]
        cat_sr = np.mean(slist) * 100
        print(f"    {cat:<16s}: {cat_sr:.1f}%  ({int(sum(slist))}/{len(slist)})")
    print()
    print(f"{'Avg Steps:':<25}{np.mean(steps):.1f}")
    print(f"{'Total Time:':<25}{elapsed:.1f}s")
    print(f"{'Time/Episode:':<25}{elapsed / n:.1f}s")
    print(f"{'Output Directory:':<25}{output_dir}")
    print("=" * 60)

    summary: dict[str, Any] = {
        "num_episodes": n,
        "metrics": {
            "SR_success_rate_pct": round(float(sr), 2),
            "SPL_pct": round(float(spl), 2),
            "SoftSPL_pct": round(float(soft_spl), 2),
            "NE_navigation_error_m": round(float(ne), 3),
        },
        "table_format": f"{sr:.1f} / {spl:.1f} / {soft_spl:.1f} / {ne:.2f}",
        "per_category_sr": {
            cat: round(np.mean(slist) * 100, 2) for cat, slist in sorted(cat_data.items())
        },
        "avg_steps": round(float(np.mean(steps)), 1),
        "total_time_sec": round(elapsed, 1),
        "episodes": [
            {
                "episode_id": r.get("episode_id", f"episode_{i:03d}"),
                "habitat_episode_id": r.get("habitat_episode_id", ""),
                "scene_id": r.get("scene_id", ""),
                "rollout_idx": r.get("rollout_idx", 0),
                "success": float(r.get("success", False)),
                "spl": round(float(r.get("spl", 0.0) or 0.0), 4),
                "soft_spl": round(float(r.get("soft_spl", 0.0) or 0.0), 4),
                "object_category": r.get("object_category", ""),
                "steps": r.get("steps", 0),
                "final_distance": round(float(r.get("final_distance", 0.0) or 0.0), 3),
                "min_distance": round(float(r.get("min_distance", 0.0) or 0.0), 3),
            }
            for i, r in enumerate(results)
        ],
    }
    if extra_info:
        summary.update(extra_info)

    _write_summary(summary, output_dir)
    return summary
