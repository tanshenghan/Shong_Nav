"""Shared pytest fixtures. CPU-only: no CUDA, no model load."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest


@pytest.fixture
def fake_centroids() -> np.ndarray:
    # (K=4, H=10, 3) -- deterministic so lookups are checkable.
    K, H = 4, 10
    arr = np.zeros((K, H, 3), dtype=np.float32)
    for k in range(K):
        arr[k, :, 0] = float(k)  # forward_m == k, easy to assert
    return arr


def write_rvq_bundle(
    d: Path,
    *,
    levels=(4, 8, 8),
    horizon: int = 10,
    representation: str = "se2_diff",
    stop_l0: int | None = 0,
    codebooks: list[np.ndarray] | None = None,
    jacobian: np.ndarray | None = None,
) -> dict:
    """Write a synthetic RVQ bundle directory (manifest + codebooks + jacobian + alpha)."""
    d.mkdir(parents=True, exist_ok=True)
    levels = list(levels)
    feat = 3 * horizon
    rng = np.random.default_rng(0)
    man = {
        "format_version": 1,
        "method": "rvq",
        "horizon": horizon,
        "representation": representation,
        "objective": "ade_v1",
        "encode": {"type": "ade", "heading_weight": 0.3},
        "levels": levels,
        "feature_dim": feat,
        "feature_space": "weighted_diff",
        "codebook_files": [f"codebook_l{i}.npy" for i in range(len(levels))],
        "jacobian_weights_file": "jacobian_weights.npy",
        "alpha_file": "alpha_per_source.json",
        "alpha_mode": "none",
        "stop": {"l0": stop_l0, "codes": [0] * len(levels)} if stop_l0 is not None else None,
        "vocab_layout": {
            "token_format": "<act_l{level}_{code}>",
            "segments": [{"level": i, "size": n} for i, n in enumerate(levels)],
            "total_rows": int(sum(levels)),
        },
        "fit": {},
    }
    for i, n in enumerate(levels):
        if codebooks is not None:
            cb = np.asarray(codebooks[i], dtype=np.float32)
        else:
            cb = rng.standard_normal((n, feat)).astype(np.float32)
        np.save(d / f"codebook_l{i}.npy", cb)
    w = np.ones(feat, np.float32) if jacobian is None else np.asarray(jacobian, dtype=np.float32)
    np.save(d / "jacobian_weights.npy", w)
    (d / "alpha_per_source.json").write_text(json.dumps({"mode": "none"}))
    (d / "manifest.json").write_text(json.dumps(man))
    return man


@pytest.fixture
def rvq_bundle_writer():
    """Factory fixture: ``rvq_bundle_writer(dir, levels=..., horizon=..., ...)`` -> manifest."""
    return write_rvq_bundle
