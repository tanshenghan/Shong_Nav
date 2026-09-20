"""RVQ action-tokenizer bundle loading and decoding, plus the SE(2) step integration.

An RVQ bundle is a directory with a ``manifest.json`` and the numpy arrays it names::

    {
        "method": "rvq",
        "horizon": 10,                        # H waypoints per chunk
        "levels": [256, 256, 256],            # codebook size per residual level (coarse -> fine)
        "feature_dim": 30,                    # == 3 * horizon
        "codebook_files": ["codebook_l0.npy", ...],   # each (levels[i], feature_dim) float32
        "jacobian_weights_file": "jacobian_weights.npy",  # (feature_dim,) per-dim weights
        "alpha_file": "alpha_per_source.json",
        "representation": "se2_diff" | "ego_abs",
        "objective": "...", "encode": {...}, "feature_space": "...",
        "stop": {"l0": <int>}                 # optional: level-0 code of the stop action
    }

Decoding sums the selected codeword of every level, divides by the Jacobian weights,
reshapes to ``(H, 3)`` and -- for ``se2_diff`` -- integrates the per-step SE(2)
differentials into absolute ego waypoints with :func:`compose_to_abs`.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np


def wrap_to_pi(theta: np.ndarray) -> np.ndarray:
    """Wrap angles to (-pi, pi]."""
    return (theta + np.pi) % (2.0 * np.pi) - np.pi


def compose_to_abs(deltas: np.ndarray) -> np.ndarray:
    """SE(2) step differentials -> absolute ego waypoints.

    A trajectory chunk is H future waypoints in the chunk-start ego frame, each a
    planar pose ``(x, y, theta)`` (x=forward, y=lateral, theta=heading [rad]). The
    chunk-start pose is the identity and is not stored; the deltas are the per-step
    relative transforms ``delta_j = T_{j-1}^{-1} . T_j`` in the previous step's local
    frame, so the first delta equals wp1's ego pose directly.

    Args:
        deltas: ``(..., H, 3)`` ``(dx_rel, dy_rel, dtheta_rel)``.

    Returns:
        ``(..., H, 3)`` absolute poses ``(x, y, theta)`` in the chunk-start frame
        (float32; accumulation in float64).
    """
    deltas = np.asarray(deltas, dtype=np.float64)
    *batch, H, C = deltas.shape
    if C != 3:
        raise ValueError(f"expected last dim 3, got {C}")

    out = np.empty((*batch, H, 3), dtype=np.float64)
    x = np.zeros(batch, dtype=np.float64)
    y = np.zeros(batch, dtype=np.float64)
    th = np.zeros(batch, dtype=np.float64)
    for j in range(H):
        dx, dy, dth = deltas[..., j, 0], deltas[..., j, 1], deltas[..., j, 2]
        cos_t, sin_t = np.cos(th), np.sin(th)
        x = x + cos_t * dx - sin_t * dy
        y = y + sin_t * dx + cos_t * dy
        th = wrap_to_pi(th + dth)
        out[..., j, 0] = x
        out[..., j, 1] = y
        out[..., j, 2] = th
    return out.astype(np.float32)


@dataclass
class RVQBundle:
    """Loaded RVQ bundle: per-level codebooks in the manifest's ``feature_space`` plus the
    Jacobian weights needed to map a codeword sum back to waypoint units."""

    path: Path
    manifest: Dict[str, Any]
    levels: List[int]
    horizon: int
    representation: str
    objective: str
    encode: Dict[str, Any]
    feature_space: str
    codebooks: List[np.ndarray]
    jacobian_weights: np.ndarray
    alpha: Dict[str, Any]
    stop_l0: Optional[int] = None  # level-0 code that 1:1 maps to the stationary (stop) tuple

    def is_stop(self, codes) -> bool:
        """True iff ``codes`` is the explicit stop action (level-0 code == stop_l0).

        The stop tuple's codeword sum is only *near* zero (k-means residual), so callers
        rely on this exact code match rather than the decoded magnitude -- unlike the
        flat vocabulary, whose centroid[0] is exactly zero.
        """
        return self.stop_l0 is not None and int(codes[0]) == int(self.stop_l0)

    def decode_waypoints(self, codes) -> np.ndarray:
        """Decode D per-level codes -> ``(H, 3)`` absolute ego waypoints.

        Per-level codeword sum in the (weighted-diff) feature space, de-weighted by the
        Jacobian, then SE(2)-composed when ``representation == 'se2_diff'``.
        """
        if len(codes) != len(self.codebooks):
            raise ValueError(f"got {len(codes)} codes, expected {len(self.codebooks)} levels")
        recon = np.zeros(self.codebooks[0].shape[1], dtype=np.float32)
        for lvl, code in enumerate(codes):
            recon += self.codebooks[lvl][int(code)]
        feat = (recon / self.jacobian_weights).reshape(self.horizon, 3)
        if self.representation == "se2_diff":
            return np.ascontiguousarray(compose_to_abs(feat[None])[0], dtype=np.float32)
        return np.ascontiguousarray(feat, dtype=np.float32)  # ego_abs


def load_rvq_bundle(
    bundle_path: Path,
    horizon: int,
    num_frames: int = 0,
    *,
    load_cluster_ids: bool = False,
    load_distances: bool = False,
) -> RVQBundle:
    """Load + validate an RVQ bundle directory (see the module docstring for the layout).

    ``num_frames``, ``load_cluster_ids`` and ``load_distances`` are accepted for
    signature compatibility and ignored: the per-frame training sidecars are not needed
    at inference.
    """
    bundle_path = Path(bundle_path)
    manifest_path = bundle_path / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"RVQ bundle manifest not found at {manifest_path}.")
    man = json.loads(manifest_path.read_text())
    if man.get("method") != "rvq":
        raise ValueError(f"{manifest_path}: manifest.method={man.get('method')!r}, expected 'rvq'.")
    if int(man["horizon"]) != int(horizon):
        raise RuntimeError(f"bundle horizon {man['horizon']} != requested horizon {horizon}.")

    levels = list(man["levels"])
    feat_dim = int(man["feature_dim"])
    codebooks = []
    for i, fn in enumerate(man["codebook_files"]):
        c = np.load(bundle_path / fn).astype(np.float32)
        if c.shape != (levels[i], feat_dim):
            raise RuntimeError(f"codebook l{i} shape {c.shape} != ({levels[i]}, {feat_dim}).")
        codebooks.append(c)
    w = np.load(bundle_path / man["jacobian_weights_file"]).astype(np.float32)
    if w.shape != (feat_dim,):
        raise RuntimeError(f"jacobian_weights shape {w.shape} != ({feat_dim},).")
    alpha = json.loads((bundle_path / man["alpha_file"]).read_text())

    stop_l0 = (man.get("stop") or {}).get("l0")
    return RVQBundle(
        path=bundle_path,
        manifest=man,
        levels=levels,
        horizon=int(man["horizon"]),
        representation=man["representation"],
        objective=man["objective"],
        encode=man["encode"],
        feature_space=man["feature_space"],
        codebooks=codebooks,
        jacobian_weights=w,
        alpha=alpha,
        stop_l0=int(stop_l0) if stop_l0 is not None else None,
    )
