"""CPU-only geometry and RVQ labels for newly rendered Habitat trajectories.

No Habitat or torch dependency: these helpers also run in the Python 3.9
simulation environment with numpy 1.23. Native Habitat body axes are right/up/back;
exported planar waypoints use forward/left/CCW, relative to each sample's body.
"""

from __future__ import annotations

import numpy as np

from lightnav.traj_vocab import RVQBundle, compose_to_abs, wrap_to_pi


def absolute_to_deltas(waypoints: np.ndarray) -> np.ndarray:
    previous = np.concatenate(
        [np.zeros_like(waypoints[..., :1, :]), waypoints[..., :-1, :]], axis=-2
    )
    xy = waypoints[..., :2] - previous[..., :2]
    c, s = np.cos(previous[..., 2]), np.sin(previous[..., 2])
    return np.stack(
        [c * xy[..., 0] + s * xy[..., 1],
         -s * xy[..., 0] + c * xy[..., 1],
         wrap_to_pi(waypoints[..., 2] - previous[..., 2])], axis=-1
    ).astype(np.float32)


def future_targets(world_from_body: np.ndarray, horizon: int) -> dict:
    """Actual future observations only; repeated tail poses are masked storage padding."""
    poses = np.asarray(world_from_body, dtype=np.float64)
    if poses.ndim != 3 or poses.shape[1:] != (4, 4) or len(poses) == 0:
        raise ValueError("world_from_body must be nonempty (N, 4, 4)")
    if horizon < 1 or not np.isfinite(poses).all():
        raise ValueError("positive horizon and finite poses required")
    rotations = poses[:, :3, :3]
    if not np.allclose(rotations.transpose(0, 2, 1) @ rotations, np.eye(3), atol=1e-5):
        raise ValueError("body rotations must be orthonormal")
    if not np.allclose(poses[:, 3], [0, 0, 0, 1]):
        raise ValueError("invalid homogeneous transforms")
    future_ids = np.arange(len(poses))[:, None] + np.arange(1, horizon + 1)
    valid = future_ids < len(poses)
    bounded = np.minimum(future_ids, len(poses) - 1)
    positions = poses[:, :3, 3]
    forward = -rotations[:, :, 2]
    left = -rotations[:, :, 0]
    displacement = positions[bounded] - positions[:, None]
    abs_targets = np.stack(
        [np.einsum("nki,ni->nk", displacement, forward),
         np.einsum("nki,ni->nk", displacement, left),
         np.arctan2(np.einsum("nki,ni->nk", forward[bounded], left),
                    np.einsum("nki,ni->nk", forward[bounded], forward))], axis=-1
    ).astype(np.float32)
    deltas = absolute_to_deltas(abs_targets)
    composed = compose_to_abs(deltas)
    np.testing.assert_allclose(composed[..., :2], abs_targets[..., :2], atol=2e-5)
    # +pi and -pi describe the same heading. Float32 composition can cross the
    # branch cut, so validate angular distance without changing the stored GT.
    np.testing.assert_allclose(wrap_to_pi(composed[..., 2] - abs_targets[..., 2]), 0., atol=2e-5)
    return {
        "waypoints": abs_targets,
        "se2_diff": deltas,
        "valid": valid,
        "frame_ids": np.where(valid, future_ids, -1),
        "vertical_offset_m": displacement[:, :, 1].astype(np.float32),
    }


def encode_rvq(waypoints: np.ndarray, bundle: RVQBundle) -> tuple:
    """Greedy residual search scored in integrated SE(2), not the unpublished encoder.

    STOP is excluded here; the collector assigns the reserved tuple only to a
    separately verified terminal STOP. FM never uses this lossy reconstruction.
    """
    target = np.asarray(waypoints, dtype=np.float32)
    if target.shape != (bundle.horizon, 3) or not np.isfinite(target).all():
        raise ValueError("trajectory shape/values do not match RVQ bundle")
    if (bundle.representation != "se2_diff" or bundle.feature_space != "weighted_diff"
            or bundle.encode.get("type") != "ade"):
        raise ValueError("collector requires a se2_diff/weighted_diff/ADE RVQ bundle")
    if np.any(bundle.jacobian_weights <= 0):
        raise ValueError("positive Jacobian weights required")
    total = np.zeros(bundle.horizon * 3, dtype=np.float32)
    codes = []
    for level, book in enumerate(bundle.codebooks):
        candidates = (total[None] + book) / bundle.jacobian_weights
        decoded = compose_to_abs(candidates.reshape(-1, bundle.horizon, 3))
        ade = np.linalg.norm(decoded[:, :, :2] - target[None, :, :2], axis=-1).mean(axis=1)
        heading = np.abs(wrap_to_pi(decoded[:, -1, 2] - target[-1, 2]))
        score = ade + float(bundle.encode["heading_weight"]) * heading
        if level == 0 and bundle.stop_l0 is not None:
            score[bundle.stop_l0] = np.inf
        if not np.isfinite(score).any():
            raise ValueError("RVQ level has no usable non-stop code")
        code = int(np.argmin(score))
        codes.append(code)
        total += book[code]
    reconstructed = bundle.decode_waypoints(codes)
    error = float(np.linalg.norm(reconstructed[:, :2] - target[:, :2], axis=1).mean())
    return codes, reconstructed, error


def pinhole_intrinsics(width: int, height: int, hfov_deg: float) -> np.ndarray:
    """Habitat pinhole K, pixel coordinates with image edges at 0 and W/H."""
    if width < 1 or height < 1 or not 0 < hfov_deg < 180:
        raise ValueError("invalid pinhole dimensions or HFOV")
    focal = width / (2 * np.tan(np.deg2rad(hfov_deg) / 2))
    return np.array([[focal, 0, width / 2], [0, focal, height / 2], [0, 0, 1]])


def route_deviation(positions: np.ndarray, route: np.ndarray) -> np.ndarray:
    """3D distance to the GT polyline, used for diagnostics only, never labels."""
    positions, route = np.asarray(positions), np.asarray(route)
    if len(route) < 2:
        raise ValueError("a reference route needs at least two positions")
    begin, delta = route[:-1], np.diff(route, axis=0)
    denom = np.maximum(np.sum(delta * delta, axis=1), 1e-12)
    offset = positions[:, None] - begin[None]
    fraction = np.clip(np.sum(offset * delta[None], axis=-1) / denom, 0, 1)
    nearest = begin[None] + fraction[:, :, None] * delta[None]
    return np.linalg.norm(positions[:, None] - nearest, axis=-1).min(axis=1)
