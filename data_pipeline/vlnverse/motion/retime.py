"""Conservative planar retiming: preserve the route and smooth each motion run."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from data_pipeline.vlnverse.scene.occupancy import OccupancyGrid
from data_pipeline.vlnverse.source.official import SourceEpisode

# Seventh-order rest-to-rest progress: velocity, acceleration AND jerk vanish
# at both ends. All joins are C3, including a translation joined to a turn.
_PROGRESS = np.array([0., 0., 0., 0., 35., -84., 70., -20.])


def _peak_derivative(order):
    coefficients = np.polynomial.polynomial.polyder(_PROGRESS, order)
    roots = np.polynomial.polynomial.polyroots(np.polynomial.polynomial.polyder(coefficients))
    candidates = [0., 1., *(r.real for r in roots if abs(r.imag) < 1e-9 and 0 < r.real < 1)]
    return float(np.max(np.abs(np.polynomial.polynomial.polyval(candidates, coefficients))))


_PEAKS = tuple(_peak_derivative(order) for order in (1, 2, 3))


def _progress_at(u):
    # Evaluate near the closer endpoint to avoid cancellation near u=1.
    near = np.minimum(u, 1 - u)
    progress = np.polynomial.polynomial.polyval(near, _PROGRESS)
    return np.where(u <= .5, progress, 1 - progress)


def _inverse_progress(progress):
    """Invert the monotone curve without dividing by its zero endpoint slope."""
    target = np.minimum(progress, 1 - progress)
    low, high = np.zeros_like(target), np.full_like(target, .5)
    for _ in range(56):
        mid = (low + high) / 2
        below = _progress_at(mid) < target
        low, high = np.where(below, mid, low), np.where(below, high, mid)
    u = (low + high) / 2
    u = np.where(progress <= .5, u, 1 - u)
    u[progress == 0], u[progress == 1] = 0., 1.
    if not np.allclose(_progress_at(u), progress, rtol=0, atol=1e-12):
        raise ValueError("could not invert source anchor progress")
    return u


@dataclass(frozen=True)
class MotionConfig:
    """Draft limits in metres, radians and synthetic seconds; tune for the robot."""

    sample_mode: str = "per_source_step"
    poses_per_step: int = 6
    sample_hz: float = 30.0  # Only used by the optional uniform_time mode.
    max_speed_m_s: float = 0.4
    max_accel_m_s2: float = 0.5
    max_jerk_m_s3: float = 1.0
    max_yaw_rate_rad_s: float = 0.8
    max_yaw_accel_rad_s2: float = 1.0
    max_yaw_jerk_rad_s3: float = 2.0
    robot_radius_m: float = 0.25
    safety_margin_m: float = 0.05
    max_heading_error_rad: float = 0.017453292519943295
    planar_tolerance_m: float = 1e-5
    max_merge_deviation_m: float = 1e-5

    def __post_init__(self):
        if self.sample_mode not in {"per_source_step", "uniform_time"}:
            raise ValueError("sample_mode must be per_source_step or uniform_time")
        if (isinstance(self.poses_per_step, bool) or not isinstance(self.poses_per_step, int)
                or self.poses_per_step not in range(5, 9)):
            raise ValueError("poses_per_step must be an integer from 5 to 8, including both endpoints")
        for name, value in asdict(self).items():
            if name in {"sample_mode", "poses_per_step"}:
                continue
            if isinstance(value, bool) or not np.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be a finite nonnegative number")
            if value == 0 and name not in {"safety_margin_m", "max_heading_error_rad"}:
                raise ValueError(f"{name} must be positive")
        if self.max_heading_error_rad > np.pi / 2:
            raise ValueError("max_heading_error_rad must not permit backward travel")

    @property
    def footprint_radius_m(self):
        return self.robot_radius_m + self.safety_margin_m


@dataclass
class MotionRecord:
    timestamp_s: np.ndarray
    positions_xyz: np.ndarray
    yaw_rad: np.ndarray
    velocity_xyz: np.ndarray
    acceleration_xyz: np.ndarray
    jerk_xyz: np.ndarray
    yaw_rate_rad_s: np.ndarray
    yaw_accel_rad_s2: np.ndarray
    yaw_jerk_rad_s3: np.ndarray
    source_progress: np.ndarray
    primitives: list[dict]

    def arrays(self):
        return {name: getattr(self, name) for name in self.__dataclass_fields__ if name != "primitives"}


def _motion_runs(positions, yaws, config):
    if np.ptp(positions[:, 2]) > config.planar_tolerance_m:
        raise ValueError("nonplanar source: stairs/slopes need a 3D planner and collision model")
    # Preserve all nonduplicate anchors; repeated source states have no known
    # physical dwell duration and are represented through source_progress only.
    keep = [0]
    for index in range(1, len(positions)):
        if np.linalg.norm(positions[index] - positions[keep[-1]]) > 1e-9 or abs(yaws[index] - yaws[keep[-1]]) > 1e-9:
            keep.append(index)
    runs = []
    for start, end in zip(keep, keep[1:]):
        delta = positions[end] - positions[start]
        distance = np.linalg.norm(delta)
        angle = yaws[end] - yaws[start]
        if distance > 1e-9:
            if abs(angle) > 1e-7:
                raise ValueError("simultaneous translation and rotation need a constrained curved-path planner")
            if np.linalg.norm(delta[:2]) <= 1e-9:
                raise ValueError("vertical motion is unsupported")
            bearing = np.arctan2(delta[1], delta[0])
            error = np.arctan2(np.sin(bearing - yaws[start]), np.cos(bearing - yaws[start]))
            if abs(error) > config.max_heading_error_rad + 1e-9:
                raise ValueError("source movement is sideways/backward relative to robot yaw")
            kind = "translate"
        else:
            kind = "rotate"
        merge = False
        if runs and runs[-1]["kind"] == kind:
            previous = runs[-1]
            a, b = previous["start"], previous["end"]
            if kind == "translate":
                direction = positions[end] - positions[a]
                squared_length = np.dot(direction, direction)
                if squared_length > 1e-18:
                    anchors = positions[a:end + 1] - positions[a]
                    fractions = anchors @ direction / squared_length
                    deviation = np.linalg.norm(anchors - fractions[:, None] * direction, axis=1)
                    merge = (deviation.max() <= config.max_merge_deviation_m
                             and (np.diff(fractions) >= 0).all()
                             and abs(yaws[start] - yaws[a]) < 1e-9)
            else:
                merge = (np.linalg.norm(positions[start] - positions[a]) < 1e-9
                         and np.sign(yaws[b] - yaws[a]) == np.sign(angle))
        if merge:
            runs[-1]["end"] = end
            runs[-1]["anchors"].append(end)
        else:
            runs.append(dict(kind=kind, start=start, end=end, anchors=[start, end]))
    return runs


def retime_episode(source: SourceEpisode, grid: OccupancyGrid, config: MotionConfig) -> MotionRecord:
    """Generate a C3 trajectory whose complete planar sweep passes the map check.

    This is a conservative stop/turn baseline, not an optimized corner-rounding
    planner or a simulator-validated dynamic rollout. No map or unsafe source
    is silently accepted, repaired or replaced by a shortest-path shortcut.
    """
    positions = np.asarray(source.positions_xyz, dtype=np.float64)
    yaws = np.unwrap(np.asarray(source.yaw_rad, dtype=np.float64))
    if not grid.path_is_free(positions[:, :2], config.footprint_radius_m):
        raise ValueError("source path or swept footprint intersects occupied/unknown space or map bounds")
    runs = _motion_runs(positions, yaws, config)
    arrays = {name: [] for name in MotionRecord.__dataclass_fields__ if name != "primitives"}
    primitives = []
    elapsed_steps = 0
    elapsed_time = 0.
    for run in runs:
        start, end = run["start"], run["end"]
        delta = positions[end] - positions[start]
        angle = yaws[end] - yaws[start]
        distance = np.linalg.norm(delta)
        required = max(
            (distance * peak / limit) ** (1 / order)
            for order, peak, limit in zip((1, 2, 3), _PEAKS,
                                          (config.max_speed_m_s, config.max_accel_m_s2, config.max_jerk_m_s3)))
        required = max(required, *(
            (abs(angle) * peak / limit) ** (1 / order)
            for order, peak, limit in zip((1, 2, 3), _PEAKS,
                (config.max_yaw_rate_rad_s, config.max_yaw_accel_rad_s2, config.max_yaw_jerk_rad_s3))))
        indices = np.asarray(run["anchors"])
        anchor_progress = ((positions[indices] - positions[start]) @ delta / distance**2
                           if distance > 1e-9 else (yaws[indices] - yaws[start]) / angle)
        anchor_progress[0], anchor_progress[-1] = 0., 1.
        if not (np.diff(anchor_progress) > 0).all():
            raise ValueError("nonduplicate source motion has nonincreasing projected progress")
        if config.sample_mode == "uniform_time":
            steps = max(1, int(np.ceil(required * config.sample_hz)))
            duration = steps / config.sample_hz
            s = np.linspace(0, 1, steps + 1)
        else:
            # Keep the whole coalesced curve. Allocate K poses to each original
            # moving interval by inverting its anchors, not by stopping there.
            anchor_times = _inverse_progress(anchor_progress)
            s = np.concatenate([np.linspace(a, b, config.poses_per_step)[:-1]
                                for a, b in zip(anchor_times, anchor_times[1:])] + [np.array([1.])])
            steps, duration = len(s) - 1, required
        progress = _progress_at(s)
        progress[0], progress[-1] = 0., 1.
        derivatives = [np.polynomial.polynomial.polyval(s, np.polynomial.polynomial.polyder(_PROGRESS, order))
                       / duration**order for order in (1, 2, 3)]
        for derivative in derivatives:
            derivative[0] = derivative[-1] = 0.
        segment_positions = positions[start] + progress[:, None] * delta
        # This checks the interpolant itself: it lies on the full straight
        # segment, which need not be safe merely because source knots are safe.
        if not grid.is_segment_free(positions[start, :2], positions[end, :2], config.footprint_radius_m):
            raise ValueError("merged motion segment is not collision-free")
        deviation = np.linalg.norm(positions[indices] - positions[start] - anchor_progress[:, None] * delta, axis=1)
        source_progress = np.interp(progress, anchor_progress, indices)
        timestamps = (elapsed_steps + np.arange(steps + 1)) / config.sample_hz \
            if config.sample_mode == "uniform_time" else elapsed_time + s * duration
        values = dict(timestamp_s=timestamps,
            positions_xyz=segment_positions, yaw_rad=yaws[start] + progress * angle,
            velocity_xyz=derivatives[0][:, None] * delta,
            acceleration_xyz=derivatives[1][:, None] * delta,
            jerk_xyz=derivatives[2][:, None] * delta,
            yaw_rate_rad_s=derivatives[0] * angle, yaw_accel_rad_s2=derivatives[1] * angle,
            yaw_jerk_rad_s3=derivatives[2] * angle, source_progress=source_progress)
        for key, value in values.items():
            arrays[key].append(value[1:] if primitives else value)
        primitives.append(dict(kind=run["kind"], source_start=start, source_end=end,
            start_frame=elapsed_steps, end_frame=elapsed_steps + steps, duration_s=duration,
            source_anchor_indices=run["anchors"], source_motion_count=len(indices) - 1,
            max_source_anchor_deviation_m=float(deviation.max())))
        elapsed_steps += steps
        elapsed_time += duration
    if not primitives:
        return MotionRecord(np.array([0.]), positions[:1].copy(), yaws[:1].copy(),
            *(np.zeros((1, 3)) for _ in range(3)), *(np.zeros(1) for _ in range(3)),
            np.array([float(len(positions) - 1)]), [])
    result = MotionRecord(**{key: np.concatenate(value) for key, value in arrays.items()}, primitives=primitives)
    result.source_progress[0] = 0.
    result.source_progress[-1] = float(len(positions) - 1)
    return result
