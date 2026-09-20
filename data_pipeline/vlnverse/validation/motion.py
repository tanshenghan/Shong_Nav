"""Checks on generated motion, separate from the retiming implementation."""

from __future__ import annotations

import numpy as np


def validate_motion(record, source, grid, config):
    arrays = record.arrays()
    count = len(record.timestamp_s)
    for name, values in arrays.items():
        expected = (count, 3) if name.endswith("_xyz") else (count,)
        if values.shape != expected or not np.isfinite(values).all():
            raise ValueError(f"invalid motion array: {name}")
    if not count or abs(record.timestamp_s[0]) > 1e-10:
        raise ValueError("motion must start at synthetic time zero")
    dt = np.diff(record.timestamp_s)
    if not (dt > 0).all():
        raise ValueError("motion timestamps must be strictly increasing")
    if (config.sample_mode == "uniform_time" and count > 1
            and not np.allclose(dt, 1 / config.sample_hz, rtol=1e-8, atol=1e-10)):
        raise ValueError("motion sampling is not uniform")
    for generated, original in [(record.positions_xyz[0], source.positions_xyz[0]),
                                (record.positions_xyz[-1], source.positions_xyz[-1])]:
        if not np.allclose(generated, original, rtol=0, atol=1e-8):
            raise ValueError("source start/goal position changed")
    yaw_error = record.yaw_rad[[0, -1]] - source.yaw_rad[[0, -1]]
    if np.max(np.abs(np.arctan2(np.sin(yaw_error), np.cos(yaw_error)))) > 1e-8:
        raise ValueError("source start/goal heading changed")
    if (np.diff(record.source_progress) < -1e-8).any():
        raise ValueError("source correspondence goes backward")
    if (record.source_progress < 0).any() or (record.source_progress > len(source.yaw_rad) - 1).any():
        raise ValueError("source correspondence is outside the episode")
    if not grid.path_is_free(record.positions_xyz[:, :2], config.footprint_radius_m):
        raise ValueError("generated swept footprint intersects obstacles or map bounds")
    limits = {
        "velocity_xyz": config.max_speed_m_s, "acceleration_xyz": config.max_accel_m_s2,
        "jerk_xyz": config.max_jerk_m_s3, "yaw_rate_rad_s": config.max_yaw_rate_rad_s,
        "yaw_accel_rad_s2": config.max_yaw_accel_rad_s2, "yaw_jerk_rad_s3": config.max_yaw_jerk_rad_s3,
    }
    measured = {}
    for name, limit in limits.items():
        values = arrays[name]
        norms = np.linalg.norm(values, axis=1) if values.ndim == 2 else np.abs(values)
        measured[name] = float(norms.max())
        if measured[name] > limit * (1 + 1e-7):
            raise ValueError(f"motion exceeds {name} limit")
        if np.max(np.abs(values[[0, -1]])) > 1e-8:
            raise ValueError(f"motion must start and end at rest: {name}")
    # Sparse samples may miss the real extrema. Bound the entire continuous
    # seventh-order curve, and verify its derivatives independently in factored
    # form instead of accepting arbitrary derivative fields from an NPZ.
    continuous_maxima = dict.fromkeys(limits, 0.)
    previous_end = 0
    for primitive in record.primitives:
        a, b = primitive["start_frame"], primitive["end_frame"]
        duration = primitive["duration_s"]
        if not (a == previous_end and a < b < count and duration > 0):
            raise ValueError("invalid motion primitive boundaries")
        if not np.isclose(record.timestamp_s[b] - record.timestamp_s[a], duration, rtol=1e-8, atol=1e-10):
            raise ValueError("primitive duration disagrees with timestamps")
        previous_end = b
        indices = np.asarray(primitive["source_anchor_indices"])
        if (indices.ndim != 1 or len(indices) < 2 or indices.dtype.kind not in "iu"
                or indices[0] < 0 or indices[-1] >= len(source.yaw_rad)
                or not (np.diff(indices) > 0).all()
                or indices[0] != primitive["source_start"] or indices[-1] != primitive["source_end"]
                or len(indices) - 1 != primitive["source_motion_count"]):
            raise ValueError("invalid source anchors")
        if config.sample_mode == "per_source_step":
            stride = config.poses_per_step - 1
            if b - a != (len(indices) - 1) * stride:
                raise ValueError("wrong pose count per source step")
            samples = a + np.arange(len(indices)) * stride
            errors = np.linalg.norm(record.positions_xyz[samples] - source.positions_xyz[indices], axis=1)
            if errors.max() > config.max_merge_deviation_m + 1e-8:
                raise ValueError("source anchor moved beyond merge tolerance")
            angle_errors = record.yaw_rad[samples] - source.yaw_rad[indices]
            if np.max(np.abs(np.arctan2(np.sin(angle_errors), np.cos(angle_errors)))) > 1e-7:
                raise ValueError("source anchor heading changed")
        delta = record.positions_xyz[b] - record.positions_xyz[a]
        angle = record.yaw_rad[b] - record.yaw_rad[a]
        u = (record.timestamp_s[a:b + 1] - record.timestamp_s[a]) / duration
        progress = u**4 * (35 - 84*u + 70*u**2 - 20*u**3)
        if (not np.allclose(record.positions_xyz[a:b + 1], record.positions_xyz[a] + progress[:, None] * delta,
                            rtol=1e-8, atol=1e-8)
                or not np.allclose(record.yaw_rad[a:b + 1], record.yaw_rad[a] + progress * angle,
                                   rtol=1e-8, atol=1e-8)):
            raise ValueError("samples disagree with continuous motion curve")
        derivatives = [140*u**3*(1-u)**3, 420*u**2*(1-u)**2*(1-2*u),
                       840*u*(1-u)*(1-5*u+5*u**2)]
        # Exact derivative maxima on u in [0,1].
        for order, peak, derivative, linear, angular in zip(
                (1, 2, 3), (35/16, 84*np.sqrt(5)/25, 105/2), derivatives,
                ("velocity_xyz", "acceleration_xyz", "jerk_xyz"),
                ("yaw_rate_rad_s", "yaw_accel_rad_s2", "yaw_jerk_rad_s3")):
            for name, amplitude in [(linear, np.linalg.norm(delta)), (angular, abs(angle))]:
                maximum = float(amplitude * peak / duration**order)
                continuous_maxima[name] = max(continuous_maxima[name], maximum)
                if maximum > limits[name] * (1 + 1e-7):
                    raise ValueError(f"continuous motion exceeds {name} limit")
            if (not np.allclose(arrays[linear][a:b + 1], derivative[:, None] * delta / duration**order,
                                rtol=1e-7, atol=1e-8)
                    or not np.allclose(arrays[angular][a:b + 1], derivative * angle / duration**order,
                                       rtol=1e-7, atol=1e-8)):
                raise ValueError("stored derivative disagrees with continuous motion curve")
    if previous_end != count - 1:
        raise ValueError("motion primitives do not cover all samples")
    # Also check finite differences using each actual sampling interval.
    if count > 1:
        for values, derivative, maximum_next_derivative, name in [
            (record.positions_xyz, record.velocity_xyz, config.max_accel_m_s2, "translation velocity"),
            (record.velocity_xyz, record.acceleration_xyz, config.max_jerk_m_s3, "translation acceleration"),
            (record.yaw_rad, record.yaw_rate_rad_s, config.max_yaw_accel_rad_s2, "yaw rate"),
            (record.yaw_rate_rad_s, record.yaw_accel_rad_s2, config.max_yaw_jerk_rad_s3, "yaw acceleration"),
        ]:
            error = np.diff(values, axis=0) / (dt[:, None] if values.ndim == 2 else dt) - derivative[:-1]
            norm = np.linalg.norm(error, axis=1) if error.ndim == 2 else np.abs(error)
            if (norm > maximum_next_derivative * dt / 2 + 1e-7).any():
                raise ValueError(f"stored derivative disagrees with sampled motion: {name}")
    return dict(valid=True, frames=count, duration_s=float(record.timestamp_s[-1]),
        sampling_mode=config.sample_mode,
        sample_hz=config.sample_hz if config.sample_mode == "uniform_time" else None,
        poses_per_source_step=config.poses_per_step if config.sample_mode == "per_source_step" else None,
        uniform_timestamps=bool(len(dt) and np.allclose(dt, dt[0], rtol=1e-8, atol=1e-10)),
        time_basis="synthetic_retimed_seconds",
        primitive_count=len(record.primitives), maxima=measured, continuous_maxima=continuous_maxima,
        collision_check="exact_swept_disc_against_closed_2d_grid_cells",
        footprint_radius_with_margin_m=config.footprint_radius_m,
        simulator_replayed=False, measured_physical_timing=False)
