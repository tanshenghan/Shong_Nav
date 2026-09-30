"""Uniform-time, forward-only motion on a regular piecewise quintic planar curve.

The spatial planner owns collision checking. This module certifies derivative
bounds with outward-rounded Bernstein intervals, then stretches a smooth
acceleration/cruise/deceleration clock. It never stops at interior Bezier joins. The poses are
synthetic reference states, not feedback from a physical robot or a simulator.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np


@dataclass(frozen=True)
class PlannedMotionConfig:
    sample_hz: float = 15.0
    time_law: str = "ramp_cruise"
    ramp_fraction: float = 0.1
    max_speed_m_s: float = 0.4
    max_accel_m_s2: float = 0.5
    max_jerk_m_s3: float = 1.0
    max_yaw_rate_rad_s: float = 0.8
    max_yaw_accel_rad_s2: float = 1.0
    max_yaw_jerk_rad_s3: float = 2.0
    certificate_subdivisions: int = 32
    max_regularity_depth: int = 18
    minimum_tangent_norm: float = 1e-8
    join_tolerance: float = 1e-6
    max_samples: int = 200_000

    def __post_init__(self):
        if self.time_law not in {"ramp_cruise", "global_septic"}:
            raise ValueError("time_law must be ramp_cruise or global_septic")
        integer_fields = {"certificate_subdivisions", "max_regularity_depth", "max_samples"}
        for name, value in asdict(self).items():
            if name == "time_law":
                continue
            if isinstance(value, bool) or not np.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be positive and finite")
            if name in integer_fields and not isinstance(value, int):
                raise ValueError(f"{name} must be an integer")
        if self.max_regularity_depth > 30:
            raise ValueError("max_regularity_depth must be at most 30")
        if self.ramp_fraction >= .5:
            raise ValueError("ramp_fraction must be less than 0.5")


_NAMES = ("speed_m_s", "accel_m_s2", "jerk_m_s3", "yaw_rate_rad_s",
          "yaw_accel_rad_s2", "yaw_jerk_rad_s3")
_ORDERS = np.array([1., 2., 3., 1., 2., 3.])
_PROGRESS_CONTROLS = np.array([0., 0., 0., 0., 1., 1., 1., 1.])


def _point(value):
    value = np.asarray(value, dtype=np.float64)
    return np.stack((value, value))


def _outward(low, high):
    return np.stack((np.nextafter(low, -np.inf), np.nextafter(high, np.inf)))


def _add(a, b):
    return _outward(a[0] + b[0], a[1] + b[1])


def _neg(a):
    return np.stack((-a[1], -a[0]))


def _sub(a, b):
    return _add(a, _neg(b))


def _mul(a, b):
    products = np.array([a[0] * b[0], a[0] * b[1], a[1] * b[0], a[1] * b[1]])
    return _outward(np.min(products, axis=0), np.max(products, axis=0))


def _scale(a, factor):
    return _mul(a, _point(factor))


def _square(a):
    minimum = np.where((a[0] <= 0) & (a[1] >= 0), 0.,
                       np.minimum(a[0] ** 2, a[1] ** 2))
    maximum = np.maximum(a[0] ** 2, a[1] ** 2)
    return np.stack((np.maximum(0., np.nextafter(minimum, -np.inf)),
                     np.nextafter(maximum, np.inf)))


def _divide_positive(a, b):
    if np.any(b[0] <= 0):
        raise ValueError("cannot certify a positive tangent norm; curve may contain a cusp")
    reciprocal = _outward(1 / b[1], 1 / b[0])
    return _mul(a, reciprocal)


def _sum(a):
    result = _point(0.)
    for index in range(a.shape[1]):
        result = _add(result, a[:, index])
    return result


def _dot(a, b):
    return _sum(_mul(a, b))


def _cross(a, b):
    return _sub(_mul(a[:, 0], b[:, 1]), _mul(a[:, 1], b[:, 0]))


def _absolute_upper(a):
    return np.maximum(np.abs(a[0]), np.abs(a[1]))


def _norm_upper(a):
    return float(np.nextafter(np.sqrt(_sum(_square(a))[1]), np.inf))


def _split_box(controls, fraction):
    """De Casteljau split with interval arithmetic; control axis is axis 1."""
    work = controls.copy()
    left, right = [work[:, 0]], [work[:, -1]]
    # Computing 1-fraction is enclosed as well: fractions need not be dyadic.
    complement = _sub(_point(1.), _point(fraction))
    while work.shape[1] > 1:
        work = _add(_mul(work[:, :-1], complement), _scale(work[:, 1:], fraction))
        left.append(work[:, 0])
        right.append(work[:, -1])
    return np.stack(left, axis=1), np.stack(right[::-1], axis=1)


def _restrict_box(controls, low, high):
    if not 0 <= low < high <= 1:
        raise ValueError("invalid Bernstein restriction interval")
    if high < 1:
        controls = _split_box(controls, high)[0]
    if low > 0:
        # Expand the lower split by one representable number so rounding a/b
        # cannot accidentally discard a sliver of the requested interval.
        fraction = max(0., float(np.nextafter(low / high, -np.inf)))
        controls = _split_box(controls, fraction)[1]
    return controls


def _hull(controls):
    return np.stack((np.min(controls[0], axis=0), np.max(controls[1], axis=0)))


def _derivative_boxes(controls, width, count):
    current = _point(controls)
    result = [current]
    for _ in range(count):
        degree = current.shape[1] - 1
        if degree == 0:
            current = _point(np.zeros_like(current[0]))
        else:
            current = _divide_positive(
                _scale(_sub(current[:, 1:], current[:, :-1]), degree), _point(width))
        result.append(current)
    return result


def _evaluate(controls, fraction):
    """Evaluate ordinary scalar/vector Bezier controls at an array of values."""
    fraction = np.asarray(fraction)
    work = np.broadcast_to(controls, (len(fraction),) + controls.shape).copy()
    blend = fraction.reshape((-1,) + (1,) * controls.ndim)
    while work.shape[1] > 1:
        work = (1 - blend) * work[:, :-1] + blend * work[:, 1:]
    return work[:, 0]


def _evaluate_controls_box(controls, fraction):
    work = np.broadcast_to(_point(controls)[:, :, None],
                           (2, len(controls), fraction.shape[1])).copy()
    complement = _sub(_point(1.), fraction)[:, None]
    blend = fraction[:, None]
    while work.shape[1] > 1:
        work = _add(_mul(work[:, :-1], complement), _mul(work[:, 1:], blend))
    return work[:, 0]


def _clock_pieces(config):
    if config.time_law == "global_septic":
        specification = [(0., 1., _PROGRESS_CONTROLS)]
    else:
        fraction = config.ramp_fraction
        rise = fraction / (1 - fraction)
        start = np.array([0., 0., 0., 0., rise / 6, rise / 3, rise / 2])
        specification = [(0., fraction, start),
                         (fraction, 1 - fraction, np.array([rise / 2, 1 - rise / 2])),
                         (1 - fraction, 1., 1 - start[::-1])]
    pieces = []
    for low, high, controls in specification:
        ordinary = [controls]
        for _ in range(3):
            degree = len(ordinary[-1]) - 1
            ordinary.append(np.diff(ordinary[-1]) * degree / (high - low)
                            if degree else np.zeros(1))
        pieces.append({"low": low, "high": high, "controls": controls,
                       "ordinary": ordinary,
                       "intervals": _derivative_boxes(controls, high - low, 3)})
    for previous, following in zip(pieces, pieces[1:]):
        for order in range(4):
            left, right = previous["ordinary"][order][-1], following["ordinary"][order][0]
            if abs(left - right) > config.join_tolerance * max(1., abs(left), abs(right)):
                raise ValueError("clock join is not C3 within tolerance; ramp_fraction is ill-conditioned")
    return pieces


def _evaluate_clock_box(fraction, pieces):
    result = np.empty((2, len(fraction)))
    for index, piece in enumerate(pieces):
        low, high = piece["low"], piece["high"]
        mask = (fraction >= low) & ((fraction < high) if index < len(pieces) - 1
                                    else (fraction <= high))
        local = _divide_positive(_sub(_point(fraction[mask]), _point(low)), _point(high - low))
        result[:, mask] = _evaluate_controls_box(piece["controls"], local)
    return result


def _evaluate_clock(fraction, pieces, order):
    result = np.empty(len(fraction))
    for index, piece in enumerate(pieces):
        low, high = piece["low"], piece["high"]
        mask = (fraction >= low) & ((fraction < high) if index < len(pieces) - 1
                                    else (fraction <= high))
        local = np.clip((fraction[mask] - low) / (high - low), 0, 1)
        result[mask] = _evaluate(piece["ordinary"][order], local)
    return result


def _clock_interval_hull(pieces, order, low, high):
    hulls = []
    for piece in pieces:
        start, stop = max(low, piece["low"]), min(high, piece["high"])
        if start >= stop:
            continue
        width = piece["high"] - piece["low"]
        local_start = max(0., (start - piece["low"]) / width - 32 * np.finfo(float).eps)
        local_stop = min(1., (stop - piece["low"]) / width + 32 * np.finfo(float).eps)
        hulls.append(_hull(_restrict_box(piece["intervals"][order], local_start, local_stop)))
    if not hulls:
        raise ValueError("empty clock certificate interval")
    return np.array([min(item[0] for item in hulls), max(item[1] for item in hulls)])


def _inverse_progress_brackets(target, pieces):
    """Enclose the inverse monotone clock, including rounded evaluations."""
    target = np.asarray(target)
    bounds = []
    for bound in (0, 1):
        low, high = np.zeros_like(target), np.ones_like(target)
        for _ in range(58):
            mid = (low + high) / 2
            values = _evaluate_clock_box(mid, pieces)
            below = values[1] < target if bound == 0 else values[0] <= target
            low, high = np.where(below, mid, low), np.where(below, high, mid)
        bounds.append(low if bound == 0 else high)
    result = np.stack(bounds)
    result[:, target == 0] = 0.
    result[:, target == 1] = 1.
    return result


def _yaw_derivative_boxes(v, a, b, c):
    denominator = _sum(_square(v))
    numerator = _cross(v, a)
    first_numerator = _cross(v, b)
    second_numerator = _add(_cross(a, b), _cross(v, c))
    first_denominator = _scale(_dot(v, a), 2.)
    second_denominator = _scale(_add(_dot(a, a), _dot(v, b)), 2.)
    theta1 = _divide_positive(numerator, denominator)
    theta2 = _divide_positive(
        _sub(first_numerator, _mul(theta1, first_denominator)), denominator)
    theta3 = _divide_positive(
        _sub(_sub(second_numerator, _scale(_mul(theta2, first_denominator), 2.)),
             _mul(theta1, second_denominator)), denominator)
    return theta1, theta2, theta3


def _curve_input(bezier_xy, parameter_knots, config):
    controls = np.asarray(bezier_xy, dtype=np.float64)
    knots = np.asarray(parameter_knots, dtype=np.float64)
    if (controls.ndim != 3 or controls.shape[1:] != (6, 2) or not len(controls)
            or not np.isfinite(controls).all()):
        raise ValueError("bezier_xy must be a finite [segments,6,2] array")
    if (knots.shape != (len(controls) + 1,) or not np.isfinite(knots).all()
            or not np.all(np.diff(knots) > 0)):
        raise ValueError("parameter_knots must be finite and strictly increasing")
    widths = np.diff(knots)
    if not np.isfinite(widths).all() or not np.isfinite(knots[-1] - knots[0]):
        raise ValueError("curve parameter range overflows")
    ordinary = []
    intervals = []
    for control, width in zip(controls, widths):
        derivatives = [control]
        for order in range(1, 5):
            derivatives.append(np.diff(derivatives[-1], axis=0) * (6 - order) / width)
        ordinary.append(derivatives)
        intervals.append(_derivative_boxes(control, width, 4))
    max_join_errors = np.zeros(5)
    for previous, following in zip(ordinary, ordinary[1:]):
        for order in range(5):
            error = np.linalg.norm(previous[order][-1] - following[order][0])
            scale = max(1., np.linalg.norm(previous[order][-1]),
                        np.linalg.norm(following[order][0]))
            max_join_errors[order] = max(max_join_errors[order], error)
            if error > config.join_tolerance * scale:
                raise ValueError(f"Bezier join is not C4: derivative order {order}, error {error}")
    if not all(np.isfinite(item).all() for group in ordinary for item in group):
        raise ValueError("curve derivatives overflow")
    return controls, knots, ordinary, intervals, max_join_errors


def _regular_leaves(derivatives, config):
    leaves = []

    def visit(low, high, depth):
        boxes = [_hull(_restrict_box(item, low, high)) for item in derivatives[1:]]
        squared_norm = _sum(_square(boxes[0]))
        if squared_norm[0] > config.minimum_tangent_norm ** 2:
            leaves.append((low, high, boxes, float(np.sqrt(squared_norm[0]))))
            return
        if depth >= config.max_regularity_depth:
            raise ValueError("curve has a cusp/zero tangent or regularity could not be certified")
        mid = (low + high) / 2
        visit(low, mid, depth + 1)
        visit(mid, high, depth + 1)

    edges = np.linspace(0, 1, config.certificate_subdivisions + 1)
    for low, high in zip(edges[:-1], edges[1:]):
        visit(float(low), float(high), 0)
    return leaves


def _certificate(knots, intervals, config, pieces):
    span = knots[-1] - knots[0]
    leaves = []
    for index, derivatives in enumerate(intervals):
        for low, high, boxes, minimum_tangent in _regular_leaves(derivatives, config):
            a = (knots[index] - knots[0] + low * (knots[index + 1] - knots[index])) / span
            b = (knots[index] - knots[0] + high * (knots[index + 1] - knots[index])) / span
            # Include arithmetic error in conversion to normalized progress.
            a = max(0., float(a - 32 * np.finfo(float).eps))
            b = min(1., float(b + 32 * np.finfo(float).eps))
            leaves.append((index, low, high, boxes, minimum_tangent, a, b))
    targets = np.array([value for leaf in leaves for value in leaf[-2:]])
    inverse = _inverse_progress_brackets(targets, pieces)
    maxima = np.zeros(6)
    certificate = []
    for index, leaf in enumerate(leaves):
        segment, low, high, boxes, minimum_tangent, _, _ = leaf
        u0, u1 = float(inverse[0, 2 * index]), float(inverse[1, 2 * index + 1])
        clock = [_scale(_clock_interval_hull(pieces, order, u0, u1), span)
                 for order in (1, 2, 3)]
        v, a, b, c = boxes
        q1, q2, q3 = clock
        q1_squared = _square(q1)
        q1_cubed = _mul(q1_squared, q1)
        velocity = _mul(v, q1)
        acceleration = _add(_mul(a, q1_squared), _mul(v, q2))
        jerk = _add(_add(_mul(b, q1_cubed), _scale(_mul(a, _mul(q1, q2)), 3.)),
                    _mul(v, q3))
        theta1, theta2, theta3 = _yaw_derivative_boxes(v, a, b, c)
        yaw_rate = _mul(theta1, q1)
        yaw_accel = _add(_mul(theta2, q1_squared), _mul(theta1, q2))
        yaw_jerk = _add(
            _add(_mul(theta3, q1_cubed), _scale(_mul(theta2, _mul(q1, q2)), 3.)),
            _mul(theta1, q3))
        bounds = np.array([_norm_upper(velocity), _norm_upper(acceleration),
                           _norm_upper(jerk), float(_absolute_upper(yaw_rate)),
                           float(_absolute_upper(yaw_accel)), float(_absolute_upper(yaw_jerk))])
        if not np.isfinite(bounds).all():
            raise ValueError("nonfinite continuous derivative bound")
        maxima = np.maximum(maxima, bounds)
        certificate.append({"segment": segment, "local_parameter_interval": [low, high],
                            "normalized_time_interval": [u0, u1],
                            "minimum_tangent_norm_lower_bound": minimum_tangent,
                            "unit_duration_derivative_upper_bounds": bounds.tolist()})
    return maxima, certificate


def retime_planned_curve(bezier_xy, parameter_knots, z_m, config=None, *, initial_yaw_rad=None):
    """Return ``(arrays, metadata)`` for a regular C4 quintic planar route.

    ``parameter_knots`` names the global parameter of the Bezier pieces; each
    piece's local parameter is in [0,1]. ``z_m`` is a constant world height.
    The optional initial yaw only chooses the unwrapped angular branch and must
    already agree with the starting tangent. Collision validation is separate.

    The interval certificate bounds each entire piece, including endpoints.
    C4 joins are checked to the recorded floating-point tolerance; certificates
    are conservative arithmetic enclosures, not claims of exact real-number
    equality between independently stored floating-point join coefficients.
    """
    config = PlannedMotionConfig() if config is None else config
    if not isinstance(config, PlannedMotionConfig):
        raise TypeError("config must be PlannedMotionConfig")
    if np.asarray(z_m).shape != () or not np.isfinite(z_m):
        raise ValueError("z_m must be one finite scalar")
    controls, knots, derivatives, intervals, join_errors = _curve_input(
        bezier_xy, parameter_knots, config)
    clock_pieces = _clock_pieces(config)
    normalized_bounds, certificate = _certificate(knots, intervals, config, clock_pieces)
    limits = np.array([getattr(config, "max_" + name) for name in _NAMES])
    required_duration = float(np.max((normalized_bounds / limits) ** (1 / _ORDERS)))
    # A margin covers subsequent ordinary evaluation/quantization arithmetic.
    sample_intervals = max(1, int(np.ceil(required_duration * config.sample_hz * (1 + 1e-10))))
    if sample_intervals + 1 > config.max_samples:
        raise ValueError(f"certified motion needs {sample_intervals + 1} samples, exceeds max_samples")
    duration = sample_intervals / config.sample_hz
    timestamps = np.arange(sample_intervals + 1) / config.sample_hz
    u = np.arange(sample_intervals + 1) / sample_intervals
    progress = _evaluate_clock(u, clock_pieces, 0)
    progress[0], progress[-1] = 0., 1.
    span = knots[-1] - knots[0]
    parameter = knots[0] + span * progress
    parameter[0], parameter[-1] = knots[0], knots[-1]
    q1, q2, q3 = [span * _evaluate_clock(u, clock_pieces, order) / duration ** order
                  for order in (1, 2, 3)]
    geometric = [np.zeros((len(u), 2)) for _ in range(5)]
    segments = np.minimum(np.searchsorted(knots, parameter, side="right") - 1, len(controls) - 1)
    for index, items in enumerate(derivatives):
        mask = segments == index
        local = np.clip((parameter[mask] - knots[index]) / (knots[index + 1] - knots[index]), 0, 1)
        for order in range(5):
            geometric[order][mask] = _evaluate(items[order], local)
    position, v, a, b, c = geometric
    denominator = np.sum(v * v, axis=1)
    def cross(left, right):
        return left[:, 0] * right[:, 1] - left[:, 1] * right[:, 0]
    numerator = cross(v, a)
    denominator1 = 2 * np.sum(v * a, axis=1)
    denominator2 = 2 * np.sum(a * a + v * b, axis=1)
    theta1 = numerator / denominator
    theta2 = (cross(v, b) - theta1 * denominator1) / denominator
    theta3 = (cross(a, b) + cross(v, c) - 2 * theta2 * denominator1
              - theta1 * denominator2) / denominator
    yaw = np.unwrap(np.arctan2(v[:, 1], v[:, 0]))
    if initial_yaw_rad is not None:
        if not np.isfinite(initial_yaw_rad):
            raise ValueError("initial_yaw_rad must be finite")
        yaw += 2 * np.pi * np.round((initial_yaw_rad - yaw[0]) / (2 * np.pi))
        if abs(yaw[0] - initial_yaw_rad) > config.join_tolerance:
            raise ValueError("initial_yaw_rad disagrees with the forward curve tangent")
    def vector3(xy):
        return np.column_stack((xy, np.zeros(len(xy))))
    arrays = {
        "timestamp_s": timestamps,
        "positions_xyz": np.column_stack((position, np.full(len(position), z_m))),
        "yaw_rad": yaw,
        "velocity_xyz": vector3(v * q1[:, None]),
        "acceleration_xyz": vector3(a * q1[:, None] ** 2 + v * q2[:, None]),
        "jerk_xyz": vector3(b * q1[:, None] ** 3 + 3 * a * (q1 * q2)[:, None]
                            + v * q3[:, None]),
        "yaw_rate_rad_s": theta1 * q1,
        "yaw_accel_rad_s2": theta2 * q1 ** 2 + theta1 * q2,
        "yaw_jerk_rad_s3": theta3 * q1 ** 3 + 3 * theta2 * q1 * q2 + theta1 * q3,
        "curve_parameter": parameter,
    }
    derivative_fields = ("velocity_xyz", "acceleration_xyz", "jerk_xyz", "yaw_rate_rad_s",
                         "yaw_accel_rad_s2", "yaw_jerk_rad_s3")
    for field in derivative_fields:
        arrays[field][[0, -1]] = 0.
    sampled = np.array([np.max(np.linalg.norm(arrays[field], axis=1))
                        if arrays[field].ndim == 2 else np.max(np.abs(arrays[field]))
                        for field in derivative_fields])
    bounds = normalized_bounds / duration ** _ORDERS
    if bounds[3] / config.sample_hz >= np.pi:
        raise ValueError("sample_hz is too low to certify an unambiguous unwrapped yaw")
    if not all(np.isfinite(value).all() for value in arrays.values()):
        raise ValueError("nonfinite sampled motion")
    if np.any(sampled > bounds * (1 + 1e-9) + 1e-10) or np.any(bounds > limits * (1 + 1e-12)):
        raise ValueError("sampled dynamics disagree with the continuous interval certificate")
    metadata = {
        "format": "vlnverse_planned_curve_motion_v1", "config": asdict(config),
        "time_semantics": "synthetic_reference_seconds", "duration_s": duration,
        "sample_count": len(timestamps), "sample_hz": config.sample_hz,
        "uniform_timestamps": True, "interior_stops": 0,
        "time_law": {"type": config.time_law,
                     "clock_pieces": [{"normalized_time_interval": [item["low"], item["high"]],
                                       "progress_bernstein_controls": item["controls"].tolist()}
                                      for item in clock_pieces],
                     "parameter_range": [float(knots[0]), float(knots[-1])]},
        "kinematics": "forward-only planar tangent yaw; no lateral reference velocity",
        "continuous_maxima_upper_bounds": dict(zip(_NAMES, bounds.tolist())),
        "sampled_maxima": dict(zip(_NAMES, sampled.tolist())),
        "required_duration_before_time_grid_s": required_duration,
        "maximum_absolute_join_errors_orders_0_to_4": join_errors.tolist(),
        "maximum_absolute_clock_join_errors_orders_0_to_3": [
            max((abs(left["ordinary"][order][-1] - right["ordinary"][order][0])
                 for left, right in zip(clock_pieces, clock_pieces[1:])), default=0.)
            for order in range(4)],
        "certificate": {
            "method": "outward-rounded Bernstein convex-hull interval bounds",
            "scope": "all points of each Bezier piece; C4 joins checked to floating-point tolerance",
            "derivative_order": list(_NAMES), "interval_count": len(certificate),
            "intervals": certificate,
            "limitations": ["This clock can be slower than curvature-adaptive local timing.",
                            "Collision geometry is validated by the separate planning stage.",
                            "This is a reference trajectory, not physical execution."],
        },
    }
    return arrays, metadata
