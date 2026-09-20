"""Continuous-curve timing, derivative certificates and differential-drive yaw."""

from dataclasses import replace
from math import comb

import numpy as np
import pytest

from data_pipeline.vlnverse.motion.planned import PlannedMotionConfig, retime_planned_curve


def polynomial_curve(x, y):
    """Independent power-to-Bernstein conversion for one polynomial piece."""
    controls = np.zeros((6, 2))
    for axis, coefficients in enumerate((x, y)):
        for row in range(6):
            controls[row, axis] = sum(value * comb(row, order) / comb(5, order)
                                      for order, value in enumerate(coefficients) if order <= row)
    return controls[None]


def split_curve(controls):
    work = controls[0].copy()
    left, right = [work[0]], [work[-1]]
    while len(work) > 1:
        work = (work[:-1] + work[1:]) / 2
        left.append(work[0])
        right.append(work[-1])
    return np.array([left, right[::-1]])


def test_straight_clock_has_known_continuous_speed_peak_and_uniform_time():
    controls = polynomial_curve([0., 14.5], [0.])
    arrays, metadata = retime_planned_curve(controls, [0., 14.5], .125)
    duration = metadata["duration_s"]
    assert 40.27 < duration < 40.4
    np.testing.assert_allclose(np.diff(arrays["timestamp_s"]), 1 / 15, atol=1e-13)
    np.testing.assert_allclose(arrays["positions_xyz"][[0, -1]], [[0, 0, .125], [14.5, 0, .125]])
    assert metadata["continuous_maxima_upper_bounds"]["speed_m_s"] == pytest.approx(
        14.5 / .9 / duration, rel=1e-12)
    for name in ("velocity_xyz", "acceleration_xyz", "jerk_xyz", "yaw_rate_rad_s",
                 "yaw_accel_rad_s2", "yaw_jerk_rad_s3"):
        assert not arrays[name][[0, -1]].any()
    assert np.all(np.linalg.norm(arrays["velocity_xyz"][1:-1], axis=1) > 0)


def test_curved_motion_has_tangent_yaw_and_analytic_yaw_derivatives():
    # p(s)=(s, s^2/2), so theta=atan(s), theta'=1/(1+s^2), etc.
    arrays, metadata = retime_planned_curve(polynomial_curve([0., 1.], [0., 0., .5]),
                                          [0., 1.], 0.)
    s = arrays["curve_parameter"]
    np.testing.assert_allclose(arrays["positions_xyz"][:, 0], s)
    np.testing.assert_allclose(arrays["positions_xyz"][:, 1], s ** 2 / 2, atol=1e-14)
    np.testing.assert_allclose(arrays["yaw_rad"], np.arctan(s), atol=1e-14)
    q1, q2, q3 = (arrays[name][:, 0]
                  for name in ("velocity_xyz", "acceleration_xyz", "jerk_xyz"))
    theta1 = 1 / (1 + s ** 2)
    theta2 = -2 * s / (1 + s ** 2) ** 2
    theta3 = (6 * s ** 2 - 2) / (1 + s ** 2) ** 3
    np.testing.assert_allclose(arrays["yaw_rate_rad_s"], theta1 * q1, atol=1e-13)
    np.testing.assert_allclose(arrays["yaw_accel_rad_s2"], theta2 * q1 ** 2 + theta1 * q2,
                               atol=1e-13)
    np.testing.assert_allclose(arrays["yaw_jerk_rad_s3"],
                               theta3 * q1 ** 3 + 3 * theta2 * q1 * q2 + theta1 * q3,
                               atol=1e-13)
    yaw = arrays["yaw_rad"]
    velocity = arrays["velocity_xyz"]
    lateral = -velocity[:, 0] * np.sin(yaw) + velocity[:, 1] * np.cos(yaw)
    np.testing.assert_allclose(lateral, 0., atol=1e-14)
    assert metadata["interior_stops"] == 0


def test_split_c4_curve_never_stops_at_piece_join():
    controls = polynomial_curve([0., 1.], [0., 0., .5])
    single, _ = retime_planned_curve(controls, [0., 1.], 0.)
    split, metadata = retime_planned_curve(split_curve(controls), [0., .5, 1.], 0.)
    near_join = np.argmin(np.abs(split["curve_parameter"] - .5))
    assert np.linalg.norm(split["velocity_xyz"][near_join]) > .1
    assert max(metadata["maximum_absolute_join_errors_orders_0_to_4"]) < 1e-10
    np.testing.assert_allclose(single["positions_xyz"][[0, -1]], split["positions_xyz"][[0, -1]])


def test_unwrapped_yaw_crosses_pi_without_jump_and_accepts_equivalent_start_branch():
    controls = polynomial_curve([0., -1.], [.025, -.1, .1])
    initial = np.arctan2(-.1, -1.) + 2 * np.pi
    arrays, _ = retime_planned_curve(controls, [0., 1.], 0., initial_yaw_rad=initial)
    assert arrays["yaw_rad"][0] > np.pi
    assert arrays["yaw_rad"][-1] < np.pi
    assert np.max(np.abs(np.diff(arrays["yaw_rad"]))) < .06
    assert arrays["yaw_rad"][0] == pytest.approx(initial)


@pytest.mark.parametrize("field,limit", [
    ("max_speed_m_s", .03), ("max_accel_m_s2", .005), ("max_jerk_m_s3", .002),
    ("max_yaw_rate_rad_s", .02), ("max_yaw_accel_rad_s2", .003),
    ("max_yaw_jerk_rad_s3", .003),
])
def test_each_motion_limit_changes_clock_and_is_continuously_bounded(field, limit):
    controls = polynomial_curve([0., 1.], [0., 0., .5])
    config = replace(PlannedMotionConfig(), **{field: limit})
    _, baseline = retime_planned_curve(controls, [0., 1.], 0.)
    _, metadata = retime_planned_curve(controls, [0., 1.], 0., config)
    assert metadata["duration_s"] > baseline["duration_s"]
    for name, bound in metadata["continuous_maxima_upper_bounds"].items():
        assert bound <= getattr(config, "max_" + name) * (1 + 1e-12)
        assert metadata["sampled_maxima"][name] <= bound + 1e-12


def test_sparse_samples_do_not_replace_continuous_certificate():
    config = PlannedMotionConfig(sample_hz=.035, time_law="global_septic")
    _, metadata = retime_planned_curve(polynomial_curve([0., 14.5], [0.]), [0., 14.5], 0., config)
    assert metadata["sample_count"] == 4
    assert metadata["continuous_maxima_upper_bounds"]["speed_m_s"] > (
        metadata["sampled_maxima"]["speed_m_s"] * 1.3)
    assert metadata["certificate"]["interval_count"] == config.certificate_subdivisions


@pytest.mark.parametrize("x,y", [([.25, -1., 1.], [0.]), ([0.], [0.]), ([0., 0., 1.], [0.])])
def test_cusps_and_zero_endpoint_tangents_are_rejected(x, y):
    with pytest.raises(ValueError, match="cusp|zero tangent|regularity"):
        retime_planned_curve(polynomial_curve(x, y), [0., 1.], 0.)


def test_join_discontinuity_is_rejected_before_timing():
    controls = split_curve(polynomial_curve([0., 1.], [0., 0., .5]))
    controls[1, 2, 1] += .001
    with pytest.raises(ValueError, match="not C4"):
        retime_planned_curve(controls, [0., .5, 1.], 0.)


def test_inconsistent_start_yaw_and_excessive_sample_count_fail_explicitly():
    controls = polynomial_curve([0., 1.], [0.])
    with pytest.raises(ValueError, match="forward curve tangent"):
        retime_planned_curve(controls, [0., 1.], 0., initial_yaw_rad=np.pi)
    with pytest.raises(ValueError, match="max_samples"):
        retime_planned_curve(controls, [0., 1.], 0., PlannedMotionConfig(max_samples=2))


@pytest.mark.parametrize("kwargs", [{"sample_hz": 0}, {"sample_hz": np.nan},
                                    {"max_speed_m_s": -1}, {"certificate_subdivisions": True},
                                    {"certificate_subdivisions": 1.5}, {"ramp_fraction": .5},
                                    {"ramp_fraction": 0}, {"time_law": "unknown"}])
def test_invalid_configuration_rejected(kwargs):
    with pytest.raises(ValueError):
        PlannedMotionConfig(**kwargs)


def test_parameter_and_shape_contract_rejected():
    controls = polynomial_curve([0., 1.], [0.])
    with pytest.raises(ValueError, match="strictly increasing"):
        retime_planned_curve(controls, [1., 0.], 0.)
    with pytest.raises(ValueError, match="segments,6,2"):
        retime_planned_curve(controls[0], [0., 1.], 0.)
    with pytest.raises(ValueError, match="finite scalar"):
        retime_planned_curve(controls, [0., 1.], np.nan)


def test_cruise_clock_uses_short_smooth_ramps_instead_of_episode_long_acceleration():
    controls = polynomial_curve([0., 14.5], [0.])
    cruise, cruise_metadata = retime_planned_curve(controls, [0., 14.5], 0.)
    septic, septic_metadata = retime_planned_curve(
        controls, [0., 14.5], 0., PlannedMotionConfig(time_law="global_septic"))
    assert cruise_metadata["duration_s"] < septic_metadata["duration_s"] * .55
    assert cruise["positions_xyz"][75, 0] > 1.
    assert septic["positions_xyz"][75, 0] < .02
    timestamps = cruise["timestamp_s"]
    middle = (timestamps > cruise_metadata["duration_s"] * .1) & (
        timestamps < cruise_metadata["duration_s"] * .9)
    np.testing.assert_allclose(cruise["velocity_xyz"][middle, 0],
                               cruise["velocity_xyz"][middle, 0][0], atol=1e-14)
    np.testing.assert_allclose(cruise["acceleration_xyz"][middle], 0., atol=1e-13)
    np.testing.assert_allclose(cruise["jerk_xyz"][middle], 0., atol=1e-13)
    assert len(cruise_metadata["time_law"]["clock_pieces"]) == 3


def test_cruise_bound_spans_clock_joins_even_with_one_geometric_interval():
    config = PlannedMotionConfig(certificate_subdivisions=1)
    arrays, metadata = retime_planned_curve(polynomial_curve([0., 14.5], [0.]),
                                          [0., 14.5], 0., config)
    assert metadata["certificate"]["interval_count"] == 1
    assert np.linalg.norm(arrays["velocity_xyz"][len(arrays["timestamp_s"]) // 2]) > 0
    for name, bound in metadata["continuous_maxima_upper_bounds"].items():
        assert metadata["sampled_maxima"][name] <= bound + 1e-12
