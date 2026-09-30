"""Route, physical-limit and no-corner-cutting checks for the smooth baseline."""

import copy
from dataclasses import replace

import numpy as np
import pytest

from data_pipeline.vlnverse.motion.retime import MotionConfig, retime_episode
from data_pipeline.vlnverse.scene.occupancy import OccupancyGrid
from data_pipeline.vlnverse.source.official import SourceEpisode
from data_pipeline.vlnverse.validation.motion import validate_motion


def free_map():
    return OccupancyGrid(np.zeros((100, 100), dtype=bool), 0.1, (-5., -5.))


def corner_episode():
    # A straight run, six left-turn samples at one position, another straight.
    positions = [[i * .25, 0., 0.] for i in range(5)]
    positions += [[1., 0., 0.]] * 6
    positions += [[1., i * .25, 0.] for i in range(1, 5)]
    yaw = [0.] * 5 + [i * np.pi / 12 for i in range(1, 7)] + [np.pi / 2] * 4
    return SourceEpisode("corner", positions, yaw)


def straight_episode(steps=1):
    return SourceEpisode("straight", [[i * .25, 0., 0.] for i in range(steps + 1)],
                         [0.] * (steps + 1))


@pytest.mark.parametrize("kind", ["translate", "rotate"])
def test_default_single_source_motion_has_exactly_six_poses(kind):
    source = (straight_episode() if kind == "translate" else
              SourceEpisode("turn", [[0., 0., 0.]] * 2, [0., np.pi / 12]))
    config = MotionConfig()
    record = retime_episode(source, free_map(), config)
    assert config.sample_mode == "per_source_step" and config.poses_per_step == 6
    assert len(record.timestamp_s) == 6
    np.testing.assert_allclose(record.positions_xyz[[0, -1]], source.positions_xyz)
    np.testing.assert_allclose(record.yaw_rad[[0, -1]], source.yaw_rad)
    assert record.primitives[0]["source_anchor_indices"] == [0, 1]
    assert record.primitives[0]["source_motion_count"] == 1
    quality = validate_motion(record, source, free_map(), config)
    assert quality["sampling_mode"] == "per_source_step"
    assert quality["poses_per_source_step"] == 6
    assert quality["sample_hz"] is None


@pytest.mark.parametrize("poses", [5, 6, 7, 8])
@pytest.mark.parametrize("kind", ["translate", "rotate"])
def test_allowed_pose_counts_include_shared_endpoints(poses, kind):
    source = (straight_episode(2) if kind == "translate" else
              SourceEpisode("turn", [[0., 0., 0.]] * 3, [0., np.pi / 12, np.pi / 6]))
    config = MotionConfig(poses_per_step=poses)
    record = retime_episode(source, free_map(), config)
    assert len(record.timestamp_s) == 1 + 2 * (poses - 1)
    anchor_frames = [0, poses - 1, 2 * (poses - 1)]
    np.testing.assert_allclose(record.positions_xyz[anchor_frames], source.positions_xyz, atol=1e-5)
    np.testing.assert_allclose(record.yaw_rad[anchor_frames], source.yaw_rad, atol=1e-5)
    assert validate_motion(record, source, free_map(), config)["valid"]


def test_two_straight_steps_share_anchor_without_stopping():
    source, config = straight_episode(2), MotionConfig()
    record = retime_episode(source, free_map(), config)
    assert len(record.timestamp_s) == 11
    assert len(record.primitives) == 1
    assert record.primitives[0]["source_anchor_indices"] == [0, 1, 2]
    assert record.primitives[0]["source_motion_count"] == 2
    np.testing.assert_allclose(record.positions_xyz[5], [.25, 0., 0.], atol=1e-5)
    assert record.velocity_xyz[5, 0] > 0
    assert record.source_progress[5] == pytest.approx(1., abs=1e-5)
    # Equal halves happen to yield uniform times, without a configured rate.
    quality = validate_motion(record, source, free_map(), config)
    assert quality["uniform_timestamps"] is True
    assert quality["sample_hz"] is None


def test_translate_turn_translate_shares_joins_and_stores_sixteen_poses():
    angle = np.pi / 12
    source = SourceEpisode("three_actions", [[0., 0., 0.], [.25, 0., 0.], [.25, 0., 0.],
        [.25 + .25 * np.cos(angle), .25 * np.sin(angle), 0.]], [0., 0., angle, angle])
    config = MotionConfig()
    record = retime_episode(source, free_map(), config)
    assert len(record.timestamp_s) == 16
    assert [run["kind"] for run in record.primitives] == ["translate", "rotate", "translate"]
    assert [(run["start_frame"], run["end_frame"]) for run in record.primitives] == [(0, 5), (5, 10), (10, 15)]
    np.testing.assert_allclose(record.positions_xyz[::5], source.positions_xyz, atol=1e-5)
    np.testing.assert_allclose(record.yaw_rad[::5], source.yaw_rad, atol=1e-5)
    assert validate_motion(record, source, free_map(), config)["valid"]


def test_duplicate_source_states_do_not_add_frames_or_motion_counts():
    source = SourceEpisode("duplicates", [[0., 0., 0.]] * 2 + [[.25, 0., 0.]] * 2 +
                           [[.5, 0., 0.]] * 2, [0.] * 6)
    config = MotionConfig()
    record = retime_episode(source, free_map(), config)
    assert len(record.timestamp_s) == 11
    assert record.primitives[0]["source_motion_count"] == 2
    assert record.primitives[0]["source_anchor_indices"] == [0, 2, 4]
    assert (np.diff(record.timestamp_s) > 0).all()
    assert validate_motion(record, source, free_map(), config)["valid"]


def test_source_step_sampling_has_variable_dt_and_ignores_sample_hz():
    # Three equal distances have unequal travel times along one smooth run.
    source = straight_episode(3)
    config = MotionConfig()
    record = retime_episode(source, free_map(), config)
    other = retime_episode(source, free_map(), replace(config, sample_hz=997.))
    assert len(record.timestamp_s) == 16
    dt = np.diff(record.timestamp_s)
    assert (dt > 0).all() and np.ptp(dt) > 1e-3
    np.testing.assert_allclose(record.positions_xyz[::5], source.positions_xyz, atol=1e-5)
    for name, array in record.arrays().items():
        np.testing.assert_array_equal(array, other.arrays()[name])
    quality = validate_motion(record, source, free_map(), config)
    assert quality["valid"] and not quality["uniform_timestamps"]


@pytest.mark.parametrize("sample_hz", [30., 60.])
def test_uniform_time_mode_retains_fixed_frequency(sample_hz):
    source = straight_episode(2)
    config = MotionConfig(sample_mode="uniform_time", sample_hz=sample_hz)
    record = retime_episode(source, free_map(), config)
    np.testing.assert_allclose(np.diff(record.timestamp_s), 1 / sample_hz, atol=1e-12)
    if sample_hz == 30:
        assert len(record.timestamp_s) == 91
        assert record.timestamp_s[-1] == pytest.approx(3.)
    else:
        assert len(record.timestamp_s) > 91
    quality = validate_motion(record, source, free_map(), config)
    assert quality["sample_hz"] == sample_hz
    assert quality["sampling_mode"] == "uniform_time"
    assert quality["poses_per_source_step"] is None


def test_continuous_limits_detect_peak_between_sparse_samples():
    source = straight_episode()
    # A six-point segment misses the velocity maximum at its time midpoint.
    config = MotionConfig(max_accel_m_s2=100., max_jerk_m_s3=100.)
    record = retime_episode(source, free_map(), config)
    assert len(record.timestamp_s) == 6
    assert not np.any(np.isclose(record.timestamp_s, record.timestamp_s[-1] / 2))
    quality = validate_motion(record, source, free_map(), config)
    sampled_peak = quality["maxima"]["velocity_xyz"]
    continuous_peak = quality["continuous_maxima"]["velocity_xyz"]
    assert continuous_peak == pytest.approx(config.max_speed_m_s)
    assert sampled_peak < continuous_peak * .9
    assert continuous_peak == pytest.approx((35 / 16) * .25 / record.timestamp_s[-1])
    # A sampled-only validator would accept this stricter limit incorrectly.
    stricter = replace(config, max_speed_m_s=(sampled_peak + continuous_peak) / 2)
    with pytest.raises(ValueError, match="velocity_xyz"):
        validate_motion(record, source, free_map(), stricter)


def test_straights_and_turns_coalesce_and_have_smooth_joins():
    source, grid, config = corner_episode(), free_map(), MotionConfig()
    record = retime_episode(source, grid, config)
    assert [p["kind"] for p in record.primitives] == ["translate", "rotate", "translate"]
    assert validate_motion(record, source, grid, config)["valid"]
    # No diagonal shortcut through the inside corner.
    assert np.all((np.abs(record.positions_xyz[:, 1]) < 1e-10) | (np.abs(record.positions_xyz[:, 0] - 1) < 1e-10))
    for segment in record.primitives:
        for frame in [segment["start_frame"], segment["end_frame"]]:
            for values in [record.velocity_xyz, record.acceleration_xyz, record.jerk_xyz,
                           record.yaw_rate_rad_s, record.yaw_accel_rad_s2, record.yaw_jerk_rad_s3]:
                np.testing.assert_allclose(values[frame], 0, atol=1e-10)
    # A source 0.25m anchor inside the first run is passed with nonzero speed.
    index = np.argmin(abs(record.positions_xyz[:record.primitives[0]["end_frame"], 0] - .25))
    assert record.velocity_xyz[index, 0] > .05


def test_float32_straight_route_is_not_split_into_stops():
    angle = .62
    distance = np.arange(25) * .25
    positions = np.column_stack([distance * np.cos(angle), distance * np.sin(angle), np.zeros(25)]).astype(np.float32)
    source = SourceEpisode("float32", positions, np.full(25, angle))
    grid = OccupancyGrid(np.zeros((160, 160), dtype=bool), .1, (-6., -6.))
    config = MotionConfig()
    record = retime_episode(source, grid, config)
    assert len(record.primitives) == 1
    assert record.primitives[0]["max_source_anchor_deviation_m"] < config.max_merge_deviation_m
    assert validate_motion(record, source, grid, config)["valid"]


def test_yaw_wrap_and_reversals_preserve_turn_order():
    yaw = np.deg2rad([170., 175., -175., -170., -175., 175.])
    source = SourceEpisode("turns", np.zeros((len(yaw), 3)), yaw)
    record = retime_episode(source, free_map(), MotionConfig())
    assert len(record.primitives) == 2
    assert record.yaw_rad.max() == pytest.approx(np.deg2rad(190))
    assert validate_motion(record, source, free_map(), MotionConfig())["valid"]


def test_no_physical_duration_inferred_from_source_timestamp():
    source = corner_episode()
    other = copy.deepcopy(source)
    other.source_timestamps = np.arange(len(source.yaw_rad)) * 1000
    config = MotionConfig()
    a = retime_episode(source, free_map(), config)
    b = retime_episode(other, free_map(), config)
    np.testing.assert_array_equal(a.timestamp_s, b.timestamp_s)
    stricter = retime_episode(source, free_map(), replace(config, max_jerk_m_s3=.01))
    assert stricter.timestamp_s[-1] > a.timestamp_s[-1]
    assert validate_motion(stricter, source, free_map(), replace(config, max_jerk_m_s3=.01))["valid"]


def test_swept_obstacle_between_safe_knots_is_rejected():
    occupied = np.zeros((100, 100), dtype=bool)
    occupied[50, 50] = True
    grid = OccupancyGrid(occupied, .1, (-5., -5.))
    source = SourceEpisode("blocked", [[-1., 0., 0.], [1., 0., 0.]], [0., 0.])
    with pytest.raises(ValueError, match="swept footprint"):
        retime_episode(source, grid, MotionConfig())


@pytest.mark.parametrize("positions,yaws,reason", [
    ([[0, 0, 0], [1, 0, 1]], [0, 0], "nonplanar"),
    ([[0, 0, 0], [0, 1, 0]], [0, 0], "sideways"),
    ([[0, 0, 0], [1, 0, 0]], [0, .2], "simultaneous"),
])
def test_incompatible_source_requires_a_different_planner(positions, yaws, reason):
    with pytest.raises(ValueError, match=reason):
        retime_episode(SourceEpisode("bad", positions, yaws), free_map(), MotionConfig())


def test_stationary_episode_and_duplicate_poses():
    source = SourceEpisode("static", [[0, 0, 0]] * 3, [0, 0, 0])
    record = retime_episode(source, free_map(), MotionConfig())
    assert len(record.timestamp_s) == 1 and not record.primitives
    assert validate_motion(record, source, free_map(), MotionConfig())["duration_s"] == 0


def test_independent_validation_catches_inconsistent_derivatives():
    source, grid, config = corner_episode(), free_map(), MotionConfig()
    record = retime_episode(source, grid, config)
    record.velocity_xyz[:] = 0
    with pytest.raises(ValueError, match="stored derivative disagrees"):
        validate_motion(record, source, grid, config)


@pytest.mark.parametrize("setting", [dict(sample_hz=0), dict(robot_radius_m=-1),
                                    dict(max_speed_m_s=np.nan), dict(max_yaw_jerk_rad_s3=0),
                                    dict(sample_mode="unknown"), dict(poses_per_step=4),
                                    dict(poses_per_step=9), dict(poses_per_step=6.0),
                                    dict(poses_per_step=True), dict(poses_per_step="6")])
def test_invalid_limits_rejected(setting):
    with pytest.raises(ValueError):
        MotionConfig(**setting)
