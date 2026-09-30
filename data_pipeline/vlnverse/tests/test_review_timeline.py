"""The video clock must preserve reference timing and original source poses."""

import numpy as np
import pytest

from data_pipeline.vlnverse.replay.compare import comparison_schedule


def test_shared_clock_holds_original_but_visits_added_poses():
    times, original, smooth, anchors = comparison_schedule(
        [0., .2, .4, .8, 1., 2.], [0., .2, .4, .6, .8, 1.], 2, 10)
    np.testing.assert_allclose(anchors, [0., 2.])
    assert len(times) == 21
    assert (original[:-1] == 0).all() and original[-1] == 1
    assert len(np.unique(smooth)) == 6
    assert smooth[-1] == 5


def test_video_endpoint_is_included_without_extrapolation():
    times, original, smooth, _ = comparison_schedule([0., .23], [0., 1.], 2, 10)
    np.testing.assert_allclose(times, [0., .1, .2, .23])
    assert original[-1] == smooth[-1] == 1


def test_pose_frames_uses_each_pose_once_and_aligns_source_anchors():
    times, original, smooth, anchors = comparison_schedule(
        [0., 1.3, 6.6, 7., 30.], [0., .03, 1. - 1e-14, 1.3, 2.], 3, 15, "pose_frames")
    np.testing.assert_allclose(times, np.arange(5) / 15.)
    np.testing.assert_array_equal(smooth, np.arange(5))
    np.testing.assert_array_equal(original, [0, 0, 1, 1, 2])
    np.testing.assert_allclose(anchors, [0., 6.6, 30.])
    assert len(times) / 15 == pytest.approx(1 / 3)


def test_unknown_playback_mode_is_rejected():
    with pytest.raises(ValueError, match="unknown playback mode"):
        comparison_schedule([0., 1.], [0., 1.], 2, 15, "invalid")


@pytest.mark.parametrize("timestamps,progress,fps", [
    ([0., 0.], [0., 1.], 20), ([0., 1.], [1., 0.], 20),
    ([0., 1.], [0., 2.], 20), ([0., 1.], [0., 1.], 0),
])
def test_invalid_video_clock_is_rejected(timestamps, progress, fps):
    with pytest.raises(ValueError):
        comparison_schedule(timestamps, progress, 2, fps)
