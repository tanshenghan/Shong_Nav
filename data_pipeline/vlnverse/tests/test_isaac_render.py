"""CPU checks for preview inputs and failed-render rejection, without Isaac."""

import subprocess
import sys

import numpy as np
import pytest

from data_pipeline.vlnverse.replay.isaac_render import _load_poses, _validate_rgba


def test_module_and_help_do_not_import_isaac():
    code = (
        "import sys; from data_pipeline.vlnverse.replay import isaac_render; "
        "assert not any(k == 'isaacsim' or k.startswith('omni.') for k in sys.modules); "
        "isaac_render.main(['--help'])"
    )
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert "--camera-height-m" in result.stdout


def test_load_poses_retains_nonuniform_input_timestamps(tmp_path):
    path = tmp_path / "poses.npz"
    positions = np.array([[0., 0., 0.], [0.2, 0., 0.], [0.25, 0., 0.]])
    timestamps = np.array([0., 0.4, 1.7])
    np.savez(path, positions_xyz=positions, yaw_rad=np.zeros(3), timestamp_s=timestamps)
    loaded, yaw, times = _load_poses(path)
    np.testing.assert_array_equal(loaded, positions)
    np.testing.assert_array_equal(yaw, np.zeros(3))
    np.testing.assert_array_equal(times, timestamps)


@pytest.mark.parametrize("positions,yaw,timestamps", [
    (np.zeros((0, 3)), np.zeros(0), None),
    (np.zeros((2, 2)), np.zeros(2), None),
    (np.zeros((2, 3)), np.zeros(1), None),
    (np.array([[np.nan, 0., 0.]]), np.zeros(1), None),
    (np.zeros((2, 3)), np.zeros(2), np.array([0., 0.])),
    (np.zeros((2, 3)), np.zeros(2), np.array([0., np.inf])),
])
def test_load_poses_rejects_ambiguous_or_nonfinite_inputs(tmp_path, positions, yaw, timestamps):
    path = tmp_path / "invalid.npz"
    fields = dict(positions_xyz=positions, yaw_rad=yaw)
    if timestamps is not None:
        fields["timestamp_s"] = timestamps
    np.savez(path, **fields)
    with pytest.raises(ValueError):
        _load_poses(path)


def test_rgba_validation_preserves_real_rgb():
    rgba = np.zeros((2, 3, 4), dtype=np.uint8)
    rgba[..., 3] = 255
    rgba[1, 2, :3] = [4, 20, 199]
    rgb = _validate_rgba(rgba, 3, 2)
    np.testing.assert_array_equal(rgb, rgba[..., :3])
    assert rgb.flags.c_contiguous


@pytest.mark.parametrize("frame", [
    None,
    np.empty((0, 0, 4), dtype=np.uint8),
    np.ones((3, 2, 4), dtype=np.uint8),
    np.ones((2, 3, 4), dtype=np.float32),
    np.full((2, 3, 4), np.nan),
    np.zeros((2, 3, 4), dtype=np.uint8),
    np.tile(np.array([0, 0, 0, 255], dtype=np.uint8), (2, 3, 1)),
    np.tile(np.array([10, 20, 30, 0], dtype=np.uint8), (2, 3, 1)),
])
def test_failed_capture_never_becomes_a_synthetic_image(frame):
    with pytest.raises(ValueError):
        _validate_rgba(frame, 3, 2)
