"""A same-length RGB file from a different episode is not aligned evidence."""

import numpy as np
import pytest

from data_pipeline.vlnverse.source.rgb import load_episode_rgb


def _files(root, array):
    parquet = root / "data/chunk-000/episode_000000.parquet"
    parquet.parent.mkdir(parents=True)
    parquet.write_bytes(b"already verified by the pose loader")
    path = root / "videos/chunk-000/observation.images.rgb/rgb.npy"
    path.parent.mkdir(parents=True)
    np.save(path, array)
    return parquet, path


def test_read_official_rgb_is_lossless_and_read_only(tmp_path):
    expected = np.arange(72, dtype=np.uint8).reshape(2, 3, 4, 3)
    parquet, path = _files(tmp_path / "scene/0_0", expected)
    actual, metadata = load_episode_rgb(path, parquet, 2)
    np.testing.assert_array_equal(actual, expected)
    assert not actual.flags.writeable
    assert metadata["camera_calibration_verified"] is False


def test_equal_frame_count_does_not_prove_episode_identity(tmp_path):
    array = np.zeros((2, 3, 4, 3), dtype=np.uint8)
    parquet, _ = _files(tmp_path / "scene/0_0", array)
    _, other_rgb = _files(tmp_path / "scene/1_0", array)
    with pytest.raises(ValueError, match="same episode"):
        load_episode_rgb(other_rgb, parquet, 2)


@pytest.mark.parametrize("array", [np.zeros((3, 3, 4, 3), dtype=np.uint8),
                                   np.zeros((2, 3, 4, 4), dtype=np.uint8),
                                   np.zeros((2, 3, 4, 3), dtype=np.float32)])
def test_wrong_count_channels_or_dtype_are_rejected(tmp_path, array):
    parquet, path = _files(tmp_path / "scene/0_0", array)
    with pytest.raises(ValueError, match="uint8"):
        load_episode_rgb(path, parquet, 2)
