"""Model Frame dimension contract.

Source Frames arrive at whatever resolution a client sends; the Model Frame is
fixed by the checkpoint's ``bundle.video_size``. These tests are CPU-only: no
ViT, no LLM, only the preprocessing seam and the policy buffer it feeds.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from lightnav.inference.frame_preprocessing import rgb_frame_to_model_tensor
from lightnav.inference.policies import NavigationPolicy

VIDEO_SIZE = (256, 320)  # (height, width)
SOURCE_SIZES = [(270, 480), (257, 321), VIDEO_SIZE]


def _source_frame(height: int, width: int) -> np.ndarray:
    rng = np.random.default_rng(height * 1000 + width)
    return rng.integers(0, 256, size=(height, width, 3), dtype=np.uint8)


def _fake_engine() -> SimpleNamespace:
    return SimpleNamespace(
        bundle=SimpleNamespace(
            num_history_frames=8,
            slowfast_tiers=None,
            video_size=VIDEO_SIZE,
        ),
        reset_episode_state=lambda: None,
    )


@pytest.mark.parametrize(("height", "width"), SOURCE_SIZES)
def test_every_source_frame_becomes_one_model_frame_size(height, width):
    tensor = rgb_frame_to_model_tensor(_source_frame(height, width), VIDEO_SIZE)

    assert tensor.dtype == torch.float32
    assert tensor.shape == (3, *VIDEO_SIZE)
    assert float(tensor.min()) >= -1.0
    assert float(tensor.max()) <= 1.0


def test_value_range_maps_uint8_to_minus_one_one():
    black = rgb_frame_to_model_tensor(np.zeros((*VIDEO_SIZE, 3), dtype=np.uint8), VIDEO_SIZE)
    white = rgb_frame_to_model_tensor(np.full((*VIDEO_SIZE, 3), 255, dtype=np.uint8), VIDEO_SIZE)
    assert torch.all(black == -1.0)
    assert torch.all(white == 1.0)


def test_concurrent_sessions_with_different_source_sizes_share_a_model_frame_size():
    sessions = []
    for height, width in ((270, 480), (257, 321)):
        policy = NavigationPolicy(_fake_engine(), num_history_frames=8)
        policy.reset(instruction="")
        policy.observe(_source_frame(height, width))
        policy.observe(_source_frame(height, width))
        sessions.append(policy)

    spatial_shapes = {tuple(policy._video_buffer.shape[-2:]) for policy in sessions}
    assert spatial_shapes == {VIDEO_SIZE}


def test_non_hwc_rgb_source_frames_fail_fast():
    with pytest.raises(ValueError, match="Model Frame"):
        rgb_frame_to_model_tensor(np.zeros((270, 480), dtype=np.uint8), VIDEO_SIZE)
    with pytest.raises(ValueError, match="Model Frame"):
        rgb_frame_to_model_tensor(np.zeros((270, 480, 4), dtype=np.uint8), VIDEO_SIZE)


def test_a_resize_that_misses_the_target_fails_fast_without_pixels(monkeypatch):
    import lightnav.inference.frame_preprocessing as module

    monkeypatch.setattr(
        module,
        "resize_video_tensor",
        lambda video, target_size: torch.zeros(3, 4, 5),
    )

    with pytest.raises(ValueError) as caught:
        rgb_frame_to_model_tensor(_source_frame(270, 480), VIDEO_SIZE)

    message = str(caught.value)
    assert "(3, 4, 5)" in message
    assert str(VIDEO_SIZE) in message
    assert "tensor(" not in message  # shapes only, never pixels
