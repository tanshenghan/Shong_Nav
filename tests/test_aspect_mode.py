"""``aspect_mode="keep"``: frames keep their aspect ratio at the checkpoint's pixel budget."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from lightnav.inference.config import InferenceConfig
from lightnav.inference.frame_preprocessing import choose_video_size
from lightnav.inference.policies import NavigationPolicy


@pytest.mark.parametrize(
    ("frame_hw", "base", "multiple", "expected"),
    [
        ((480, 640), (256, 448), 32, (288, 384)),  # 4:3 camera on the released checkpoint
        ((270, 480), (256, 448), 32, (256, 448)),  # 16:9 source -> the training size itself
        ((256, 448), (256, 448), 32, (256, 448)),  # identical -> unchanged
        ((512, 512), (256, 448), 32, (352, 352)),  # square
        ((480, 640), (256, 448), 64, (320, 448)),  # pre-ViT pooling by 2 needs multiples of 64
        ((480, 640), (224, 320), 32, (224, 288)),
        ((1080, 1920), (256, 448), 32, (256, 448)),  # resolution does not matter, aspect does
    ],
)
def test_choose_video_size_keeps_aspect_at_the_pixel_budget(frame_hw, base, multiple, expected):
    got = choose_video_size(frame_hw, base, multiple=multiple)
    assert got == expected
    assert got[0] % multiple == 0 and got[1] % multiple == 0
    # within a comfortable margin of the training token budget
    assert abs(got[0] * got[1] - base[0] * base[1]) <= 0.25 * base[0] * base[1]


def _engine(aspect_mode, *, pool_stage="post_vit", pool_spatial=2, tiers=None):
    vp = SimpleNamespace(patch_size=16, merge_size=2, temporal_patch_size=2)
    bundle = SimpleNamespace(
        video_size=(256, 448), slowfast_tiers=tiers, num_history_frames=8,
        pool_stage=pool_stage, pool_spatial=pool_spatial, processor=SimpleNamespace(video_processor=vp),
    )
    return SimpleNamespace(
        bundle=bundle, aspect_mode=aspect_mode, new_vit_cache=lambda: None,
        reset_episode_state=lambda: None,
    )


def test_stretch_mode_always_uses_the_checkpoint_size():
    policy = NavigationPolicy(_engine("stretch"), num_history_frames=8)
    policy.reset("go")
    policy.observe(np.zeros((480, 640, 3), dtype=np.uint8))
    assert policy.video_size == (256, 448)
    assert tuple(policy._get_video_tensor().shape[-2:]) == (256, 448)


def test_keep_mode_picks_the_size_from_the_first_frame_and_resets_per_episode():
    policy = NavigationPolicy(_engine("keep"), num_history_frames=8)
    policy.reset("go")
    policy.observe(np.zeros((480, 640, 3), dtype=np.uint8))
    policy.observe(np.zeros((480, 640, 3), dtype=np.uint8))
    assert policy.video_size == (288, 384)
    video = policy._get_video_tensor()
    assert tuple(video.shape) == (2, 3, 288, 384) and video.dtype == torch.float32

    policy.reset("next episode")  # a new episode may come from another camera
    assert policy.video_size is None
    policy.observe(np.zeros((270, 480, 3), dtype=np.uint8))
    assert policy.video_size == (256, 448)


def test_keep_mode_respects_pre_vit_pooling_factors():
    tiers = [{"name": "long", "age_lo": 2, "age_hi": 10, "mode": "dense", "pool_spatial": 4}]
    policy = NavigationPolicy(
        _engine("keep", pool_stage="pre_vit", pool_spatial=2, tiers=tiers), num_history_frames=8
    )
    assert policy._size_multiple() == 128  # 32 * 4 (2 divides 4)
    policy.observe(np.zeros((480, 640, 3), dtype=np.uint8))
    h, w = policy.video_size
    assert h % 128 == 0 and w % 128 == 0


def test_engine_config_default_is_stretch_and_validates():
    assert InferenceConfig().aspect_mode == "stretch"
    from lightnav.inference.engine import VLNInferenceEngine

    bundle = SimpleNamespace(slowfast_tiers=None, num_history_frames=4)
    with pytest.raises(ValueError, match="aspect_mode"):
        VLNInferenceEngine(bundle, backend="hf", aspect_mode="crop")
    assert VLNInferenceEngine(bundle, backend="hf", aspect_mode="keep").aspect_mode == "keep"
