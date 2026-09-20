"""The ViT-cache selective path must not change the model input.

When the engine hands ``_vit_cached_keys`` to the data processor (so that only
cache-miss tubelets are patchified), the rendered prompt -- in particular the
``<X.X seconds>`` timestamps -- has to stay identical to the plain path: a
non-SlowFast checkpoint was trained with window-relative frame positions, and
the hf backend (which never uses the cache) renders exactly those. Pre-ViT
pooled checkpoints additionally must not receive miss-only pixel rows, which
the pooling step cannot consume.

These tests need the public ``Qwen/Qwen3-VL-2B-Instruct`` processor config in
the local Hugging Face cache and ``torchvision``; they are skipped otherwise.
"""

from __future__ import annotations

import glob
import os
from types import SimpleNamespace

import pytest
import torch

pytest.importorskip("torchvision")

_SNAPSHOTS = sorted(
    glob.glob(
        os.path.expanduser(
            "~/.cache/huggingface/hub/models--Qwen--Qwen3-VL-2B-Instruct/snapshots/*/"
        )
    )
)
if not _SNAPSHOTS:
    pytest.skip("public Qwen3-VL processor config not cached", allow_module_level=True)
SNAP = _SNAPSHOTS[-1]


@pytest.fixture(scope="module")
def processor():
    from lightnav.processing import VLNQwen3VLProcessor

    return VLNQwen3VLProcessor.from_pretrained(SNAP, padding_side="left")


@pytest.fixture(scope="module")
def pos_id_func():
    from lightnav.inference.model import _position_id_func_from_config

    return _position_id_func_from_config(SNAP)


def _make_dp(processor, pos_id_func, *, pool_enable, pool_spatial, pool_stage, video_size):
    from lightnav.data_processor import Qwen3VLDataProcessor

    return Qwen3VLDataProcessor(
        processor=processor,
        model_max_length=8192,
        video_fps=4,
        video_pool_enable=pool_enable,
        video_pool_spatial=pool_spatial,
        video_pool_mode="avg",
        video_pool_stage=pool_stage,
        video_size=video_size,
        position_id_func=pos_id_func,
    )


def _make_bundle(processor, *, pool_enable, pool_spatial, video_size):
    return SimpleNamespace(
        video_size=video_size,
        video_fps=4,
        pool_enable=pool_enable,
        pool_spatial=pool_spatial,
        pool_mode="avg",
        slowfast_tiers=None,
        predict_horizon=10,
        action_method="flat",
        processor=processor,
    )


def _video(num_frames: int, video_size):
    torch.manual_seed(0)
    return torch.rand(num_frames, 3, *video_size) * 2.0 - 1.0


def _decode(processor, out):
    return processor.tokenizer.decode(out["input_ids"].tolist(), skip_special_tokens=False)


def test_selective_path_keeps_window_relative_timestamps(processor, pos_id_func):
    """Absolute frame ids (100..107) must not leak into the ``<X.X seconds>`` labels."""
    from lightnav.inference.samples import build_tracking_sample

    video_size = (224, 320)
    dp = _make_dp(
        processor, pos_id_func, pool_enable=False, pool_spatial=1, pool_stage="pre_vit",
        video_size=video_size,
    )
    bundle = _make_bundle(processor, pool_enable=False, pool_spatial=1, video_size=video_size)
    video = _video(8, video_size)

    plain = build_tracking_sample(video, "Follow the person.", list(range(100, 108)), bundle)
    ref = dp.process_sample(plain, add_generation_prompt=True, validate_video_shapes=False)

    cached = build_tracking_sample(video, "Follow the person.", list(range(100, 108)), bundle)
    cached["_vit_cached_keys"] = set()  # cold cache: every tubelet is a miss
    out = dp.process_sample(cached, add_generation_prompt=True, validate_video_shapes=False)

    assert torch.equal(out["input_ids"], ref["input_ids"])
    assert torch.equal(out["video_grid_thw"], ref["video_grid_thw"])
    text = _decode(processor, out)
    assert "<0.1 seconds>" in text and "<25.1 seconds>" not in text


def test_selective_path_matches_plain_for_pooled_two_segment_sample(processor, pos_id_func):
    """Two-segment (pooled history + current) samples: same tokens with and without keys."""
    from lightnav.inference.samples import build_tracking_sample

    video_size = (224, 320)
    dp = _make_dp(
        processor, pos_id_func, pool_enable=True, pool_spatial=2, pool_stage="post_vit",
        video_size=video_size,
    )
    bundle = _make_bundle(processor, pool_enable=True, pool_spatial=2, video_size=video_size)
    video = _video(8, video_size)

    plain = build_tracking_sample(video, "Follow the person.", list(range(8)), bundle)
    assert len(plain["video_segments"]) == 2
    ref = dp.process_sample(plain, add_generation_prompt=True, validate_video_shapes=False)

    cached = build_tracking_sample(video, "Follow the person.", list(range(8)), bundle)
    cached["_vit_cached_keys"] = set()
    out = dp.process_sample(cached, add_generation_prompt=True, validate_video_shapes=False)

    assert torch.equal(out["input_ids"], ref["input_ids"])


def test_pre_vit_pooled_sample_ignores_the_selective_mask(processor, pos_id_func):
    """Pre-ViT pooling needs full pixel rows: a warm cache must not break the pooling step."""
    from lightnav.inference.samples import build_tracking_sample

    video_size = (256, 512)  # merged grid 8x16, divisible by the pooling factor
    dp = _make_dp(
        processor, pos_id_func, pool_enable=True, pool_spatial=2, pool_stage="pre_vit",
        video_size=video_size,
    )
    bundle = _make_bundle(processor, pool_enable=True, pool_spatial=2, video_size=video_size)
    video = _video(8, video_size)

    plain = build_tracking_sample(video, "Follow the person.", list(range(8)), bundle)
    ref = dp.process_sample(plain, add_generation_prompt=True, validate_video_shapes=False)

    cached = build_tracking_sample(video, "Follow the person.", list(range(8)), bundle)
    # Pretend the first tubelets of the pooled history are already cached.
    gh, gw = video_size[0] // 16, video_size[1] // 16
    cached["_vit_cached_keys"] = {(0, 1, gh, gw), (2, 3, gh, gw)}
    out = dp.process_sample(cached, add_generation_prompt=True, validate_video_shapes=False)

    assert torch.equal(out["input_ids"], ref["input_ids"])
    assert torch.equal(out["pixel_values_videos"], ref["pixel_values_videos"])


def test_keep_mode_sized_frames_go_through_the_selective_path(processor, pos_id_func):
    """A 4:3 session (288x384 frames) must produce a consistent grid with and without keys."""
    from lightnav.inference.samples import build_tracking_sample

    video_size = (256, 448)  # the checkpoint's size; the session picked 288x384 instead
    dp = _make_dp(
        processor, pos_id_func, pool_enable=True, pool_spatial=2, pool_stage="post_vit",
        video_size=video_size,
    )
    bundle = _make_bundle(processor, pool_enable=True, pool_spatial=2, video_size=video_size)
    video = _video(8, (288, 384))

    plain = build_tracking_sample(video, "Follow the person.", list(range(8)), bundle)
    ref = dp.process_sample(plain, add_generation_prompt=True, validate_video_shapes=False)
    # ViT grid = 288/16 = 18 rows x 384/16 = 24 cols for every segment (video_grid_thw
    # itself carries the post-ViT pooled grid for the pooled history segment)
    assert all(int(g[1]) == 18 and int(g[2]) == 24 for g in ref["_original_video_grid_thw"])

    cached = build_tracking_sample(video, "Follow the person.", list(range(8)), bundle)
    gh, gw = 288 // 16, 384 // 16
    cached["_vit_cached_keys"] = {(0, 1, gh, gw)}  # first tubelet of the history is cached
    out = dp.process_sample(cached, add_generation_prompt=True, validate_video_shapes=False)
    assert torch.equal(out["input_ids"], ref["input_ids"])
    assert torch.equal(out["video_grid_thw"], ref["video_grid_thw"])
    # one tubelet's rows (18 * 24 patches) were dropped from the pixel tensor
    assert ref["pixel_values_videos"].shape[0] - out["pixel_values_videos"].shape[0] == gh * gw
