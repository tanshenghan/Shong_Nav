"""NavigationPolicy: the per-session frame buffer (ring window or full SlowFast episode)."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from lightnav.inference.policies import NavigationPolicy


class _FakeCache:
    def __init__(self):
        self.cleared = 0

    def clear(self):
        self.cleared += 1


class FakeEngine:
    def __init__(self, slowfast_tiers=None):
        self.bundle = SimpleNamespace(video_size=(4, 4), slowfast_tiers=slowfast_tiers, num_history_frames=4)
        self.reset_count = 0
        self.caches: list[_FakeCache] = []

    def reset_episode_state(self):
        self.reset_count += 1

    def new_vit_cache(self):
        cache = _FakeCache()
        self.caches.append(cache)
        return cache


def _frame(value: int = 0) -> np.ndarray:
    return np.full((4, 4, 3), value, dtype=np.uint8)


def _marker_convert(self, frame) -> torch.Tensor:
    # Skip the real preprocessing; encode the frame's marker value so ordering is checkable.
    return torch.full((3, 4, 4), float(frame[0, 0, 0]), dtype=torch.float32)


def test_observe_converts_the_frame_into_the_model_tensor():
    policy = NavigationPolicy(FakeEngine(), num_history_frames=4)
    policy.observe(_frame(255))
    policy.observe(_frame(0))

    video = policy._get_video_tensor()
    assert tuple(video.shape) == (2, 3, 4, 4)
    assert video.dtype == torch.float32
    assert torch.all(video[0] == 1.0) and torch.all(video[1] == -1.0)
    assert policy._history_frame_ids == [0, 1]
    assert policy._buffer_len == 2


def test_ring_buffer_keeps_the_newest_frames_in_chronological_order(monkeypatch):
    monkeypatch.setattr(NavigationPolicy, "_convert_frame", _marker_convert)
    policy = NavigationPolicy(FakeEngine(), num_history_frames=3)
    for value in (1, 2, 3, 4):
        policy.observe(_frame(value))

    video = policy._get_video_tensor()
    assert [int(f[0, 0, 0].item()) for f in video] == [2, 3, 4]
    assert policy._history_frame_ids == [1, 2, 3]  # absolute ids, oldest first
    assert policy._buffer_len == 3
    assert len(policy._history) == 3  # raw history truncated to the window too

    policy.observe(_frame(5))
    assert [int(f[0, 0, 0].item()) for f in policy._get_video_tensor()] == [3, 4, 5]
    assert policy._history_frame_ids == [2, 3, 4]


def test_video_tensor_is_a_prefix_view_while_the_window_is_not_full(monkeypatch):
    monkeypatch.setattr(NavigationPolicy, "_convert_frame", _marker_convert)
    policy = NavigationPolicy(FakeEngine(), num_history_frames=4)
    policy.observe(_frame(7))
    video = policy._get_video_tensor()
    assert tuple(video.shape) == (1, 3, 4, 4)
    assert int(video[0, 0, 0, 0].item()) == 7


def test_get_video_tensor_before_any_frame_raises():
    policy = NavigationPolicy(FakeEngine(), num_history_frames=4)
    with pytest.raises(RuntimeError, match="No frames"):
        policy._get_video_tensor()


def test_reset_clears_state_and_the_session_vit_cache():
    engine = FakeEngine()
    policy = NavigationPolicy(engine, num_history_frames=4)
    assert policy._vit_cache is engine.caches[0]
    policy.observe(_frame())
    policy.observe(_frame())

    policy.reset("go left")

    assert policy.instruction == "go left"
    assert policy._history == [] and policy._history_frame_ids == []
    assert policy._buffer_len == 0
    assert engine.caches[0].cleared == 1
    assert engine.reset_count == 1
    with pytest.raises(RuntimeError):
        policy._get_video_tensor()

    policy.observe(_frame())
    assert policy._history_frame_ids == [0]  # frame ids restart after reset


def test_reset_defaults_to_an_empty_instruction():
    policy = NavigationPolicy(FakeEngine(), num_history_frames=4)
    policy.reset("something")
    policy.reset()
    assert policy.instruction == ""


def test_engine_without_a_vit_cache_factory_is_supported():
    engine = SimpleNamespace(
        bundle=SimpleNamespace(video_size=(4, 4), slowfast_tiers=None),
        reset_episode_state=lambda: None,
    )
    policy = NavigationPolicy(engine, num_history_frames=2)
    assert policy._vit_cache is None
    policy.observe(_frame())
    policy.reset()
    assert policy._buffer_len == 0


def test_observe_stores_a_copy_of_the_source_frame():
    policy = NavigationPolicy(FakeEngine(), num_history_frames=2)
    frame = _frame(3)
    policy.observe(frame)
    frame[:] = 9
    assert int(policy._history[0][0, 0, 0]) == 3


def test_slowfast_policy_keeps_the_whole_episode(monkeypatch):
    monkeypatch.setattr(NavigationPolicy, "_convert_frame", _marker_convert)
    tiers = [{"age_lo": 0, "age_hi": 1, "mode": "dense", "pool_spatial": 1}]
    policy = NavigationPolicy(FakeEngine(slowfast_tiers=tiers), num_history_frames=4)
    assert policy.slowfast is True

    for value in range(70):  # past the initial 64-slot allocation -> buffer grows
        policy.observe(_frame(value))

    video = policy._get_video_tensor()
    assert tuple(video.shape) == (70, 3, 4, 4)
    assert [int(f[0, 0, 0].item()) for f in video] == list(range(70))
    assert policy._buffer_len == 70
    assert policy._history_frame_ids == list(range(70))
    assert len(policy._history) == 70

    policy.reset()
    assert policy._sf_buffer is None and policy._buffer_len == 0


def test_slowfast_flag_is_false_for_a_plain_window():
    assert NavigationPolicy(FakeEngine(), num_history_frames=4).slowfast is False
