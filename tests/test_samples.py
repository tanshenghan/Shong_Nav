"""Sample builders: prompt selection, video_segments layout and the engine's task dispatch."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from lightnav.inference.engine import VLNInferenceEngine
from lightnav.inference.samples import (
    _build_slowfast_eval_sample,
    build_tracking_sample,
    build_vln_traj_sample,
)
from lightnav.prompts import (
    TRACKING_PROMPT_TEMPLATE,
    TRACKING_PROMPT_TEMPLATE_POOLED,
    UNIFIED_TRAJ_PROMPT_TEMPLATE,
    VLN_TRAJ_PROMPT_TEMPLATE,
    VLN_TRAJ_PROMPT_TEMPLATE_POOLED,
    build_video_block,
    to_rvq_prompt,
)
from lightnav.slowfast import DEFAULT_SLOWFAST_TIERS_SPAN, validate_slowfast_tiers


def _bundle(**overrides) -> SimpleNamespace:
    fields = dict(
        pool_enable=True,
        pool_spatial=2,
        pool_mode="avg",
        video_fps=4,
        num_history_frames=4,
        predict_horizon=1,
    )
    fields.update(overrides)
    return SimpleNamespace(**fields)


def _bundle_without_pooling() -> SimpleNamespace:
    return _bundle(pool_enable=False)


def _check_common(sample: dict, video_fps: int = 4) -> None:
    assert sample["video_fps"] == video_fps
    assert sample["_allow_vit_cache"] is True
    assert sample["_skip_normalize"] is True
    assert sample["conversations"][0]["from"] == "human"
    assert sample["conversations"][1] == {"from": "gpt", "value": "placeholder"}
    assert sample["conversations"][0]["value"].count("<video>") == len(sample["video_segments"])


# -- tracking -------------------------------------------------------------------------


def test_tracking_sample_splits_pooled_history_and_current_segments():
    video = torch.zeros(5, 3, 224, 320)
    frame_ids = [10, 11, 12, 13, 14]
    sample = build_tracking_sample(video, "follow the person in red", frame_ids, _bundle())

    prompt = sample["conversations"][0]["value"]
    assert prompt == TRACKING_PROMPT_TEMPLATE_POOLED.format(task="follow the person in red")
    assert prompt.count("<video>") == 2
    assert "history observations" in prompt
    assert "current observation" in prompt
    assert "<navigation_task>follow the person in red</navigation_task>" in prompt
    assert "trajectory id token" in prompt
    assert "next" not in prompt.lower()
    assert "actions" not in prompt.lower()

    history, current = sample["video_segments"]
    assert torch.equal(history["video"], video[:-2])
    assert history["frame_indices"] == frame_ids[:-2]
    assert history["total_frames"] == 5
    assert history["pool_spatial"] == 2
    assert history["pool_mode"] == "avg"
    assert torch.equal(current["video"], video[-2:])
    assert current["frame_indices"] == frame_ids[-2:]
    assert current["total_frames"] == 5
    assert current["pool_spatial"] == 1
    assert current["pool_mode"] == "avg"
    _check_common(sample)
    assert "slowfast_abs_frame_indices" not in sample


def test_tracking_sample_uses_single_unpooled_segment_when_pooling_disabled():
    video = torch.zeros(5, 3, 224, 320)
    sample = build_tracking_sample(video, "follow the person in red", None, _bundle_without_pooling())

    prompt = sample["conversations"][0]["value"]
    assert prompt == TRACKING_PROMPT_TEMPLATE.format(task="follow the person in red")
    assert prompt.count("<video>") == 1
    assert "sequence of observations" in prompt
    assert "history observations" not in prompt
    assert "trajectory id token" in prompt

    segment = sample["video_segments"][0]
    assert segment["video"] is video
    assert segment["frame_indices"] == list(range(5))  # frame_ids=None -> positional ids
    assert segment["total_frames"] == 5
    assert segment["pool_spatial"] == 1
    assert segment["pool_mode"] == "avg"
    _check_common(sample)


def test_tracking_sample_uses_single_unpooled_segment_for_short_window():
    video = torch.zeros(2, 3, 224, 320)
    sample = build_tracking_sample(video, "follow", [4, 5], _bundle())

    assert sample["conversations"][0]["value"] == TRACKING_PROMPT_TEMPLATE.format(task="follow")
    segment = sample["video_segments"][0]
    assert len(sample["video_segments"]) == 1
    assert segment["video"] is video
    assert segment["frame_indices"] == [4, 5]
    assert segment["total_frames"] == 2
    assert segment["pool_spatial"] == 1
    _check_common(sample)


def test_pool_spatial_one_disables_the_history_split_even_when_pooling_is_enabled():
    video = torch.zeros(5, 3, 224, 320)
    sample = build_tracking_sample(video, "follow", None, _bundle(pool_spatial=1))
    assert len(sample["video_segments"]) == 1
    assert sample["conversations"][0]["value"] == TRACKING_PROMPT_TEMPLATE.format(task="follow")


def test_pool_mode_and_fps_are_taken_from_the_bundle():
    video = torch.zeros(5, 3, 224, 320)
    sample = build_tracking_sample(video, "follow", None, _bundle(pool_mode="max", video_fps=8))
    assert [s["pool_mode"] for s in sample["video_segments"]] == ["max", "max"]
    assert sample["video_fps"] == 8


# -- vlnce_traj -------------------------------------------------------------------------


def test_vln_traj_sample_splits_pooled_history_and_current_segments():
    video = torch.zeros(64, 3, 224, 320)
    frame_ids = list(range(100, 164))
    sample = build_vln_traj_sample(video, "go to the kitchen", frame_ids, _bundle())

    prompt = sample["conversations"][0]["value"]
    assert prompt == VLN_TRAJ_PROMPT_TEMPLATE_POOLED.format(task="go to the kitchen")
    assert "history observations <video> and current observation <video>" in prompt
    assert "<navigation_task>go to the kitchen</navigation_task>" in prompt
    assert "single trajectory id token" in prompt
    assert "next" not in prompt.lower().replace("navigation", "")

    history, current = sample["video_segments"]
    assert tuple(history["video"].shape) == (62, 3, 224, 320)
    assert history["frame_indices"] == frame_ids[:-2]
    assert history["total_frames"] == 64
    assert history["pool_spatial"] == 2
    assert tuple(current["video"].shape) == (2, 3, 224, 320)
    assert current["frame_indices"] == frame_ids[-2:]
    assert current["pool_spatial"] == 1
    _check_common(sample)


def test_vln_traj_sample_uses_single_unpooled_segment_when_pooling_disabled():
    video = torch.zeros(64, 3, 224, 320)
    sample = build_vln_traj_sample(video, "go to the kitchen", None, _bundle_without_pooling())

    prompt = sample["conversations"][0]["value"]
    assert prompt == VLN_TRAJ_PROMPT_TEMPLATE.format(task="go to the kitchen")
    assert "sequence of observations <video>" in prompt
    assert "history observations" not in prompt

    segment = sample["video_segments"][0]
    assert segment["video"] is video
    assert segment["frame_indices"] == list(range(64))
    assert segment["total_frames"] == 64
    assert segment["pool_spatial"] == 1
    _check_common(sample)


def test_tracking_and_vln_builders_differ_only_in_the_prompt():
    video = torch.zeros(5, 3, 224, 320)
    a = build_tracking_sample(video, "task", [1, 2, 3, 4, 5], _bundle())
    b = build_vln_traj_sample(video, "task", [1, 2, 3, 4, 5], _bundle())
    assert a["conversations"][0]["value"] != b["conversations"][0]["value"]
    for sa, sb in zip(a["video_segments"], b["video_segments"]):
        assert torch.equal(sa["video"], sb["video"])
        assert {k: v for k, v in sa.items() if k != "video"} == {
            k: v for k, v in sb.items() if k != "video"
        }


# -- slowfast -------------------------------------------------------------------------


def _marked_video(n: int) -> torch.Tensor:
    return torch.arange(n, dtype=torch.float32).reshape(n, 1, 1, 1) * torch.ones(n, 3, 4, 4)


def _slowfast_bundle(**overrides) -> SimpleNamespace:
    return _bundle(slowfast_tiers=validate_slowfast_tiers(DEFAULT_SLOWFAST_TIERS_SPAN), **overrides)


@pytest.mark.parametrize("builder", [build_tracking_sample, build_vln_traj_sample])
def test_slowfast_bundle_routes_both_builders_to_the_unified_sample(builder):
    video = _marked_video(30)
    sample = builder(video, "follow the person", list(range(30)), _slowfast_bundle())

    assert sample["slowfast_abs_frame_indices"] is True
    segments = sample["video_segments"]
    prompt = sample["conversations"][0]["value"]
    assert prompt == UNIFIED_TRAJ_PROMPT_TEMPLATE.format(
        task="follow the person", videos=build_video_block(len(segments)), horizon=1
    )
    assert prompt.endswith("Predict your future trajectory as a trajectory id token. ")
    _check_common(sample)

    flat = [i for s in segments for i in s["frame_indices"]]
    assert 0 in flat and 29 in flat  # anchor keeps the episode start; current frame present
    assert flat == sorted(flat)  # oldest segment first, absolute indices
    for s in segments:
        assert s["total_frames"] == 30
        assert s["video"][:, 0, 0, 0].tolist() == [float(i) for i in s["frame_indices"]]
        assert s["pool_spatial"] in (1, 2, 4)
    assert segments[-1]["pool_spatial"] == 1  # current tier keeps full resolution


def test_slowfast_sample_ignores_frame_ids_and_indexes_by_position():
    video = _marked_video(12)
    a = _build_slowfast_eval_sample(video, "t", None, _slowfast_bundle())
    b = _build_slowfast_eval_sample(video, "t", [100 + i for i in range(12)], _slowfast_bundle())
    assert [s["frame_indices"] for s in a["video_segments"]] == [
        s["frame_indices"] for s in b["video_segments"]
    ]
    assert a["conversations"] == b["conversations"]


def test_slowfast_rvq_checkpoint_swaps_the_output_sentence():
    video = _marked_video(12)
    flat = _build_slowfast_eval_sample(video, "t", None, _slowfast_bundle(action_method="flat"))
    rvq = _build_slowfast_eval_sample(video, "t", None, _slowfast_bundle(action_method="rvq"))

    flat_prompt = flat["conversations"][0]["value"]
    rvq_prompt = rvq["conversations"][0]["value"]
    assert flat_prompt.endswith("as a trajectory id token. ")
    assert rvq_prompt.endswith("as a sequence of coarse-to-fine trajectory tokens. ")
    assert rvq_prompt == to_rvq_prompt(flat_prompt)
    assert [s["frame_indices"] for s in flat["video_segments"]] == [
        s["frame_indices"] for s in rvq["video_segments"]
    ]


def test_to_rvq_prompt_rejects_the_per_task_templates():
    with pytest.raises(ValueError):
        to_rvq_prompt(TRACKING_PROMPT_TEMPLATE)


# -- engine dispatch --------------------------------------------------------------------


def _capturing_engine(bundle):
    engine = VLNInferenceEngine(bundle, backend="hf")
    captured = {}

    def fake_generate(sample, max_new_tokens=None):
        captured["sample"] = sample
        captured["max_new_tokens"] = max_new_tokens
        return "<traj_1>", 3.0

    engine.generate = fake_generate
    return engine, captured


def test_generate_from_frames_tracking_builds_tracking_sample():
    engine, captured = _capturing_engine(_bundle())
    video = torch.zeros(5, 3, 4, 4)

    text, latency_ms = engine.generate_from_frames(
        video,
        "follow the target",
        predict_horizon=9,
        frame_ids=[20, 21, 22, 23, 24],
        max_new_tokens=6,
        task_type="tracking",
    )

    assert text == "<traj_1>"
    assert latency_ms == 3.0
    assert captured["max_new_tokens"] == 6
    sample = captured["sample"]
    prompt = sample["conversations"][0]["value"]
    assert prompt == TRACKING_PROMPT_TEMPLATE_POOLED.format(task="follow the target")
    assert "next 9 actions" not in prompt
    history, current = sample["video_segments"]
    assert torch.equal(history["video"], video[:-2])
    assert history["frame_indices"] == [20, 21, 22]
    assert history["pool_spatial"] == 2
    assert torch.equal(current["video"], video[-2:])
    assert current["frame_indices"] == [23, 24]
    assert current["pool_spatial"] == 1


def test_generate_from_frames_vlnce_traj_builds_vln_sample():
    engine, captured = _capturing_engine(_bundle())
    video = torch.zeros(5, 3, 4, 4)

    text, _ = engine.generate_from_frames(video, "go left", frame_ids=[0, 1, 2, 3, 4], task_type="vlnce_traj")

    assert text == "<traj_1>"
    assert captured["max_new_tokens"] is None
    prompt = captured["sample"]["conversations"][0]["value"]
    assert prompt == VLN_TRAJ_PROMPT_TEMPLATE_POOLED.format(task="go left")


def test_generate_from_frames_defaults_to_tracking():
    engine, captured = _capturing_engine(_bundle())
    engine.generate_from_frames(torch.zeros(5, 3, 4, 4), "follow")
    assert captured["sample"]["conversations"][0]["value"].startswith(
        "You are a mobile robot performing a person-tracking task."
    )


def test_generate_from_frames_rejects_unknown_task_types():
    engine, _ = _capturing_engine(_bundle())
    with pytest.raises(ValueError, match="task_type"):
        engine.generate_from_frames(torch.zeros(5, 3, 4, 4), "x", task_type="vlnce")


def test_generate_rejects_unknown_backends():
    engine = VLNInferenceEngine(_bundle(), backend="vllm_server")
    with pytest.raises(ValueError, match="backend"):
        engine.generate({"video_segments": [], "conversations": []})


def test_continuous_sample_matches_fm_only_training_prompt():
    from lightnav.inference.samples import build_continuous_traj_sample
    from lightnav.prompts import CONTINUOUS_TRAJ_PROMPT_TEMPLATE

    bundle = _bundle(slowfast_tiers=validate_slowfast_tiers(DEFAULT_SLOWFAST_TIERS_SPAN),
                     action_method="rvq")
    sample = build_continuous_traj_sample(torch.zeros(8, 3, 4, 4), "walk to door", None, bundle)
    expected = CONTINUOUS_TRAJ_PROMPT_TEMPLATE.format(
        task="walk to door", videos=build_video_block(len(sample["video_segments"]))
    )
    assert sample["conversations"] == [
        {"from": "human", "value": expected}, {"from": "gpt", "value": ""}
    ]
    assert "trajectory tokens" not in expected
    assert sample["slowfast_abs_frame_indices"] is True
    assert max(f for segment in sample["video_segments"] for f in segment["frame_indices"]) == 7
