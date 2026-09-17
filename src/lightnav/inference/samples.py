"""Sample builders: (video tensor, instruction, frame ids, bundle) -> processor sample dict.

The sample dict shape and the prompt strings are trained-in contracts; see
``lightnav.prompts`` for the templates and ``lightnav.data_processor``
for the consumer.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from lightnav.prompts import (
    CONTINUOUS_TRAJ_PROMPT_TEMPLATE,
    TRACKING_PROMPT_TEMPLATE,
    TRACKING_PROMPT_TEMPLATE_POOLED,
    UNIFIED_TRAJ_PROMPT_TEMPLATE,
    VLN_TRAJ_PROMPT_TEMPLATE,
    VLN_TRAJ_PROMPT_TEMPLATE_POOLED,
    build_video_block,
    to_rvq_prompt,
)
from lightnav.slowfast import slowfast_video_segments

if TYPE_CHECKING:
    from lightnav.inference.model import ModelBundle


def _build_slowfast_eval_sample(
    video_tensor: torch.Tensor,
    instruction: str,
    frame_ids: list[int] | None,
    bundle: "ModelBundle",
    task_prompt: str = UNIFIED_TRAJ_PROMPT_TEMPLATE,
) -> dict:
    """SlowFast sample, mirroring the training-time slowfast sample layout.

    ``video_tensor`` must be the FULL episode buffer (frame i at index i); the
    NavigationPolicy keeps all observed frames when slowfast is enabled so the
    span tier can reach frame 0. Uses the same ``slowfast_video_segments`` +
    unified ``{videos}`` prompt + absolute frame_indices as training.

    ``frame_ids`` is accepted for signature symmetry with the windowed builders
    but unused: slowfast indexes the episode buffer by absolute position.
    """
    template = task_prompt
    # rvq checkpoints train with the coarse-to-fine output sentence (to_rvq_prompt);
    # apply the same swap so the inference prompt equals the training prompt. flat
    # runs keep the base unified wording untouched.
    if getattr(bundle, "action_method", "flat") == "rvq":
        template = to_rvq_prompt(template)
    total = int(video_tensor.shape[0])
    segments = slowfast_video_segments(video_tensor, total - 1, total, bundle.slowfast_tiers)
    user_message = template.format(
        task=instruction,
        videos=build_video_block(len(segments)),
        horizon=getattr(bundle, "predict_horizon", 1),
    )
    return {
        "video_segments": segments,
        "conversations": [
            {"from": "human", "value": user_message},
            {"from": "gpt", "value": "placeholder"},
        ],
        "video_fps": bundle.video_fps,
        "slowfast_abs_frame_indices": True,
        "_allow_vit_cache": True,
        "_skip_normalize": True,
    }


def _build_windowed_sample(
    video_tensor: torch.Tensor,
    instruction: str,
    frame_ids: list[int] | None,
    bundle: "ModelBundle",
    template: str,
    template_pooled: str,
) -> dict:
    """Shared body of the tracking / vln_traj builders (they differ only in templates).

    The POOLED variant (history segment pooled, last two frames unpooled) is
    selected when the bundle has ``pool_enable and pool_spatial > 1`` and the
    clip has more than two frames, mirroring the training-time prompt selection.
    """
    total_frames = int(video_tensor.shape[0])
    ids = list(frame_ids) if frame_ids is not None else list(range(total_frames))
    pool_enable = bool(getattr(bundle, "pool_enable", False))
    pool_spatial = int(bundle.pool_spatial)
    pool_mode = bundle.pool_mode

    if pool_enable and pool_spatial > 1 and total_frames > 2:
        user_message = template_pooled.format(task=instruction)
        video_segments = [
            {
                "video": video_tensor[:-2],
                "frame_indices": ids[:-2],
                "total_frames": total_frames,
                "pool_spatial": pool_spatial,
                "pool_mode": pool_mode,
            },
            {
                "video": video_tensor[-2:],
                "frame_indices": ids[-2:],
                "total_frames": total_frames,
                "pool_spatial": 1,
                "pool_mode": pool_mode,
            },
        ]
    else:
        user_message = template.format(task=instruction)
        video_segments = [
            {
                "video": video_tensor,
                "frame_indices": ids,
                "total_frames": total_frames,
                "pool_spatial": 1,
                "pool_mode": pool_mode,
            }
        ]

    return {
        "video_segments": video_segments,
        "conversations": [
            {"from": "human", "value": user_message},
            {"from": "gpt", "value": "placeholder"},
        ],
        "video_fps": bundle.video_fps,
        "_allow_vit_cache": True,
        "_skip_normalize": True,
    }


def build_vln_traj_sample(
    video_tensor: torch.Tensor,
    instruction: str,
    frame_ids: list[int] | None,
    bundle: "ModelBundle",
) -> dict:
    """Sample for the ``vlnce_traj`` task (single trajectory token; VLN_TRAJ templates)."""
    if getattr(bundle, "slowfast_tiers", None):
        return _build_slowfast_eval_sample(video_tensor, instruction, frame_ids, bundle)
    return _build_windowed_sample(
        video_tensor,
        instruction,
        frame_ids,
        bundle,
        VLN_TRAJ_PROMPT_TEMPLATE,
        VLN_TRAJ_PROMPT_TEMPLATE_POOLED,
    )


def build_continuous_traj_sample(
    video_tensor: torch.Tensor,
    instruction: str,
    frame_ids: list[int] | None,
    bundle: "ModelBundle",
) -> dict:
    """FM-only prompt and SlowFast inputs, matching the training loader exactly."""
    if not getattr(bundle, "slowfast_tiers", None):
        raise ValueError("FM-only evaluation requires the training SlowFast configuration")
    total = int(video_tensor.shape[0])
    segments = slowfast_video_segments(video_tensor, total - 1, total, bundle.slowfast_tiers)
    text = CONTINUOUS_TRAJ_PROMPT_TEMPLATE.format(
        task=instruction, videos=build_video_block(len(segments))
    )
    return {
        "video_segments": segments,
        "conversations": [
            {"from": "human", "value": text},
            {"from": "gpt", "value": ""},
        ],
        "video_fps": bundle.video_fps,
        "slowfast_abs_frame_indices": True,
        "_skip_normalize": True,
    }


def build_tracking_sample(
    video_tensor: torch.Tensor,
    instruction: str,
    frame_ids: list[int] | None,
    bundle: "ModelBundle",
) -> dict:
    """Sample for the ``tracking`` task (single trajectory token; TRACKING templates)."""
    if getattr(bundle, "slowfast_tiers", None):
        return _build_slowfast_eval_sample(video_tensor, instruction, frame_ids, bundle)
    return _build_windowed_sample(
        video_tensor,
        instruction,
        frame_ids,
        bundle,
        TRACKING_PROMPT_TEMPLATE,
        TRACKING_PROMPT_TEMPLATE_POOLED,
    )
