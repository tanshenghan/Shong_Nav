"""Frozen Qwen3-VL prefix keys/values for the layer-wise FM action expert.

Only the selected attention layers are retained. The prefix still runs through
Qwen's unmodified causal attention, vision tower and DeepStack connections;
there is no generated answer, LM-head projection or full decoder KV cache.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor, nn

from lightnav.inference.stop_head import last_prefix_features


@dataclass(frozen=True)
class PrefixKVContext:
    """Keys already include native Qwen k-norm and multimodal RoPE.

    ``layer_kv`` entries are [B,kv_heads,N,head_dim]; ``valid`` retains the
    original [B,N] padding layout. Action cos/sin are [B,H,head_dim] and apply
    only to newly projected action Q/K. Layer indices are one-based.
    """

    layer_kv: tuple[tuple[Tensor, Tensor], ...]
    valid: Tensor
    action_cos: Tensor
    action_sin: Tensor
    layer_indices: tuple[int, ...]


@torch.no_grad()
def encode_prefix_kv(
    vlm: nn.Module, inputs: dict[str, Any], layer_indices: tuple[int, ...], horizon: int,
    *, return_summary: bool = False,
) -> PrefixKVContext | tuple[PrefixKVContext, Tensor]:
    """Encode a prompt-only prefix and extend its native RoPE for action tokens.

    The caller owns multimodal pooling setup and must exclude all answer tokens.
    No train/eval mode is restored: this API is exclusively for a frozen VLM.
    Hooks are local to this synchronous forward and are removed even on errors.
    Selected K/V projections are recomputed from the native pre-attention hidden
    states so that nonselected layers never allocate persistent cache tensors.
    ``return_summary`` copies the final valid token feature from that same call;
    no full hidden-state history or second backbone forward is needed.
    """
    from transformers.models.qwen3_vl.modeling_qwen3_vl import apply_rotary_pos_emb

    backbone = getattr(vlm, "model", None)
    language = getattr(backbone, "language_model", None)
    if language is None or not hasattr(language, "rotary_emb"):
        raise TypeError("prefix KV requires an HF Qwen3-VL conditional-generation model")
    if (not isinstance(horizon, int) or isinstance(horizon, bool) or horizon < 1):
        raise ValueError("horizon must be a positive integer")
    if (not layer_indices or any(not isinstance(i, int) or isinstance(i, bool)
                                 or not 1 <= i <= len(language.layers) for i in layer_indices)
            or tuple(sorted(set(layer_indices))) != tuple(layer_indices)):
        raise ValueError("layer_indices must be unique increasing one-based VLM layer indices")

    allowed = {
        "input_ids", "attention_mask", "position_ids", "mm_token_type_ids",
        "pixel_values", "pixel_values_videos", "image_grid_thw", "video_grid_thw",
    }
    if inputs.keys() - allowed:
        raise ValueError("prefix KV accepts only prompt inputs, without labels or past_key_values")
    ids = inputs.get("input_ids")
    if (not isinstance(ids, Tensor) or ids.dtype != torch.long or ids.ndim != 2
            or not ids.shape[0] or not ids.shape[1]):
        raise ValueError("input_ids must be nonempty int64 [B,N]")
    valid = inputs.get("attention_mask", torch.ones_like(ids, dtype=torch.bool))
    if (not isinstance(valid, Tensor) or valid.shape != ids.shape or valid.device != ids.device
            or not ((valid == 0) | (valid == 1)).all() or not valid.bool().any(dim=1).all()):
        raise ValueError("attention_mask must be binary [B,N] with a valid token in every row")
    valid = valid.bool()
    kwargs = {**inputs, "attention_mask": valid}
    has_multimodal = any(inputs.get(k) is not None for k in ("image_grid_thw", "video_grid_thw"))
    if kwargs.get("position_ids") is None and not has_multimodal:
        # HF's text-only fallback uses absolute padded indices. Explicit compact
        # positions make the suffix independent of left/right/batch padding.
        kwargs["position_ids"] = (valid.long().cumsum(-1) - 1).masked_fill(~valid, 0)

    vlm.eval()
    captured: dict[int, tuple[Tensor, Tensor]] = {}
    positions: list[Tensor] = []
    hooks = []

    def capture_positions(_module, args, hook_kwargs):
        position_ids = hook_kwargs.get("position_ids", args[1] if len(args) > 1 else None)
        if position_ids is None:
            raise RuntimeError("Qwen rotary embedding did not receive position_ids")
        if positions:
            raise RuntimeError("expected exactly one prefix rotary embedding call")
        positions.append(position_ids.detach())

    def capture_layer(index):
        def capture(attention, args, hook_kwargs):
            if index in captured:
                raise RuntimeError("selected prefix attention layer was evaluated more than once")
            hidden = hook_kwargs.get("hidden_states", args[0] if args else None)
            rotary = hook_kwargs.get("position_embeddings", args[1] if len(args) > 1 else None)
            if hidden is None or rotary is None:
                raise RuntimeError("selected Qwen attention is missing hidden states or rotary embeddings")
            shape = (*hidden.shape[:-1], -1, attention.head_dim)
            key = attention.k_norm(attention.k_proj(hidden).view(shape)).transpose(1, 2)
            value = attention.v_proj(hidden).view(shape).transpose(1, 2)
            # Using the native routine preserves Qwen's rotate_half convention.
            # Q is irrelevant here; pass K to avoid computing another projection.
            _, key = apply_rotary_pos_emb(key, key, *rotary)
            mask = valid[:, None, :, None]
            captured[index] = (
                torch.where(mask, key, 0.0).detach(),
                torch.where(mask, value, 0.0).detach(),
            )
        return capture

    try:
        hooks.append(language.rotary_emb.register_forward_pre_hook(capture_positions, with_kwargs=True))
        for index in layer_indices:
            hooks.append(language.layers[index - 1].self_attn.register_forward_pre_hook(
                capture_layer(index), with_kwargs=True,
            ))
        output = backbone(
            **kwargs, use_cache=False, return_dict=True,
            output_hidden_states=False, output_attentions=False,
        )
    finally:
        for hook in hooks:
            hook.remove()

    if len(captured) != len(layer_indices) or len(positions) != 1:
        raise RuntimeError("Qwen forward did not visit every selected layer and rotary embedding")
    prefix_positions = positions[0]
    if prefix_positions.ndim == 2:
        prefix_positions = prefix_positions[None].expand(3, -1, -1)
    if prefix_positions.shape != (3, *ids.shape):
        raise RuntimeError("expected native Qwen multimodal positions [3,B,N]")
    # Qwen continues after the largest effective temporal/height/width index,
    # rather than after the padded token count or the count of visual patches.
    end = prefix_positions.masked_fill(~valid[None], -1).amax(dim=(0, 2)) + 1
    action_positions = end[:, None] + torch.arange(horizon, device=ids.device)[None]
    key = captured[layer_indices[0]][0]
    action_cos, action_sin = language.rotary_emb(
        key.new_empty((ids.shape[0], horizon, key.shape[-1])), action_positions,
    )
    context = PrefixKVContext(
        layer_kv=tuple(captured[i] for i in layer_indices), valid=valid.detach(),
        action_cos=action_cos.detach(), action_sin=action_sin.detach(),
        layer_indices=tuple(layer_indices),
    )
    if return_summary:
        return context, last_prefix_features(output.last_hidden_state, valid)
    return context
