"""Flow expert with layerwise prefix KV, joint attention and time AdaRMSNorm.

The frozen prefix has already received Qwen's native key normalization and
multimodal RoPE. Only the action queries and keys are normalized/rotated here.
Action tokens read every valid prefix token and every action token through one
softmax; future-action loss masks never enter this module.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any

import torch
from torch import Tensor, nn
from torch.nn import functional as F

if TYPE_CHECKING:
    from lightnav.inference.prefix_kv import PrefixKVContext


class _RMSNorm(nn.Module):
    """Qwen-compatible float32 RMS statistics, with optional learned weights."""

    def __init__(self, width: int, eps: float, *, affine: bool = True):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(width)) if affine else None

    def forward(self, x: Tensor) -> Tensor:
        normalized = x.float() * torch.rsqrt(x.float().square().mean(-1, keepdim=True) + self.eps)
        normalized = normalized.to(x.dtype)
        # Expert master weights stay FP32 under autocast. Preserve projection
        # dtype so normalized Q/K do not upcast the entire long prefix cache.
        return normalized if self.weight is None else self.weight.to(x.dtype) * normalized


class TimeAdaptiveRMSNorm(nn.Module):
    """Time-conditioned scale/shift, optionally including a residual gate.

    Zero initialization starts residual branches at the identity. Their gates
    learn on the first update; attention/FFN parameters learn once gates open.
    """

    def __init__(self, width: int, eps: float, *, gated: bool = True):
        super().__init__()
        self.gated = gated
        self.norm = _RMSNorm(width, eps, affine=False)
        self.modulation = nn.Sequential(nn.SiLU(), nn.Linear(width, (3 if gated else 2) * width))
        nn.init.zeros_(self.modulation[-1].weight)
        nn.init.zeros_(self.modulation[-1].bias)

    def forward(self, x: Tensor, time: Tensor) -> tuple[Tensor, Tensor | None]:
        parts = self.modulation(time).chunk(3 if self.gated else 2, dim=-1)
        scale, shift = parts[:2]
        hidden = self.norm(x) * (1 + scale[:, None]) + shift[:, None]
        gate = parts[2][:, None] if self.gated else None
        return hidden, gate


def _rotate_half(x: Tensor) -> Tensor:
    first, second = x.chunk(2, dim=-1)
    return torch.cat((-second, first), dim=-1)


class PiJointAttention(nn.Module):
    """One GQA operation over frozen prefix KV and trainable action KV.

    Residual width and attention projection width are independent. For the
    production LightNav model, a 512-wide residual projects to 32 query heads
    and 8 key/value heads, each of width 128.
    """

    def __init__(self, config: Any, *, num_kv_heads: int, head_dim: int, rms_norm_eps: float):
        super().__init__()
        self.num_heads = config.num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.dropout = config.dropout
        self.q_proj = nn.Linear(config.hidden_size, self.num_heads * head_dim, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, num_kv_heads * head_dim, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, num_kv_heads * head_dim, bias=False)
        self.q_norm = _RMSNorm(head_dim, rms_norm_eps)
        self.k_norm = _RMSNorm(head_dim, rms_norm_eps)
        self.o_proj = nn.Linear(self.num_heads * head_dim, config.hidden_size, bias=False)

    def forward(
        self, hidden: Tensor, prefix_kv: tuple[Tensor, Tensor], action_cos: Tensor,
        action_sin: Tensor, attention_mask: Tensor,
    ) -> Tensor:
        batch, horizon, _ = hidden.shape
        q = self.q_norm(self.q_proj(hidden).view(batch, horizon, self.num_heads, self.head_dim))
        k = self.k_norm(self.k_proj(hidden).view(batch, horizon, self.num_kv_heads, self.head_dim))
        q, k = q.transpose(1, 2), k.transpose(1, 2)
        v = self.v_proj(hidden).view(batch, horizon, self.num_kv_heads, self.head_dim).transpose(1, 2)
        cos = action_cos.detach()[:, None].to(q.dtype)
        sin = action_sin.detach()[:, None].to(q.dtype)
        q, k = q * cos + _rotate_half(q) * sin, k * cos + _rotate_half(k) * sin
        # Prefix K already has native k_norm + mRoPE; never apply either twice.
        # Enforce gradient isolation even for contexts constructed by callers.
        key = torch.cat((prefix_kv[0].detach().to(k.dtype), k), dim=2)
        value = torch.cat((prefix_kv[1].detach().to(v.dtype), v), dim=2)
        attended = F.scaled_dot_product_attention(
            q, key, value, attn_mask=attention_mask, is_causal=False,
            dropout_p=self.dropout if self.training else 0.0,
            enable_gqa=self.num_heads != self.num_kv_heads,
        )
        attended = attended.transpose(1, 2).reshape(batch, horizon, self.num_heads * self.head_dim)
        return self.o_proj(attended)


class _SwiGLU(nn.Module):
    def __init__(self, width: int, intermediate_size: int):
        super().__init__()
        self.gate_proj = nn.Linear(width, intermediate_size, bias=False)
        self.up_proj = nn.Linear(width, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, width, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class _PiActionBlock(nn.Module):
    def __init__(self, config: Any, *, num_kv_heads: int, head_dim: int, rms_norm_eps: float):
        super().__init__()
        self.attention_norm = TimeAdaptiveRMSNorm(config.hidden_size, rms_norm_eps)
        self.ffn_norm = TimeAdaptiveRMSNorm(config.hidden_size, rms_norm_eps)
        self.attention = PiJointAttention(
            config, num_kv_heads=num_kv_heads, head_dim=head_dim, rms_norm_eps=rms_norm_eps,
        )
        self.ffn = _SwiGLU(config.hidden_size, config.intermediate_size)
        self.dropout = nn.Dropout(config.dropout)

    def forward(
        self, actions: Tensor, time: Tensor, prefix_kv: tuple[Tensor, Tensor],
        action_cos: Tensor, action_sin: Tensor, attention_mask: Tensor,
    ) -> Tensor:
        normalized, gate = self.attention_norm(actions, time)
        actions = actions + gate * self.dropout(self.attention(
            normalized, prefix_kv, action_cos, action_sin, attention_mask,
        ))
        normalized, gate = self.ffn_norm(actions, time)
        return actions + gate * self.dropout(self.ffn(normalized))


class PiFlowActionExpert(nn.Module):
    """Predict normalized SE(2) flow from noisy actions and frozen prefix KV."""

    def __init__(self, config: Any, *, num_kv_heads: int, head_dim: int, rms_norm_eps: float):
        super().__init__()
        for name, value in (
            ("num_kv_heads", num_kv_heads), ("head_dim", head_dim),
            ("num_heads", config.num_heads), ("horizon", config.horizon),
            ("num_layers", config.num_layers), ("hidden_size", config.hidden_size),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if config.num_heads % num_kv_heads:
            raise ValueError("num_heads must be divisible by num_kv_heads")
        if head_dim % 2 or config.hidden_size % 2:
            raise ValueError("head_dim and hidden_size must be even")
        if not math.isfinite(rms_norm_eps) or rms_norm_eps <= 0:
            raise ValueError("rms_norm_eps must be finite and positive")
        self.config = config
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        width = config.hidden_size
        self.action_in = nn.Linear(3, width)
        self.step_embedding = nn.Parameter(torch.randn(config.horizon, width) * 0.02)
        self.time_mlp = nn.Sequential(nn.Linear(width, width), nn.SiLU(), nn.Linear(width, width))
        self.blocks = nn.ModuleList([
            _PiActionBlock(
                config, num_kv_heads=num_kv_heads, head_dim=head_dim, rms_norm_eps=rms_norm_eps,
            ) for _ in range(config.num_layers)
        ])
        self.out_norm = TimeAdaptiveRMSNorm(width, rms_norm_eps, gated=False)
        self.action_out = nn.Linear(width, 3)
        self.register_buffer(
            "time_frequencies",
            torch.exp(-math.log(10000) * torch.arange(width // 2).float() / max(width // 2 - 1, 1)),
            persistent=False,
        )

    def _validate_context(self, context: PrefixKVContext) -> tuple[int, int]:
        if context.valid.ndim != 2 or context.valid.dtype != torch.bool:
            raise ValueError("context.valid must be bool [B,N]")
        batch, length = context.valid.shape
        if batch < 1 or length < 1:
            raise ValueError("context must contain at least one sample and prefix token")
        if len(context.layer_kv) != self.config.num_layers:
            raise ValueError("context must provide one KV pair per action layer")
        if len(context.layer_indices) != self.config.num_layers:
            raise ValueError("context layer_indices must match the number of action layers")
        expected_layers = getattr(self.config, "prefix_layers", ())
        if expected_layers and tuple(context.layer_indices) != tuple(expected_layers):
            raise ValueError("context layer_indices must match the configured prefix_layers in order")
        kv_shape = (batch, self.num_kv_heads, length, self.head_dim)
        position_shape = (batch, self.config.horizon, self.head_dim)
        for kv in context.layer_kv:
            if len(kv) != 2 or any(x.shape != kv_shape for x in kv):
                raise ValueError(f"prefix KV must have shape {kv_shape}")
        if any(x.shape != position_shape for x in (context.action_cos, context.action_sin)):
            raise ValueError(f"action cosine/sine must have shape {position_shape}")
        if any(
            x.device != context.valid.device
            for x in (context.action_cos, context.action_sin, *(t for kv in context.layer_kv for t in kv))
        ):
            raise ValueError("all context tensors must share one device")
        return batch, length

    def forward(
        self, noisy_actions: Tensor, flow_time: Tensor, context: PrefixKVContext,
    ) -> Tensor:
        batch, _ = self._validate_context(context)
        if noisy_actions.shape != (batch, self.config.horizon, 3):
            raise ValueError("noisy_actions must be [B,horizon,3]")
        if flow_time.shape != (batch,):
            raise ValueError("flow_time must be [B]")
        device = self.action_in.weight.device
        if any(t.device != device for t in (noisy_actions, flow_time, context.valid)):
            raise ValueError("noisy_actions, flow_time and context must be on the expert device")
        dtype = self.action_in.weight.dtype
        actions = self.action_in(noisy_actions.to(dtype)) + self.step_embedding[None]
        phase = flow_time.float()[:, None] * self.time_frequencies.float()[None] * 1000
        time = self.time_mlp(torch.cat((phase.sin(), phase.cos()), dim=-1).to(dtype))
        # This mask is independent of GT validity: all future actions are visible.
        action_valid = torch.ones(batch, self.config.horizon, device=device, dtype=torch.bool)
        attention_mask = torch.cat((context.valid, action_valid), dim=1)[:, None, None]
        for block, kv in zip(self.blocks, context.layer_kv, strict=True):
            actions = block(actions, time, kv, context.action_cos, context.action_sin, attention_mask)
        actions, _ = self.out_norm(actions, time)
        return self.action_out(actions)

    @torch.no_grad()
    def sample(
        self, context: PrefixKVContext, *, num_steps: int | None = None,
        noise: Tensor | None = None, generator: torch.Generator | None = None,
    ) -> Tensor:
        """Euler integration with one reusable prefix, from noise at 0 to GT at 1."""
        if self.training:
            raise RuntimeError("call eval() before sampling actions")
        steps = self.config.num_flow_steps if num_steps is None else num_steps
        if isinstance(steps, bool) or not isinstance(steps, int) or steps < 1:
            raise ValueError("num_steps must be a positive integer")
        batch, _ = self._validate_context(context)
        shape = (batch, self.config.horizon, 3)
        device = context.valid.device
        if noise is None:
            x = torch.randn(shape, device=device, dtype=torch.float32, generator=generator)
        else:
            if noise.shape != shape or not torch.isfinite(noise).all():
                raise ValueError(f"noise must be finite with shape {shape}")
            x = noise.detach().to(device=device, dtype=torch.float32)
        for k in range(steps):
            tau = torch.full((batch,), k / steps, device=device, dtype=torch.float32)
            x = x + self(x, tau, context).float() / steps
        return x
