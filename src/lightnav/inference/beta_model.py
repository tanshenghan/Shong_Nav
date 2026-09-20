"""Beta LightNav: token CE trains the VLM; flow matching trains an action expert.

The HF Qwen3-VL backbone and its LM head retain their existing token vocabulary.
Only input-prefix hidden states condition the expert, through a stop-gradient
boundary. Actions are local SE(2) increments, not robot velocities: the default
horizon is 30 recorded action steps, independently of the flow integration steps.

This module is a model API, not a dataset builder or a vLLM serving backend.
See docs/BETA_MODEL.md for batch construction, optimizer and checkpoint contracts.
"""

from __future__ import annotations

import json
import math
from collections.abc import Iterator, Mapping
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.nn.utils.rnn import pad_sequence

from lightnav.vln_utils import parse_rvq_action_tokens

if TYPE_CHECKING:
    from lightnav.inference.config import InferenceConfig
    from lightnav.inference.model import ModelBundle
    from lightnav.tracking import TrackingAgent


@dataclass(frozen=True)
class BetaModelConfig:
    """Expert architecture; action_scale must be fixed from training data only.

    Scaling is zero-preserving: normalized = local_SE2_increment / action_scale.
    Supply positive floors for dimensions with zero variance (e.g. lateral motion).
    No dataset fitting or clipping is performed by the model.
    """

    action_scale: tuple[float, float, float]
    horizon: int = 30
    hidden_size: int = 512
    num_layers: int = 6
    num_heads: int = 8
    intermediate_size: int = 2048
    dropout: float = 0.0
    num_flow_steps: int = 10
    training_mode: str = "ce_fm"
    flow_time_distribution: str = "uniform"
    flow_time_alpha: float = 1.0
    flow_time_beta: float = 1.0

    def __post_init__(self):
        if len(self.action_scale) != 3 or any(
            not math.isfinite(s) or s <= 0 for s in self.action_scale
        ):
            raise ValueError("action_scale must contain three finite positive values")
        for name in ("horizon", "hidden_size", "num_layers", "num_heads",
                     "intermediate_size", "num_flow_steps"):
            if not isinstance(getattr(self, name), int) or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.hidden_size % self.num_heads or self.hidden_size % 2:
            raise ValueError("hidden_size must be even and divisible by num_heads")
        if not 0 <= self.dropout < 1:
            raise ValueError("dropout must be in [0, 1)")
        if self.training_mode not in {"ce_fm", "fm_only"}:
            raise ValueError("training_mode must be ce_fm or fm_only")
        if self.flow_time_distribution not in {"uniform", "beta"}:
            raise ValueError("flow_time_distribution must be uniform or beta")
        for name in ("flow_time_alpha", "flow_time_beta"):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be finite and positive")


@dataclass
class BetaTrainBatch:
    """Batched tensors; answer_start is an index in the PADDED token sequence.

    model_inputs follows LightNav's processed HF inputs, with input_ids [B,L].
    labels [B,L] already incorporates each token channel's validity (-100 means
    ignore). answer_start [B] is the first answer position BEFORE loss masking;
    it must never be inferred from the first remaining supervised label.
    fm_clean_se2_diff [B,H,3] is in metres/radians; fm_valid_mask [B,H] is bool.
    """

    model_inputs: Mapping[str, Any]
    labels: Tensor
    answer_start: Tensor
    fm_clean_se2_diff: Tensor
    fm_valid_mask: Tensor


@dataclass
class BetaContext:
    memory: Tensor  # [B,N,expert_width], projected AFTER detaching VLM features
    valid: Tensor  # [B,N], True = real context; opposite of MHA key_padding_mask


@dataclass
class BetaTrainOutput:
    loss: Tensor
    ce_loss: Tensor
    fm_loss: Tensor
    ce_token_count: int
    fm_step_count: int
    context: BetaContext | None
    velocity: Tensor | None


@dataclass
class BetaPrediction:
    token_ids: Tensor
    texts: list[str]
    stop: Tensor  # [B] bool, determined by the existing RVQ decoder
    explicit_stop: Tensor  # [B] bool, canonical RVQ STOP tuple (not merely near-zero)
    decode_failed: Tensor  # [B] bool, malformed/incomplete RVQ token sequence
    rvq_waypoints: Tensor  # [B,rvq_horizon,3], comparison branch
    se2_deltas: Tensor  # [B,H,3], per-step local increments in metres/radians
    waypoints: Tensor  # [B,H,3], composed relative to the current body


class ActionNormalizer(nn.Module):
    def __init__(self, scale: tuple[float, float, float]):
        super().__init__()
        self.register_buffer("scale", torch.tensor(scale, dtype=torch.float32))

    def normalize(self, actions: Tensor) -> Tensor:
        return actions.float() / self.scale.float()

    def denormalize(self, actions: Tensor) -> Tensor:
        return actions.float() * self.scale.float()


class _ActionBlock(nn.Module):
    """Bidirectional action attention, input-context attention, and time FiLM."""

    def __init__(self, config: BetaModelConfig):
        super().__init__()
        d = config.hidden_size
        self.norms = nn.ModuleList([nn.LayerNorm(d, elementwise_affine=False) for _ in range(3)])
        self.time_modulation = nn.Sequential(nn.SiLU(), nn.Linear(d, 6 * d))
        self.self_attention = nn.MultiheadAttention(
            d, config.num_heads, dropout=config.dropout, batch_first=True
        )
        self.cross_attention = nn.MultiheadAttention(
            d, config.num_heads, dropout=config.dropout, batch_first=True
        )
        self.ffn = nn.Sequential(
            nn.Linear(d, config.intermediate_size), nn.GELU(), nn.Dropout(config.dropout),
            nn.Linear(config.intermediate_size, d),
        )
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, actions: Tensor, time: Tensor, context: BetaContext) -> Tensor:
        modulation = self.time_modulation(time).chunk(6, dim=-1)

        def modulate(x, index):
            shift, scale = modulation[2 * index:2 * index + 2]
            return self.norms[index](x) * (1 + scale[:, None]) + shift[:, None]

        x = modulate(actions, 0)
        actions = actions + self.dropout(self.self_attention(x, x, x, need_weights=False)[0])
        x = modulate(actions, 1)
        actions = actions + self.dropout(self.cross_attention(
            x, context.memory, context.memory,
            key_padding_mask=~context.valid, need_weights=False,
        )[0])
        return actions + self.dropout(self.ffn(modulate(actions, 2)))


class FlowActionExpert(nn.Module):
    def __init__(self, context_size: int, config: BetaModelConfig):
        super().__init__()
        self.config = config
        d = config.hidden_size
        self.context_proj = nn.Linear(context_size, d)
        self.action_in = nn.Linear(3, d)
        self.step_embedding = nn.Parameter(torch.randn(config.horizon, d) * 0.02)
        self.time_mlp = nn.Sequential(nn.Linear(d, d), nn.SiLU(), nn.Linear(d, d))
        self.blocks = nn.ModuleList([_ActionBlock(config) for _ in range(config.num_layers)])
        self.out_norm = nn.LayerNorm(d)
        self.action_out = nn.Linear(d, 3)
        self.register_buffer(
            "time_frequencies",
            torch.exp(-math.log(10000) * torch.arange(d // 2).float() / max(d // 2 - 1, 1)),
            persistent=False,
        )

    def prepare_context(self, hidden: Tensor, valid: Tensor) -> BetaContext:
        if hidden.ndim != 3 or valid.shape != hidden.shape[:2] or valid.dtype != torch.bool:
            raise ValueError("context requires hidden [B,N,D] and bool valid [B,N]")
        if not valid.any(dim=1).all():
            raise ValueError("each sample needs at least one valid context token")
        # Detach BEFORE projection: the adapter must receive the FM gradient.
        hidden = hidden.detach().to(dtype=self.context_proj.weight.dtype)
        hidden = torch.where(valid[..., None], hidden, 0.0)
        return BetaContext(self.context_proj(hidden), valid)

    def forward(
        self, noisy_actions: Tensor, flow_time: Tensor, context: BetaContext,
    ) -> Tensor:
        b = context.memory.shape[0]
        if noisy_actions.shape != (b, self.config.horizon, 3):
            raise ValueError("noisy_actions must be [B,horizon,3]")
        if flow_time.shape != (b,):
            raise ValueError("flow_time must be [B]")
        dtype = self.action_in.weight.dtype
        actions = self.action_in(noisy_actions.to(dtype)) + self.step_embedding[None]
        phase = flow_time.float()[:, None] * self.time_frequencies.float()[None] * 1000
        time = self.time_mlp(torch.cat([phase.sin(), phase.cos()], dim=-1).to(dtype))
        for block in self.blocks:
            actions = block(actions, time, context)
        return self.action_out(self.out_norm(actions))

    @torch.no_grad()
    def sample(
        self, context: BetaContext, *, num_steps: int | None = None,
        noise: Tensor | None = None, generator: torch.Generator | None = None,
    ) -> Tensor:
        """Euler integration, noise at tau=0 -> normalized clean actions at tau=1."""
        if self.training:
            raise RuntimeError("call eval() before sampling actions")
        steps = self.config.num_flow_steps if num_steps is None else num_steps
        if not isinstance(steps, int) or steps < 1:
            raise ValueError("num_steps must be a positive integer")
        shape = (context.memory.shape[0], self.config.horizon, 3)
        device = context.memory.device
        x = _noise(shape, device, noise, generator)
        for k in range(steps):
            tau = torch.full((shape[0],), k / steps, device=device, dtype=torch.float32)
            x = x + self(x, tau, context).float() / steps
        return x


def _noise(shape, device, noise, generator):
    if noise is None:
        return torch.randn(shape, device=device, dtype=torch.float32, generator=generator)
    if noise.shape != shape or not torch.isfinite(noise).all():
        raise ValueError(f"noise must be finite with shape {shape}")
    return noise.detach().to(device=device, dtype=torch.float32)


def compose_se2_deltas(deltas: Tensor) -> Tensor:
    """Compose [B,H,3] local forward/left/CCW increments into current-body waypoints."""
    if deltas.ndim != 3 or deltas.shape[-1] != 3:
        raise ValueError("deltas must be [B,H,3]")
    deltas = deltas.float()
    yaw = deltas[..., 2].cumsum(dim=1)
    previous_yaw = F.pad(yaw[:, :-1], (1, 0))
    c, s = previous_yaw.cos(), previous_yaw.sin()
    x, y = deltas[..., 0], deltas[..., 1]
    return torch.stack([
        (c * x - s * y).cumsum(dim=1),
        (s * x + c * y).cumsum(dim=1),
        torch.atan2(yaw.sin(), yaw.cos()),
    ], dim=-1)


class BetaModel(nn.Module):
    """HF LightNav wrapper with a gradient-isolated continuous action expert.

    Use ``model(batch)`` for training (including DDP), ``encode_context`` and
    ``sample_actions`` for continuous-only inference, or ``predict`` to retain
    the existing token branch and its RVQ STOP decision.
    """

    def __init__(self, vlm: nn.Module, config: BetaModelConfig):
        super().__init__()
        if not hasattr(vlm, "model") or not hasattr(vlm, "lm_head"):
            raise TypeError("BetaModel requires the full HF VLM, not a vLLM/ViT-only bundle")
        self.vlm = vlm
        self.config = config
        hidden_size = vlm.config.text_config.hidden_size
        self.expert = FlowActionExpert(hidden_size, config)
        self.normalizer = ActionNormalizer(config.action_scale)
        device = vlm.get_input_embeddings().weight.device
        # Keep newly initialized expert weights FP32; mixed precision is optional.
        self.expert.to(device=device)
        self.normalizer.to(device=device)
        if config.training_mode == "fm_only":
            self.vlm.requires_grad_(False)
        self.train(vlm.training)

    def train(self, mode: bool = True):
        super().train(mode)
        if self.config.training_mode == "fm_only":
            self.vlm.eval()
        return self

    def optimizer_parameters(self) -> dict[str, list[nn.Parameter]]:
        """Disjoint groups; use separate optimizers/clipping for the two losses."""
        return {
            "vlm": [p for p in self.vlm.parameters() if p.requires_grad],
            "fm": [p for p in self.expert.parameters() if p.requires_grad],
        }

    def _sample_flow_time(self, batch_size: int, device: torch.device) -> Tensor:
        """One time per action chunk; small tau means more noise in our convention.

        Uses the rank-local Torch RNG, captured by the existing resume checkpoints.
        Explicit forward(flow_time=...) bypasses this sampler.
        """
        if self.config.flow_time_distribution == "uniform":
            return torch.rand(batch_size, device=device, dtype=torch.float32)
        alpha = torch.tensor(self.config.flow_time_alpha, device=device, dtype=torch.float32)
        beta = torch.tensor(self.config.flow_time_beta, device=device, dtype=torch.float32)
        return torch.distributions.Beta(alpha, beta).sample((batch_size,))

    @contextmanager
    def _hf_inputs(self, source: Mapping[str, Any]) -> Iterator[dict[str, Tensor]]:
        """Reuse LightNav pooling; let pinned HF compute its native multimodal RoPE.

        Accept batched token tensors; pixels and grids concatenate across samples
        in token order (not torch.stack). Never trim using the CE loss mask.
        """
        allowed = {
            "input_ids", "attention_mask", "pixel_values", "pixel_values_videos",
            "image_grid_thw", "video_grid_thw",
        }
        metadata = {
            "labels", "position_ids", "mm_token_type_ids", "image_mask", "video_mask",
            "_original_video_grid_thw", "_post_vit_pool_factors", "_post_vit_pool_spatial",
            "motion_sample_mask", "episode_index", "frame_index", "raw_action_text",
            "dataset_name",
        }
        unknown = source.keys() - allowed - metadata
        if unknown:
            raise ValueError(f"unsupported HF input keys: {sorted(unknown)}")
        weight = self.vlm.get_input_embeddings().weight
        inputs = {
            k: v.to(device=weight.device, dtype=weight.dtype if v.is_floating_point() else v.dtype)
            for k, v in source.items() if k in allowed and v is not None
        }
        ids = inputs.get("input_ids")
        if ids is None or ids.ndim != 2 or ids.dtype != torch.long:
            raise ValueError("input_ids must be batched int64 [B,L]")
        if ids.shape[0] == 0 or ids.shape[1] == 0:
            raise ValueError("input_ids must be nonempty")
        attention = inputs.get("attention_mask", torch.ones_like(ids, dtype=torch.bool))
        if attention.shape != ids.shape or not ((attention == 0) | (attention == 1)).all():
            raise ValueError("attention_mask must be a binary [B,L] padding mask")
        inputs["attention_mask"] = attention.bool()
        if not inputs["attention_mask"].any(dim=1).all():
            raise ValueError("each sample needs at least one input token")
        # DataProcessor drops this field, but native Qwen3-VL needs it for mRoPE.
        mm_types = torch.zeros_like(ids)
        mm_types[ids == self.vlm.config.image_token_id] = 1
        mm_types[ids == self.vlm.config.video_token_id] = 2
        inputs["mm_token_type_ids"] = mm_types

        target = getattr(self.vlm, "_post_vit_target", None)
        original_grid = source.get("_original_video_grid_thw")
        factors = source.get("_post_vit_pool_factors")
        if (original_grid is not None or factors is not None) and target is None:
            raise ValueError("post-ViT metadata requires load_hf_model's pooling patch")
        previous = None
        if target is not None:
            previous = (getattr(target, "_post_vit_original_grid", None),
                        getattr(target, "_post_vit_pool_factors", None))
            target._post_vit_original_grid = (
                original_grid.to(weight.device) if original_grid is not None else None
            )
            target._post_vit_pool_factors = factors
        try:
            yield inputs
        finally:
            if target is not None:
                target._post_vit_original_grid, target._post_vit_pool_factors = previous

    def _context(self, hidden: Tensor, prefix_valid: Tensor) -> BetaContext:
        # Pack away answer/padding positions before projection and cross-attention.
        rows = [h[valid] for h, valid in zip(hidden, prefix_valid, strict=True)]
        packed = pad_sequence(rows, batch_first=True)
        lengths = prefix_valid.sum(dim=1)
        valid = torch.arange(packed.shape[1], device=hidden.device)[None] < lengths[:, None]
        return self.expert.prepare_context(packed, valid)

    def forward(
        self, batch: BetaTrainBatch, *, lambda_fm: float = 1.0,
        noise: Tensor | None = None, flow_time: Tensor | None = None,
    ) -> BetaTrainOutput:
        if not math.isfinite(lambda_fm) or lambda_fm < 0:
            raise ValueError("lambda_fm must be finite and nonnegative")
        with self._hf_inputs(batch.model_inputs) as inputs:
            ids, attention = inputs["input_ids"], inputs["attention_mask"]
            b, length = ids.shape
            labels = batch.labels.to(ids.device)
            start = batch.answer_start.to(ids.device)
            if labels.shape != ids.shape or labels.dtype != torch.long:
                raise ValueError("labels must be int64 [B,L]")
            if start.shape != (b,) or start.dtype != torch.long or not ((start > 0) & (start <= length)).all():
                raise ValueError("answer_start must be int64 [B] in (0,L]")
            prefix = (torch.arange(length, device=ids.device)[None] < start[:, None]) & attention
            if not prefix.any(dim=1).all():
                raise ValueError("answer_start leaves an empty input prefix")
            if ((labels != -100) & (prefix | ~attention)).any():
                raise ValueError("CE labels must ignore all input-prefix and padding positions")
            if self.config.training_mode == "fm_only" and (
                (labels != -100).any() or (attention & ~prefix).any()
            ):
                raise ValueError("fm_only requires prompt-only inputs and entirely ignored CE labels")
            with torch.no_grad() if self.config.training_mode == "fm_only" else nullcontext():
                hidden = self.vlm.model(**inputs, use_cache=False, return_dict=True).last_hidden_state

        ce_valid = labels[:, 1:] != -100
        ce_count = int(ce_valid.sum())
        # A disconnected zero keeps all-ignored branches out of AdamW decay too,
        # provided the caller uses zero_grad(set_to_none=True).
        def zero():
            return hidden.new_zeros((), dtype=torch.float32, requires_grad=torch.is_grad_enabled())

        if ce_count:
            # Shift once and only project supervised positions, avoiding [B,L,156k] logits.
            logits = self.vlm.lm_head(hidden[:, :-1][ce_valid])
            loss_ce = F.cross_entropy(logits.float(), labels[:, 1:][ce_valid])
        else:
            loss_ce = zero()

        valid = batch.fm_valid_mask.to(ids.device)
        clean = batch.fm_clean_se2_diff.to(ids.device)
        if valid.dtype != torch.bool or valid.shape != (b, self.config.horizon):
            raise ValueError("fm_valid_mask must be bool [B,horizon]")
        if clean.shape != (b, self.config.horizon, 3) or not torch.isfinite(clean[valid]).all():
            raise ValueError("fm_clean_se2_diff must be [B,horizon,3] with finite valid targets")
        fm_count = int(valid.sum())
        context, velocity = None, None
        loss_fm = zero()
        if fm_count and lambda_fm > 0:
            context = self._context(hidden, prefix)
            mask = valid[..., None]
            clean = self.normalizer.normalize(torch.where(mask, clean, 0.0).detach())
            eps = _noise(clean.shape, ids.device, noise, None)
            tau = self._sample_flow_time(b, ids.device) if flow_time is None else flow_time.detach().to(ids.device)
            if tau.shape != (b,) or not torch.isfinite(tau).all() or not ((tau >= 0) & (tau <= 1)).all():
                raise ValueError("flow_time must be finite [B] in [0,1]")
            t = tau.float()[:, None, None]
            noisy = torch.where(mask, (1 - t) * eps + t * clean, eps)
            velocity = self.expert(noisy, tau, context)
            error = (velocity.float() - (clean - eps)).square()
            loss_fm = torch.where(mask, error, 0.0).sum() / (3 * fm_count)
        elif self.config.training_mode == "fm_only" and lambda_fm > 0:
            # DDP needs parameter hooks even when this rank's final accumulation
            # microbatch has no valid actions. The trainer clears these gradients
            # and skips AdamW when the *global* valid count is zero.
            loss_fm = sum(p.reshape(-1)[0] * 0 for p in self.expert.parameters() if p.requires_grad)
        return BetaTrainOutput(
            loss_ce + lambda_fm * loss_fm, loss_ce, loss_fm,
            ce_count, fm_count, context, velocity,
        )

    @torch.no_grad()
    def encode_context(self, prefix_inputs: Mapping[str, Any]) -> BetaContext:
        """Encode prompt-only inputs (no GT answer); call eval() first."""
        self._check_inference(prefix_inputs)
        with self._hf_inputs(prefix_inputs) as inputs:
            hidden = self.vlm.model(**inputs, use_cache=False, return_dict=True).last_hidden_state
            return self._context(hidden, inputs["attention_mask"])

    @torch.no_grad()
    def sample_actions(
        self, context: BetaContext, *, num_steps: int | None = None,
        noise: Tensor | None = None, generator: torch.Generator | None = None,
    ) -> tuple[Tensor, Tensor]:
        """Return physical SE(2) deltas and composed waypoints; no STOP inference."""
        normalized = self.expert.sample(context, num_steps=num_steps, noise=noise, generator=generator)
        deltas = self.normalizer.denormalize(normalized)
        return deltas, compose_se2_deltas(deltas)

    def _check_inference(self, inputs):
        if self.training:
            raise RuntimeError("call eval() before inference")
        labels = inputs.get("labels")
        if labels is not None and (labels != -100).any():
            raise ValueError("inference requires prompt-only inputs, not a teacher-forced answer")

    @torch.no_grad()
    def predict(
        self, prefix_inputs: Mapping[str, Any], *, tokenizer: Any,
        action_decoder: TrackingAgent, max_new_tokens: int = 16,
        num_steps: int | None = None, generator: torch.Generator | None = None,
        continue_on_decode_error: bool = False, explicit_stop_only: bool = False,
        action_output: str = "fm",
    ) -> BetaPrediction:
        """Greedy token generation + existing RVQ STOP + FM for continuing rows.

        With action_output="token", return decoded RVQ trajectories without calling
        the FM context projection or expert. STOP semantics are unchanged.

        Capture the last hidden layer of generate's FIRST (prefill) call, so FM
        never receives generated answer tokens and does not rerun the VLM prefill.
        HF eager generation only; concurrent calls on one model are not supported.
        """
        self._check_inference(prefix_inputs)
        if self.config.training_mode == "fm_only":
            raise ValueError("fm_only has no RVQ/STOP branch; use encode_context and sample_actions")
        if action_output not in {"token", "fm"}:
            raise ValueError("action_output must be token or fm")
        if action_output == "token" and (continue_on_decode_error or explicit_stop_only):
            raise ValueError("token output requires baseline RVQ STOP semantics")
        if getattr(action_decoder, "rvq", None) is None:
            raise ValueError("predict requires the existing TrackingAgent RVQ decoder")
        if max_new_tokens < 1:
            raise ValueError("max_new_tokens must be positive")
        captured = []

        def capture_prefill(_module, _args, output):
            if not captured:
                captured.append(output.last_hidden_state)

        with self._hf_inputs(prefix_inputs) as inputs:
            if not inputs["attention_mask"][:, -1].all():
                raise ValueError("batched generation requires left padding")
            hook = self.vlm.model.register_forward_hook(capture_prefill)
            try:
                sequences = self.vlm.generate(
                    **inputs, use_cache=True, do_sample=False, num_beams=1,
                    num_return_sequences=1, return_dict_in_generate=False,
                    output_hidden_states=False, output_attentions=False,
                    max_new_tokens=max_new_tokens, min_new_tokens=1,
                    pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id,
                )
            finally:
                hook.remove()
            if not captured or captured[0].shape[:2] != inputs["input_ids"].shape:
                raise RuntimeError("HF generation did not expose the full input prefill")
            tokens = sequences[:, inputs["input_ids"].shape[1]:]
            texts = [tokenizer.decode(row, skip_special_tokens=False) for row in tokens]
            rvq_rows = []
            explicit_stop_rows = []
            decode_failed_rows = []
            for text in texts:
                try:
                    codes = parse_rvq_action_tokens(text)
                    decoded = action_decoder.decode_waypoints(text)[0]
                    rvq_rows.append(torch.as_tensor(decoded, device=tokens.device))
                    explicit_stop_rows.append(bool(action_decoder.rvq.is_stop(codes)))
                    decode_failed_rows.append(False)
                except ValueError:
                    if not continue_on_decode_error:
                        raise
                    rvq_rows.append(
                        torch.zeros(
                            (int(action_decoder.H), 3),
                            dtype=torch.float32,
                            device=tokens.device,
                        )
                    )
                    explicit_stop_rows.append(False)
                    decode_failed_rows.append(True)
            rvq = torch.stack(rvq_rows)
            explicit_stop = torch.tensor(
                explicit_stop_rows, dtype=torch.bool, device=tokens.device
            )
            decode_failed = torch.tensor(
                decode_failed_rows, dtype=torch.bool, device=tokens.device
            )
            # TrackingAgent already maps explicit STOP and near-zero RVQ to exact zero.
            decoded_stop = (rvq == 0).all(dim=-1).all(dim=-1) & ~decode_failed
            stop = explicit_stop if explicit_stop_only else decoded_stop
            deltas = torch.zeros((len(texts), self.config.horizon, 3), device=tokens.device)
            continuing = ~stop
            if action_output == "fm" and continuing.any():
                context = self._context(captured[0][continuing], inputs["attention_mask"][continuing])
                deltas[continuing], _ = self.sample_actions(context, num_steps=num_steps, generator=generator)
            return BetaPrediction(
                tokens,
                texts,
                stop,
                explicit_stop,
                decode_failed,
                rvq,
                deltas,
                compose_se2_deltas(deltas) if action_output == "fm" else rvq,
            )


def load_beta_model(
    inference_config: InferenceConfig, beta_config: BetaModelConfig, *, task_type: str = "vlnce",
) -> tuple[BetaModel, ModelBundle]:
    """Load the existing HF checkpoint/processing bundle and add a NEW untrained expert.

    The returned bundle retains the original VLM/processor. Call model.train()
    for training; restore the full Beta state_dict to resume a trained Beta model.
    """
    from lightnav.inference.model import load_hf_model

    if inference_config.backend != "hf":
        raise ValueError("BetaModel requires backend='hf'; vLLM cannot train this expert")
    bundle = load_hf_model(inference_config, task_type=task_type)
    return BetaModel(bundle.model, beta_config), bundle


def resolve_beta_checkpoint(checkpoint: str | Path) -> tuple[Path, dict[str, Any]]:
    """Resolve a model file, committed checkpoint dir, or BetaNav run dir.

    A run directory may contain both ``latest_model.json`` and ``latest.json``.
    The highest committed step wins; at an equal step the model-only archive is
    preferred because it contains exactly the inference state and no optimizer
    payload.  Incomplete directories are never accepted.
    """
    requested = Path(checkpoint).expanduser()
    if requested.is_file():
        if requested.name != "model.pt":
            raise ValueError(f"Beta checkpoint file must be named model.pt, got {requested}")
        requested = requested.parent

    def validate(path: Path, expected_step: int | None = None) -> tuple[Path, dict[str, Any]]:
        missing = [name for name in ("model.pt", "metadata.json", "COMPLETE") if not (path / name).is_file()]
        if missing:
            raise FileNotFoundError(f"incomplete Beta checkpoint {path}: missing {', '.join(missing)}")
        metadata = json.loads((path / "metadata.json").read_text())
        if metadata.get("format") not in {
            "betanav_model_v1", "betanav_checkpoint_v1", "betanav_checkpoint_v2",
        }:
            raise ValueError(f"unsupported Beta checkpoint format in {path}: {metadata.get('format')!r}")
        if "beta_config" not in metadata:
            raise ValueError(f"Beta checkpoint metadata has no beta_config: {path}")
        step = int(metadata.get("step", expected_step if expected_step is not None else -1))
        if expected_step is not None and step != expected_step:
            raise ValueError(
                f"Beta checkpoint pointer says step {expected_step}, metadata says {step}: {path}"
            )
        return path.resolve(), metadata

    if requested.is_dir() and (requested / "model.pt").is_file():
        return validate(requested)
    if not requested.is_dir():
        raise FileNotFoundError(f"Beta checkpoint path does not exist: {requested}")

    candidates: list[tuple[int, int, Path, dict[str, Any]]] = []
    pointer_specs = (
        ("latest_model.json", "model_checkpoints", 1),
        ("latest.json", "checkpoints", 0),
    )
    pointer_errors: list[str] = []
    for pointer_name, subdir, inference_preference in pointer_specs:
        pointer_path = requested / pointer_name
        if not pointer_path.is_file():
            continue
        try:
            pointer = json.loads(pointer_path.read_text())
            step = int(pointer["step"])
            path, metadata = validate(requested / subdir / str(pointer["checkpoint"]), step)
            candidates.append((step, inference_preference, path, metadata))
        except (KeyError, OSError, TypeError, ValueError) as exc:
            pointer_errors.append(f"{pointer_name}: {exc}")
    if not candidates:
        details = f" ({'; '.join(pointer_errors)})" if pointer_errors else ""
        raise FileNotFoundError(
            f"no committed Beta checkpoint found under run directory {requested}{details}"
        )
    _step, _preference, path, metadata = max(candidates, key=lambda item: (item[0], item[1]))
    return path, metadata


def load_beta_checkpoint(
    inference_config: InferenceConfig,
    checkpoint: str | Path,
    *,
    task_type: str = "vlnce",
) -> tuple[BetaModel, ModelBundle, Path, dict[str, Any]]:
    """Build the HF BetaModel and restore a committed inference checkpoint."""
    path, metadata = resolve_beta_checkpoint(checkpoint)
    recorded_model_path = str(
        ((metadata.get("contract") or {}).get("config") or {}).get("model_path") or ""
    )
    config = inference_config
    if not config.model_path:
        if not recorded_model_path:
            raise ValueError("model_path is absent from both inference config and checkpoint metadata")
        config = replace(config, model_path=recorded_model_path)
    beta_config = BetaModelConfig(**metadata["beta_config"])
    model, bundle = load_beta_model(config, beta_config, task_type=task_type)

    # mmap avoids materialising a second ~18 GB CPU copy before the tensors are
    # copied into the already allocated model parameters.
    state = torch.load(path / "model.pt", map_location="cpu", weights_only=True, mmap=True)
    model.load_state_dict(state, strict=True)
    del state
    model.eval()
    return model, bundle, path, metadata
