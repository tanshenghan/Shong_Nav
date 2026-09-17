"""Strict, reviewable BetaNav training configuration (no implicit LR scaling)."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path

import yaml

EVALUATION_OPTIONS = frozenset({"eval_every", "eval_max_batches", "eval_sample_batches", "save_best"})


@dataclass
class TrainConfig:
    dataset_root: str
    model_path: str
    data_manifest: str
    output_dir: str
    seed: int = 42
    batch_size: int = 8  # per rank; tune to GPU capacity and history length
    gradient_accumulation_steps: int = 1
    max_steps: int = 30_000  # completed optimizer update windows
    num_epochs: int | None = None  # if set, derive max_steps from the actual distributed loader
    training_mode: str = "ce_fm"  # fm_only freezes the complete VLM and omits token targets
    warmup_steps: int = 1_000
    min_lr_ratio: float = 0.1
    lr_vlm: float = 1e-5
    lr_fm: float = 1e-4
    weight_decay: float = 0.01
    adam_betas: list[float] = field(default_factory=lambda: [0.9, 0.95])
    adam_eps: float = 1e-8
    clip_vlm: float = 1.0
    clip_fm: float = 1.0
    lambda_fm: float = 2.0
    freeze_vision: bool = True
    gradient_checkpointing: bool = True
    mixed_precision: str = "bf16"  # FP32 master parameters; AMP computation
    optimizer: str = "zero1"  # AdamW states sharded with native PyTorch, or adamw
    attention: str = "sdpa"
    expert_hidden_size: int = 512
    expert_num_layers: int = 6
    expert_num_heads: int = 8
    expert_intermediate_size: int = 2048
    expert_dropout: float = 0.0
    num_flow_steps: int = 10
    num_workers: int = 2
    prefetch_factor: int = 1
    cache_row_groups: int = 4
    cache_frames: int = 64
    cpu_threads: int = 4
    log_every: int = 10
    eval_every: int = 500
    eval_max_batches: int = 64  # per rank; 0 means the full validation split
    eval_sample_batches: int = 4  # per rank, additional FM integration metrics
    save_every: int = 1_000
    checkpoint_mode: str = "full"  # legacy full, model_only, or asynchronous dual retention
    model_save_every: int = 3_000  # permanent model archive cadence in dual mode
    keep_last: int = 3
    save_best: bool = True
    tracker: str = "none"  # JSONL is always written; optional tensorboard / wandb
    project_name: str = "betanav"
    wandb_mode: str = "offline"
    cpu: bool = False
    tiny_model: bool = False  # real processor/vocabulary, small RANDOM model

    def validate(self):
        for name in ("dataset_root", "model_path", "data_manifest", "output_dir"):
            if not isinstance(getattr(self, name), str) or not getattr(self, name):
                raise ValueError(f"{name} must be a nonempty path string")
        positive = ("batch_size", "gradient_accumulation_steps", "max_steps", "expert_hidden_size",
                    "expert_num_layers", "expert_num_heads", "expert_intermediate_size", "num_flow_steps",
                    "prefetch_factor", "cache_row_groups", "cpu_threads", "log_every", "save_every", "model_save_every", "keep_last")
        nonnegative = ("seed", "warmup_steps", "num_workers", "cache_frames", "eval_every",
                       "eval_max_batches", "eval_sample_batches")
        for name in positive + nonnegative:
            value = getattr(self, name)
            if type(value) is not int or value < (1 if name in positive else 0):
                raise ValueError(f"invalid integer option: {name}")
        for name in ("lr_vlm", "lr_fm", "adam_eps", "clip_vlm", "clip_fm"):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be finite and positive")
        for name in ("weight_decay", "lambda_fm"):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
        if self.warmup_steps >= self.max_steps or not 0 <= self.min_lr_ratio <= 1:
            raise ValueError("require warmup_steps < max_steps and min_lr_ratio in [0,1]")
        if self.num_epochs is not None and (type(self.num_epochs) is not int or self.num_epochs < 1):
            raise ValueError("num_epochs must be a positive integer or null")
        if self.training_mode not in {"ce_fm", "fm_only"}:
            raise ValueError("training_mode must be ce_fm or fm_only")
        if self.training_mode == "fm_only" and self.lambda_fm <= 0:
            raise ValueError("fm_only requires lambda_fm > 0")
        if len(self.adam_betas) != 2 or any(not 0 <= b < 1 for b in self.adam_betas):
            raise ValueError("adam_betas must contain two values in [0,1)")
        if self.mixed_precision not in {"no", "bf16"}:
            raise ValueError("supported mixed_precision: no / bf16 (no FP16 GradScaler)")
        if self.optimizer not in {"adamw", "zero1"} or self.attention not in {"sdpa", "eager"}:
            raise ValueError("unsupported optimizer or attention implementation")
        if self.checkpoint_mode not in {"full", "model_only", "dual"}:
            raise ValueError("checkpoint_mode must be full, model_only or dual")
        if self.checkpoint_mode in {"model_only", "dual"} and self.save_best:
            raise ValueError("asynchronous checkpoints use the step cadence; set save_best=false")
        if self.checkpoint_mode == "dual" and self.model_save_every % self.save_every:
            raise ValueError("model_save_every must be a multiple of save_every in dual mode")
        if self.tracker not in {"none", "tensorboard", "wandb"} or self.wandb_mode not in {"offline", "online"}:
            raise ValueError("unsupported tracker / wandb_mode")
        if self.expert_hidden_size % 2 or self.expert_hidden_size % self.expert_num_heads:
            raise ValueError("expert_hidden_size must be even and divisible by expert_num_heads")
        if not 0 <= self.expert_dropout < 1:
            raise ValueError("expert_dropout must be in [0,1)")
        for name in ("freeze_vision", "gradient_checkpointing", "save_best", "cpu", "tiny_model"):
            if type(getattr(self, name)) is not bool:
                raise ValueError(f"{name} must be a YAML boolean")
        return self

    def resume_contract(self):
        # Evaluation is RNG-isolated and does not update parameters or the training data cursor.
        ignored = {"output_dir", "num_workers", "prefetch_factor", "cache_row_groups", "cache_frames",
                   "cpu_threads", "log_every", "save_every", "model_save_every", "keep_last", "checkpoint_mode",
                   "tracker", "project_name", "wandb_mode"} | EVALUATION_OPTIONS
        return {k: v for k, v in asdict(self).items() if k not in ignored}


def resume_contracts_match(saved, current):
    """Accept evaluation settings in older v1 contracts without relaxing training identity."""
    def normalize(contract):
        if not isinstance(contract, dict) or not isinstance(contract.get("config"), dict):
            return contract
        options = {"training_mode": "ce_fm", "num_epochs": None, **contract["config"]}
        return {**contract, "config": {k: v for k, v in options.items() if k not in EVALUATION_OPTIONS}}

    return normalize(saved) == normalize(current)


def load_config(path: str | Path, overrides: dict | None = None) -> TrainConfig:
    data = yaml.safe_load(Path(path).read_text())
    if not isinstance(data, dict):
        raise ValueError("training YAML must be a mapping")
    data.update({k: v for k, v in (overrides or {}).items() if v is not None})
    unknown = data.keys() - {f.name for f in fields(TrainConfig)}
    if unknown:
        raise ValueError(f"unknown training options: {sorted(unknown)}")
    return TrainConfig(**data).validate()


def learning_rate_factor(step: int, warmup: int, total: int, floor: float) -> float:
    """Rate for zero-based update step; total/floor are fixed across resume."""
    if step < warmup:
        return (step + 1) / warmup
    progress = min(1., (step - warmup) / max(1, total - warmup - 1))
    return floor + (1 - floor) * (1 + math.cos(math.pi * progress)) / 2
