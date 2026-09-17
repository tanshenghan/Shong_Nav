"""Beta-specific optimizer groups, exact masked gradients and held-out metrics."""

from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path

import torch
from torch import Tensor

from lightnav.dataloader.Train_Lerobot_DataLoader import LightNavTrainingSpec
from lightnav.inference.beta_model import BetaModel, BetaModelConfig, compose_se2_deltas
from lightnav.processing import enable_post_vit_pool, get_post_vit_target


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for part in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(part)
    return digest.hexdigest()


def read_data_manifest(config):
    data = json.loads(Path(config.data_manifest).read_text())
    if data.get("format") != "betanav_training_data_v1":
        raise ValueError("prepare data with python -m lightnav.training.prepare_data")
    root = Path(config.dataset_root).resolve()
    if Path(data["dataset_root"]).resolve() != root:
        raise ValueError("training data manifest points at a different dataset")
    train, val = data["train_episodes"], data["val_episodes"]
    if not train or len(set(train)) != len(train) or len(set(val)) != len(val) or set(train) & set(val):
        raise ValueError("training episode IDs must be nonempty; split IDs must be unique and disjoint")
    if data.get("split_by") == "all":
        info = json.loads((root / "meta/info.json").read_text())
        if val or data["val_scenes"] or data["val_frames"] or config.eval_every:
            raise ValueError("all-training manifests require empty validation metadata and eval_every=0")
        if len(train) != info["total_episodes"] or data["train_frames"] != info["total_frames"]:
            raise ValueError("all-training manifest must cover every episode and frame")
    elif data.get("split_by") != "scene" or not val or set(data["train_scenes"]) & set(data["val_scenes"]):
        raise ValueError("require all-training or scene-separated metadata")
    # Only rank 0 calls this. Verify metadata without hashing hundreds of GB of pixels.
    for name, expected in data["source_sha256"].items():
        path = (root / name).resolve()
        if not path.is_relative_to(root) or file_hash(path) != expected:
            raise ValueError(f"dataset metadata changed since training preparation: {name}")
    scale = torch.tensor(data["normalization"]["action_scale"])
    if scale.shape != (3,) or not torch.isfinite(scale).all() or not (scale > 0).all():
        raise ValueError("invalid training-only action scale")
    return data


def model_contract(config, spec):
    paths = sorted(Path(spec.rvq_path).glob("*")) if spec.rvq_path else []
    # These are the codebooks/config actually loaded; weight identity includes size/mtime.
    from lightnav.inference.model import resolve_model_paths

    config_dir, weights_dir, _ = resolve_model_paths(config.model_path)
    assets = {str(p.name): file_hash(p) for p in paths if p.is_file()}
    weight_files = sorted(Path(weights_dir).glob("*.safetensors"))
    if not weight_files and not config.tiny_model:
        raise ValueError("base model has no safetensors weights")
    weights = {p.name: [p.stat().st_size, p.stat().st_mtime_ns] for p in weight_files}
    contract = dict(config=config.resume_contract(), data_sha256=file_hash(config.data_manifest),
                    processing=spec.params, config_sha256=file_hash(Path(config_dir) / "config.json"),
                    rvq_assets=assets, base_weight_files=weights)
    return json.loads(json.dumps(contract))


def build_model(config, action_scale):
    from transformers import (
        AutoConfig,
        AutoModelForImageTextToText,
        Qwen3VLForConditionalGeneration,
    )

    from lightnav.inference.model import resolve_model_paths

    spec = LightNavTrainingSpec.from_checkpoint(config.model_path, training_mode=config.training_mode)
    config_dir, weights_dir, _ = resolve_model_paths(config.model_path)
    if config.tiny_model:
        hf = AutoConfig.from_pretrained(config_dir, local_files_only=True)
        for k, v in dict(hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                         num_attention_heads=4, num_key_value_heads=2, head_dim=8,
                         rope_parameters={"rope_type": "default", "rope_theta": 10000.,
                                          "mrope_section": [1, 1, 2]}).items():
            setattr(hf.text_config, k, v)
        for k, v in dict(depth=2, hidden_size=16, intermediate_size=32, num_heads=2,
                         out_hidden_size=32, deepstack_visual_indexes=[0, 1]).items():
            setattr(hf.vision_config, k, v)
        hf._attn_implementation = config.attention
        vlm = Qwen3VLForConditionalGeneration(hf)
    else:
        # FP32 master parameters and Adam states; BF16 is an autocast COMPUTE mode.
        # Local CPU load first: no device_map='auto', inference eval(), or warm-up.
        frozen_bf16 = config.training_mode == "fm_only" and config.mixed_precision == "bf16"
        vlm = AutoModelForImageTextToText.from_pretrained(
            weights_dir, dtype=torch.bfloat16 if frozen_bf16 else torch.float32, attn_implementation=config.attention,
            local_files_only=True,
        )
    if config.tiny_model or config.training_mode != "fm_only" or config.mixed_precision != "bf16":
        vlm.float()  # tiny HF configs may themselves request BF16 construction
    vlm.config.use_cache = False
    if spec.params["pool_enable"] and spec.params["pool_stage"] == "post_vit":
        enable_post_vit_pool(vlm, spatial_factor=spec.params["pool_spatial"])
        vlm._post_vit_target = get_post_vit_target(vlm)
    beta = BetaModelConfig(action_scale=tuple(action_scale), horizon=30,
                           hidden_size=config.expert_hidden_size, num_layers=config.expert_num_layers,
                           num_heads=config.expert_num_heads, intermediate_size=config.expert_intermediate_size,
                           dropout=config.expert_dropout, num_flow_steps=config.num_flow_steps,
                           training_mode=config.training_mode)
    model = BetaModel(vlm, beta)
    if config.freeze_vision:
        model.vlm.model.visual.requires_grad_(False)
    if config.gradient_checkpointing and config.training_mode != "fm_only":
        model.vlm.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.train()
    if config.freeze_vision:
        model.vlm.model.visual.eval()
    return model


def make_optimizers(model, config, world_size):
    result = {}
    for name, module, lr in (("vlm", model.vlm, config.lr_vlm), ("fm", model.expert, config.lr_fm)):
        decay, no_decay = [], []
        for key, parameter in module.named_parameters():
            if parameter.requires_grad:
                (no_decay if parameter.ndim < 2 or key.endswith("bias") else decay).append(parameter)
        if not decay and not no_decay:
            continue
        groups = [{"params": decay, "weight_decay": config.weight_decay},
                  {"params": no_decay, "weight_decay": 0.0}]
        kwargs = dict(lr=lr, betas=tuple(config.adam_betas), eps=config.adam_eps, foreach=False)
        if config.optimizer == "zero1" and world_size > 1:
            from torch.distributed.optim import ZeroRedundancyOptimizer

            result[name] = ZeroRedundancyOptimizer(groups, optimizer_class=torch.optim.AdamW, **kwargs)
        else:
            result[name] = torch.optim.AdamW(groups, **kwargs)
    return result


def move_batch(batch, device):
    return replace(batch, model_inputs={k: v.to(device, non_blocking=True) if isinstance(v, Tensor) else v
                                       for k, v in batch.model_inputs.items()},
                   labels=batch.labels.to(device, non_blocking=True), answer_start=batch.answer_start.to(device),
                   fm_clean_se2_diff=batch.fm_clean_se2_diff.to(device, non_blocking=True),
                   fm_valid_mask=batch.fm_valid_mask.to(device, non_blocking=True))


def backward_sums(output, accelerator, lambda_fm):
    """Accumulate SUM gradients; normalize disjoint CE/FM groups after the whole window.

    This avoids retaining a window of huge video batches just to count their masks.
    Accelerator.gradient_accumulation_steps MUST be 1; our loop owns accumulation.
    DDP averages across W ranks, so the final multiplier is W/global_valid_count.
    """
    loss = output.ce_loss * output.ce_token_count + lambda_fm * output.fm_loss * output.fm_step_count
    accelerator.backward(loss)


def normalize_gradients(groups, counts, world_size):
    for name, count in zip(("vlm", "fm"), counts, strict=True):
        for parameter in groups[name]:
            if not count:
                parameter.grad = None
            elif parameter.grad is not None:
                parameter.grad.mul_(world_size / count)


def clip_and_step(groups, optimizers, counts, config):
    norms = {}
    for name, count, maximum in zip(("vlm", "fm"), counts, (config.clip_vlm, config.clip_fm), strict=True):
        norms[name] = float(torch.nn.utils.clip_grad_norm_(groups[name], maximum, error_if_nonfinite=True)) if count else 0.
    # Check both groups before changing either optimizer's state.
    for name, count in zip(("vlm", "fm"), counts, strict=True):
        if count:
            optimizers[name].step()
    for optimizer in optimizers.values():
        optimizer.zero_grad(set_to_none=True)
    return norms


@torch.no_grad()
def evaluate(model, loader, accelerator, config):
    """Deterministic held-out losses plus integrated FM ADE/FDE; no navigation rollout."""
    bare = accelerator.unwrap_model(model)
    bare.eval()
    # Unwrapped model: validation shards may have different lengths; no DDP collectives per forward.
    totals = torch.zeros(9, dtype=torch.float64, device=accelerator.device)
    generator = torch.Generator(device=accelerator.device).manual_seed(config.seed + 10_000 + accelerator.process_index)
    try:
        for i, batch in enumerate(loader):
            if config.eval_max_batches and i >= config.eval_max_batches:
                break
            batch = move_batch(batch, accelerator.device)
            noise = torch.randn(batch.fm_clean_se2_diff.shape, generator=generator, device=accelerator.device)
            tau = torch.rand(len(batch.labels), generator=generator, device=accelerator.device)
            with accelerator.autocast():
                out = bare(batch, lambda_fm=config.lambda_fm, noise=noise, flow_time=tau)
            totals[:4] += torch.tensor([float(out.ce_loss) * out.ce_token_count, out.ce_token_count,
                                       float(out.fm_loss) * out.fm_step_count, out.fm_step_count],
                                      device=totals.device, dtype=totals.dtype)
            totals[8] += len(batch.labels)
            if i < config.eval_sample_batches and out.context is not None:
                with accelerator.autocast():
                    _, predicted = bare.sample_actions(out.context, num_steps=config.num_flow_steps, noise=noise)
                mask = batch.fm_valid_mask
                target = compose_se2_deltas(torch.where(mask[..., None], batch.fm_clean_se2_diff, 0.))
                distance = torch.linalg.vector_norm(predicted[..., :2] - target[..., :2], dim=-1)
                rows = mask.any(dim=1)
                last = mask.sum(dim=1)[rows] - 1
                totals[4] += distance[mask].sum()
                totals[5] += mask.sum()
                totals[6] += distance[rows].gather(1, last[:, None]).sum()
                totals[7] += rows.sum()
        totals = accelerator.reduce(totals, reduction="sum").tolist()
    finally:
        bare.train()
        if config.freeze_vision:
            bare.vlm.model.visual.eval()
    ce, nc, fm, nf, ade, na, fde, nr, samples = totals
    result = dict(val_ce=ce / max(nc, 1), val_fm=fm / max(nf, 1), val_ce_tokens=int(nc),
                  val_fm_steps=int(nf), val_samples=int(samples), val_rollout_samples=int(nr))
    if na:
        result.update(val_fm_ade_m=ade / na, val_fm_fde_m=fde / nr)
    result["val_loss"] = result["val_ce"] + config.lambda_fm * result["val_fm"]
    return result
