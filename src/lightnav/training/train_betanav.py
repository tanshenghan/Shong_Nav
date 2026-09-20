"""Accelerate DDP + optional ZeRO-1, with separate CE/FM optimizer semantics."""

from __future__ import annotations

import argparse
import json
import os
import platform
import signal
import time
import uuid
from contextlib import nullcontext
from dataclasses import asdict
from datetime import timedelta
from importlib.metadata import version
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset

from lightnav.dataloader.Train_Lerobot_DataLoader import Train_Lerobot_DataLoader
from lightnav.training.checkpoint import (
    TrainState,
    atomic_json,
    load_checkpoint,
    on_main,
    resolve_resume,
    save_checkpoint,
)
from lightnav.training.config import learning_rate_factor, load_config, resume_contracts_match
from lightnav.training.full_checkpoint import AsyncFullCheckpointer
from lightnav.training.model_checkpoint import AsyncModelCheckpointer
from lightnav.training.runtime import (
    backward_sums,
    build_model,
    clip_and_step,
    evaluate,
    file_hash,
    make_optimizers,
    model_contract,
    move_batch,
    normalize_gradients,
    read_data_manifest,
)


class StopRequest:
    def __init__(self):
        self.requested = False
        self.previous = {}

    def __enter__(self):
        for sig in (signal.SIGINT, signal.SIGTERM):
            self.previous[sig] = signal.signal(sig, self._handle)
        return self

    def _handle(self, _signum, _frame):
        self.requested = True

    def __exit__(self, *_args):
        for sig, handler in self.previous.items():
            signal.signal(sig, handler)


class RunLogger:
    def __init__(self, accelerator, config, state):
        self.accelerator, self.config = accelerator, config
        self.path = Path(config.output_dir) / "metrics.jsonl"
        if config.tracker != "none":
            extra = {}
            if config.tracker == "wandb":
                extra = {"wandb": dict(id=state.run_id, resume="allow", mode=config.wandb_mode)}
            accelerator.init_trackers(config.project_name, config=asdict(config), init_kwargs=extra)

    def write(self, values, step):
        record = dict(step=step, time_unix=time.time(), **values)

        def append():
            with self.path.open("a") as stream:
                stream.write(json.dumps(record, allow_nan=False) + "\n")
            print(json.dumps(record, ensure_ascii=False, allow_nan=False), flush=True)

        on_main(self.accelerator, append)
        if self.config.tracker != "none":
            scalars = {}
            for name, value in values.items():
                if isinstance(value, list):
                    scalars.update({f"{name}/rank_{i}": v for i, v in enumerate(value)})
                else:
                    scalars[name] = value
            self.accelerator.log(scalars, step=step)


def build_loaders(config, manifest, accelerator):
    common = dict(dataset_root=config.dataset_root, model_path=config.model_path, batch_size=config.batch_size,
                  num_workers=config.num_workers, prefetch_factor=config.prefetch_factor,
                  cache_row_groups=config.cache_row_groups, cache_frames=config.cache_frames,
                  training_mode=config.training_mode, fm_horizon=config.fm_horizon,
                  pin_memory=accelerator.device.type == "cuda", seed=config.seed)
    train = Train_Lerobot_DataLoader(**common, episode_indices=manifest["train_episodes"],
                                     rank=accelerator.process_index, world_size=accelerator.num_processes)
    if len(train) < 1:
        raise ValueError("training split yields no batches")
    if not config.eval_every:
        return train, None
    # No DistributedSampler padding in validation: every selected frame is scored once.
    validation = Train_Lerobot_DataLoader(**common, episode_indices=manifest["val_episodes"], shuffle=False)
    indices = range(accelerator.process_index, len(validation.dataset), accelerator.num_processes)
    val = DataLoader(Subset(validation.dataset, indices), batch_size=config.batch_size,
                     collate_fn=validation.collate_fn, num_workers=0,
                     generator=torch.Generator().manual_seed(config.seed + 12345))
    return train, val


def validate_training_data_expansion(saved, current, manifest):
    """One explicit transition: the original train+validation union, with the old action scale."""
    old_path = Path(saved["config"]["data_manifest"])
    if file_hash(old_path) != saved["data_sha256"]:
        raise ValueError("original frozen data manifest changed")
    old = json.loads(old_path.read_text())
    adjusted = {**saved, "config": {**saved["config"], "data_manifest": current["config"]["data_manifest"]},
                "data_sha256": current["data_sha256"]}
    if not resume_contracts_match(adjusted, current):
        raise ValueError("training-data expansion cannot change model, batch, optimizer or schedule")
    if (old.get("split_by") != "scene" or not old["val_episodes"] or manifest.get("split_by") != "all"
            or manifest["val_episodes"] or manifest["val_scenes"] or manifest["val_frames"]
            or set(manifest["train_episodes"]) != set(old["train_episodes"]) | set(old["val_episodes"])
            or len(manifest["train_episodes"]) != len(old["train_episodes"]) + len(old["val_episodes"])
            or set(manifest["train_scenes"]) != set(old["train_scenes"]) | set(old["val_scenes"])
            or manifest["train_frames"] != old["train_frames"] + old["val_frames"]
            or any(manifest[k] != old[k] for k in ("dataset_root", "normalization", "source_sha256"))):
        raise ValueError("expansion must use the unchanged dataset's full union and preserve normalization")


def _initialize_run(config, manifest, contract, resume_path, *, expand_training_data=False, override_max_steps=False):
    output = Path(config.output_dir)
    if resume_path is not None:
        meta = json.loads((resume_path / "metadata.json").read_text())
        checked = meta.get("contract")
        old_max_steps = checked.get("config", {}).get("max_steps")
        max_steps_changed = old_max_steps is not None and old_max_steps != config.max_steps
        if override_max_steps:
            if config.max_steps <= json.loads((resume_path / "trainer.json").read_text())["step"]:
                raise ValueError("new max_steps must exceed the resumed step")
            checked = {**checked, "config": {**checked["config"], "max_steps": config.max_steps}}
        expanded = not resume_contracts_match(checked, contract)
        if expanded:
            if not expand_training_data:
                raise ValueError("resume contract differs; do not overwrite this experiment")
            validate_training_data_expansion(checked, contract, manifest)
        source = Path(__file__).parent
        resume_dir = output / "resumes"
        resume_dir.mkdir(exist_ok=True)
        atomic_json(resume_dir / f"{time.time_ns()}-{uuid.uuid4().hex[:8]}.json", dict(
            resume_from=str(resume_path), run_id=meta["run_id"], config=asdict(config), contract=contract,
            training_data_expanded=expanded, data_cursor_reset=expanded,
            max_steps_override=override_max_steps and max_steps_changed, previous_max_steps=old_max_steps,
            source_sha256={p.name: file_hash(p) for p in source.glob("*.py")}))
        atomic_json(output / "active_config.json", asdict(config))
        atomic_json(output / "active_data_manifest.json", manifest)
        return meta["run_id"]
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("output_dir is nonempty; select a new run or pass --resume auto")
    output.mkdir(parents=True, exist_ok=True)
    run_id = uuid.uuid4().hex
    atomic_json(output / "config.json", asdict(config))
    atomic_json(output / "active_config.json", asdict(config))
    atomic_json(output / "data_manifest.json", manifest)
    atomic_json(output / "active_data_manifest.json", manifest)
    source = Path(__file__).parent
    files = [*source.glob("*.py"), source.parent / "inference/beta_model.py",
             source.parent / "dataloader/Train_Lerobot_DataLoader.py"]
    atomic_json(output / "run.json", dict(run_id=run_id, contract=contract,
                python=platform.python_version(), packages={k: version(k) for k in ("torch", "transformers", "accelerate", "pyarrow")},
                source_sha256={str(p.relative_to(source.parent)): file_hash(p) for p in files}))
    return run_id


def train(config, *, resume=None, dry_run=False, stop_after=None, expand_training_data=False, override_max_steps=False):
    from accelerate import Accelerator
    from accelerate.utils import (
        DistributedDataParallelKwargs,
        DistributedType,
        InitProcessGroupKwargs,
        set_seed,
    )

    if stop_after is not None and stop_after < 1:
        raise ValueError("--stop-after must be a positive completed step")
    if expand_training_data and not resume:
        raise ValueError("--expand-training-data requires an explicit resume checkpoint")
    if override_max_steps and not resume:
        raise ValueError("--override-max-steps requires an explicit resume checkpoint")
    torch.set_num_threads(config.cpu_threads)
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("OMP_NUM_THREADS", str(config.cpu_threads))
    if not config.cpu and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; no automatic CPU fallback. Use --cpu for an explicit CPU check.")
    accelerator = Accelerator(
        cpu=config.cpu, mixed_precision=config.mixed_precision, gradient_accumulation_steps=1,
        log_with=None if config.tracker == "none" else config.tracker,
        project_dir=config.output_dir,
        kwargs_handlers=[DistributedDataParallelKwargs(find_unused_parameters=True,
                         broadcast_buffers=False, gradient_as_bucket_view=True),
                         InitProcessGroupKwargs(timeout=timedelta(minutes=30))],
    )
    if accelerator.distributed_type not in {DistributedType.NO, DistributedType.MULTI_GPU, DistributedType.MULTI_CPU}:
        raise ValueError("this trainer supports DDP/ZeRO-1; do not launch it with a DeepSpeed/FSDP config")
    if config.mixed_precision == "bf16" and accelerator.device.type == "cuda" and not torch.cuda.is_bf16_supported():
        raise RuntimeError("BF16 requested but unavailable on this GPU")
    if accelerator.device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
    set_seed(config.seed)  # identical initial expert on every rank
    manifest = on_main(accelerator, lambda: read_data_manifest(config))
    loader, val_loader = build_loaders(config, manifest, accelerator)
    updates_per_epoch = (len(loader) + config.gradient_accumulation_steps - 1) // config.gradient_accumulation_steps
    if config.num_epochs is not None:
        config.max_steps = config.num_epochs * updates_per_epoch
        config.validate()
        if override_max_steps:
            raise ValueError("num_epochs cannot be combined with --override-max-steps")
    contract = on_main(accelerator, lambda: model_contract(config, loader.spec))
    plan = dict(world_size=accelerator.num_processes, device=str(accelerator.device),
                training_mode=config.training_mode, num_epochs=config.num_epochs, fm_horizon=config.fm_horizon,
                flow_time_distribution=config.flow_time_distribution,
                flow_time_alpha=config.flow_time_alpha, flow_time_beta=config.flow_time_beta,
                max_steps=config.max_steps, checkpoint_mode=config.checkpoint_mode,
                save_every=config.save_every, model_save_every=config.model_save_every,
                train_frames=len(loader.dataset), val_frames=manifest["val_frames"],
                train_episodes=len(manifest["train_episodes"]), val_episodes=len(manifest["val_episodes"]),
                effective_batch_size=config.batch_size * config.gradient_accumulation_steps * accelerator.num_processes,
                updates_per_epoch=(len(loader) + config.gradient_accumulation_steps - 1) // config.gradient_accumulation_steps,
                action_scale=manifest["normalization"]["action_scale"],
                optimizer=config.optimizer if accelerator.num_processes > 1 else "adamw",
                validation_enabled=bool(config.eval_every),
                pretrained_weights_loaded=not config.tiny_model, config=asdict(config))
    if dry_run:
        # No model weights/optimizer/run directory are created in a dry run.
        on_main(accelerator, lambda: print(json.dumps({**plan, "pretrained_weights_loaded": False}, indent=2)))
        accelerator.end_training()
        return plan
    checkpoint = on_main(accelerator, lambda: str(resolve_resume(Path(config.output_dir), resume)) if resume else None)
    checkpoint = Path(checkpoint) if checkpoint else None
    run_id = on_main(accelerator, lambda: _initialize_run(config, manifest, contract, checkpoint,
                                                        expand_training_data=expand_training_data,
                                                        override_max_steps=override_max_steps))
    model = build_model(config, manifest["normalization"]["action_scale"])
    model = accelerator.prepare_model(model)
    bare = accelerator.unwrap_model(model)
    groups = bare.optimizer_parameters()
    optimizers = make_optimizers(bare, config, accelerator.num_processes)
    state = TrainState(run_id=run_id)
    logger = RunLogger(accelerator, config, state)
    if checkpoint:
        source_contract = on_main(accelerator, lambda: json.loads((checkpoint / "metadata.json").read_text())["contract"])
        state = load_checkpoint(checkpoint, accelerator, model, optimizers,
                                source_contract if expand_training_data or override_max_steps else contract)
        checked = source_contract
        if override_max_steps:
            checked = {**checked, "config": {**checked["config"], "max_steps": config.max_steps}}
            accelerator.print(json.dumps(dict(event="max_steps_overridden", step=state.step,
                                               previous_max_steps=source_contract["config"]["max_steps"],
                                               max_steps=config.max_steps)))
        if not resume_contracts_match(checked, contract):
            state.epoch, state.batch_offset, state.best_validation = 0, 0, float("inf")
            accelerator.print(json.dumps(dict(event="training_data_expanded", step=state.step,
                                               train_frames=len(loader.dataset), data_cursor_reset=True)))
    else:
        set_seed(config.seed + accelerator.process_index)  # independent rank noise/dropout after DDP broadcast
    accelerator.print(json.dumps({k: v for k, v in plan.items() if k != "config"}, indent=2))
    accelerator.print("Trainable parameters:", {k: sum(p.numel() for p in v) for k, v in groups.items()})
    for optimizer in optimizers.values():
        optimizer.zero_grad(set_to_none=True)
    last_saved = state.step if checkpoint else -1
    model_writer = AsyncModelCheckpointer(accelerator, config, contract) if config.checkpoint_mode == "model_only" else None
    full_writer = AsyncFullCheckpointer(accelerator, config, contract) if config.checkpoint_mode == "dual" else None

    def checkpoint_now(best=False):
        if full_writer is not None:
            return full_writer.submit(model, optimizers, state, permanent=state.step % config.model_save_every == 0)
        if model_writer is not None:
            return model_writer.submit(model, state)
        saved = save_checkpoint(accelerator, model, optimizers, state, config, contract, best=best)
        accelerator.print(f"Committed checkpoint: {saved}")
        return saved

    try:
        with StopRequest() as stop:
            while state.step < config.max_steps:
                if stop_after is not None and state.step >= stop_after:
                    break
                loader.set_epoch(state.epoch)
                if not 0 <= state.batch_offset < len(loader):
                    raise ValueError("invalid checkpoint data cursor")
                # Skip the SAMPLER indices, never replay/decode all previous RGB batches.
                active = accelerator.skip_first_batches(loader, state.batch_offset) if state.batch_offset else loader
                iterator = iter(active)
                while state.batch_offset < len(loader) and state.step < config.max_steps:
                    log_update = state.step == 0 or (state.step + 1) % config.log_every == 0
                    if log_update and accelerator.device.type == "cuda":
                        torch.cuda.synchronize(accelerator.device)
                    start_time = time.perf_counter()
                    data_wait = 0.
                    factor = learning_rate_factor(state.step, config.warmup_steps, config.max_steps, config.min_lr_ratio)
                    for name, base_lr in (("vlm", config.lr_vlm), ("fm", config.lr_fm)):
                        if name not in optimizers:
                            continue
                        for group in optimizers[name].param_groups:
                            group["lr"] = base_lr * factor
                    n_micro = min(config.gradient_accumulation_steps, len(loader) - state.batch_offset)
                    totals = torch.zeros(8, dtype=torch.float64, device=accelerator.device)
                    for micro in range(n_micro):
                        wait_start = time.perf_counter()
                        cpu_batch = next(iterator)
                        data_wait += time.perf_counter() - wait_start
                        batch = move_batch(cpu_batch, accelerator.device)
                        del cpu_batch
                        # A final synchronized backward also reduces preceding no_sync gradients.
                        sync_context = accelerator.no_sync(model) if micro < n_micro - 1 else nullcontext()
                        with sync_context:
                            out = model(batch, lambda_fm=config.lambda_fm)
                            finite = accelerator.reduce(torch.isfinite(out.loss).to(torch.int64), reduction="sum")
                            if int(finite) != accelerator.num_processes:
                                raise FloatingPointError("nonfinite loss; aborting before optimizer update on every rank")
                            backward_sums(out, accelerator, config.lambda_fm)
                        nc, nf = out.ce_token_count, out.fm_step_count if config.lambda_fm > 0 else 0
                        totals[0] += out.ce_loss.detach().double() * nc
                        totals[1] += nc
                        totals[2] += out.fm_loss.detach().double() * nf
                        totals[3] += nf
                        totals[4] += len(batch.labels)
                        totals[5] += batch.model_inputs["attention_mask"].sum()
                        totals[6] += batch.labels.numel()
                        totals[7] += (~batch.fm_valid_mask.any(dim=1)).sum()
                        state.batch_offset += 1
                        del out, batch
                    sums = accelerator.reduce(totals, reduction="sum").tolist()
                    counts = [int(sums[1]), int(sums[3])]
                    normalize_gradients(groups, counts, accelerator.num_processes)
                    norms = clip_and_step(groups, optimizers, counts, config)
                    state.step += 1
                    state.samples_seen += int(sums[4])
                    state.vlm_updates += bool(counts[0])
                    state.fm_updates += bool(counts[1])
                    epoch_end = state.batch_offset == len(loader)
                    if epoch_end:
                        state.epoch += 1
                        state.batch_offset = 0
                    if log_update:
                        if accelerator.device.type == "cuda":
                            torch.cuda.synchronize(accelerator.device)
                        elapsed = time.perf_counter() - start_time
                        cuda = accelerator.device.type == "cuda"
                        perf = torch.tensor([[elapsed, data_wait,
                            torch.cuda.max_memory_allocated(accelerator.device) / 2**30 if cuda else 0.,
                            torch.cuda.max_memory_reserved(accelerator.device) / 2**30 if cuda else 0.]],
                            device=accelerator.device, dtype=torch.float64)
                        per_rank = accelerator.gather(perf)
                        slowest, max_wait, max_allocated, max_reserved = per_rank.max(dim=0).values.tolist()
                        ce, fm = sums[0] / max(counts[0], 1), sums[2] / max(counts[1], 1)
                        logger.write(dict(train_loss=ce + config.lambda_fm * fm, train_ce=ce, train_fm=fm,
                                     lr_vlm=config.lr_vlm * factor if "vlm" in optimizers else 0.,
                                     lr_fm=config.lr_fm * factor,
                                     grad_norm_vlm=norms["vlm"], grad_norm_fm=norms["fm"],
                                     ce_tokens=counts[0], fm_valid_steps=counts[1], samples=int(sums[4]),
                                     samples_seen=state.samples_seen, epoch=state.epoch,
                                     batch_offset=state.batch_offset, no_future_samples=int(sums[7]),
                                     padding_fraction=1 - sums[5] / max(sums[6], 1),
                                     update_seconds=slowest, samples_per_second=sums[4] / slowest,
                                     input_tokens_per_second=sums[5] / slowest,
                                     data_wait_seconds_max=max_wait,
                                     update_seconds_by_rank=per_rank[:, 0].tolist(),
                                     data_wait_seconds_by_rank=per_rank[:, 1].tolist(),
                                     peak_memory_gib=max_allocated,
                                     peak_reserved_memory_gib=max_reserved), state.step)
                    best = False
                    if model_writer is not None:
                        model_writer.poll()
                    if full_writer is not None:
                        full_writer.poll()
                    if config.eval_every and state.step % config.eval_every == 0:
                        metrics = evaluate(model, val_loader, accelerator, config)
                        logger.write(metrics, state.step)
                        best = metrics["val_loss"] < state.best_validation
                        if best:
                            state.best_validation = metrics["val_loss"]
                    stopping = bool(int(accelerator.reduce(torch.tensor(int(stop.requested), device=accelerator.device),
                                                          reduction="sum")))
                    stopping |= state.step >= config.max_steps or (stop_after is not None and state.step >= stop_after)
                    if state.step % config.save_every == 0 or (best and config.save_best) or stopping:
                        checkpoint_now(best=best and config.save_best)
                        last_saved = state.step
                    if stopping or epoch_end:
                        break
                if stopping:
                    break
            if state.step and last_saved != state.step:
                checkpoint_now()
            if model_writer is not None:
                model_writer.poll(wait=True)
            if full_writer is not None:
                full_writer.poll(wait=True)
    finally:
        if model_writer is not None:
            model_writer.close()
        if full_writer is not None:
            full_writer.close()
        # A hard failure retains the last committed checkpoint; never serialize partial gradients.
        accelerator.end_training()
    return asdict(state)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir")
    parser.add_argument("--data-manifest")
    parser.add_argument("--resume", nargs="?", const="auto", help="auto or a committed checkpoint path")
    parser.add_argument("--expand-training-data", action="store_true",
                        help="explicitly resume on the original train+validation union; preserve Adam/step, reset data cursor")
    parser.add_argument("--override-max-steps", action="store_true",
                        help="explicitly change the remaining stopping point and cosine schedule when resuming")
    parser.add_argument("--dry-run", action="store_true", help="validate configuration/data, without loading weights")
    parser.add_argument("--cpu", action="store_true", default=None, help="explicitly allow CPU execution")
    parser.add_argument("--tiny-model", action="store_true", default=None, help="use a small RANDOM model for infrastructure checks")
    parser.add_argument("--stop-after", type=int, help="stop/save at this completed step without changing the LR schedule")
    args = parser.parse_args()
    config = load_config(args.config, dict(output_dir=args.output_dir, data_manifest=args.data_manifest,
                                          cpu=args.cpu, tiny_model=args.tiny_model))
    train(config, resume=args.resume, dry_run=args.dry_run, stop_after=args.stop_after,
          expand_training_data=args.expand_training_data, override_max_steps=args.override_max_steps)


if __name__ == "__main__":
    main()
