"""Committed checkpoints, rank RNG state and exact data-cursor resume for DDP/ZeRO-1."""

from __future__ import annotations

import json
import os
import random
import shutil
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist

from lightnav.training.config import resume_contracts_match


@dataclass
class TrainState:
    run_id: str
    step: int = 0
    epoch: int = 0
    batch_offset: int = 0
    samples_seen: int = 0
    vlm_updates: int = 0
    fm_updates: int = 0
    stop_updates: int = 0
    best_validation: float = float("inf")


def broadcast(value, accelerator):
    box = [value]
    if accelerator.num_processes > 1:
        dist.broadcast_object_list(box, src=0)
    return box[0]


def on_main(accelerator, function):
    """Propagate filesystem errors before peers enter their next collective."""
    result, error = None, None
    if accelerator.is_main_process:
        try:
            result = function()
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
    result, error = broadcast((result, error), accelerator)
    if error:
        raise RuntimeError(error)
    return result


def atomic_json(path: Path, value):
    temp = path.with_name(path.name + ".tmp")
    with temp.open("w") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    temp.replace(path)


def rng_state(device):
    return {"python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state(device) if device.type == "cuda" else None}


def restore_rng(state, device):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if device.type == "cuda" and state["cuda"] is not None:
        torch.cuda.set_rng_state(state["cuda"], device)


def resolve_resume(output: Path, resume: str | None):
    if resume is None:
        return None
    if resume == "auto":
        model_pointer = output / "latest_model.json"
        if model_pointer.exists():
            model_step = json.loads(model_pointer.read_text())["step"]
            full_step = json.loads((output / "latest.json").read_text())["step"] if (output / "latest.json").exists() else -1
            if model_step > full_step:
                raise ValueError("latest model-only checkpoint has no Adam/RNG state; choose an explicit full checkpoint for exact resume")
        pointer = json.loads((output / "latest.json").read_text())
        path = (output / "checkpoints" / pointer["checkpoint"]).resolve()
    else:
        path = Path(resume).resolve()
    if not path.is_relative_to((output / "checkpoints").resolve()):
        raise ValueError("resume must point to a checkpoint in this output directory")
    if not (path / "COMPLETE").is_file():
        raise ValueError("refusing to resume an incomplete checkpoint")
    return path


def _cpu_state(model):
    # Qwen may tie embeddings, and the post-ViT target aliases an existing module.
    # Preserve aliases during GPU->CPU copying instead of duplicating full VLM weights.
    cache, result = {}, {}
    for name, value in model.state_dict().items():
        key = (value.device, value.dtype, value.untyped_storage().data_ptr(), value.storage_offset(),
               tuple(value.shape), tuple(value.stride()))
        if key not in cache:
            cache[key] = value.detach().to(device="cpu", copy=True)
        result[name] = cache[key]
    return result


def save_checkpoint(accelerator, model, optimizers, state, config, contract, *, best=False):
    root = Path(config.output_dir) / "checkpoints"
    name = f"step-{state.step:08d}"

    def begin():
        root.mkdir(parents=True, exist_ok=True)
        if (root / name).exists():
            raise FileExistsError(root / name)
        temp = root / f".{name}.{uuid.uuid4().hex}.incomplete"
        temp.mkdir()
        return str(temp)

    temp = Path(on_main(accelerator, begin))
    error = None
    try:
        torch.save(rng_state(accelerator.device), temp / f"rng-{accelerator.process_index:05d}.pt")
    except Exception as exc:
        error = repr(exc)
    errors = [None] * accelerator.num_processes
    if accelerator.num_processes > 1:
        dist.all_gather_object(errors, error)
    else:
        errors[0] = error
    if any(errors):
        raise RuntimeError(f"rank RNG save failed: {errors}")
    for optimizer in optimizers.values():
        if hasattr(optimizer, "consolidate_state_dict"):
            optimizer.consolidate_state_dict(to=0)  # ALL ranks; state is collected onto rank 0 CPU

    def commit():
        bare = accelerator.unwrap_model(model)
        torch.save(_cpu_state(bare), temp / "model.pt")
        for key, optimizer in optimizers.items():
            torch.save(optimizer.state_dict(), temp / f"optimizer-{key}.pt")
        atomic_json(temp / "trainer.json", {**asdict(state), "best_validation":
                    state.best_validation if np.isfinite(state.best_validation) else None})
        atomic_json(temp / "metadata.json", dict(format="betanav_checkpoint_v1", contract=contract,
                    world_size=accelerator.num_processes, config=asdict(config),
                    beta_config=asdict(bare.config), run_id=state.run_id))
        (temp / "COMPLETE").write_text("betanav_checkpoint_v1\n")
        destination = root / name
        temp.replace(destination)
        output = Path(config.output_dir)
        atomic_json(output / "latest.json", dict(checkpoint=name, step=state.step))
        if best:
            atomic_json(output / "best.json", dict(checkpoint=name, metric=state.best_validation))
        protected = {name}
        if (output / "best.json").exists():
            protected.add(json.loads((output / "best.json").read_text())["checkpoint"])
        committed = sorted(p for p in root.glob("step-*") if (p / "COMPLETE").is_file())
        protected.update(p.name for p in committed[-config.keep_last:])
        for old in committed:
            meta = json.loads((old / "metadata.json").read_text())
            if old.name not in protected and meta.get("run_id") == state.run_id:
                shutil.rmtree(old)
        return str(destination)

    return on_main(accelerator, commit)


def load_checkpoint(path, accelerator, model, optimizers, contract):
    meta = json.loads((path / "metadata.json").read_text())
    if meta.get("format") not in {"betanav_checkpoint_v1", "betanav_checkpoint_v2"} or not resume_contracts_match(meta["contract"], contract):
        raise ValueError("resume configuration/data/model contract differs from checkpoint")
    if meta["world_size"] != accelerator.num_processes:
        raise ValueError("exact data/RNG resume requires the original world size")
    bare = accelerator.unwrap_model(model)
    bare.load_state_dict(torch.load(path / "model.pt", map_location="cpu", weights_only=True), strict=True)
    if meta["format"] == "betanav_checkpoint_v2":
        from lightnav.training.full_checkpoint import load_optimizer_shards

        rng = load_optimizer_shards(path, accelerator, bare, optimizers)
    else:
        for key, optimizer in optimizers.items():
            # Full state is read per rank and the ZeRO optimizer retains its own partition.
            optimizer.load_state_dict(torch.load(path / f"optimizer-{key}.pt", map_location="cpu", weights_only=True))
        # Locally generated trusted checkpoints contain Python/NumPy RNG state.
        rng = torch.load(path / f"rng-{accelerator.process_index:05d}.pt", map_location="cpu", weights_only=False)
    raw = json.loads((path / "trainer.json").read_text())
    if raw["best_validation"] is None:
        raw["best_validation"] = float("inf")
    state = TrainState(**raw)
    restore_rng(rng, accelerator.device)
    return state
