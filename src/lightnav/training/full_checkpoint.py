"""Background full checkpoints with rank-local Adam/RNG shards and bounded CPU snapshots."""

from __future__ import annotations

import copy
import json
import os
import shutil
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from torch.distributed.optim import ZeroRedundancyOptimizer

from lightnav.training.checkpoint import _cpu_state, atomic_json, on_main, rng_state


def _cpu_tree(value):
    if isinstance(value, torch.Tensor):
        return value.detach().to(device="cpu", copy=True)
    if isinstance(value, dict):
        return {k: _cpu_tree(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_cpu_tree(v) for v in value]
    if isinstance(value, tuple):
        return tuple(_cpu_tree(v) for v in value)
    return copy.deepcopy(value)


def _optimizer_parts(optimizer, names):
    sharded = isinstance(optimizer, ZeroRedundancyOptimizer)
    local = optimizer.optim if sharded else optimizer
    signature = [[names[id(p)] for p in group["params"]] for group in local.param_groups]
    return local, "zero1" if sharded else "replicated", signature


def capture_optimizer_shards(model, optimizers, rank):
    names = {id(p): name for name, p in model.named_parameters()}
    result = {}
    for key, optimizer in optimizers.items():
        local, kind, signature = _optimizer_parts(optimizer, names)
        if kind == "replicated" and rank:
            continue
        result[key] = dict(kind=kind, param_names=signature, state_dict=_cpu_tree(local.state_dict()),
                           global_param_groups=_cpu_tree([{k: v for k, v in g.items() if k != "params"}
                                                          for g in optimizer.param_groups]))
    return result


def load_optimizer_shards(path, accelerator, model, optimizers):
    # Trusted local training state includes Python/NumPy RNG objects, like the v1 RNG file.
    payload = torch.load(path / f"rank-{accelerator.process_index:05d}.pt", map_location="cpu", weights_only=False)
    if payload["rank"] != accelerator.process_index or payload["world_size"] != accelerator.num_processes:
        raise ValueError("optimizer shard rank/world size mismatch")
    names = {id(p): name for name, p in model.named_parameters()}
    primary, restores = None, []
    for key, optimizer in optimizers.items():
        local, kind, signature = _optimizer_parts(optimizer, names)
        source = payload
        if kind == "replicated" and accelerator.process_index:
            if primary is None:
                primary = torch.load(path / "rank-00000.pt", map_location="cpu", weights_only=False)
            source = primary
        saved = source["optimizers"][key]
        if saved["kind"] != kind or saved["param_names"] != signature:
            raise ValueError(f"optimizer partition/parameter order differs for {key}")
        if len(saved["global_param_groups"]) != len(optimizer.param_groups):
            raise ValueError(f"optimizer group count differs for {key}")
        restores.append((optimizer, local, saved))
    for optimizer, local, saved in restores:
        local.load_state_dict(saved["state_dict"])
        for group, options in zip(optimizer.param_groups, saved["global_param_groups"], strict=True):
            group.update(options)
        if local is not optimizer:
            # ZeRO.step uses its local optimizer; discard any stale wrapper state from a prior load.
            optimizer.state.clear()
    return payload["rng"]


def _write_rank(temp, rank, payload, weights, archive, metadata, snapshot_seconds):
    """CPU/file work only: no CUDA, distributed communication, or live optimizer references."""
    started = time.perf_counter()
    torch.save(payload, temp / f"rank-{rank:05d}.pt")
    if rank == 0:
        torch.save(weights, temp / "model.pt")
        if archive is not None:
            archive.mkdir(parents=True)
            try:
                os.link(temp / "model.pt", archive / "model.pt")
            except OSError:
                shutil.copyfile(temp / "model.pt", archive / "model.pt")
            atomic_json(archive / "metadata.json", {**metadata, "format": "betanav_model_v1",
                                                     "permanent": True, "training_state_complete": False})
            (archive / "COMPLETE").write_text("betanav_model_v1\n")
    timing = dict(snapshot_seconds=snapshot_seconds, write_seconds=time.perf_counter() - started)
    atomic_json(temp / f"timing-{rank:05d}.json", timing)
    return timing


def prune_full_checkpoints(output, run_id, keep_last):
    """Exactly the newest N full states for this run; permanent model archives live elsewhere."""
    root = output / "checkpoints"
    committed = [p for p in sorted(root.glob("step-*")) if (p / "COMPLETE").is_file()
                 and json.loads((p / "metadata.json").read_text()).get("run_id") == run_id]
    removed = []
    for path in committed[:-keep_last]:
        shutil.rmtree(path)
        removed.append(path.name)
    best = output / "best.json"
    if best.exists() and json.loads(best.read_text())["checkpoint"] in removed:
        best.unlink()
    return removed


class AsyncFullCheckpointer:
    """One immutable CPU snapshot per rank; optimizer shards never gather onto rank zero."""

    def __init__(self, accelerator, config, contract):
        self.accelerator, self.config, self.contract = accelerator, config, contract
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="full-checkpoint")
        self.pending = None

    def _gather(self, value):
        rows = [None] * self.accelerator.num_processes
        if len(rows) > 1:
            dist.all_gather_object(rows, value)
        else:
            rows[0] = value
        return rows

    def submit(self, model, optimizers, state, *, permanent=False):
        self.poll(wait=True)
        output = Path(self.config.output_dir)
        name = f"step-{state.step:08d}"
        destination = output / "checkpoints" / name
        model_destination = output / "model_checkpoints" / name

        def begin():
            destination.parent.mkdir(parents=True, exist_ok=True)
            if destination.exists() or (permanent and model_destination.exists()):
                raise FileExistsError(f"checkpoint step already exists: {name}")
            nonce = uuid.uuid4().hex
            temp = destination.with_name(f".{name}.{nonce}.incomplete")
            temp.mkdir()
            archive = model_destination.with_name(f".{name}.{nonce}.incomplete") if permanent else None
            return str(temp), str(archive) if archive else None

        temp, archive = on_main(self.accelerator, begin)
        temp, archive = Path(temp), Path(archive) if archive else None
        started, error = time.perf_counter(), None
        weights = payload = metadata = None
        try:
            bare = self.accelerator.unwrap_model(model)
            payload = dict(rank=self.accelerator.process_index, world_size=self.accelerator.num_processes,
                           rng=_cpu_tree(rng_state(self.accelerator.device)),
                           optimizers=capture_optimizer_shards(bare, optimizers, self.accelerator.process_index))
            if self.accelerator.is_main_process:
                weights = _cpu_state(bare)
            metadata = dict(format="betanav_checkpoint_v2", optimizer_storage="rank_shards",
                            training_state_complete=True, step=state.step, epoch=state.epoch,
                            samples_seen=state.samples_seen, run_id=state.run_id,
                            contract=self.contract, world_size=self.accelerator.num_processes,
                            config=asdict(self.config), beta_config=asdict(bare.config))
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
        captures = self._gather(dict(error=error, snapshot_seconds=time.perf_counter() - started))
        if any(row["error"] for row in captures):
            raise RuntimeError(f"checkpoint snapshot failed: {captures}")
        raw_state = {**asdict(state), "best_validation": state.best_validation if np.isfinite(state.best_validation) else None}
        future, error = None, None
        try:
            future = self.executor.submit(_write_rank, temp, self.accelerator.process_index, payload, weights,
                                          archive, metadata, captures[self.accelerator.process_index]["snapshot_seconds"])
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
        launches = self._gather(error)
        if any(launches):
            raise RuntimeError(f"checkpoint writer launch failed: {launches}")
        self.pending = dict(future=future, temp=temp, destination=destination, archive=archive,
                            model_destination=model_destination, metadata=metadata, state=raw_state)
        if self.accelerator.is_main_process:
            print(json.dumps(dict(event="full_checkpoint_queued", step=state.step, permanent_model=permanent,
                                  snapshot_seconds=max(r["snapshot_seconds"] for r in captures))), flush=True)
        return str(destination)

    def poll(self, *, wait=False):
        if self.pending is None:
            return None
        future = self.pending["future"]
        done, error, timing = False, None, None
        if wait or future.done():
            try:
                timing = future.result()
                done = True
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"
        rows = self._gather(dict(done=done, error=error, timing=timing))
        if any(r["error"] for r in rows):
            raise RuntimeError(f"background full checkpoint failed: {rows}")
        if not all(r["done"] for r in rows):
            return None

        def commit():
            pending = self.pending
            temp, destination, metadata = pending["temp"], pending["destination"], pending["metadata"]
            atomic_json(temp / "metadata.json", metadata)
            atomic_json(temp / "trainer.json", pending["state"])
            (temp / "COMPLETE").write_text("betanav_checkpoint_v2\n")
            temp.replace(destination)
            output = Path(self.config.output_dir)
            if pending["archive"] is not None:
                pending["archive"].replace(pending["model_destination"])
                atomic_json(output / "latest_model.json", dict(checkpoint=destination.name, step=metadata["step"],
                                                              format="betanav_model_v1"))
            atomic_json(output / "latest.json", dict(checkpoint=destination.name, step=metadata["step"]))
            removed = prune_full_checkpoints(output, metadata["run_id"], self.config.keep_last)
            return dict(path=str(destination), step=metadata["step"], pruned=removed,
                        permanent_model=pending["archive"] is not None,
                        write_seconds=max(r["timing"]["write_seconds"] for r in rows))

        result = on_main(self.accelerator, commit)
        self.pending = None
        if self.accelerator.is_main_process:
            print(json.dumps(dict(event="full_checkpoint_committed", **result)), flush=True)
        return result

    def close(self):
        self.executor.shutdown(wait=True)
