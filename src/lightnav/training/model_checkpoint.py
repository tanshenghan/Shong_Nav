"""Immutable model snapshots with bounded background I/O; no optimizer collectives."""

from __future__ import annotations

import json
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from pathlib import Path

import torch

from lightnav.training.checkpoint import _cpu_state, atomic_json, on_main


def _write_snapshot(temp, destination, weights, metadata, output):
    """Background thread owns CPU-only copies and never calls CUDA or distributed APIs."""
    started = time.perf_counter()
    torch.save(weights, temp / "model.pt")
    atomic_json(temp / "metadata.json", metadata)
    timing = dict(snapshot_seconds=metadata["snapshot_seconds"], write_seconds=time.perf_counter() - started)
    atomic_json(temp / "timing.json", timing)
    (temp / "COMPLETE").write_text("betanav_model_v1\n")
    temp.replace(destination)
    atomic_json(output / "latest_model.json", dict(checkpoint=destination.name, step=metadata["step"],
                                                  format="betanav_model_v1"))
    return dict(path=str(destination), step=metadata["step"], **timing)


class AsyncModelCheckpointer:
    """All ranks enter submit/poll; at most one CPU snapshot is resident per writer.

    Snapshot capture blocks updates to keep a single step's weights consistent.
    The disk write overlaps later updates. Completed exports are never pruned.
    If disk falls behind, the next snapshot waits for the previous write.
    """

    def __init__(self, accelerator, config, contract):
        self.accelerator, self.config, self.contract = accelerator, config, contract
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="model-checkpoint") if accelerator.is_main_process else None
        self.pending = None

    def poll(self, *, wait=False):
        def collect():
            if self.pending is None or (not wait and not self.pending.done()):
                return None
            result = self.pending.result()  # on_main propagates writer failures to all ranks
            self.pending = None
            return result

        result = on_main(self.accelerator, collect)
        if result and self.accelerator.is_main_process:
            print(json.dumps(dict(event="model_checkpoint_committed", **result)), flush=True)
        return result

    def submit(self, model, state):
        self.poll(wait=True)

        def capture():
            output = Path(self.config.output_dir)
            root = output / "model_checkpoints"
            root.mkdir(parents=True, exist_ok=True)
            destination = root / f"step-{state.step:08d}"
            if destination.exists():
                raise FileExistsError(destination)
            temp = root / f".{destination.name}.{uuid.uuid4().hex}.incomplete"
            temp.mkdir()
            started = time.perf_counter()
            bare = self.accelerator.unwrap_model(model)
            weights = _cpu_state(bare)  # synchronous independent copies, including on CPU
            seconds = time.perf_counter() - started
            metadata = dict(format="betanav_model_v1", permanent=True, training_state_complete=False,
                            step=state.step, epoch=state.epoch, samples_seen=state.samples_seen, run_id=state.run_id,
                            world_size=self.accelerator.num_processes, config=asdict(self.config), contract=self.contract,
                            beta_config=asdict(bare.config), snapshot_seconds=seconds)
            self.pending = self.executor.submit(_write_snapshot, temp, destination, weights, metadata, output)
            print(json.dumps(dict(event="model_checkpoint_queued", step=state.step, snapshot_seconds=seconds,
                                  path=str(destination))), flush=True)
            return str(destination)

        return on_main(self.accelerator, capture)

    def close(self):
        # No collectives here: exception cleanup may run after a peer's unrelated hard failure.
        if self.executor is not None:
            self.executor.shutdown(wait=True)
