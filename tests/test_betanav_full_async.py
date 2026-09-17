"""Rolling background checkpoints must restore Adam/RNG exactly and retain model milestones."""

import copy
import json
import random
import threading
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch import nn

from lightnav.training.checkpoint import TrainState, load_checkpoint
from lightnav.training.config import TrainConfig
from lightnav.training.full_checkpoint import AsyncFullCheckpointer
from lightnav.training.runtime import make_optimizers


class Branches(nn.Module):
    def __init__(self):
        super().__init__()
        self.vlm = nn.Linear(4, 3)
        self.expert = nn.Linear(4, 3)
        self.config = TrainState("small")

    def forward(self, x):
        return self.vlm(x).square().mean() + 2 * self.expert(x).square().mean()


def _update(model, optimizers, rank, device):
    for opt in optimizers.values():
        opt.zero_grad(set_to_none=True)
    x = torch.randn(3, 4, device=device) + rank * .1 + random.random() + np.random.random()
    model(x).backward()
    for opt in optimizers.values():
        opt.step()


def _random(device):
    return random.random(), np.random.random(), torch.rand(2), torch.rand(2, device=device).cpu()


def _worker(rank, directory, cuda, optimizer_kind, fm_only):
    from lightnav.training import full_checkpoint as module

    torch.set_num_threads(1)
    device = torch.device(f"cuda:{rank}" if cuda else "cpu")
    if cuda:
        torch.cuda.set_device(device)
    dist.init_process_group("nccl" if cuda else "gloo", init_method=f"file://{directory}/rendezvous", rank=rank, world_size=2)
    release, writer = threading.Event(), None
    try:
        torch.manual_seed(42)
        bare = Branches().to(device)
        if fm_only:
            bare.vlm.requires_grad_(False)
        frozen_before = copy.deepcopy(bare.vlm.state_dict())
        model = nn.parallel.DistributedDataParallel(bare, device_ids=[rank] if cuda else None)
        accel = SimpleNamespace(num_processes=2, process_index=rank, is_main_process=rank == 0,
                                device=device, unwrap_model=lambda m: m.module)
        cfg = TrainConfig(dataset_root="data", model_path="model", data_manifest="data.json", output_dir=directory,
                          optimizer=optimizer_kind, checkpoint_mode="dual", save_best=False, keep_last=3)
        opts = make_optimizers(bare, cfg, 2)
        assert set(opts) == ({"fm"} if fm_only else {"vlm", "fm"})
        random.seed(42 + rank)
        np.random.seed(42 + rank)
        torch.manual_seed(42 + rank)
        _update(model, opts, rank, device)
        original = module._write_rank

        def delayed(*args):
            assert release.wait(30), "writer not released after training updates"
            return original(*args)

        def forbidden_consolidation(*_args, **_kwargs):
            raise AssertionError("full async checkpoint must never consolidate optimizer states")

        module.ZeroRedundancyOptimizer.consolidate_state_dict = forbidden_consolidation
        module._write_rank = delayed
        writer = AsyncFullCheckpointer(accel, cfg, {})
        path = Path(writer.submit(model, opts, TrainState("rolling", step=1000, epoch=1, batch_offset=7, samples_seen=64)))
        expected_rng = _random(device)
        _update(model, opts, rank, device)
        expected_next = copy.deepcopy(bare.state_dict())
        _update(model, opts, rank, device)
        assert writer.poll() is None
        assert not path.exists()
        dist.barrier()
        release.set()
        writer.poll(wait=True)
        state = load_checkpoint(path, accel, model, opts, {})
        assert (state.step, state.epoch, state.batch_offset, state.samples_seen) == (1000, 1, 7, 64)
        actual_rng = _random(device)
        assert actual_rng[:2] == expected_rng[:2]
        for actual, expected in zip(actual_rng[2:], expected_rng[2:], strict=True):
            torch.testing.assert_close(actual, expected, atol=0, rtol=0)
        _update(model, opts, rank, device)
        for name, value in bare.state_dict().items():
            torch.testing.assert_close(value, expected_next[name], atol=0, rtol=0)
        if fm_only:
            for name, value in bare.vlm.state_dict().items():
                torch.testing.assert_close(value, frozen_before[name], atol=0, rtol=0)

        module._write_rank = original
        for step in (2000, 3000, 4000, 5000, 6000):
            writer.submit(model, opts, TrainState("rolling", step=step), permanent=step % 3000 == 0)
            writer.poll(wait=True)
            if step == 3000 and rank == 0:
                (Path(directory) / "best.json").write_text(json.dumps({"checkpoint": "step-00003000"}))
            dist.barrier()
        root = Path(directory)
        assert {p.name for p in (root / "checkpoints").glob("step-*")} == {"step-00004000", "step-00005000", "step-00006000"}
        assert not (root / "best.json").exists()
        for step in (3000, 6000):
            archive = root / "model_checkpoints" / f"step-{step:08d}"
            assert {p.name for p in archive.iterdir()} == {"model.pt", "metadata.json", "COMPLETE"}
            torch.testing.assert_close(torch.load(archive / "model.pt", weights_only=True)["vlm.weight"],
                                       bare.vlm.weight.detach().cpu(), atol=0, rtol=0)
        payload = torch.load(root / "checkpoints/step-00006000" / f"rank-{rank:05d}.pt", weights_only=False)
        assert payload["rank"] == rank and "rng" in payload
        assert (root / "checkpoints/step-00006000/model.pt").stat().st_size > 0

        def fail_one_rank(*args):
            if args[1] == 1:
                raise OSError("rank-one disk error")
            return original(*args)

        module._write_rank = fail_one_rank
        writer.submit(model, opts, TrainState("rolling", step=7000))
        with pytest.raises(RuntimeError, match="rank-one disk error"):
            writer.poll(wait=True)
        assert json.loads((root / "latest.json").read_text())["step"] == 6000
        assert not (root / "checkpoints/step-00007000/COMPLETE").exists()
    finally:
        release.set()
        if writer is not None:
            writer.close()
        dist.destroy_process_group()


@pytest.mark.parametrize("cuda,optimizer_kind,fm_only", [
    (False, "zero1", False), (False, "adamw", False), (False, "zero1", True),
    pytest.param(True, "zero1", False, marks=pytest.mark.gpu),
])
def test_full_async_resume_retention_and_failures(tmp_path, cuda, optimizer_kind, fm_only):
    if cuda and torch.cuda.device_count() < 2:
        pytest.skip("requires two CUDA GPUs")
    mp.spawn(_worker, args=(str(tmp_path), cuda, optimizer_kind, fm_only), nprocs=2, join=True)
