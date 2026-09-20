"""Actual DDP updates overlap a blocked model writer, including CUDA snapshot capture."""

import threading
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch import nn

from lightnav.training.checkpoint import TrainState
from lightnav.training.config import TrainConfig
from lightnav.training.model_checkpoint import AsyncModelCheckpointer


def _worker(rank, directory, cuda):
    from lightnav.training import model_checkpoint as module

    torch.set_num_threads(1)
    device = torch.device(f"cuda:{rank}" if cuda else "cpu")
    if cuda:
        torch.cuda.set_device(device)
    dist.init_process_group("nccl" if cuda else "gloo", init_method=f"file://{directory}/rendezvous",
                            rank=rank, world_size=2)
    writer = None
    release = threading.Event()
    try:
        torch.manual_seed(42)
        bare = nn.Linear(32, 32).to(device)
        bare.config = TrainState("small-model")
        model = nn.parallel.DistributedDataParallel(bare, device_ids=[rank] if cuda else None)
        optimizer = torch.optim.SGD(model.parameters(), lr=.1)
        accelerator = SimpleNamespace(num_processes=2, process_index=rank, is_main_process=rank == 0,
                                      device=device, unwrap_model=lambda m: m.module)
        cfg = TrainConfig(dataset_root="data", model_path="model", data_manifest="data.json", output_dir=directory,
                          checkpoint_mode="model_only", save_best=False)
        original = module._write_snapshot

        def delayed(*args):
            assert all(t.device.type == "cpu" for t in args[2].values())
            assert release.wait(30), "writer was not released after DDP updates"
            return original(*args)

        module._write_snapshot = delayed
        writer = AsyncModelCheckpointer(accelerator, cfg, {})
        expected = {k: v.detach().cpu().clone() for k, v in bare.state_dict().items()}
        path = Path(writer.submit(model, TrainState("ddp-async", step=3000)))
        for _ in range(3):
            optimizer.zero_grad()
            model(torch.ones(2, 32, device=device)).square().mean().backward()
            optimizer.step()
            assert writer.poll() is None
        assert not path.exists()
        dist.barrier()
        release.set()
        writer.poll(wait=True)
        actual = torch.load(path / "model.pt", map_location="cpu", weights_only=True)
        for name, value in expected.items():
            torch.testing.assert_close(actual[name], value, atol=0, rtol=0)
        assert not torch.equal(bare.weight.detach().cpu(), actual["weight"])

        def fail(*_args):
            raise OSError("distributed writer failure")

        module._write_snapshot = fail
        writer.submit(model, TrainState("ddp-async", step=6000))
        with pytest.raises(RuntimeError, match="distributed writer failure"):
            writer.poll(wait=True)
        assert not (Path(directory) / "model_checkpoints/step-00006000/COMPLETE").exists()
    finally:
        release.set()
        if writer is not None:
            writer.close()
        dist.destroy_process_group()


@pytest.mark.parametrize("cuda", [False, pytest.param(True, marks=pytest.mark.gpu)])
def test_ddp_training_overlaps_model_write_and_propagates_failure(tmp_path, cuda):
    if cuda and torch.cuda.device_count() < 2:
        pytest.skip("requires two CUDA GPUs")
    mp.spawn(_worker, args=(str(tmp_path), cuda), nprocs=2, join=True)
