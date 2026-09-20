"""Real two-rank Gloo/DDP/ZeRO test with unequal masks and an all-empty final FM microbatch."""

from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch import nn

pytest.importorskip("accelerate")

from lightnav.inference.beta_model import BetaTrainOutput  # noqa: E402
from lightnav.training.checkpoint import TrainState, load_checkpoint, save_checkpoint  # noqa: E402
from lightnav.training.config import TrainConfig  # noqa: E402
from lightnav.training.runtime import (  # noqa: E402
    backward_sums,
    clip_and_step,
    make_optimizers,
    normalize_gradients,
)


class TwoBranches(nn.Module):
    def __init__(self, fm_only=False):
        super().__init__()
        self.vlm = nn.Linear(2, 1)
        self.expert = nn.Linear(2, 1)
        self.config = TrainState("test-config")  # dataclass checkpoint metadata
        self.fm_only = fm_only
        if fm_only:
            self.vlm.requires_grad_(False)

    def forward(self, rank, micro):
        x = torch.tensor([[1., 2.], [3., 1.], [-1., 2.]]) + rank * .1
        nc = 0 if self.fm_only else [[3, 1], [1, 2]][rank][micro]
        nf = [[2, 0], [0, 0]][rank][micro]
        ce = self.vlm(x[:nc]).square().mean() if nc else torch.zeros((), requires_grad=True)
        fm = self.expert(x[:nf]).square().mean() if nf else torch.zeros((), requires_grad=True)
        if not nf and self.fm_only:
            fm = sum(p.reshape(-1)[0] * 0 for p in self.expert.parameters())
        return BetaTrainOutput(ce + fm, ce, fm, nc, nf, None, None)


def _worker(rank, directory, fm_only):
    torch.set_num_threads(1)
    dist.init_process_group("gloo", init_method=f"file://{directory}/rendezvous", rank=rank, world_size=2)
    try:
        torch.manual_seed(11)
        bare = TwoBranches(fm_only)
        reference = TwoBranches(fm_only)
        reference.load_state_dict(bare.state_dict())
        model = nn.parallel.DistributedDataParallel(bare, find_unused_parameters=True, broadcast_buffers=False)
        accelerator = SimpleNamespace(num_processes=2, process_index=rank, is_main_process=rank == 0,
                                      device=torch.device("cpu"), unwrap_model=lambda m: m.module,
                                      backward=lambda loss: loss.backward())
        cfg = TrainConfig(dataset_root="data", model_path="model", data_manifest="meta", output_dir=directory,
                          optimizer="zero1", clip_vlm=100., clip_fm=100.)
        optimizers = make_optimizers(bare, cfg, 2)
        groups = {"vlm": [p for p in bare.vlm.parameters() if p.requires_grad],
                  "fm": list(bare.expert.parameters())}
        assert ("vlm" in optimizers) != fm_only
        totals = torch.zeros(2, dtype=torch.long)
        for micro in range(2):
            with model.no_sync() if micro == 0 else nullcontext():
                out = model(rank, micro)
                backward_sums(out, accelerator, .7)
                totals += torch.tensor([out.ce_token_count, out.fm_step_count])
        dist.all_reduce(totals)
        normalize_gradients(groups, totals.tolist(), 2)
        all_outputs = [reference(r, m) for r in range(2) for m in range(2)]
        sum(o.ce_loss * o.ce_token_count / max(totals[0], 1) + .7 * o.fm_loss * o.fm_step_count / totals[1]
            for o in all_outputs).backward()
        for actual, expected in zip(bare.parameters(), reference.parameters(), strict=True):
            if expected.grad is None:
                assert actual.grad is None
            else:
                torch.testing.assert_close(actual.grad, expected.grad, rtol=1e-5, atol=1e-6)
        clip_and_step(groups, optimizers, totals.tolist(), cfg)
        path = Path(save_checkpoint(accelerator, model, optimizers, TrainState("ddp", step=1), cfg, {"test": 1}))
        expected = {k: v.clone() for k, v in bare.state_dict().items()}
        with torch.no_grad():
            for parameter in bare.parameters():
                parameter.add_(10.)
        load_checkpoint(path, accelerator, model, optimizers, {"test": 1})
        for key, value in bare.state_dict().items():
            torch.testing.assert_close(value, expected[key], rtol=0, atol=0)
        # A second update proves loaded ZeRO partitions contain usable Adam moments.
        out = model(rank, 0)
        backward_sums(out, accelerator, .7)
        counts = torch.tensor([out.ce_token_count, out.fm_step_count])
        dist.all_reduce(counts)
        normalize_gradients(groups, counts.tolist(), 2)
        clip_and_step(groups, optimizers, counts.tolist(), cfg)
        for parameter in bare.parameters():
            other = parameter.detach().clone()
            dist.broadcast(other, src=0)
            torch.testing.assert_close(parameter, other, atol=0, rtol=0)
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("fm_only", [False, True])
def test_two_rank_mask_weighting_zero_state_and_resume(tmp_path, fm_only):
    mp.spawn(_worker, args=(str(tmp_path), fm_only), nprocs=2, join=True)
