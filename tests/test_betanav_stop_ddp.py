"""Real two-rank STOP/FM accumulation, weighted normalization and optimizer skips."""

from __future__ import annotations

import copy
from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch import nn

pytest.importorskip("accelerate")

from lightnav.inference.beta_model import BetaModel, BetaModelConfig, BetaTrainBatch  # noqa: E402
from lightnav.training.config import TrainConfig  # noqa: E402
from lightnav.training.runtime import (  # noqa: E402
    backward_sums,
    clip_and_step,
    make_optimizers,
    normalize_gradients,
)


def _model(architecture):
    from transformers import Qwen3VLConfig, Qwen3VLForConditionalGeneration

    config = Qwen3VLConfig(
        text_config=dict(
            vocab_size=64, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
            num_attention_heads=4, num_key_value_heads=2, head_dim=8, pad_token_id=0,
            rope_parameters={"rope_type": "default", "rope_theta": 10000.,
                             "mrope_section": [1, 1, 2]},
        ),
        vision_config=dict(
            depth=2, hidden_size=16, intermediate_size=32, num_heads=2,
            patch_size=2, temporal_patch_size=2, spatial_merge_size=2,
            out_hidden_size=32, num_position_embeddings=16, deepstack_visual_indexes=[0, 1],
        ),
        image_token_id=60, video_token_id=61, vision_start_token_id=59,
        vision_end_token_id=62, tie_word_embeddings=True,
    )
    return BetaModel(Qwen3VLForConditionalGeneration(config), BetaModelConfig(
        action_scale=(.25, .25, .5), horizon=2, hidden_size=16, num_layers=2,
        num_heads=4, intermediate_size=32, training_mode="fm_stop",
        expert_architecture=architecture,
        prefix_layers=(1, 2) if architecture == "prefix_kv_v1" else (),
        stop_head_type="last_prefix_mlp_v1", stop_hidden_size=8, stop_positive_weight=2.5,
    ))


def _batch(rank, micro, phase):
    """Phase 0: joint; 1: STOP only; 2: FM only; 3: entirely invalid."""
    ids = torch.tensor([[1, 2 + rank, 3 + micro], [0, 7 + micro, 9 + rank]])
    stop_targets = torch.tensor([0., 1.]) if rank == 0 else torch.tensor([1., float(micro == 0)])
    stop_valid = torch.tensor([[True, True], [False, False]][micro]) if rank == 0 else torch.tensor(
        [[True, False], [True, True]][micro],
    )
    fm_valid = torch.tensor(
        [[[True, True], [True, False]], [[False, False], [False, False]]][micro]
        if rank == 0 else
        [[[False, False], [False, False]], [[True, False], [False, False]]][micro],
    )
    if phase in {1, 3}:
        fm_valid.zero_()
    if phase in {2, 3}:
        stop_valid.zero_()
    clean = torch.arange(12).reshape(2, 2, 3).float() * .01 + (rank + micro) * .05
    clean[~fm_valid] = float("nan")
    stop_targets[~stop_valid] = float("nan")
    return BetaTrainBatch(
        {"input_ids": ids, "attention_mask": ids != 0},
        torch.full_like(ids, -100), torch.tensor([3, 3]), clean, fm_valid,
        stop_target=stop_targets, stop_valid_mask=stop_valid,
    )


def _flow_inputs(rank, micro, phase):
    generator = torch.Generator().manual_seed(73 + 100 * phase + 10 * rank + micro)
    return dict(noise=torch.randn(2, 2, 3, generator=generator), flow_time=torch.tensor([.2, .7]))


def _local_optimizer_state(optimizer):
    # ZeRO keeps the local Adam shard in .optim; the ordinary optimizer is its
    # own local state. Inspecting it does not consolidate or communicate state.
    return copy.deepcopy(getattr(optimizer, "optim", optimizer).state_dict())


def _worker(rank, directory, architecture):
    torch.set_num_threads(1)
    dist.init_process_group("gloo", init_method=f"file://{directory}/rendezvous", rank=rank, world_size=2)
    try:
        torch.manual_seed(71)
        bare = _model(architecture)
        reference = copy.deepcopy(bare)
        model = nn.parallel.DistributedDataParallel(bare, find_unused_parameters=True, broadcast_buffers=False)
        accelerator = SimpleNamespace(num_processes=2, backward=lambda loss: loss.backward())
        config = TrainConfig(
            dataset_root="data", model_path="model", data_manifest="manifest", output_dir=directory,
            training_mode="fm_stop", stop_head_type="last_prefix_mlp_v1", optimizer="zero1",
            lambda_fm=.7, lambda_stop=1.3, weight_decay=.2,
            clip_vlm=100., clip_fm=100., clip_stop=100.,
        )
        optimizers = make_optimizers(bare, config, 2)
        reference_optimizers = make_optimizers(reference, config, 1)
        assert set(optimizers) == {"fm", "stop"}
        groups = bare.optimizer_parameters()
        reference_groups = reference.optimizer_parameters()
        for phase in range(4):
            before = {k: v.detach().clone() for k, v in bare.state_dict().items()}
            optimizer_before = {name: _local_optimizer_state(opt) for name, opt in optimizers.items()}
            totals = torch.zeros(3, dtype=torch.float64)
            for micro in range(2):
                with model.no_sync() if micro == 0 else nullcontext():
                    output = model(_batch(rank, micro, phase), **_flow_inputs(rank, micro, phase))
                    backward_sums(output, accelerator, config.lambda_fm, config.lambda_stop)
                    totals += torch.tensor(
                        [output.ce_token_count, output.fm_step_count, output.stop_weight_sum], dtype=totals.dtype,
                    )
            dist.all_reduce(totals)
            assert totals.tolist() == [0., 4. if phase in {0, 2} else 0., 9.5 if phase in {0, 1} else 0.]
            normalize_gradients(groups, totals.tolist(), 2)

            # Independent global-reference objective, with actual fractional
            # positive weights, not the mean of rank/microbatch mean losses.
            loss = torch.zeros((), requires_grad=True)
            for source_rank in range(2):
                for micro in range(2):
                    output = reference(_batch(source_rank, micro, phase), **_flow_inputs(source_rank, micro, phase))
                    if totals[1]:
                        loss = loss + config.lambda_fm * output.fm_loss * output.fm_step_count / totals[1]
                    if totals[2]:
                        loss = loss + config.lambda_stop * output.stop_loss * output.stop_weight_sum / totals[2]
            loss.backward()
            for (name, actual), (_, expected) in zip(bare.named_parameters(), reference.named_parameters(), strict=True):
                if expected.grad is None:
                    assert actual.grad is None, name
                else:
                    torch.testing.assert_close(actual.grad, expected.grad, rtol=2e-5, atol=2e-6, msg=name)
            clip_and_step(groups, optimizers, totals.tolist(), config)
            clip_and_step(reference_groups, reference_optimizers, totals.tolist(), config)
            for name, value in bare.state_dict().items():
                torch.testing.assert_close(value, reference.state_dict()[name], rtol=2e-5, atol=2e-6, msg=name)
                if name.startswith("vlm."):
                    torch.testing.assert_close(value, before[name], rtol=0, atol=0)

            # STOP-only, FM-only and entirely empty windows preserve the other
            # branch's parameters AND Adam moments, including weight decay.
            inactive = {"fm"} if phase == 1 else {"stop"} if phase == 2 else {"fm", "stop"} if phase == 3 else set()
            for group in inactive:
                prefix = "expert." if group == "fm" else "stop_head."
                for name, value in bare.state_dict().items():
                    if name.startswith(prefix):
                        torch.testing.assert_close(value, before[name], rtol=0, atol=0)
                torch.testing.assert_close(
                    _local_optimizer_state(optimizers[group]), optimizer_before[group], rtol=0, atol=0,
                )
            for parameter in bare.parameters():
                other = parameter.detach().clone()
                dist.broadcast(other, src=0)
                torch.testing.assert_close(parameter, other, atol=0, rtol=0)
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("architecture", ["cross_attention_v1", "prefix_kv_v1"])
def test_stop_weighted_ddp_accumulation_and_independent_zero_optimizer_skips(tmp_path, architecture):
    mp.spawn(_worker, args=(str(tmp_path), architecture), nprocs=2, join=True)
