"""Frozen native-prefix KV, FM-30 training and checkpoint integration on tiny Qwen."""

from __future__ import annotations

import copy
import json
from dataclasses import asdict, replace
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from test_beta_model import batch as old_batch_fixture  # noqa: F401
from test_beta_model import model as old_model_fixture  # noqa: F401

from lightnav.inference.beta_model import BetaModel, BetaModelConfig, BetaTrainBatch
from lightnav.inference.prefix_kv import PrefixKVContext


@pytest.fixture(autouse=True, scope="module")
def cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


@pytest.fixture
def model(request):
    old = request.getfixturevalue("old_model_fixture")
    return BetaModel(old.vlm, replace(
        old.config, training_mode="fm_only", horizon=30, expert_architecture="prefix_kv_v1",
        prefix_layers=(1, 2), num_heads=4, flow_time_distribution="beta", flow_time_beta=1.5,
    ))


@pytest.fixture
def batch(request):
    old = request.getfixturevalue("old_batch_fixture")
    return replace(
        old, model_inputs={k: v[:, :3] for k, v in old.model_inputs.items()},
        labels=torch.full_like(old.labels[:, :3], -100),
        fm_clean_se2_diff=torch.randn(2, 30, 3) * .1,
        fm_valid_mask=torch.tensor([[True] * 30, [True] * 17 + [False] * 13]),
    )


def test_thirty_step_training_freezes_vlm_and_uses_only_fm(model, batch, monkeypatch):
    frozen = {n: p.detach().clone() for n, p in model.vlm.named_parameters()}
    seen = []
    handle = model.vlm.model.register_forward_pre_hook(
        lambda module, args: seen.append((module.training, torch.is_grad_enabled())))

    def forbidden(*args, **kwargs):
        raise AssertionError("FM-only training must not call the LM head")

    monkeypatch.setattr(model.vlm.lm_head, "forward", forbidden)
    optimizer = torch.optim.AdamW(model.optimizer_parameters()["fm"], lr=1e-3)
    before = model.expert.action_out.weight.detach().clone()
    noise = torch.randn(2, 30, 3)
    tau = torch.tensor([.1, .8])
    out = model(batch, noise=noise, flow_time=tau)
    handle.remove()
    assert seen == [(False, False)]
    assert out.velocity.shape == (2, 30, 3)
    assert isinstance(out.context, PrefixKVContext)
    assert out.context.layer_indices == (1, 2)
    assert all(not t.requires_grad for kv in out.context.layer_kv for t in kv)
    assert out.ce_token_count == 0 and out.fm_step_count == 47
    assert not model.optimizer_parameters()["vlm"]
    mask = batch.fm_valid_mask[..., None]
    clean = model.normalizer.normalize(torch.where(mask, batch.fm_clean_se2_diff, 0))
    expected = torch.where(mask, (out.velocity.float() - (clean - noise)).square(), 0).sum() / (3 * 47)
    torch.testing.assert_close(out.loss, expected)
    out.loss.backward()
    assert model.expert.blocks[0].attention_norm.modulation[-1].weight.grad.abs().sum() > 0
    optimizer.step()
    assert not torch.equal(before, model.expert.action_out.weight)
    # Zero-initialized modulation learns first, then opens the time/attention path.
    optimizer.zero_grad(set_to_none=True)
    model(batch, noise=noise, flow_time=tau).loss.backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.expert.time_mlp.parameters())
    assert all(block.attention.q_proj.weight.grad.abs().sum() > 0 for block in model.expert.blocks)
    assert all(not p.requires_grad and p.grad is None and torch.equal(p, frozen[n])
               for n, p in model.vlm.named_parameters())


def test_gt_mask_changes_supervision_without_changing_action_context(model, batch):
    noise, tau = torch.randn(2, 30, 3), torch.tensor([.2, .6])
    first = model(batch, noise=noise, flow_time=tau)
    clean = batch.fm_clean_se2_diff.clone()
    clean[~batch.fm_valid_mask] = float("nan")
    second = model(replace(batch, fm_clean_se2_diff=clean), noise=noise, flow_time=tau)
    torch.testing.assert_close(first.loss, second.loss, rtol=0, atol=0)
    torch.testing.assert_close(first.velocity, second.velocity, rtol=0, atol=0)
    assert first.context.valid.shape == batch.model_inputs["input_ids"].shape
    with pytest.raises(ValueError, match="prompt-only"):
        model(replace(batch, answer_start=torch.tensor([2, 2])))


def test_prefix_cached_once_for_sampling_and_empty_gt_keeps_ddp_hooks(model, batch, monkeypatch):
    empty = model(replace(batch, fm_valid_mask=torch.zeros_like(batch.fm_valid_mask)))
    assert empty.fm_step_count == 0 and empty.context is None and empty.velocity is None
    empty.loss.backward()
    assert all(p.grad is not None and not p.grad.any() for p in model.expert.parameters())
    model.eval()
    context = model.encode_prefix_kv(batch.model_inputs)
    snapshots = [(k.clone(), v.clone()) for k, v in context.layer_kv]

    def forbidden(*args, **kwargs):
        raise AssertionError("Euler integration must reuse the already encoded prefix")

    monkeypatch.setattr(model.vlm.model, "forward", forbidden)
    deltas, waypoints = model.sample_actions(context, num_steps=3, noise=torch.randn(2, 30, 3))
    assert deltas.shape == waypoints.shape == (2, 30, 3)
    assert torch.isfinite(deltas).all() and torch.isfinite(waypoints).all()
    for actual, expected in zip(context.layer_kv, snapshots, strict=True):
        for a, b in zip(actual, expected, strict=True):
            torch.testing.assert_close(a, b, rtol=0, atol=0)


def test_native_attention_config_is_saved_and_mismatch_is_rejected(model):
    saved = json.loads(json.dumps(asdict(model.config)))
    assert saved["prefix_kv_config"]["head_dim"] == 8
    assert saved["prefix_kv_config"]["num_key_value_heads"] == 2
    assert saved["prefix_layers"] == [1, 2]
    changed = copy.deepcopy(saved)
    changed["prefix_kv_config"]["rope_parameters"]["rope_theta"] = 123.0
    with pytest.raises(ValueError, match="attention/RoPE"):
        BetaModel(model.vlm, BetaModelConfig(**changed))
    with pytest.raises(ValueError, match="query heads"):
        BetaModel(model.vlm, replace(model.config, num_heads=2))
    with pytest.raises(ValueError, match="decoder layers"):
        BetaModel(model.vlm, replace(model.config, prefix_layers=(1, 3)))


def test_full_checkpoint_restores_prefix_architecture_and_adam(model, batch, tmp_path):
    pytest.importorskip("accelerate")
    from lightnav.training.checkpoint import TrainState, load_checkpoint, save_checkpoint
    from lightnav.training.config import TrainConfig
    from lightnav.training.runtime import make_optimizers

    cfg = TrainConfig(dataset_root="data", model_path="model", data_manifest="manifest",
                      output_dir=str(tmp_path), training_mode="fm_only", fm_horizon=30,
                      expert_architecture="prefix_kv_v1", prefix_layers=[1, 2],
                      expert_num_layers=2, expert_num_heads=4, optimizer="adamw")
    accel = SimpleNamespace(num_processes=1, process_index=0, is_main_process=True,
                            device=torch.device("cpu"), unwrap_model=lambda m: m)
    optimizers = make_optimizers(model, cfg, 1)
    model(batch).loss.backward()
    optimizers["fm"].step()
    optimizers["fm"].zero_grad(set_to_none=True)
    contract = {"config": cfg.resume_contract()}
    path = Path(save_checkpoint(accel, model, optimizers, TrainState("prefix-test", step=1), cfg, contract))
    metadata = json.loads((path / "metadata.json").read_text())
    assert metadata["beta_config"]["expert_architecture"] == "prefix_kv_v1"
    fresh = type(model.vlm)(copy.deepcopy(model.vlm.config))
    restored = BetaModel(fresh, BetaModelConfig(**metadata["beta_config"]))
    restored_opts = make_optimizers(restored, cfg, 1)
    state = load_checkpoint(path, accel, restored, restored_opts, contract)
    assert state.step == 1 and restored_opts["fm"].state
    for key, value in model.state_dict().items():
        torch.testing.assert_close(restored.state_dict()[key], value, rtol=0, atol=0)
    model.eval()
    restored.eval()
    noise, tau = torch.randn(2, 30, 3), torch.tensor([.3, .7])
    torch.testing.assert_close(model(batch, noise=noise, flow_time=tau).velocity,
                               restored(batch, noise=noise, flow_time=tau).velocity, rtol=0, atol=0)


def test_video_pooling_native_mrope_and_bfloat16(model):
    from lightnav.processing import enable_post_vit_pool

    enable_post_vit_pool(model.vlm, spatial_factor=2, merge_size=2)
    model.vlm._post_vit_target = model.vlm.model
    ids = torch.tensor([[1, 59, 61, 62, 4]])
    inputs = {
        "input_ids": ids, "attention_mask": torch.ones_like(ids),
        "pixel_values_videos": torch.randn(16, 24),
        "video_grid_thw": torch.tensor([[1, 2, 2]]),
        "_original_video_grid_thw": torch.tensor([[1, 4, 4]]),
        "_post_vit_pool_factors": [2],
        "position_ids": torch.full((3, 1, 5), -999),
    }
    batch = BetaTrainBatch(inputs, torch.full_like(ids, -100), torch.tensor([5]),
                           torch.randn(1, 30, 3) * .1, torch.ones(1, 30, dtype=torch.bool))
    with torch.autocast("cpu", dtype=torch.bfloat16):
        out = model(batch)
    assert torch.isfinite(out.loss)
    assert out.context.action_cos.shape == (1, 30, 8)
    out.loss.backward()
    assert all(p.grad is None for p in model.vlm.parameters())
    assert model.vlm.model._post_vit_original_grid is None
    assert model.vlm.model._post_vit_pool_factors is None
