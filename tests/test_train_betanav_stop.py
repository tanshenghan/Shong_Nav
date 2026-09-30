"""Independent STOP accumulation, skipped optimizers, and exact asynchronous resume."""

from __future__ import annotations

import copy
import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from test_beta_stop_model import batch as stop_batch_fixture  # noqa: F401
from test_beta_stop_model import cpu_threads, old_batch_fixture, old_model_fixture  # noqa: F401

from lightnav.inference.beta_model import BetaModel
from lightnav.training.checkpoint import TrainState, load_checkpoint
from lightnav.training.config import TrainConfig, resume_contracts_match
from lightnav.training.full_checkpoint import AsyncFullCheckpointer
from lightnav.training.runtime import (
    backward_sums,
    clip_and_step,
    make_optimizers,
    normalize_gradients,
    validate_stop_warm_start_contract,
)


@pytest.fixture(params=["cross_attention_v1", "prefix_kv_v1"])
def model(request):
    base = request.getfixturevalue("old_model_fixture")
    prefix = request.param == "prefix_kv_v1"
    return BetaModel(base.vlm, replace(base.config, training_mode="fm_stop",
        expert_architecture=request.param, prefix_layers=(1, 2) if prefix else (),
        num_heads=4 if prefix else 2, stop_head_type="last_prefix_mlp_v1",
        stop_hidden_size=8, stop_positive_weight=3.0))


@pytest.fixture
def batch(request):
    return request.getfixturevalue("stop_batch_fixture")


@pytest.fixture
def config(tmp_path):
    return TrainConfig(dataset_root="data", model_path="model", data_manifest="manifest.json",
                       output_dir=str(tmp_path / "run"), training_mode="fm_stop",
                       stop_head_type="last_prefix_mlp_v1", optimizer="adamw", max_steps=4,
                       warmup_steps=1, lambda_stop=1.7)


@pytest.fixture
def accelerator():
    return SimpleNamespace(num_processes=1, process_index=0, is_main_process=True,
                           device=torch.device("cpu"), unwrap_model=lambda m: m,
                           backward=lambda loss: loss.backward())


def update(model, batch, config, accelerator, optimizers):
    out = model(batch, lambda_fm=config.lambda_fm, lambda_stop=config.lambda_stop)
    backward_sums(out, accelerator, config.lambda_fm, config.lambda_stop)
    counts = [out.ce_token_count, out.fm_step_count, out.stop_weight_sum]
    normalize_gradients(model.optimizer_parameters(), counts, 1)
    return clip_and_step(model.optimizer_parameters(), optimizers, counts, config)


@pytest.mark.parametrize("changes", [
    {"stop_head_type": "none"}, {"lambda_stop": 0.}, {"lambda_stop": float("nan")},
    {"stop_hidden_size": 0}, {"stop_hidden_size": True}, {"lr_stop": 0.}, {"clip_stop": -1.},
])
def test_invalid_stop_training_settings(config, changes):
    with pytest.raises(ValueError):
        replace(config, **changes).validate()


def test_stop_fields_preserve_old_contract_defaults_and_reject_mode_changes(config):
    old = replace(config, training_mode="fm_only", stop_head_type="none", lambda_stop=1.)
    fields = {"stop_head_type", "stop_hidden_size", "lambda_stop", "lr_stop", "clip_stop"}
    legacy = {"config": {k: v for k, v in old.resume_contract().items() if k not in fields}}
    assert resume_contracts_match(legacy, {"config": old.resume_contract()})
    assert not resume_contracts_match(legacy, {"config": config.resume_contract()})
    assert not resume_contracts_match({"config": config.resume_contract()},
                                     {"config": replace(config, lambda_stop=2.).resume_contract()})


def test_stop_accumulation_uses_own_weighted_denominator(model, batch, config, accelerator):
    reference = copy.deepcopy(model)
    terminal = replace(batch, fm_valid_mask=torch.zeros_like(batch.fm_valid_mask),
                       stop_target=torch.ones(2), stop_valid_mask=torch.tensor([False, True]))
    unlabeled = replace(batch, stop_valid_mask=torch.zeros(2, dtype=torch.bool))
    batches = [batch, terminal, unlabeled]
    noise, tau = torch.randn_like(batch.fm_clean_se2_diff), torch.tensor([.2, .7])
    outputs = [reference(b, noise=noise, flow_time=tau) for b in batches]
    fm_count = sum(o.fm_step_count for o in outputs)
    stop_weight = sum(o.stop_weight_sum for o in outputs)
    assert stop_weight == 7.0
    sum(config.lambda_fm * o.fm_loss * o.fm_step_count / fm_count
        + config.lambda_stop * o.stop_loss * o.stop_weight_sum / stop_weight for o in outputs).backward()
    for b in batches:
        backward_sums(model(b, noise=noise, flow_time=tau), accelerator, config.lambda_fm, config.lambda_stop)
    normalize_gradients(model.optimizer_parameters(), [0, fm_count, stop_weight], 1)
    for actual, expected in zip(model.parameters(), reference.parameters(), strict=True):
        if expected.grad is None:
            assert actual.grad is None
        else:
            torch.testing.assert_close(actual.grad, expected.grad, atol=3e-6, rtol=3e-5)


@pytest.mark.parametrize("branch", ["fm", "stop"])
def test_empty_branch_does_not_decay_or_advance_adam(model, batch, config, accelerator, branch):
    optimizers = make_optimizers(model, config, 1)
    assert set(optimizers) == {"fm", "stop"}
    update(model, batch, config, accelerator, optimizers)
    module = model.expert if branch == "fm" else model.stop_head
    before = copy.deepcopy(module.state_dict())
    adam = copy.deepcopy(optimizers[branch].state_dict())
    masked = replace(batch, **({"fm_valid_mask": torch.zeros_like(batch.fm_valid_mask)}
                               if branch == "fm" else {"stop_valid_mask": torch.zeros(2, dtype=torch.bool)}))
    norms = update(model, masked, config, accelerator, optimizers)
    assert norms[branch] == 0.
    for key, value in module.state_dict().items():
        torch.testing.assert_close(value, before[key], rtol=0, atol=0)
    for key, state in optimizers[branch].state_dict()["state"].items():
        for name, value in state.items():
            torch.testing.assert_close(value, adam["state"][key][name], rtol=0, atol=0)


def test_stop_async_checkpoint_restores_exact_next_update(model, batch, config, accelerator):
    optimizers = make_optimizers(model, config, 1)
    update(model, batch, config, accelerator, optimizers)
    state = TrainState("stop-test", step=1, stop_updates=1, fm_updates=1, batch_offset=3)
    contract = {"config": config.resume_contract(), "stop_supervision": {"version": "test", "positive_weight": 3.}}
    writer = AsyncFullCheckpointer(accelerator, config, contract)
    try:
        saved = Path(writer.submit(model, optimizers, state, permanent=True))
        writer.poll(wait=True)
    finally:
        writer.close()
    expected_random = torch.rand(3)
    update(model, batch, config, accelerator, optimizers)
    expected = copy.deepcopy(model.state_dict())
    restored = copy.deepcopy(model)
    restored_opts = make_optimizers(restored, config, 1)
    restored_state = load_checkpoint(saved, accelerator, restored, restored_opts, contract)
    assert restored_state.stop_updates == 1 and restored_state.batch_offset == 3
    torch.testing.assert_close(torch.rand(3), expected_random, rtol=0, atol=0)
    update(restored, batch, config, accelerator, restored_opts)
    for key, value in restored.state_dict().items():
        torch.testing.assert_close(value, expected[key], rtol=0, atol=0)
    archive = Path(config.output_dir) / "model_checkpoints" / saved.name
    metadata = json.loads((archive / "metadata.json").read_text())
    assert metadata["contract"]["stop_supervision"] == contract["stop_supervision"]
    assert metadata["beta_config"]["stop_head_type"] == "last_prefix_mlp_v1"
    assert any(k.startswith("stop_head.") for k in torch.load(archive / "model.pt", weights_only=True))
    with pytest.raises(ValueError):
        load_checkpoint(archive, accelerator, restored, restored_opts, contract)


def test_stop_warm_start_requires_same_backbone_processing():
    contract = dict(processing={"video_size": [280, 470]}, config_sha256="config", base_weight_files={"a": [1, 2]})
    validate_stop_warm_start_contract({"contract": contract}, contract)
    for key in contract:
        with pytest.raises(ValueError, match=key):
            validate_stop_warm_start_contract({"contract": {**contract, key: None}}, contract)
