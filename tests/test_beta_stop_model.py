"""STOP supervision and one-pass continuous inference on tiny CPU Qwen models."""

from __future__ import annotations

import copy
import json
from dataclasses import asdict, replace

import pytest
import torch
from test_beta_model import batch as old_batch_fixture  # noqa: F401
from test_beta_model import model as old_model_fixture  # noqa: F401
from torch.nn import functional as F

from lightnav.inference.beta_model import (
    BetaModel,
    BetaModelConfig,
    warm_start_fm_stop,
)
from lightnav.inference.prefix_kv import PrefixKVContext
from lightnav.inference.stop_head import last_prefix_features


@pytest.fixture(autouse=True, scope="module")
def cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


@pytest.fixture(params=["cross_attention_v1", "prefix_kv_v1"])
def model(request):
    old = request.getfixturevalue("old_model_fixture")
    prefix = request.param == "prefix_kv_v1"
    return BetaModel(old.vlm, replace(
        old.config, training_mode="fm_stop", expert_architecture=request.param,
        prefix_layers=(1, 2) if prefix else (), num_heads=4 if prefix else 2,
        stop_head_type="last_prefix_mlp_v1", stop_hidden_size=8, stop_positive_weight=3.0,
    ))


@pytest.fixture
def batch(request):
    old = request.getfixturevalue("old_batch_fixture")
    return replace(
        old, model_inputs={k: v[:, :3] for k, v in old.model_inputs.items()},
        labels=torch.full_like(old.labels[:, :3], -100),
        stop_target=torch.tensor([0., 1.]), stop_valid_mask=torch.ones(2, dtype=torch.bool),
    )


def test_single_frozen_forward_and_independent_weighted_stop_loss(model, batch, monkeypatch):
    model.train()
    calls = []
    handle = model.vlm.model.register_forward_pre_hook(
        lambda module, _args: calls.append((torch.is_grad_enabled(), module.training)),
    )

    def forbidden(*args, **kwargs):
        raise AssertionError("fm_stop must not invoke the RVQ language-model head")

    monkeypatch.setattr(model.vlm.lm_head, "forward", forbidden)
    out = model(batch, lambda_fm=2.0, lambda_stop=1.5)
    handle.remove()
    assert calls == [(False, False)]
    assert out.stop_valid_count == 2 and out.stop_positive_count == 1
    assert out.stop_weight_sum == 4.0
    expected = F.binary_cross_entropy_with_logits(out.stop_logits, batch.stop_target, reduction="none")
    torch.testing.assert_close(out.stop_loss, (expected * torch.tensor([1., 3.])).sum() / 4)
    torch.testing.assert_close(out.loss, 2 * out.fm_loss + 1.5 * out.stop_loss)
    groups = model.optimizer_parameters()
    assert not groups["vlm"] and groups["stop"]
    assert not {id(p) for p in groups["stop"]} & {id(p) for p in groups["fm"]}
    assert all(p.dtype == torch.float32 for p in groups["stop"])


def test_stop_and_fm_gradients_are_mutually_isolated(model, batch):
    model(batch).stop_loss.backward()
    assert all(p.grad is None for p in model.vlm.parameters())
    assert all(p.grad is None for p in model.expert.parameters())
    assert all(p.grad is not None for p in model.stop_head.parameters())
    assert model.stop_head.layers[-1].weight.grad.abs().sum() > 0
    model.zero_grad(set_to_none=True)
    model(batch).fm_loss.backward()
    assert all(p.grad is None for p in model.vlm.parameters())
    assert all(p.grad is None for p in model.stop_head.parameters())
    assert model.expert.action_out.weight.grad.abs().sum() > 0


def test_terminal_only_stop_supervision_and_invalid_only_ddp_hooks(model, batch):
    terminal = replace(batch, fm_valid_mask=torch.zeros_like(batch.fm_valid_mask),
                       stop_target=torch.ones_like(batch.stop_target))
    out = model(terminal)
    assert out.fm_step_count == 0 and out.context is None
    assert out.stop_valid_count == out.stop_positive_count == 2
    out.loss.backward()
    assert model.stop_head.layers[-1].weight.grad.abs().sum() > 0
    assert all(p.grad is not None and not p.grad.any() for p in model.expert.parameters())
    model.zero_grad(set_to_none=True)
    invalid = replace(terminal, stop_valid_mask=torch.zeros_like(batch.stop_valid_mask),
                      stop_target=torch.full_like(batch.stop_target, float("nan")))
    out = model(invalid)
    assert out.stop_valid_count == out.stop_positive_count == 0
    assert out.stop_weight_sum == out.stop_loss.item() == 0
    out.loss.backward()
    assert all(p.grad is not None and not p.grad.any() for p in model.stop_head.parameters())
    assert all(p.grad is not None and not p.grad.any() for p in model.expert.parameters())


def test_stop_ignores_invalid_targets_and_fm_mask(model, batch):
    partial = replace(batch, stop_valid_mask=torch.tensor([True, False]))
    first = model(partial)
    second = model(replace(
        partial, stop_target=torch.tensor([0., float("nan")]),
        fm_valid_mask=torch.zeros_like(batch.fm_valid_mask),
    ))
    torch.testing.assert_close(first.stop_loss, second.stop_loss, rtol=0, atol=0)
    torch.testing.assert_close(first.stop_logits, second.stop_logits, rtol=0, atol=0)
    assert second.stop_weight_sum == 1 and second.stop_valid_count == 1


def test_stop_summary_excludes_padding_and_owns_only_selected_storage(model, batch):
    model.eval()
    with model._hf_inputs(batch.model_inputs) as inputs:
        if model.config.expert_architecture == "prefix_kv_v1":
            valid = inputs["attention_mask"]
            inputs["position_ids"] = (valid.long().cumsum(-1) - 1).masked_fill(~valid, 0)
        expected = model.vlm.model(**inputs, use_cache=False, return_dict=True).last_hidden_state
    context = model.encode_stop_context(batch.model_inputs)
    torch.testing.assert_close(context.stop_features, expected[:, -1])
    assert not context.stop_features.requires_grad
    assert context.stop_features.untyped_storage().nbytes() == (
        context.stop_features.numel() * context.stop_features.element_size()
    )
    # Changing GT padding/labels is not an input to STOP, and supplied answers
    # are rejected instead of being accidentally read by the summary position.
    with pytest.raises(ValueError, match="prompt-only"):
        model(replace(batch, answer_start=torch.tensor([2, 2])))


def test_last_prefix_selection_handles_answers_and_internal_padding():
    hidden = torch.arange(2 * 6 * 3).reshape(2, 6, 3).float().requires_grad_()
    attention = torch.tensor([[False, True, False, True, True, False], [True, True, False, True, True, True]])
    answer_start = torch.tensor([4, 2])
    valid = attention & (torch.arange(6)[None] < answer_start[:, None])
    summary = last_prefix_features(hidden, valid)
    torch.testing.assert_close(summary, torch.stack([hidden[0, 3], hidden[1, 1]]))
    changed = hidden.detach().clone()
    changed[~valid] = float("nan")
    torch.testing.assert_close(summary, last_prefix_features(changed, valid), rtol=0, atol=0)
    assert not summary.requires_grad


def test_bfloat16_backbone_keeps_stop_parameters_and_loss_fp32(model, batch):
    model.vlm.to(dtype=torch.bfloat16)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        out = model(batch)
    assert out.stop_logits.dtype == out.stop_loss.dtype == torch.float32
    assert torch.isfinite(out.loss)
    out.stop_loss.backward()
    assert all(p.dtype == torch.float32 for p in model.stop_head.parameters())
    assert all(p.grad is None for p in model.vlm.parameters())


def test_all_stop_encodes_once_skips_fm_and_consumes_no_noise(model, batch, monkeypatch):
    model.eval()
    calls = []
    handle = model.vlm.model.register_forward_pre_hook(lambda *_: calls.append(1))
    monkeypatch.setattr(model.stop_head, "forward", lambda h: h.new_full((len(h),), 10.0))

    def forbidden(*args, **kwargs):
        raise AssertionError("all STOP must never sample FM actions")

    monkeypatch.setattr(model.expert, "sample", forbidden)
    generator = torch.Generator().manual_seed(101)
    before = generator.get_state().clone()
    prediction = model.predict_continuous(batch.model_inputs, generator=generator)
    handle.remove()
    assert calls == [1] and prediction.stop.all()
    assert not prediction.se2_deltas.any() and not prediction.waypoints.any()
    assert torch.equal(before, generator.get_state())


def test_mixed_batch_slices_context_after_original_batch_noise(model, batch, monkeypatch):
    model.eval()
    expected_context = model.encode_stop_context(batch.model_inputs).motion
    calls, sampled = [], []
    handle = model.vlm.model.register_forward_pre_hook(lambda *_: calls.append(1))
    monkeypatch.setattr(model.stop_head, "forward", lambda h: h.new_tensor([10., -10.]))

    def sample(context, *, num_steps=None, noise=None, generator=None):
        sampled.append((context, noise.clone()))
        assert generator is None
        return noise

    monkeypatch.setattr(model.expert, "sample", sample)
    generator = torch.Generator().manual_seed(123)
    expected_noise = torch.randn(2, model.config.horizon, 3, generator=torch.Generator().manual_seed(123))
    prediction = model.predict_continuous(batch.model_inputs, generator=generator)
    handle.remove()
    assert calls == [1] and prediction.stop.tolist() == [True, False]
    context, noise = sampled[0]
    torch.testing.assert_close(noise, expected_noise[1:], rtol=0, atol=0)
    torch.testing.assert_close(context.valid, expected_context.valid[1:])
    if isinstance(context, PrefixKVContext):
        for (k, v), (full_k, full_v) in zip(context.layer_kv, expected_context.layer_kv, strict=True):
            torch.testing.assert_close(k, full_k[1:])
            torch.testing.assert_close(v, full_v[1:])
        torch.testing.assert_close(context.action_cos, expected_context.action_cos[1:])
        torch.testing.assert_close(context.action_sin, expected_context.action_sin[1:])
    else:
        torch.testing.assert_close(context.memory, expected_context.memory[1:])
    assert not prediction.se2_deltas[0].any()
    torch.testing.assert_close(prediction.se2_deltas[1:], model.normalizer.denormalize(expected_noise[1:]))
    monkeypatch.setattr(model.stop_head, "forward", lambda h: h.new_full((len(h),), -10.))
    all_continue = model.predict_continuous(
        batch.model_inputs, generator=torch.Generator().manual_seed(123),
    )
    torch.testing.assert_close(prediction.se2_deltas[1], all_continue.se2_deltas[1], rtol=0, atol=0)


def test_unconfirmed_stop_samples_real_fm_from_the_single_shared_encoding(model, batch, monkeypatch):
    model.eval()
    calls, sampled = [], []
    handle = model.vlm.model.register_forward_pre_hook(lambda *_: calls.append(1))
    monkeypatch.setattr(model.stop_head, "forward", lambda h: h.new_full((len(h),), 10.0))

    def sample(context, *, num_steps=None, noise=None, generator=None):
        sampled.append(context)
        return noise

    monkeypatch.setattr(model.expert, "sample", sample)
    noise = torch.full((2, model.config.horizon, 3), 0.25)
    prediction = model.predict_continuous(
        batch.model_inputs, noise=noise, stop_eligible=torch.tensor([False, True]),
    )
    handle.remove()
    assert calls == [1] and len(sampled) == 1
    assert prediction.stop.tolist() == [False, True]
    assert (prediction.stop_scores > 0.99).all()
    assert sampled[0].valid.shape[0] == 1
    torch.testing.assert_close(prediction.se2_deltas[:1], model.normalizer.denormalize(noise[:1]))
    assert not prediction.se2_deltas[1].any()


def test_stop_eligibility_cannot_override_a_below_threshold_score(model, batch, monkeypatch):
    model.eval()
    monkeypatch.setattr(model.stop_head, "forward", lambda h: h.new_full((len(h),), -10.0))
    prediction = model.predict_continuous(
        batch.model_inputs, stop_eligible=torch.tensor([True, True]),
        noise=torch.ones(2, model.config.horizon, 3), num_steps=1,
    )
    assert not prediction.stop.any()


@pytest.mark.parametrize("mask", [torch.tensor([True]), torch.ones(2), [True, True]])
def test_stop_eligibility_validates_shape_and_boolean_type(model, batch, mask):
    model.eval()
    with pytest.raises(ValueError, match="boolean tensor with shape"):
        model.predict_continuous(batch.model_inputs, stop_eligible=mask)


def _save_fm_source(model, path, *, remove_key=None, config_change=None):
    source_config = asdict(replace(model.config, training_mode="fm_only", stop_head_type="none"))
    source_config.update(config_change or {})
    state = {k: v for k, v in model.state_dict().items() if not k.startswith("stop_head.")}
    if remove_key is not None:
        state.pop(remove_key)
    torch.save(state, path / "model.pt")
    (path / "metadata.json").write_text(json.dumps({
        "format": "betanav_model_v1", "step": 12, "beta_config": source_config,
    }))
    (path / "COMPLETE").touch()


def test_warm_start_loads_all_fm_weights_and_keeps_only_new_head(model, tmp_path):
    _save_fm_source(model, tmp_path)
    fresh_vlm = type(model.vlm)(copy.deepcopy(model.vlm.config))
    target = BetaModel(fresh_vlm, model.config)
    head_before = {k: v.clone() for k, v in target.stop_head.state_dict().items()}
    path, metadata = warm_start_fm_stop(target, tmp_path)
    assert path == tmp_path.resolve() and metadata["step"] == 12
    for key, tensor in model.state_dict().items():
        if key.startswith("stop_head."):
            torch.testing.assert_close(target.state_dict()[key], head_before[key.removeprefix("stop_head.")])
        else:
            torch.testing.assert_close(target.state_dict()[key], tensor, rtol=0, atol=0)


def test_warm_start_rejects_other_missing_weights_and_changed_horizon(model, tmp_path):
    _save_fm_source(model, tmp_path, remove_key="expert.action_out.weight")
    with pytest.raises(ValueError, match="only new stop_head"):
        warm_start_fm_stop(model, tmp_path)
    _save_fm_source(model, tmp_path, config_change={"horizon": 17})
    with pytest.raises(ValueError, match="horizon"):
        warm_start_fm_stop(model, tmp_path)


def test_old_mode_cannot_request_random_learned_stop(request, batch):
    old = request.getfixturevalue("old_model_fixture")
    old.eval()
    assert old.stop_head is None
    assert "stop" not in old.optimizer_parameters()
    assert not any(k.startswith("stop_head.") for k in old.state_dict())
    with pytest.raises(ValueError, match="trained stop_head"):
        old.predict_continuous(batch.model_inputs)


@pytest.mark.parametrize("kwargs", [
    {"training_mode": "fm_stop"},
    {"training_mode": "fm_only", "stop_head_type": "last_prefix_mlp_v1"},
    {"stop_positive_weight": 0.0}, {"stop_hidden_size": 0},
])
def test_invalid_stop_config(kwargs):
    with pytest.raises(ValueError):
        BetaModelConfig(action_scale=(1., 1., 1.), **kwargs)
