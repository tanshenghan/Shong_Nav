"""Beta's real HF interface and gradient routing, with a tiny CPU Qwen3-VL."""

from __future__ import annotations

from dataclasses import asdict, replace

import numpy as np
import pytest
import torch
from torch.nn import functional as F

from lightnav.inference.beta_model import (
    BetaContext,
    BetaModel,
    BetaModelConfig,
    BetaTrainBatch,
    compose_se2_deltas,
    load_beta_model,
)
from lightnav.inference.config import InferenceConfig


@pytest.fixture(autouse=True, scope="module")
def cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


@pytest.fixture
def model():
    from transformers import Qwen3VLConfig, Qwen3VLForConditionalGeneration

    torch.manual_seed(7)
    config = Qwen3VLConfig(
        text_config=dict(
            vocab_size=64, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
            num_attention_heads=4, num_key_value_heads=2, head_dim=8, pad_token_id=0,
            rope_parameters={"rope_type": "default", "rope_theta": 10000.0,
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
    vlm = Qwen3VLForConditionalGeneration(config)
    return BetaModel(vlm, BetaModelConfig(
        action_scale=(0.25, 0.25, np.pi / 12), horizon=4, hidden_size=16,
        num_layers=2, num_heads=2, intermediate_size=32, num_flow_steps=3,
    ))


@pytest.fixture
def batch():
    ids = torch.tensor([[1, 2, 3, 4, 5, 6], [0, 1, 2, 3, 4, 5]])
    labels = ids.clone()
    labels[:, :3] = -100
    # First answer token is masked for one channel. The boundary stays at 3.
    labels[0, 3] = -100
    return BetaTrainBatch(
        model_inputs={"input_ids": ids, "attention_mask": ids != 0},
        labels=labels, answer_start=torch.tensor([3, 3]),
        fm_clean_se2_diff=torch.randn(2, 4, 3) * 0.1,
        fm_valid_mask=torch.tensor([[True, True, True, True], [True, True, False, False]]),
    )


def test_fm_gradient_never_reaches_backbone_but_trains_adapter(model, batch):
    out = model(batch)
    out.fm_loss.backward()
    assert all(p.grad is None for p in model.vlm.parameters())
    assert model.expert.context_proj.weight.grad.abs().sum() > 0
    assert model.expert.action_in.weight.grad.abs().sum() > 0
    assert model.expert.blocks[0].cross_attention.in_proj_weight.grad.abs().sum() > 0
    assert model.expert.action_out.weight.grad.abs().sum() > 0


def test_ce_gradient_trains_backbone_and_never_expert(model, batch):
    out = model(batch)
    out.ce_loss.backward()
    assert model.vlm.get_input_embeddings().weight.grad.abs().sum() > 0
    assert any(p.grad is not None and p.grad.abs().sum() > 0
               for p in model.vlm.model.language_model.layers.parameters())
    assert all(p.grad is None for p in model.expert.parameters())


def test_fm_only_freezes_vlm_and_never_computes_ce(model, batch, monkeypatch):
    frozen = BetaModel(model.vlm, replace(model.config, training_mode="fm_only"))
    frozen.train()
    assert frozen.expert.training and not frozen.vlm.training
    assert not frozen.optimizer_parameters()["vlm"]
    prefix = replace(batch, model_inputs={k: v[:, :3] for k, v in batch.model_inputs.items()},
                     labels=torch.full_like(batch.labels[:, :3], -100))
    before = {k: p.detach().clone() for k, p in frozen.vlm.named_parameters()}
    seen = []
    hook = frozen.vlm.model.register_forward_pre_hook(
        lambda module, _args: seen.append((torch.is_grad_enabled(), module.training)))

    def forbidden(*args, **kwargs):
        raise AssertionError("FM-only must never invoke the LM head")

    monkeypatch.setattr(frozen.vlm.lm_head, "forward", forbidden)
    opt = torch.optim.AdamW(frozen.optimizer_parameters()["fm"], lr=1e-3)
    expert_before = frozen.expert.context_proj.weight.detach().clone()
    out = frozen(prefix, lambda_fm=2.)
    out.loss.backward()
    opt.step()
    hook.remove()
    assert seen == [(False, False)]
    assert out.ce_token_count == 0 and out.ce_loss.item() == 0
    torch.testing.assert_close(out.loss, 2 * out.fm_loss)
    assert all(not p.requires_grad and p.grad is None and torch.equal(p, before[k])
               for k, p in frozen.vlm.named_parameters())
    assert not torch.equal(expert_before, frozen.expert.context_proj.weight)
    with pytest.raises(ValueError, match="fm_only"):
        frozen(batch)


def test_fm_only_rejects_hidden_answers_even_with_ignored_labels(model, batch):
    frozen = BetaModel(model.vlm, replace(model.config, training_mode="fm_only"))
    with pytest.raises(ValueError, match="prompt-only"):
        frozen(replace(batch, labels=torch.full_like(batch.labels, -100)))


def test_joint_optimizer_step_updates_both_disjoint_groups(model, batch):
    groups = model.optimizer_parameters()
    vlm_ids, fm_ids = [{id(p) for p in groups[name]} for name in ("vlm", "fm")]
    assert not vlm_ids & fm_ids
    assert vlm_ids | fm_ids == {id(p) for p in model.parameters() if p.requires_grad}
    optimizers = [torch.optim.AdamW(groups[name], lr=1e-3) for name in ("vlm", "fm")]
    before = [model.vlm.lm_head.weight.detach().clone(), model.expert.context_proj.weight.detach().clone()]
    for opt in optimizers:
        opt.zero_grad(set_to_none=True)
    model(batch).loss.backward()
    for opt in optimizers:
        opt.step()
    assert not torch.equal(before[0], model.vlm.lm_head.weight)
    assert not torch.equal(before[1], model.expert.context_proj.weight)


def test_ce_matches_shifted_hf_loss_with_independent_masks(model, batch):
    model.eval()
    with model._hf_inputs(batch.model_inputs) as inputs:
        expected = model.vlm(**inputs, labels=batch.labels, use_cache=False).loss
    actual = model(batch).ce_loss
    torch.testing.assert_close(actual, expected)


def test_bfloat16_autocast_keeps_fm_loss_finite_and_backbone_isolated(model, batch):
    model.vlm.to(dtype=torch.bfloat16)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        out = model(batch)
    assert out.fm_loss.dtype == torch.float32 and torch.isfinite(out.loss)
    out.fm_loss.backward()
    assert all(p.grad is None for p in model.vlm.parameters())
    assert torch.isfinite(model.expert.context_proj.weight.grad).all()


def test_fm_condition_cannot_read_any_answer_tokens_including_ignored_ones(model, batch):
    model.eval()
    noise, tau = torch.randn(2, 4, 3), torch.tensor([0.2, 0.7])
    before = model(batch, noise=noise, flow_time=tau)
    changed = batch.model_inputs["input_ids"].clone()
    changed[:, 3:] = torch.tensor([[21, 22, 23], [24, 25, 26]])
    other = replace(batch, model_inputs={**batch.model_inputs, "input_ids": changed})
    after = model(other, noise=noise, flow_time=tau)
    torch.testing.assert_close(before.context.memory, after.context.memory, rtol=0, atol=0)
    torch.testing.assert_close(before.velocity, after.velocity, rtol=0, atol=0)
    # Comparing with actual inference on the prefix also checks the causal boundary.
    context = model.encode_context({k: v[:, :3] for k, v in batch.model_inputs.items()})
    torch.testing.assert_close(before.context.memory, context.memory)


def test_prefix_changes_do_change_fm_condition(model, batch):
    model.eval()
    before = model(batch)
    changed = batch.model_inputs["input_ids"].clone()
    changed[0, 1] = 31
    after = model(replace(batch, model_inputs={**batch.model_inputs, "input_ids": changed}))
    assert not torch.equal(before.context.memory, after.context.memory)


def test_padding_is_not_condition_and_padding_nan_is_safe(model):
    hidden = torch.randn(2, 3, 32, requires_grad=True)
    valid = torch.tensor([[True, True, True], [False, True, True]])
    a = model.expert.prepare_context(hidden, valid)
    altered = hidden.detach().clone()
    altered[~valid] = float("nan")
    b = model.expert.prepare_context(altered, valid)
    noise, tau = torch.randn(2, 4, 3), torch.tensor([0.3, 0.5])
    torch.testing.assert_close(model.expert(noise, tau, a), model.expert(noise, tau, b))


def test_fm_tail_is_loss_masked_and_never_a_zero_action_target(model, batch):
    model.eval()
    noise, tau = torch.randn(2, 4, 3), torch.tensor([0.2, 0.6])
    before = model(batch, noise=noise, flow_time=tau)
    dirty = batch.fm_clean_se2_diff.clone()
    dirty[~batch.fm_valid_mask] = float("nan")
    after = model(replace(batch, fm_clean_se2_diff=dirty), noise=noise, flow_time=tau)
    torch.testing.assert_close(before.velocity, after.velocity, rtol=0, atol=0)
    clean = model.normalizer.normalize(batch.fm_clean_se2_diff)
    expected = F.mse_loss(before.velocity[batch.fm_valid_mask], (clean - noise)[batch.fm_valid_mask])
    torch.testing.assert_close(after.fm_loss, expected)
    assert after.fm_step_count == 6


def test_terminal_only_batch_leaves_expert_out_of_weight_decay(model, batch):
    terminal = replace(batch, fm_valid_mask=torch.zeros_like(batch.fm_valid_mask))
    optimizer = torch.optim.AdamW(model.expert.parameters(), lr=1e-3, weight_decay=0.1)
    optimizer.zero_grad(set_to_none=True)
    before = model.expert.context_proj.weight.detach().clone()
    out = model(terminal)
    assert out.fm_loss.item() == 0 and out.fm_step_count == 0 and out.context is None
    out.loss.backward()
    assert all(p.grad is None for p in model.expert.parameters())
    optimizer.step()
    torch.testing.assert_close(before, model.expert.context_proj.weight, rtol=0, atol=0)


def test_all_ce_labels_ignored_still_allows_fm_without_backbone_grads(model, batch):
    out = model(replace(batch, labels=torch.full_like(batch.labels, -100)))
    assert out.ce_loss.item() == 0 and out.ce_token_count == 0
    out.loss.backward()
    assert all(p.grad is None for p in model.vlm.parameters())
    assert model.expert.context_proj.weight.grad is not None


def test_fm_only_empty_batch_has_zero_ddp_hooks_without_weight_decay(model, batch):
    from lightnav.training.config import TrainConfig
    from lightnav.training.runtime import clip_and_step, normalize_gradients

    frozen = BetaModel(model.vlm, replace(model.config, training_mode="fm_only"))
    prefix = replace(batch, model_inputs={
        "input_ids": batch.model_inputs["input_ids"][:, :3],
        "attention_mask": torch.ones_like(batch.labels[:, :3], dtype=torch.bool),
    }, labels=torch.full_like(batch.labels[:, :3], -100),
        answer_start=torch.full_like(batch.answer_start, 3),
        fm_valid_mask=torch.zeros_like(batch.fm_valid_mask))
    before = {k: v.detach().clone() for k, v in frozen.state_dict().items()}
    out = frozen(prefix, lambda_fm=2)
    assert out.fm_step_count == 0 and out.context is None
    out.loss.backward()
    assert all(p.grad is not None and not p.grad.any() for p in frozen.expert.parameters())
    groups = frozen.optimizer_parameters()
    optimizers = {"fm": torch.optim.AdamW(groups["fm"], lr=.1, weight_decay=.1)}
    normalize_gradients(groups, [0, 0], 1)
    cfg = TrainConfig(dataset_root="data", model_path="model", data_manifest="meta", output_dir="out")
    clip_and_step(groups, optimizers, [0, 0], cfg)
    assert not optimizers["fm"].state
    for k, v in frozen.state_dict().items():
        torch.testing.assert_close(v, before[k], rtol=0, atol=0)


def test_zero_fm_weight_keeps_expert_gradients_none(model, batch):
    out = model(batch, lambda_fm=0)
    out.loss.backward()
    assert out.context is None
    assert all(p.grad is None for p in model.expert.parameters())


@pytest.mark.parametrize("distribution,beta,expected_mean,expected_noisy_fraction", [
    ("uniform", 2., .5, .3), ("beta", 1.5, .4, 1 - .7**1.5),
    ("beta", 2., 1 / 3, .51), ("beta", 3., .25, .657),
])
def test_flow_time_sampling_distribution_and_rng_replay(model, distribution, beta, expected_mean, expected_noisy_fraction):
    model.config = replace(model.config, flow_time_distribution=distribution, flow_time_beta=beta)
    torch.manual_seed(1234)
    state = torch.get_rng_state()
    times = model._sample_flow_time(50000, torch.device("cpu"))
    assert times.shape == (50000,) and times.dtype == torch.float32
    assert not times.requires_grad and ((times >= 0) & (times <= 1)).all()
    assert abs(float(times.mean()) - expected_mean) < .006
    # Under X_tau = (1-tau)*noise + tau*GT this means at least 70% noise coefficient.
    assert abs(float((times <= .3).float().mean()) - expected_noisy_fraction) < .012
    torch.set_rng_state(state)
    torch.testing.assert_close(model._sample_flow_time(50000, torch.device("cpu")), times, rtol=0, atol=0)


@pytest.mark.parametrize("beta", [1.5, 2., 3.])
def test_ten_step_fm_only_trains_and_samples_with_frozen_vlm(model, batch, beta):
    model = BetaModel(model.vlm, replace(model.config, training_mode="fm_only", horizon=10,
                                        num_flow_steps=3, flow_time_distribution="beta", flow_time_beta=beta))
    batch = replace(batch, model_inputs={k: v[:, :3] for k, v in batch.model_inputs.items()},
                    labels=torch.full_like(batch.labels[:, :3], -100),
                    fm_clean_se2_diff=torch.randn(2, 10, 3) * .1,
                    fm_valid_mask=torch.tensor([[True] * 10, [True] * 5 + [False] * 5]))
    out = model(batch, lambda_fm=1.)
    assert torch.isfinite(out.loss) and out.fm_step_count == 15
    torch.testing.assert_close(out.loss, out.fm_loss)
    out.loss.backward()
    assert model.expert.step_embedding.grad.shape == (10, model.config.hidden_size)
    assert model.expert.context_proj.weight.grad.abs().sum() > 0
    assert all(p.grad is None for p in model.vlm.parameters())
    model.eval()
    context = model.encode_context(batch.model_inputs)
    actions, waypoints = model.sample_actions(context)
    assert actions.shape == (2, 10, 3) and torch.isfinite(actions).all()
    assert waypoints.shape == (2, 10, 3) and torch.isfinite(waypoints).all()


def test_beta_flow_time_conditions_the_whole_chunk_and_allows_fixed_time(model, batch, monkeypatch):
    model.config = replace(model.config, flow_time_distribution="beta", flow_time_beta=2.)
    model.eval()
    noise = torch.randn_like(batch.fm_clean_se2_diff)
    seen = []

    def capture(_module, args):
        seen.append((args[0].detach().clone(), args[1].detach().clone()))

    hook = model.expert.register_forward_pre_hook(capture)
    out = model(batch, noise=noise)
    hook.remove()
    noisy, tau = seen[0]
    assert tau.shape == (len(batch.labels),)
    mask = batch.fm_valid_mask[..., None]
    clean = model.normalizer.normalize(torch.where(mask, batch.fm_clean_se2_diff, 0.))
    t = tau[:, None, None]
    torch.testing.assert_close(noisy, torch.where(mask, (1 - t) * noise + t * clean, noise))

    def forbidden_sampling(*_args):
        raise AssertionError("explicit times must bypass the configured sampler")

    monkeypatch.setattr(model, "_sample_flow_time", forbidden_sampling)
    fixed = model(batch, noise=noise, flow_time=tau)
    torch.testing.assert_close(fixed.velocity, out.velocity, rtol=0, atol=0)


def test_sampling_uses_noise_to_data_direction_and_denormalizes(model, monkeypatch):
    model.eval()
    context = BetaContext(torch.zeros(2, 1, 16), torch.ones(2, 1, dtype=torch.bool))
    noise = torch.randn(2, 4, 3)
    endpoint = torch.randn_like(noise)
    times = []

    def constant_field(noisy, tau, _context):
        times.append(tau.clone())
        return endpoint - noise

    monkeypatch.setattr(model.expert, "forward", constant_field)
    deltas, waypoints = model.sample_actions(context, noise=noise, num_steps=4)
    torch.testing.assert_close(deltas, model.normalizer.denormalize(endpoint))
    torch.testing.assert_close(waypoints, compose_se2_deltas(deltas))
    torch.testing.assert_close(torch.stack(times)[:, 0], torch.tensor([0., .25, .5, .75]))


def test_composition_rotates_local_motion_and_matches_rvq_utility():
    from lightnav.traj_vocab import compose_to_abs

    deltas = torch.tensor([[[1., 0., np.pi / 2], [1., 0., 0.], [0., 1., 0.]]])
    actual = compose_se2_deltas(deltas)
    torch.testing.assert_close(actual[0, :, :2], torch.tensor([[1., 0.], [1., 1.], [0., 1.]]), atol=1e-6, rtol=0)
    np.testing.assert_allclose(actual.numpy(), compose_to_abs(deltas.numpy()), atol=1e-6)


def test_checkpoint_roundtrip_includes_vlm_expert_and_action_scale(model, batch, tmp_path):
    model.eval()
    noise, tau = torch.randn(2, 4, 3), torch.tensor([0.2, 0.6])
    expected = model(batch, noise=noise, flow_time=tau).velocity
    path = tmp_path / "beta.pt"
    torch.save({"beta_config": asdict(model.config), "model": model.state_dict()}, path)
    saved = torch.load(path, weights_only=True)
    fresh_vlm = type(model.vlm)(model.vlm.config)
    assert not torch.equal(fresh_vlm.lm_head.weight, model.vlm.lm_head.weight)
    restored = BetaModel(fresh_vlm, BetaModelConfig(**saved["beta_config"])).eval()
    restored.load_state_dict(saved["model"])
    torch.testing.assert_close(restored(batch, noise=noise, flow_time=tau).velocity, expected, rtol=0, atol=0)
    torch.testing.assert_close(restored.normalizer.scale, model.normalizer.scale)


def test_real_video_pooling_and_native_mrope_support_training(model):
    from lightnav.processing import enable_post_vit_pool

    enable_post_vit_pool(model.vlm, spatial_factor=2, merge_size=2)
    model.vlm._post_vit_target = model.vlm.model
    ids = torch.tensor([[1, 59, 61, 62, 4, 5, 6]])
    labels = torch.full_like(ids, -100)
    labels[:, 5:] = ids[:, 5:]
    inputs = {
        "input_ids": ids, "attention_mask": torch.ones_like(ids),
        "pixel_values_videos": torch.randn(16, 24),
        "video_grid_thw": torch.tensor([[1, 2, 2]]),
        "_original_video_grid_thw": torch.tensor([[1, 4, 4]]),
        "_post_vit_pool_factors": [2],
        # The processor's incompatible position IDs must not reach HF.
        "position_ids": torch.full((3, 1, 7), -999),
    }
    batch = BetaTrainBatch(inputs, labels, torch.tensor([5]), torch.randn(1, 4, 3), torch.ones(1, 4, dtype=torch.bool))
    out = model(batch)
    out.loss.backward()
    assert torch.isfinite(out.loss)
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.vlm.model.visual.parameters())
    assert model.expert.context_proj.weight.grad.abs().sum() > 0
    assert model.vlm.model._post_vit_original_grid is None
    assert model.vlm.model._post_vit_pool_factors is None


class _Tokenizer:
    pad_token_id = 0
    eos_token_id = None

    def __init__(self, texts):
        self.texts = iter(texts)

    def decode(self, _tokens, skip_special_tokens):
        assert not skip_special_tokens
        return next(self.texts)


@pytest.mark.parametrize("action_output", ["fm", "token"])
def test_predict_reuses_prefill_and_preserves_explicit_and_near_zero_stop(
    model, rvq_bundle_writer, tmp_path, monkeypatch, action_output
):
    from lightnav.tracking import TrackingAgent
    from lightnav.traj_vocab import load_rvq_bundle

    # Non-stop L0=1 but exactly zero reconstruction exercises the legacy fallback.
    codebooks = [np.zeros((4, 30), np.float32), np.zeros((8, 30), np.float32), np.zeros((8, 30), np.float32)]
    codebooks[0][2] = 0.2
    rvq_bundle_writer(tmp_path / "rvq", codebooks=codebooks)
    decoder = TrackingAgent(engine=None, rvq_bundle=load_rvq_bundle(tmp_path / "rvq", horizon=10))
    model.eval()
    if action_output == "token":
        def forbid_fm(*args, **kwargs):
            raise AssertionError("RVQ-only inference must not call FM")
        monkeypatch.setattr(model, "_context", forbid_fm)
        monkeypatch.setattr(model, "sample_actions", forbid_fm)
    inputs = {"input_ids": torch.tensor([[0, 1, 2], [1, 2, 3], [1, 2, 3]])}
    inputs["attention_mask"] = inputs["input_ids"] != 0
    calls = []
    hook = model.vlm.model.register_forward_pre_hook(lambda _m, _a, kw: calls.append(kw["input_ids"].shape[1]), with_kwargs=True)
    try:
        prediction = model.predict(
            inputs, action_decoder=decoder, max_new_tokens=2,
            tokenizer=_Tokenizer(["<act_l0_0><act_l1_2><act_l2_4>",
                                  "<act_l0_1><act_l1_0><act_l2_0>",
                                  "<act_l0_2><act_l1_0><act_l2_0>"]),
            generator=torch.Generator().manual_seed(21),
            action_output=action_output,
        )
    finally:
        hook.remove()
    assert calls == [3, 1]  # prefill once, then one cached decode; no FM VLM calls
    assert prediction.stop.tolist() == [True, True, False]
    assert prediction.waypoints.shape == (3, 4 if action_output == "fm" else 10, 3)
    if action_output == "token":
        torch.testing.assert_close(prediction.waypoints, prediction.rvq_waypoints)
        assert prediction.se2_deltas.count_nonzero() == 0
    assert prediction.rvq_waypoints.shape == (3, 10, 3)
    assert prediction.waypoints[:2].count_nonzero() == 0
    assert prediction.waypoints[2].count_nonzero() > 0


def test_bad_token_generation_raises_and_removes_capture_hook(model, rvq_bundle_writer, tmp_path):
    from lightnav.tracking import TrackingAgent
    from lightnav.traj_vocab import load_rvq_bundle

    rvq_bundle_writer(tmp_path / "rvq")
    decoder = TrackingAgent(engine=None, rvq_bundle=load_rvq_bundle(tmp_path / "rvq", horizon=10))
    model.eval()
    hooks_before = len(model.vlm.model._forward_hooks)
    with pytest.raises(ValueError):
        model.predict({"input_ids": torch.tensor([[1, 2, 3]])}, tokenizer=_Tokenizer(["invalid"]),
                      action_decoder=decoder, max_new_tokens=1)
    assert len(model.vlm.model._forward_hooks) == hooks_before

    prediction = model.predict(
        {"input_ids": torch.tensor([[1, 2, 3]])},
        tokenizer=_Tokenizer(["invalid"]),
        action_decoder=decoder,
        max_new_tokens=1,
        continue_on_decode_error=True,
        generator=torch.Generator().manual_seed(9),
    )
    assert prediction.decode_failed.tolist() == [True]
    assert prediction.explicit_stop.tolist() == [False]
    assert prediction.stop.tolist() == [False]
    assert prediction.waypoints.count_nonzero() > 0


@pytest.mark.parametrize("change", [
    {"action_scale": (1., 0., 1.)}, {"action_scale": (1., float("nan"), 1.)},
    {"hidden_size": 17}, {"num_flow_steps": 0}, {"horizon": 0},
    {"flow_time_distribution": "unknown"}, {"flow_time_alpha": 0},
    {"flow_time_beta": -1}, {"flow_time_alpha": float("nan")}, {"flow_time_beta": float("inf")},
])
def test_invalid_config_rejected(change):
    with pytest.raises(ValueError):
        BetaModelConfig(**({"action_scale": (1., 1., 1.)} | change))


def test_rejects_loss_boundary_errors_and_training_inputs_in_inference(model, batch):
    labels = batch.labels.clone()
    labels[0, 1] = 2
    with pytest.raises(ValueError, match="prefix"):
        model(replace(batch, labels=labels))
    with pytest.raises(ValueError, match="empty input prefix"):
        model(replace(batch, answer_start=torch.tensor([1, 1])))
    model.eval()
    with pytest.raises(ValueError, match="teacher-forced"):
        model.encode_context({**batch.model_inputs, "labels": batch.labels})


def test_missing_pooling_patch_and_nonpadding_attention_rejected(model, batch):
    with pytest.raises(ValueError, match="pooling patch"):
        model(replace(batch, model_inputs={**batch.model_inputs, "_original_video_grid_thw": torch.ones(1, 3)}))
    with pytest.raises(ValueError, match="padding mask"):
        model(replace(batch, model_inputs={**batch.model_inputs, "attention_mask": torch.ones(2, 1, 6, 6)}))


def test_vllm_rejected_before_loading_weights():
    with pytest.raises(ValueError, match="backend='hf'"):
        load_beta_model(InferenceConfig(backend="vllm_local"), BetaModelConfig(action_scale=(1., 1., 1.)))
