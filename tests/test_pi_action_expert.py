"""CPU checks for native-KV joint attention and time-adaptive flow training."""

from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch
from torch.nn import functional as F

from lightnav.inference.pi_action_expert import (
    PiFlowActionExpert,
    PiJointAttention,
    TimeAdaptiveRMSNorm,
)
from lightnav.inference.prefix_kv import PrefixKVContext


@pytest.fixture(autouse=True, scope="module")
def cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def expert_config(**overrides):
    fields = dict(horizon=30, hidden_size=16, num_layers=2, num_heads=4,
                  intermediate_size=32, dropout=0.0, num_flow_steps=3)
    fields.update(overrides)
    return SimpleNamespace(**fields)


def make_context(config, *, batch=2, length=5, num_kv_heads=2, head_dim=8):
    valid = torch.ones(batch, length, dtype=torch.bool)
    valid[0, -2:] = False
    position = torch.arange(config.horizon, dtype=torch.float32)[None, :, None] + length
    frequency = 1 / (10000 ** (torch.arange(head_dim // 2) / (head_dim // 2)))
    phase = position * frequency[None, None]
    phase = torch.cat((phase, phase), dim=-1).expand(batch, -1, -1)
    return PrefixKVContext(
        layer_kv=tuple(
            tuple(torch.randn(batch, num_kv_heads, length, head_dim) for _ in range(2))
            for _ in range(config.num_layers)
        ),
        valid=valid, action_cos=phase.cos(), action_sin=phase.sin(),
        layer_indices=tuple(range(1, config.num_layers + 1)),
    )


def attention_mask(context, horizon):
    return torch.cat((context.valid, torch.ones(
        len(context.valid), horizon, dtype=torch.bool,
    )), dim=1)[:, None, None]


def reference_attention(module, hidden, context):
    """Explicit normalize -> rotate -> one softmax reference, without SDPA."""
    batch, horizon, _ = hidden.shape
    q = module.q_proj(hidden).view(batch, horizon, module.num_heads, module.head_dim)
    k = module.k_proj(hidden).view(batch, horizon, module.num_kv_heads, module.head_dim)
    v = module.v_proj(hidden).view(batch, horizon, module.num_kv_heads, module.head_dim)
    q = q * (q.square().mean(-1, keepdim=True) + module.q_norm.eps).rsqrt() * module.q_norm.weight
    k = k * (k.square().mean(-1, keepdim=True) + module.k_norm.eps).rsqrt() * module.k_norm.weight
    q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
    cos, sin = context.action_cos[:, None], context.action_sin[:, None]
    half = module.head_dim // 2
    q = q * cos + torch.cat((-q[..., half:], q[..., :half]), dim=-1) * sin
    k = k * cos + torch.cat((-k[..., half:], k[..., :half]), dim=-1) * sin
    prefix_k, prefix_v = context.layer_kv[0]
    k, v = torch.cat((prefix_k, k), dim=2), torch.cat((prefix_v, v), dim=2)
    groups = module.num_heads // module.num_kv_heads
    k, v = k.repeat_interleave(groups, dim=1), v.repeat_interleave(groups, dim=1)
    logits = q @ k.transpose(-2, -1) / (module.head_dim ** 0.5)
    logits = logits.masked_fill(~attention_mask(context, horizon), -torch.inf)
    attended = logits.softmax(dim=-1) @ v
    return module.o_proj(attended.transpose(1, 2).reshape(batch, horizon, -1))


@pytest.mark.parametrize("query_heads", [2, 4, 8])
def test_joint_attention_matches_one_softmax_reference(query_heads):
    torch.manual_seed(70)
    config = expert_config(num_heads=query_heads)
    context = make_context(config)
    module = PiJointAttention(config, num_kv_heads=2, head_dim=8, rms_norm_eps=1e-6).eval()
    hidden = torch.randn(2, 30, 16)
    expected = reference_attention(module, hidden, context)
    actual = module(hidden, context.layer_kv[0], context.action_cos,
                    context.action_sin, attention_mask(context, 30))
    torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-6)


def test_prefix_padding_never_changes_action_attention():
    torch.manual_seed(71)
    config = expert_config()
    context = make_context(config)
    module = PiJointAttention(config, num_kv_heads=2, head_dim=8, rms_norm_eps=1e-6).eval()
    hidden = torch.randn(2, 30, 16)
    changed = tuple(x.masked_fill(~context.valid[:, None, :, None], 1e5)
                    for x in context.layer_kv[0])
    mask = attention_mask(context, 30)
    actual = module(hidden, context.layer_kv[0], context.action_cos, context.action_sin, mask)
    ignored = module(hidden, changed, context.action_cos, context.action_sin, mask)
    torch.testing.assert_close(actual, ignored)


def test_all_future_actions_are_visible_and_prefix_kv_receives_no_gradient():
    torch.manual_seed(72)
    config = expert_config()
    context = make_context(config)
    prefix = tuple(x.clone().requires_grad_() for x in context.layer_kv[0])
    module = PiJointAttention(config, num_kv_heads=2, head_dim=8, rms_norm_eps=1e-6)
    hidden = torch.randn(2, 30, 16, requires_grad=True)
    result = module(hidden, prefix, context.action_cos, context.action_sin,
                    attention_mask(context, 30))
    # In causal attention the first action could not depend on the last action.
    result[:, 0].square().sum().backward()
    assert hidden.grad[:, -1].abs().sum() > 0
    assert all(x.grad is None for x in prefix)
    assert module.q_norm.weight.grad.abs().sum() > 0
    assert module.k_norm.weight.grad.abs().sum() > 0


def test_attention_projection_width_is_independent_of_residual_width():
    config = expert_config(num_heads=32, hidden_size=16)
    context = make_context(config, num_kv_heads=8, head_dim=128)
    expert = PiFlowActionExpert(config, num_kv_heads=8, head_dim=128, rms_norm_eps=1e-6)
    assert expert.blocks[0].attention.q_proj.weight.shape == (4096, 16)
    assert expert.blocks[0].attention.k_proj.weight.shape == (1024, 16)
    result = expert(torch.randn(2, 30, 3), torch.tensor([0.1, 0.7]), context)
    assert result.shape == (2, 30, 3)
    assert torch.isfinite(result).all()


def test_adaptive_rms_norm_has_separate_scale_shift_and_residual_gate():
    module = TimeAdaptiveRMSNorm(4, 1e-6)
    x, time = torch.randn(2, 3, 4), torch.randn(2, 4)
    normalized, gate = module(x, time)
    rms = x * (x.square().mean(-1, keepdim=True) + 1e-6).rsqrt()
    torch.testing.assert_close(normalized, rms)
    assert torch.count_nonzero(gate) == 0
    with torch.no_grad():
        module.modulation[-1].bias.copy_(torch.tensor([2.] * 4 + [3.] * 4 + [4.] * 4))
    normalized, gate = module(x, time)
    torch.testing.assert_close(normalized, rms * 3 + 3)
    torch.testing.assert_close(gate, torch.full((2, 1, 4), 4.))


def test_time_and_each_attention_layer_train_after_zero_gates_open():
    torch.manual_seed(73)
    config = expert_config()
    context = make_context(config)
    prefix = tuple(tuple(t.clone().requires_grad_() for t in kv) for kv in context.layer_kv)
    context = replace(context, layer_kv=prefix)
    expert = PiFlowActionExpert(config, num_kv_heads=2, head_dim=8, rms_norm_eps=1e-6)
    optimizer = torch.optim.SGD(expert.parameters(), lr=0.1)
    actions = torch.randn(2, 30, 3)
    target = torch.randn_like(actions)
    time = torch.tensor([0.1, 0.7])
    # Zero gates learn first; attention branches begin learning on update two.
    F.mse_loss(expert(actions, time, context), target).backward()
    assert expert.blocks[0].attention_norm.modulation[-1].weight.grad.abs().sum() > 0
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    velocity = expert(actions, time, context)
    F.mse_loss(velocity, target).backward()
    assert velocity.shape == (2, 30, 3) and torch.isfinite(velocity).all()
    for block in expert.blocks:
        assert block.attention.q_proj.weight.grad.abs().sum() > 0
        assert block.attention.k_proj.weight.grad.abs().sum() > 0
        assert block.attention.v_proj.weight.grad.abs().sum() > 0
        assert block.ffn.up_proj.weight.grad.abs().sum() > 0
        assert block.ffn_norm.modulation[-1].weight.grad.abs().sum() > 0
    assert expert.time_mlp[0].weight.grad.abs().sum() > 0
    assert all(t.grad is None for kv in prefix for t in kv)
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in expert.parameters())
    with torch.no_grad():
        earlier = expert(actions, torch.zeros(2), context)
        later = expert(actions, torch.ones(2), context)
    assert not torch.allclose(earlier, later)


def test_autocast_preserves_qkv_dtype_for_sdpa(monkeypatch):
    config = expert_config()
    context = make_context(config)
    expert = PiFlowActionExpert(config, num_kv_heads=2, head_dim=8, rms_norm_eps=1e-6)
    real_sdpa = F.scaled_dot_product_attention
    dtypes = []

    def record(q, k, v, **kwargs):
        dtypes.append((q.dtype, k.dtype, v.dtype))
        return real_sdpa(q, k, v, **kwargs)

    monkeypatch.setattr(F, "scaled_dot_product_attention", record)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        output = expert(torch.randn(2, 30, 3), torch.tensor([0.1, 0.7]), context)
        output.float().square().mean().backward()
    assert dtypes == [(torch.bfloat16,) * 3] * config.num_layers
    assert torch.isfinite(output).all()


def test_sample_uses_forward_euler_time_direction_and_reuses_context(monkeypatch):
    config = expert_config()
    context = make_context(config)
    before = tuple(tuple(t.clone() for t in kv) for kv in context.layer_kv)
    expert = PiFlowActionExpert(config, num_kv_heads=2, head_dim=8, rms_norm_eps=1e-6).eval()
    noise = torch.randn(2, 30, 3, requires_grad=True)
    noise_before = noise.detach().clone()
    observed_times = []

    def constant_velocity(actions, tau, seen_context):
        assert seen_context is context and not torch.is_grad_enabled()
        observed_times.append(tau.clone())
        return torch.full_like(actions, 2.5)

    monkeypatch.setattr(expert, "forward", constant_velocity)
    result = expert.sample(context, num_steps=4, noise=noise)
    torch.testing.assert_close(result, noise.detach() + 2.5)
    torch.testing.assert_close(noise, noise_before)
    assert not result.requires_grad
    torch.testing.assert_close(torch.stack(observed_times)[:, 0], torch.tensor([0., .25, .5, .75]))
    for kv_before, kv_after in zip(before, context.layer_kv, strict=True):
        for expected, actual in zip(kv_before, kv_after, strict=True):
            torch.testing.assert_close(expected, actual, atol=0, rtol=0)


def test_sampling_is_deterministic_with_generator_or_explicit_noise():
    config = expert_config(horizon=10)
    context = make_context(config)
    expert = PiFlowActionExpert(config, num_kv_heads=2, head_dim=8, rms_norm_eps=1e-6).eval()
    left = expert.sample(context, generator=torch.Generator().manual_seed(33))
    right = expert.sample(context, generator=torch.Generator().manual_seed(33))
    noise = torch.randn(2, 10, 3, generator=torch.Generator().manual_seed(33))
    explicit = expert.sample(context, noise=noise)
    torch.testing.assert_close(left, right)
    torch.testing.assert_close(left, explicit)
    assert left.shape == (2, 10, 3) and torch.isfinite(left).all()


@pytest.mark.parametrize("steps", [0, -1, 1.2, True])
def test_sampling_rejects_invalid_steps(steps):
    config = expert_config()
    expert = PiFlowActionExpert(config, num_kv_heads=2, head_dim=8, rms_norm_eps=1e-6).eval()
    with pytest.raises(ValueError, match="num_steps"):
        expert.sample(make_context(config), num_steps=steps)


def test_sampling_requires_eval_and_valid_noise():
    config = expert_config()
    context = make_context(config)
    expert = PiFlowActionExpert(config, num_kv_heads=2, head_dim=8, rms_norm_eps=1e-6)
    with pytest.raises(RuntimeError, match="eval"):
        expert.sample(context)
    expert.eval()
    for noise in (torch.zeros(2, 10, 3), torch.full((2, 30, 3), float("nan"))):
        with pytest.raises(ValueError, match="noise"):
            expert.sample(context, noise=noise)


@pytest.mark.parametrize("case", ["actions", "time", "valid", "layers", "indices", "kv", "positions"])
def test_rejects_incompatible_context_and_action_shapes(case):
    config = expert_config()
    context = make_context(config)
    expert = PiFlowActionExpert(config, num_kv_heads=2, head_dim=8, rms_norm_eps=1e-6)
    actions, time = torch.zeros(2, 30, 3), torch.zeros(2)
    if case == "actions":
        actions = torch.zeros(2, 10, 3)
    elif case == "time":
        time = torch.zeros(2, 1)
    elif case == "valid":
        context = replace(context, valid=context.valid.float())
    elif case == "layers":
        context = replace(context, layer_kv=context.layer_kv[:1])
    elif case == "indices":
        context = replace(context, layer_indices=(1,))
    elif case == "kv":
        context = replace(context, layer_kv=tuple((k[:, :, :-1], v) for k, v in context.layer_kv))
    elif case == "positions":
        context = replace(context, action_cos=context.action_cos[:, :-1])
    with pytest.raises(ValueError):
        expert(actions, time, context)


@pytest.mark.parametrize("field,value", [("num_kv_heads", 0), ("num_kv_heads", 3),
                                        ("head_dim", 7), ("rms_norm_eps", 0.)])
def test_rejects_invalid_native_attention_dimensions(field, value):
    options = dict(num_kv_heads=2, head_dim=8, rms_norm_eps=1e-6)
    options[field] = value
    with pytest.raises(ValueError):
        PiFlowActionExpert(expert_config(), **options)


def test_context_layer_order_must_match_configured_prefix_layers():
    config = expert_config(prefix_layers=(1, 2))
    context = make_context(config)
    expert = PiFlowActionExpert(config, num_kv_heads=2, head_dim=8, rms_norm_eps=1e-6)
    with pytest.raises(ValueError, match="configured prefix_layers in order"):
        expert(torch.zeros(2, 30, 3), torch.zeros(2),
               replace(context, layer_indices=(2, 1), layer_kv=context.layer_kv[::-1]))
