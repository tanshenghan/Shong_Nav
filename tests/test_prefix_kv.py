"""Selected native Qwen prefix KV, without weights or a GPU."""

from __future__ import annotations

import pytest
import torch

from lightnav.inference.prefix_kv import encode_prefix_kv


@pytest.fixture(autouse=True, scope="module")
def cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


@pytest.fixture
def vlm():
    from transformers import Qwen3VLConfig, Qwen3VLForConditionalGeneration

    torch.manual_seed(37)
    config = Qwen3VLConfig(
        text_config=dict(
            vocab_size=64, hidden_size=32, intermediate_size=64, num_hidden_layers=4,
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
    return Qwen3VLForConditionalGeneration(config)


def _inputs(ids):
    ids = torch.tensor(ids, dtype=torch.long)
    return {"input_ids": ids, "attention_mask": ids != 0}


def _assert_native_cache(vlm, inputs, context):
    kwargs = dict(inputs)
    if (kwargs.get("position_ids") is None
            and kwargs.get("image_grid_thw") is None and kwargs.get("video_grid_thw") is None):
        valid = kwargs["attention_mask"].bool()
        kwargs["position_ids"] = (valid.long().cumsum(-1) - 1).masked_fill(~valid, 0)
    with torch.no_grad():
        native = vlm.model(**kwargs, use_cache=True, return_dict=True).past_key_values
    for index, (key, value) in zip(context.layer_indices, context.layer_kv, strict=True):
        native_key, native_value = native.layers[index - 1].keys, native.layers[index - 1].values
        mask = context.valid[:, None, :, None]
        torch.testing.assert_close(key, torch.where(mask, native_key, 0.0), rtol=0, atol=0)
        torch.testing.assert_close(value, torch.where(mask, native_value, 0.0), rtol=0, atol=0)


def test_selected_layers_match_native_rotated_cache_exactly(vlm):
    inputs = _inputs([[1, 2, 3, 4], [0, 0, 7, 8]])
    context = encode_prefix_kv(vlm, inputs, (2, 4), 30)
    assert context.layer_indices == (2, 4)
    assert len(context.layer_kv) == 2
    assert context.action_cos.shape == context.action_sin.shape == (2, 30, 8)
    assert all(k.shape == v.shape == (2, 2, 4, 8) for k, v in context.layer_kv)
    _assert_native_cache(vlm, inputs, context)


def test_no_full_cache_lm_head_or_gradient_and_eval_mode(vlm, monkeypatch):
    from transformers.models.qwen3_vl import modeling_qwen3_vl

    def forbidden(*args, **kwargs):
        raise AssertionError("prefix encoding must not create a full KV cache or LM logits")

    monkeypatch.setattr(modeling_qwen3_vl, "DynamicCache", forbidden)
    monkeypatch.setattr(vlm.lm_head, "forward", forbidden)
    seen = []
    hook = vlm.model.register_forward_pre_hook(
        lambda module, args: seen.append((torch.is_grad_enabled(), module.training)),
    )
    vlm.train()
    context = encode_prefix_kv(vlm, _inputs([[1, 2, 3]]), (1, 4), 30)
    hook.remove()
    assert seen == [(False, False)]
    assert not vlm.training and not any(m.training for m in vlm.modules())
    tensors = [context.valid, context.action_cos, context.action_sin]
    tensors.extend(t for pair in context.layer_kv for t in pair)
    assert all(not t.requires_grad and t.grad_fn is None for t in tensors)
    assert all(p.grad is None for p in vlm.parameters())


@pytest.mark.parametrize("padded", [[[0, 0, 1, 2, 3]], [[1, 2, 3, 0, 0]], [[0, 1, 0, 2, 3]]])
def test_padding_does_not_change_effective_kv_or_action_rope(vlm, padded):
    bare = encode_prefix_kv(vlm, _inputs([[1, 2, 3]]), (1, 3), 30)
    context = encode_prefix_kv(vlm, _inputs(padded), (1, 3), 30)
    for (key, value), (base_key, base_value) in zip(context.layer_kv, bare.layer_kv, strict=True):
        torch.testing.assert_close(key[:, :, context.valid[0]], base_key)
        torch.testing.assert_close(value[:, :, context.valid[0]], base_value)
        assert (key[:, :, ~context.valid[0]] == 0).all()
        assert (value[:, :, ~context.valid[0]] == 0).all()
    torch.testing.assert_close(context.action_cos, bare.action_cos, rtol=0, atol=0)
    torch.testing.assert_close(context.action_sin, bare.action_sin, rtol=0, atol=0)


def test_prefix_keeps_native_causal_attention(vlm):
    first = encode_prefix_kv(vlm, _inputs([[1, 2, 3, 4]]), (1, 4), 30)
    second = encode_prefix_kv(vlm, _inputs([[1, 2, 30, 31]]), (1, 4), 30)
    for (key, value), (other_key, other_value) in zip(first.layer_kv, second.layer_kv, strict=True):
        torch.testing.assert_close(key[:, :, :2], other_key[:, :, :2], rtol=0, atol=0)
        torch.testing.assert_close(value[:, :, :2], other_value[:, :, :2], rtol=0, atol=0)
    assert not torch.equal(first.layer_kv[-1][0][:, :, 2:], second.layer_kv[-1][0][:, :, 2:])


def test_multimodal_native_mrope_cache_and_action_extension(vlm):
    # Four image tokens span a 2x2 post-merge grid. Their sequence length and
    # largest spatial position differ, so token-count suffix positions are wrong.
    inputs = _inputs([[0, 1, 59, 60, 60, 60, 60, 62, 2]])
    inputs.update(
        pixel_values=torch.randn(16, 24), image_grid_thw=torch.tensor([[1, 4, 4]]),
        mm_token_type_ids=torch.tensor([[0, 0, 0, 1, 1, 1, 1, 0, 0]]),
    )
    context = encode_prefix_kv(vlm, inputs, (2, 4), 30)
    _assert_native_cache(vlm, inputs, context)
    positions, _ = vlm.model.get_rope_index(
        input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"],
        image_grid_thw=inputs["image_grid_thw"], mm_token_type_ids=inputs["mm_token_type_ids"],
    )
    start = positions.masked_fill(~context.valid[None], -1).amax(dim=(0, 2)) + 1
    assert start.item() != context.valid.sum().item()
    expected_positions = start[:, None] + torch.arange(30)[None]
    expected = vlm.model.language_model.rotary_emb(torch.empty(1, 30, 8), expected_positions)
    torch.testing.assert_close(context.action_cos, expected[0], rtol=0, atol=0)
    torch.testing.assert_close(context.action_sin, expected[1], rtol=0, atol=0)
    # Padding values, including their would-be large positions, never advance
    # the action suffix. Multimodal positions come from native valid tokens.
    unpadded = {k: v[:, 1:] if k in {"input_ids", "attention_mask", "mm_token_type_ids"} else v
                for k, v in inputs.items()}
    other = encode_prefix_kv(vlm, unpadded, (2, 4), 30)
    torch.testing.assert_close(context.action_cos, other.action_cos, rtol=0, atol=0)
    for (key, value), (other_key, other_value) in zip(context.layer_kv, other.layer_kv, strict=True):
        torch.testing.assert_close(key[:, :, 1:], other_key)
        torch.testing.assert_close(value[:, :, 1:], other_value)


def test_explicit_mrope_uses_all_three_axes_and_ignores_padding(vlm):
    inputs = _inputs([[0, 1, 2]])
    inputs["position_ids"] = torch.tensor([[[10000, 2, 3]], [[20000, 2, 8]], [[30000, 4, 5]]])
    context = encode_prefix_kv(vlm, inputs, (1, 4), 3)
    cos, sin = vlm.model.language_model.rotary_emb(torch.empty(1, 3, 8), torch.tensor([[9, 10, 11]]))
    torch.testing.assert_close(context.action_cos, cos, rtol=0, atol=0)
    torch.testing.assert_close(context.action_sin, sin, rtol=0, atol=0)
    _assert_native_cache(vlm, inputs, context)


def test_video_frames_keep_native_deepstack_and_mrope(vlm):
    # Qwen represents a video as separately timestamped frame blocks, while
    # video_grid_thw still records a single two-frame input video.
    inputs = _inputs([[1, 59, 61, 61, 61, 61, 62, 2, 59, 61, 61, 61, 61, 62, 3]])
    inputs.update(
        pixel_values_videos=torch.randn(32, 24), video_grid_thw=torch.tensor([[2, 4, 4]]),
        mm_token_type_ids=(inputs["input_ids"] == 61).long() * 2,
    )
    context = encode_prefix_kv(vlm, inputs, (1, 3, 4), 30)
    _assert_native_cache(vlm, inputs, context)
    positions, _ = vlm.model.get_rope_index(
        input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"],
        video_grid_thw=inputs["video_grid_thw"], mm_token_type_ids=inputs["mm_token_type_ids"],
    )
    start = positions.amax(dim=(0, 2)) + 1
    cos, sin = vlm.model.language_model.rotary_emb(
        torch.empty(1, 30, 8), start[:, None] + torch.arange(30)[None],
    )
    torch.testing.assert_close(context.action_cos, cos, rtol=0, atol=0)
    torch.testing.assert_close(context.action_sin, sin, rtol=0, atol=0)


def test_bfloat16_autocast_keeps_native_kv_and_action_rotary_dtype(vlm):
    vlm.to(dtype=torch.bfloat16)
    inputs = _inputs([[1, 2, 3], [0, 7, 8]])
    with torch.autocast("cpu", dtype=torch.bfloat16):
        context = encode_prefix_kv(vlm, inputs, (1, 4), 30)
        _assert_native_cache(vlm, inputs, context)
    assert context.action_cos.dtype == context.action_sin.dtype == torch.bfloat16
    assert all(k.dtype == v.dtype == torch.bfloat16 for k, v in context.layer_kv)
    assert all(torch.isfinite(t).all() for pair in context.layer_kv for t in pair)


def test_hooks_removed_on_failure_and_next_call_works(vlm, monkeypatch):
    language = vlm.model.language_model
    original = language.layers[2].forward

    def fail(*args, **kwargs):
        raise RuntimeError("synthetic decoder failure")

    monkeypatch.setattr(language.layers[2], "forward", fail)
    with pytest.raises(RuntimeError, match="synthetic decoder failure"):
        encode_prefix_kv(vlm, _inputs([[1, 2, 3]]), (1, 4), 30)
    assert not language.rotary_emb._forward_pre_hooks
    assert all(not layer.self_attn._forward_pre_hooks for layer in language.layers)
    monkeypatch.setattr(language.layers[2], "forward", original)
    assert len(encode_prefix_kv(vlm, _inputs([[1, 2, 3]]), (1, 4), 30).layer_kv) == 2
    assert not language.rotary_emb._forward_pre_hooks


@pytest.mark.parametrize("layers", [(), (0,), (5,), (1, 1), (2, 1), (True,), (1.5,)])
def test_invalid_layer_selection(vlm, layers):
    with pytest.raises(ValueError, match="layer_indices"):
        encode_prefix_kv(vlm, _inputs([[1, 2, 3]]), layers, 30)


@pytest.mark.parametrize("horizon", [0, -1, True, 1.5])
def test_invalid_horizon(vlm, horizon):
    with pytest.raises(ValueError, match="horizon"):
        encode_prefix_kv(vlm, _inputs([[1, 2, 3]]), (1,), horizon)


def test_rejects_labels_cache_and_empty_prefix(vlm):
    inputs = _inputs([[1, 2, 3]])
    for key in ("labels", "past_key_values", "inputs_embeds"):
        with pytest.raises(ValueError, match="prompt inputs"):
            encode_prefix_kv(vlm, {**inputs, key: None}, (1,), 30)
    with pytest.raises(ValueError, match="attention_mask"):
        encode_prefix_kv(vlm, _inputs([[0, 0]]), (1,), 30)
