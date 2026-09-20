import pytest
import torch

from lightnav.training.performance import NativeFlopCounter, attention_flops, flop_rates


def test_flop_rate_uses_cluster_time_and_number_of_gpus():
    result = flop_rates([1e12, 3e12], 2, 10)
    assert result["estimated_cluster_tflops"] == 2
    assert result["estimated_per_gpu_tflops"] == 1
    assert result["estimated_hfu"] == .1
    assert flop_rates([1e12], 2)["estimated_hfu"] is None
    for args in (([1], 0), ([1], float("nan")), ([1], 1, 0), ([], 1)):
        with pytest.raises(ValueError):
            flop_rates(*args)


def test_flops_count_frozen_forward_and_trainable_backward():
    frozen = torch.nn.Linear(4, 8, bias=False).requires_grad_(False)
    trainable = torch.nn.Linear(8, 2, bias=False)
    with NativeFlopCounter() as counter:
        trainable(frozen(torch.ones(3, 4))).sum().backward()
    # Frozen forward: 2*3*4*8. Trainable forward and weight gradient: 2*(2*3*8*2).
    assert counter.get_total_flops() == 384


def test_gqa_count_matches_explicit_dense_head_expansion():
    from torch.utils.flop_counter import sdpa_backward_flop_count, sdpa_flop_count

    q, k, v = (2, 16, 7, 8), (2, 4, 9, 8), (2, 4, 9, 6)
    expanded_k, expanded_v = (2, 16, 9, 8), (2, 16, 9, 6)
    assert attention_flops(q, k, v) == sdpa_flop_count(q, expanded_k, expanded_v)
    assert attention_flops(q, k, v, backward=True) == sdpa_backward_flop_count(
        (2, 16, 7, 6), q, expanded_k, expanded_v)
    with pytest.raises(ValueError):
        attention_flops((2, 15, 7, 8), k, v)


def test_native_counter_preserves_checkpointed_gradients_and_counts_recompute():
    from torch.utils.checkpoint import checkpoint

    layer = torch.nn.Linear(4, 8, bias=False)
    x = torch.randn(3, 4, requires_grad=True)
    with NativeFlopCounter() as plain:
        layer(x).square().sum().backward()
    expected, expected_x = layer.weight.grad.clone(), x.grad.clone()
    layer.zero_grad(set_to_none=True)
    x.grad = None
    with NativeFlopCounter() as recompute:
        checkpoint(lambda a: layer(a).square(), x, use_reentrant=False).sum().backward()
    torch.testing.assert_close(layer.weight.grad, expected, rtol=0, atol=0)
    torch.testing.assert_close(x.grad, expected_x, rtol=0, atol=0)
    assert recompute.get_total_flops() == plain.get_total_flops() + 2 * 3 * 4 * 8
