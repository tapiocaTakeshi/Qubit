"""Exact local attention, repeatable inference, and memory-saving training."""

import copy
from unittest.mock import patch

import pytest
import torch
import torch.nn.functional as F

from neuroquantum_layered import LocalAttention, NeuroQuantum, NeuroQuantumConfig, QBNNLayer


@pytest.fixture(autouse=True)
def deterministic_cpu_tests():
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    with torch.random.fork_rng():
        torch.manual_seed(31)
        yield
    torch.set_num_threads(threads)


def dense_reference(q, k, v, window, mask=None):
    """Independent dense definition, including completely masked query rows."""
    positions = torch.arange(q.shape[-2], device=q.device)
    distance = positions[:, None] - positions[None, :]
    allowed = (distance >= 0) & (distance < window)
    if mask is not None:
        allowed = allowed & (mask != 0)
    scores = (q @ k.transpose(-2, -1)) / q.shape[-1] ** 0.5
    scores = scores.masked_fill(~allowed, float("-inf"))
    scores = scores.masked_fill(~allowed.any(dim=-1, keepdim=True), 0)
    weights = scores.softmax(dim=-1).masked_fill(~allowed, 0)
    return weights @ v


@pytest.mark.parametrize("length,window", [(1, 1), (3, 8), (4, 4), (11, 1), (13, 4)])
@pytest.mark.parametrize("mask_kind", ["none", "square", "padding", "per_head"])
def test_local_attention_matches_dense_outputs_and_gradients(length, window, mask_kind):
    attention = LocalAttention(8, 2, attention_window=window, dropout=0).double()
    q, k, v = [torch.randn(2, 2, length, 4, dtype=torch.float64, requires_grad=True)
               for _ in range(3)]
    mask = None
    if mask_kind == "square":
        mask = torch.rand(length, length) > 0.3
        mask[0] = False
    elif mask_kind == "padding":
        mask = torch.ones(2, 1, 1, length, dtype=torch.int64)
        mask[0, ..., :2] = 0
        mask[1, ..., -2:] = 0
    elif mask_kind == "per_head":
        mask = (torch.rand(2, 2, length, length) > 0.3).float()
        mask[1, 0] = 0
    expected = dense_reference(q, k, v, window, mask)
    actual = attention._local_attention(q, k, v, mask)
    torch.testing.assert_close(actual, expected, rtol=1e-9, atol=1e-10)
    probe = torch.randn_like(actual)
    expected_grads = torch.autograd.grad((expected * probe).sum(), (q, k, v))
    actual_grads = torch.autograd.grad((actual * probe).sum(), (q, k, v))
    for actual_grad, expected_grad in zip(actual_grads, expected_grads):
        torch.testing.assert_close(actual_grad, expected_grad, rtol=1e-9, atol=1e-10)


def test_long_attention_only_computes_bounded_windows_and_keeps_no_dense_cache():
    window = 8
    attention = LocalAttention(16, 2, attention_window=window, dropout=0).eval()
    sdpa = F.scaled_dot_product_attention
    for length in (33, 34, 35):
        with patch.object(F, "scaled_dot_product_attention", wraps=sdpa) as calls:
            attention(torch.randn(1, length, 16))
        assert len(calls.call_args_list) > 1
        for call in calls.call_args_list:
            q, k, v = call.args[:3]
            assert q.shape[-2] <= window
            assert k.shape[-2] <= 2 * window - 1
            assert v.shape[-2] == k.shape[-2]
    assert not getattr(attention, "_window_mask_cache", {})


def test_short_attention_uses_native_causal_path_without_allocating_a_mask():
    attention = LocalAttention(16, 2, attention_window=16, dropout=0).eval()
    with patch.object(F, "scaled_dot_product_attention",
                      wraps=F.scaled_dot_product_attention) as calls:
        attention(torch.randn(2, 11, 16))
    assert calls.call_count == 1
    assert calls.call_args.kwargs.get("attn_mask") is None
    assert calls.call_args.kwargs.get("is_causal") is True


@pytest.mark.parametrize("use_core", [False, True])
def test_qbnn_inference_is_independent_of_prior_requests_and_checkpoint_counter(use_core):
    layer = QBNNLayer(8, 12, use_qbnn_layered=use_core)
    if use_core:
        assert layer.use_qbnn_layered, "Install requests to exercise the EQBNN backend"
    inputs = torch.randn(2, 7, 8)
    trained_output = layer(inputs)
    layer.eval()
    first = layer(inputs)
    layer(torch.randn_like(inputs))
    layer.call_count.fill_(37)  # Simulate restoring an old serving checkpoint.
    restored = copy.deepcopy(layer)
    restored.load_state_dict(layer.state_dict(), strict=True)
    torch.testing.assert_close(first, trained_output, rtol=0, atol=0)
    torch.testing.assert_close(layer(inputs), first, rtol=0, atol=0)
    torch.testing.assert_close(restored(inputs), first, rtol=0, atol=0)
    expected_lambda = layer.lambda_min + (layer.lambda_max - layer.lambda_min) * (
        0.7 * torch.sigmoid(layer.lambda_base).item() + 0.15
    )
    assert layer.get_quantum_info()["lambda_eff"] == pytest.approx(expected_lambda)


def tiny_model(checkpointing=False, dropout=0):
    return NeuroQuantum(NeuroQuantumConfig(
        vocab_size=32, embed_dim=16, hidden_dim=24, num_heads=2,
        num_layers=2, max_seq_len=32, attention_window=4,
        dropout=dropout, gradient_checkpointing=checkpointing,
    ))


def test_model_preserves_prefix_logits_and_uses_implicit_causality():
    model = tiny_model().eval()
    tokens = torch.randint(0, 32, (2, 11))
    with patch.object(model.transformer_blocks[0], "forward",
                      wraps=model.transformer_blocks[0].forward) as block:
        full = model(tokens)
    assert block.call_args.args[1] is None
    prefix = model(tokens[:, :7])
    torch.testing.assert_close(prefix, full[:, :7], rtol=1e-5, atol=1e-6)
    altered = tokens.clone()
    altered[:, 7:] = torch.randint(0, 32, (2, 4))
    torch.testing.assert_close(model(altered)[:, :7], prefix, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(model.forward_with_details(tokens)["logits"], full)


def test_checkpointing_reduces_saved_activations_and_preserves_dropout_gradients():
    eager = tiny_model(dropout=0.2).train()
    checkpointed = copy.deepcopy(eager)
    checkpointed.config.gradient_checkpointing = True
    tokens = torch.randint(0, 32, (2, 11))
    results = []
    for model in (eager, checkpointed):
        saved_sizes = []

        def pack(tensor):
            saved_sizes.append(tensor.numel())
            return tensor

        torch.manual_seed(99)
        with torch.autograd.graph.saved_tensors_hooks(pack, lambda tensor: tensor):
            logits = model(tokens)
            loss = F.cross_entropy(logits[:, :-1].reshape(-1, 32), tokens[:, 1:].reshape(-1))
        loss.backward()
        grads = {name: p.grad for name, p in model.named_parameters() if p.grad is not None}
        results.append((logits, grads, sum(saved_sizes)))
    torch.testing.assert_close(results[0][0], results[1][0], rtol=0, atol=0)
    assert results[0][1].keys() == results[1][1].keys()
    for name in results[0][1]:
        torch.testing.assert_close(results[0][1][name], results[1][1][name], rtol=1e-5, atol=1e-6)
    assert results[1][2] < results[0][2] / 2


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires a Runpod/CUDA GPU")
def test_cuda_mixed_precision_forward_and_backward(dtype):
    if dtype == torch.bfloat16 and not torch.cuda.is_bf16_supported():
        pytest.skip("GPU does not support BF16")
    model = tiny_model(checkpointing=True, dropout=0.1).cuda().train()
    tokens = torch.randint(0, 32, (2, 11), device="cuda")
    with torch.autocast("cuda", dtype=dtype):
        logits = model(tokens)
        loss = F.cross_entropy(logits[:, :-1].reshape(-1, 32), tokens[:, 1:].reshape(-1))
    assert torch.isfinite(loss)
    loss.backward()
    for p in model.parameters():
        if p.grad is not None:
            assert torch.isfinite(p.grad).all()
