"""Regression tests for trainable QBNN gates and APQB numerical stability."""

import math

import pytest
import torch
import torch.nn.functional as F

from apqb_qbnn_v2 import APQBv2, QBNNLayerV2, StochasticQBNNLayerV2


@pytest.mark.parametrize("rank", [None, 3])
@pytest.mark.parametrize("stochastic", [False, True])
def test_default_correlation_branches_learn_from_task_loss(rank, stochastic):
    with torch.random.fork_rng():
        torch.manual_seed(17)
        cls = StochasticQBNNLayerV2 if stochastic else QBNNLayerV2
        kwargs = {"beta": 0.0} if stochastic else {}
        layer = cls(4, 6, rank=rank, **kwargs)
        h = torch.randn(3, 5, 4)
        target = torch.randn(3, 5, 6)

    # Zero-initialized J must preserve the initial plain-layer output.
    torch.testing.assert_close(layer(h), torch.tanh(layer.W(h)), rtol=0, atol=0)
    optimizer = torch.optim.SGD(layer.parameters(), lr=0.1)
    for step in range(2):
        optimizer.zero_grad()
        F.mse_loss(layer(h), target).backward()
        names = ("J_r", "J_q") if rank is None else ("V_r", "V_q")
        if step == 1:
            names += ("P.weight", "lambda_r", "lambda_q")
            if rank is not None:
                names += ("U_r", "U_q")
        for name in names:
            gradient = dict(layer.named_parameters())[name].grad
            assert gradient is not None, name
            assert torch.isfinite(gradient).all(), name
            assert torch.count_nonzero(gradient) > 0, name
        optimizer.step()

    assert not torch.equal(layer(h), torch.tanh(layer.W(h)))


@pytest.mark.parametrize("rank", [None, 3])
def test_explicit_zero_gates_and_existing_state_dict_remain_supported(rank):
    original = QBNNLayerV2(4, 6, rank=rank, lambda_r_init=0, lambda_q_init=0)
    restored = QBNNLayerV2(4, 6, rank=rank)
    restored.load_state_dict(original.state_dict(), strict=True)
    h = torch.randn(5, 4)
    assert restored.lambda_r.item() == restored.lambda_q.item() == 0
    torch.testing.assert_close(restored(h), torch.tanh(original.W(h)), rtol=0, atol=0)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64])
def test_large_latents_have_finite_outputs_and_gradients(dtype):
    a = torch.tensor([-1000, -100, -20, -2, 0, 2, 20, 100, 1000],
                     dtype=dtype, requires_grad=True)
    r, q, theta = APQBv2.from_latent(a)
    for value in (r, q, theta):
        assert value.dtype == dtype
        assert torch.isfinite(value).all()
        gradient, = torch.autograd.grad(value.sum(), a, retain_graph=True)
        assert torch.isfinite(gradient).all()
    assert ((q >= 0) & (q <= 1)).all()
    assert ((theta >= 0) & (theta <= math.pi / 2)).all()
    torch.testing.assert_close(r.square() + q.square(), torch.ones_like(a),
                               atol=4 * torch.finfo(dtype).eps, rtol=0)


def test_latent_parameterization_preserves_values_and_derivatives():
    a = torch.tensor([-5.0, -1.0, 0.0, 1.0, 5.0], dtype=torch.float64,
                     requires_grad=True)
    r, q, theta = APQBv2.from_latent(a)
    torch.testing.assert_close(q, 1 / torch.cosh(a))
    torch.testing.assert_close(theta, torch.atan(torch.exp(-a)))
    for value, expected in ((r, q.square()), (q, -r * q), (theta, -q / 2)):
        gradient, = torch.autograd.grad(value.sum(), a, retain_graph=True)
        torch.testing.assert_close(gradient, expected)
    assert torch.autograd.gradcheck(APQBv2.from_latent, (a,))
    assert torch.autograd.gradgradcheck(APQBv2.from_latent, (a,))


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64])
def test_entropy_is_finite_and_symmetric_at_deterministic_endpoints(dtype):
    r = torch.tensor([-1, -0.5, 0, 0.5, 1], dtype=dtype, requires_grad=True)
    entropy = APQBv2.entropy_z(r)
    expected = torch.tensor([0, 0.8112781244591328, 1, 0.8112781244591328, 0],
                            dtype=dtype)
    torch.testing.assert_close(entropy, expected, rtol=0, atol=4 * torch.finfo(dtype).eps)
    assert entropy[0].item() == entropy[-1].item() == 0
    assert torch.isfinite(entropy).all()
    entropy.sum().backward()
    assert torch.isfinite(r.grad).all()


def test_entropy_retains_interior_derivatives():
    r = torch.tensor([-0.9, -0.2, 0.0, 0.2, 0.9], dtype=torch.float64,
                     requires_grad=True)
    gradient, = torch.autograd.grad(APQBv2.entropy_z(r).sum(), r)
    expected = 0.5 * torch.log2((1 - r) / (1 + r))
    torch.testing.assert_close(gradient, expected)


def test_layer_with_large_latent_clip_has_finite_backward():
    layer = QBNNLayerV2(2, 4, a_clip=1000.0)
    with torch.no_grad():
        layer.P.weight.zero_()
        layer.P.bias.copy_(torch.tensor([-100.0, -20.0, 20.0, 100.0]))
        layer.J_r.fill_(0.2)
        layer.J_q.fill_(0.3)
    h = torch.ones(3, 2, requires_grad=True)
    layer(h).square().mean().backward()
    for name, parameter in layer.named_parameters():
        assert parameter.grad is not None, name
        assert torch.isfinite(parameter.grad).all(), name
    assert torch.isfinite(h.grad).all()
