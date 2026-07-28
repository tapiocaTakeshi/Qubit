#!/usr/bin/env python3
"""
Tests for apqb_qbnn_v2.py against the propositions and equations of the
APQB/QBNN v2 research draft (Appendix A proofs, Eq. 4-26).
"""

import math
import warnings

import torch
import numpy as np
from numpy.polynomial import chebyshev as npcheb

from apqb_qbnn_v2 import (
    APQBv2,
    IndependentGateLayerV2,
    chebyshev_features,
    subset_product_features,
    num_subset_terms,
    QBNNLayerV2,
    StochasticQBNNLayerV2,
    qbnn_regularization_loss,
    apqb_correlation_bank,
    nearest_correlation_matrix,
    calibrated_temperature,
)

torch.manual_seed(0)


def check(name, cond):
    status = "PASS" if cond else "FAIL"
    print(f"[{status}] {name}")
    assert cond, name


def test_prop1_normalization_and_recovery():
    r = torch.linspace(-0.999, 0.999, 101)
    theta = APQBv2.r_to_theta(r)
    state = APQBv2.theta_to_state(theta)
    norm = (state ** 2).sum(dim=-1)
    check("Prop.1 normalization: ||psi_r||^2 == 1", torch.allclose(norm, torch.ones_like(norm), atol=1e-5))

    p0 = torch.cos(theta) ** 2
    p1 = torch.sin(theta) ** 2
    z_expect = p0 - p1
    check("Prop.1 recovery: <Z> = P(0)-P(1) == r", torch.allclose(z_expect, r, atol=1e-4))


def test_bloch_constraint():
    r = torch.linspace(-1, 1, 50)
    q = APQBv2.q_from_r(r)
    check("Sec.3.3: r^2+q^2 == 1 (endpoint embedding)", torch.allclose(r ** 2 + q ** 2, torch.ones_like(r), atol=1e-6))


def test_stable_latent_parameterization():
    a = torch.linspace(-5, 5, 200)
    r, q, theta = APQBv2.from_latent(a)
    check("Eq.12: tanh^2(a)+sech^2(a) == 1", torch.allclose(r ** 2 + q ** 2, torch.ones_like(a), atol=1e-6))

    # dr/da = q^2, dq/da = -rq
    a_grad = a.clone().requires_grad_(True)
    r_g = torch.tanh(a_grad)
    q_g = 1.0 / torch.cosh(a_grad)
    dr_da = torch.autograd.grad(r_g.sum(), a_grad, retain_graph=True)[0]
    dq_da = torch.autograd.grad(q_g.sum(), a_grad)[0]
    check("Eq.12: dr/da == q^2", torch.allclose(dr_da, q_g ** 2, atol=1e-4))
    check("Eq.12: dq/da == -r*q", torch.allclose(dq_da, -r_g * q_g, atol=1e-4))


def test_entropy_and_coherence_endpoints():
    r_center = torch.tensor([0.0])
    r_edge = torch.tensor([1.0 - 1e-6])
    H_center = APQBv2.entropy_z(r_center)
    H_edge = APQBv2.entropy_z(r_edge)
    check("Eq.10: H_Z(0) ~ 1 bit", torch.allclose(H_center, torch.ones_like(H_center), atol=1e-3))
    check("Eq.10: H_Z(r->1) ~ 0", H_edge.item() < 1e-3)

    coherence = APQBv2.coherence_l1(torch.tensor([0.6]))
    check("Sec.3.4: coherence C_l1 == q", torch.allclose(coherence, APQBv2.q_from_r(torch.tensor([0.6]))))


def test_prop2_chebyshev_relation():
    r = torch.tensor(np.random.uniform(-0.99, 0.99, size=20), dtype=torch.float64)
    q = APQBv2.q_from_r(r)
    K = 6
    reals, imags = chebyshev_features(r, q, K)

    ok_real, ok_imag = True, True
    for k in range(1, K + 1):
        # T_k has a Chebyshev coefficient vector with a single 1 at position k.
        coeffs = np.zeros(k + 1)
        coeffs[k] = 1.0
        Tk_val = npcheb.chebval(r.numpy(), coeffs)
        if not np.allclose(reals[:, k - 1].numpy(), Tk_val, atol=1e-6):
            ok_real = False

        if k >= 1:
            # U_{k-1} via sin(k*arccos(r)) / sin(arccos(r))  (standard identity)
            acos_r = np.arccos(r.numpy())
            sin_acos = np.sin(acos_r)
            with np.errstate(divide='ignore', invalid='ignore'):
                Uk_1 = np.where(np.abs(sin_acos) > 1e-9, np.sin(k * acos_r) / sin_acos, k)
            expected_imag = q.numpy() * Uk_1
            if not np.allclose(imags[:, k - 1].numpy(), expected_imag, atol=1e-6):
                ok_imag = False

    check("Prop.2: Re(z^k) == T_k(r)", ok_real)
    check("Prop.2: Im(z^k) == q*U_{k-1}(r)", ok_imag)


def test_single_z_is_not_2n_degrees_of_freedom():
    # Sec. 4.3: a degree-K single-variable expansion has K+1 complex
    # coefficients, not 2^n for any n > 1 relevant here.
    K = 5
    r = torch.rand(3)
    q = APQBv2.q_from_r(r)
    reals, imags = chebyshev_features(r, q, K)
    check("Sec.4.3: single-z expansion has K coefficients, not 2^d",
          reals.shape[-1] == K and reals.shape[-1] != 2 ** 4)


def test_prop3_subset_product_count():
    d = 4
    rs = [torch.rand(5) * 2 - 1 for _ in range(d)]
    qs = [APQBv2.q_from_r(r) for r in rs]
    zs = [APQBv2.z_from_r_q(r, q) for r, q in zip(rs, qs)]

    feats_full = subset_product_features(zs)
    check("Prop.3: full subset count == 2^d", len(feats_full) == 2 ** d == num_subset_terms(d, d))
    check("Prop.3: empty set Phi_{} == 1", torch.allclose(feats_full[()].real, torch.ones(5)) and
          torch.allclose(feats_full[()].imag, torch.zeros(5)))

    feats_k2 = subset_product_features(zs, max_degree=2)
    expected_k2 = sum(math.comb(d, k) for k in range(3))
    check("Sec.4.5: degree-limited count == sum_{k<=K} C(d,k)", len(feats_k2) == expected_k2 == num_subset_terms(d, 2))

    # spot-check one product against direct multiplication
    S = (0, 2)
    direct = zs[0] * zs[2]
    check("Subset product Phi_{0,2} == z0*z2", torch.allclose(feats_full[S].real, direct.real, atol=1e-6)
          and torch.allclose(feats_full[S].imag, direct.imag, atol=1e-6))


def test_qbnn_layer_reduces_to_plain_layer_at_lambda_zero():
    torch.manual_seed(1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        layer = QBNNLayerV2(8, 16, lambda_r_init=0.0, lambda_q_init=0.0)
    h = torch.randn(4, 8)
    out = layer(h)
    expected = torch.tanh(layer.W(h))
    check("Sec.6.1(i): lambda_r=lambda_q=0 reduces to plain affine+activation",
          torch.allclose(out, expected, atol=1e-6))

    # r,q recorded and satisfy the constraint
    err = layer.constraint_error()
    check("QBNNLayerV2: r^2+q^2 ~= 1 on the forward pass", torch.allclose(err, torch.zeros_like(err), atol=1e-4))


def test_qbnn_layer_low_rank_matches_shapes():
    layer = QBNNLayerV2(8, 16, rank=4, lambda_r_init=0.1, lambda_q_init=0.1)
    h = torch.randn(3, 8)
    out = layer(h)
    check("Low-rank QBNNLayerV2 forward shape", out.shape == (3, 16))
    loss = out.pow(2).mean() + qbnn_regularization_loss(layer, beta_J=1e-3)
    loss.backward()
    check("Low-rank QBNNLayerV2 backward runs and populates grads",
          layer.U_r.grad is not None and layer.V_r.grad is not None)


def test_dead_gate_at_zero_lambda_and_zero_J():
    """The paper's Appendix C allows lambda=0 and J=0 simultaneously, which
    is a zero-gradient saddle: the gate can never activate. The layer must
    warn about it, and the gradients must actually be zero."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        layer = QBNNLayerV2(6, 6, rank=3, lambda_r_init=0.0, lambda_q_init=0.0)
    check("Dead-gate init raises a RuntimeWarning",
          any(issubclass(w.category, RuntimeWarning) for w in caught))

    layer(torch.randn(8, 6)).pow(2).sum().backward()
    check("Dead-gate init: grad(lambda_r) == 0", layer.lambda_r.grad.abs().item() == 0.0)
    check("Dead-gate init: grad(V_r) == 0", layer.V_r.grad.abs().sum().item() == 0.0)

    # A small nonzero lambda restores gradient flow into J.
    layer2 = QBNNLayerV2(6, 6, rank=3, lambda_r_init=0.01, lambda_q_init=0.01)
    layer2(torch.randn(8, 6)).pow(2).sum().backward()
    check("lambda=0.01 restores grad(V_r) != 0", layer2.V_r.grad.abs().sum().item() > 0.0)


def test_independent_gate_control_breaks_constraint():
    """Sec. 8.5 Independent-gates control: same gate arithmetic, but r and q
    must NOT satisfy r^2+q^2=1 (that is the whole point of the control)."""
    layer = IndependentGateLayerV2(8, 8, rank=4, lambda_r_init=0.01, lambda_q_init=0.01)
    layer(torch.randn(16, 8))
    err = (layer.last_r ** 2 + layer.last_q ** 2 - 1.0).abs()
    check("Independent-gates: r^2+q^2 != 1 (constraint removed)", err.max().item() > 1e-3)
    check("Independent-gates: r in [-1,1]", layer.last_r.abs().max().item() <= 1.0)
    check("Independent-gates: q in [0,1]",
          layer.last_q.min().item() >= 0.0 and layer.last_q.max().item() <= 1.0)


def test_ablation_flags():
    """Sec. 8.5 rows QBNN-r / QBNN-q / Random-J."""
    h = torch.randn(4, 8)

    layer_r = QBNNLayerV2(8, 8, rank=4, lambda_r_init=0.5, lambda_q_init=0.5, use_q=False)
    layer_q = QBNNLayerV2(8, 8, rank=4, lambda_r_init=0.5, lambda_q_init=0.5, use_r=False)
    check("QBNN-r / QBNN-q forward shapes", layer_r(h).shape == (4, 8) and layer_q(h).shape == (4, 8))

    random_j = QBNNLayerV2(8, 8, rank=4, lambda_r_init=0.5, lambda_q_init=0.5, learnable_J=False)
    j_names = {n for n, _ in random_j.named_parameters()}
    check("Random-J: J factors are buffers, not parameters",
          'V_r' not in j_names and 'U_r' not in j_names)
    random_j(h).pow(2).sum().backward()
    check("Random-J: gate is non-trivial (lambda receives gradient)",
          random_j.lambda_r.grad.abs().item() > 0.0)


def test_stochastic_layer_noise_only_in_training():
    torch.manual_seed(2)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        layer = StochasticQBNNLayerV2(6, 6, beta=1.0, lambda_r_init=0.0, lambda_q_init=0.0)
    h = torch.randn(2, 6)

    layer.eval()
    out1 = layer(h)
    out2 = layer(h)
    check("Stochastic layer deterministic in eval mode", torch.allclose(out1, out2))

    layer.train()
    out3 = layer(h)
    out4 = layer(h)
    check("Stochastic layer stochastic in train mode", not torch.allclose(out3, out4))


def test_regularization_loss_terms():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        layer = QBNNLayerV2(4, 4, lambda_r_init=0.0, lambda_q_init=0.0)
    h = torch.randn(2, 4)
    layer(h)
    r_target = torch.zeros_like(layer.last_r)
    reg = qbnn_regularization_loss(layer, r_target=r_target, alpha=1.0, beta_J=0.0, gamma=0.0)
    manual = torch.nn.functional.mse_loss(layer.last_r, r_target)
    check("Eq.25: L_corr term matches manual MSE", torch.allclose(reg, manual, atol=1e-6))


def test_correlation_bank_and_nearest_correlation_matrix():
    R = torch.tensor([[1.0, 0.9, -0.9], [0.9, 1.0, 0.9], [-0.9, 0.9, 1.0]])
    r, q = apqb_correlation_bank(R)
    check("Sec.5.2: r bank == R", torch.allclose(r, R))
    check("Sec.5.2: q bank == sqrt(1-R^2)", torch.allclose(q, torch.sqrt(1 - R ** 2)))

    # R above is not PSD (check), nearest_correlation_matrix should fix it
    eigvals = torch.linalg.eigvalsh(R)
    check("sanity: seed matrix is not PSD", eigvals.min().item() < -1e-6)

    R_fixed = nearest_correlation_matrix(R, iters=200)
    eigvals_fixed = torch.linalg.eigvalsh(R_fixed)
    diag = torch.diagonal(R_fixed)
    check("Higham projection: result is PSD", eigvals_fixed.min().item() > -1e-5)
    check("Higham projection: unit diagonal", torch.allclose(diag, torch.ones_like(diag), atol=1e-4))
    check("Higham projection: symmetric", torch.allclose(R_fixed, R_fixed.t(), atol=1e-6))


def test_calibrated_temperature_mapping():
    q = torch.tensor([0.0, 0.5, 1.0])
    tau = calibrated_temperature(q, tau_min=0.3, tau_max=1.5, gamma=1.0)
    check("Eq.26: tau(q=0) == tau_min", math.isclose(tau[0].item(), 0.3, abs_tol=1e-6))
    check("Eq.26: tau(q=1) == tau_max", math.isclose(tau[2].item(), 1.5, abs_tol=1e-6))
    check("Eq.26: tau is monotonic in q", tau[0] < tau[1] < tau[2])


def main():
    tests = [
        test_prop1_normalization_and_recovery,
        test_bloch_constraint,
        test_stable_latent_parameterization,
        test_entropy_and_coherence_endpoints,
        test_prop2_chebyshev_relation,
        test_single_z_is_not_2n_degrees_of_freedom,
        test_prop3_subset_product_count,
        test_qbnn_layer_reduces_to_plain_layer_at_lambda_zero,
        test_qbnn_layer_low_rank_matches_shapes,
        test_dead_gate_at_zero_lambda_and_zero_J,
        test_independent_gate_control_breaks_constraint,
        test_ablation_flags,
        test_stochastic_layer_noise_only_in_training,
        test_regularization_loss_terms,
        test_correlation_bank_and_nearest_correlation_matrix,
        test_calibrated_temperature_mapping,
    ]
    failures = 0
    for t in tests:
        print(f"--- {t.__name__} ---")
        try:
            t()
        except AssertionError as e:
            failures += 1
            print(f"  -> ASSERTION FAILED: {e}")
    print()
    if failures:
        print(f"{failures}/{len(tests)} test functions had failing checks")
        raise SystemExit(1)
    print(f"All {len(tests)} test functions passed")


if __name__ == "__main__":
    main()
