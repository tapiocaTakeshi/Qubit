"""Checks for the revised APQB/QBNN paper (2026-10-09): Prop. 1-5,
Eq. (3)-(12), (16), and the numerical tables 2-3."""

import math

import numpy as np
import pytest
import torch

from apqb_qbnn_v2 import (
    APQBv2,
    boolean_cube,
    boolean_fourier_coefficients,
    boolean_fourier_eval,
    chebyshev_features,
    concurrence_pure,
    ghz_state,
    nand_qbnn_layer,
    QBNNLayerV2,
    ThresholdCircuitCalculator,
    qbnn_mac_counts,
    reduced_purity,
    subset_product_features,
    three_tangle,
    two_qubit_correlated_state,
    verify_threshold_calculator,
    zz_correlation,
)

TOL = 1e-12  # Appendix B: pass threshold is max absolute error < 1e-12
R_GRID = torch.linspace(-1.0, 1.0, 1001, dtype=torch.float64)


def assert_max_abs(a, b, tol=TOL):
    """Table 2 / Appendix B criterion: max |a - b| < tol (no relative slack)."""
    a, b = torch.as_tensor(a), torch.as_tensor(b)
    err = (a - b).abs().max().item()
    assert err < tol, f"max abs error {err:.3e} >= {tol:.0e}"


def test_prop1_z_component_is_cos_2theta_with_exact_endpoints():
    theta = APQBv2.r_to_theta(R_GRID)
    state = APQBv2.theta_to_state(theta)
    q = APQBv2.q_from_r(R_GRID)
    p0, p1 = APQBv2.probabilities(R_GRID)
    assert_max_abs((state ** 2).sum(-1), torch.ones_like(R_GRID))
    assert_max_abs(state[..., 0] ** 2, p0)
    assert_max_abs(state[..., 1] ** 2, p1)
    assert_max_abs(APQBv2.theta_to_r(theta), R_GRID)
    assert_max_abs(APQBv2.theta_to_q(theta), q)
    assert_max_abs(R_GRID ** 2 + q ** 2, torch.ones_like(R_GRID))
    # Endpoints are exact unless a numerical guard is requested explicitly.
    assert APQBv2.r_to_theta(torch.tensor(1.0, dtype=torch.float64)) == 0
    assert APQBv2.r_to_theta(torch.tensor(-1.0, dtype=torch.float64)) == pytest.approx(math.pi / 2)
    assert APQBv2.r_to_theta(torch.tensor(1.0, dtype=torch.float64), eps=1e-7) > 0
    # The earlier draft's cos(theta) is not the Z expectation.
    assert not torch.allclose(torch.cos(theta), R_GRID, atol=1e-3)


def test_eq3_joint_distribution_has_pearson_r():
    r = torch.linspace(-1, 1, 21, dtype=torch.float64)
    P = APQBv2.joint_distribution(r)
    s = torch.tensor([1.0, 1.0, -1.0, -1.0], dtype=torch.float64)
    t = torch.tensor([1.0, -1.0, 1.0, -1.0], dtype=torch.float64)
    assert (P >= 0).all()
    assert_max_abs(P.sum(-1), torch.ones_like(r))
    assert_max_abs(P @ s, torch.zeros_like(r))
    assert_max_abs(P @ t, torch.zeros_like(r))
    assert_max_abs(P @ (s * t), r)


def test_eq4_density_matrix_is_pure_with_coherence_q():
    rho00, rho01, rho11 = APQBv2.density_matrix(R_GRID)
    rho = torch.stack([torch.stack([rho00, rho01], -1), torch.stack([rho01, rho11], -1)], -2)
    assert_max_abs(rho @ rho, rho)
    assert_max_abs(APQBv2.coherence_l1(R_GRID), APQBv2.q_from_r(R_GRID))


def test_eq5_latent_constraint_is_finite_and_exact():
    a = torch.tensor([-1000, -100, -4, -1, 0, 1, 4, 100, 1000], dtype=torch.float64)
    r, q, theta = APQBv2.from_latent(a)
    assert torch.isfinite(r).all() and torch.isfinite(q).all() and torch.isfinite(theta).all()
    assert_max_abs(r ** 2 + q ** 2, torch.ones_like(a))


def test_prop2_two_qubit_family_concurrence_is_abs_r():
    psi = two_qubit_correlated_state(R_GRID)
    q = APQBv2.q_from_r(R_GRID)
    assert_max_abs((psi ** 2).sum(-1), torch.ones_like(R_GRID))
    assert_max_abs(zz_correlation(psi), R_GRID)
    C2 = concurrence_pure(psi)
    assert_max_abs(C2, R_GRID.abs())
    assert_max_abs(C2 ** 2 + q ** 2, torch.ones_like(R_GRID))
    assert_max_abs(reduced_purity(psi), (1 + q ** 2) / 2)


def test_sec31_counterexample_r_does_not_determine_concurrence():
    entangled = torch.tensor([1.0, 1.0, 1.0, -1.0], dtype=torch.float64) / 2
    product_pp = torch.full((4,), 0.5, dtype=torch.float64)
    assert zz_correlation(entangled) == pytest.approx(0.0, abs=TOL)
    assert zz_correlation(product_pp) == pytest.approx(0.0, abs=TOL)
    assert concurrence_pure(entangled) == pytest.approx(1.0, abs=TOL)
    assert concurrence_pure(product_pp) == pytest.approx(0.0, abs=TOL)


def test_eq8_ghz_three_tangle_is_q_squared():
    theta = APQBv2.r_to_theta(R_GRID)
    G = ghz_state(theta)
    q = APQBv2.q_from_r(R_GRID)
    tau3 = three_tangle(G)
    assert_max_abs(tau3, q ** 2)
    assert_max_abs(R_GRID ** 2 + tau3, torch.ones_like(R_GRID))
    assert (tau3 >= 0).all()
    # W state has zero three-tangle (CKW).
    W = torch.zeros(8, dtype=torch.float64)
    W[[1, 2, 4]] = 1 / math.sqrt(3)
    assert three_tangle(W) == pytest.approx(0.0, abs=TOL)


def test_prop3_boolean_expansion_is_unique_and_reconstructs():
    d = 4
    gen = torch.Generator().manual_seed(17)
    f = torch.randn(2 ** d, generator=gen, dtype=torch.float64)
    subsets, coeffs = boolean_fourier_coefficients(f, d)
    assert len(subsets) == 2 ** d
    x = boolean_cube(d)
    assert_max_abs(boolean_fourier_eval(subsets, coeffs, x), f)
    # Integer +-1 inputs must not truncate the fractional coefficients.
    assert_max_abs(boolean_fourier_eval(subsets, coeffs, x.long()), f)


def test_sec43_subset_products_match_boolean_characters_at_endpoints():
    d = 3
    x = boolean_cube(d)
    z = [torch.complex(x[:, i], torch.zeros_like(x[:, i])) for i in range(d)]
    feats = subset_product_features(z)
    subsets, _ = boolean_fourier_coefficients(torch.zeros(2 ** d, dtype=torch.float64), d)
    for S in subsets:
        expected = x[:, list(S)].prod(-1) if S else torch.ones(2 ** d, dtype=torch.float64)
        assert_max_abs(feats[S].real, expected)
        assert torch.all(feats[S].imag == 0)


def test_eq11_chebyshev_on_grid_including_endpoints():
    r = R_GRID
    reals, imags = chebyshev_features(r, APQBv2.q_from_r(r), 6)
    two_theta = torch.acos(r)
    for k in range(1, 7):
        assert_max_abs(reals[:, k - 1], torch.cos(k * two_theta))
        assert_max_abs(imags[:, k - 1], torch.sin(k * two_theta))


def test_eq16_mac_counts():
    c = qbnn_mac_counts(batch=8, d=64, m=64, rank=4)
    assert c["plain"] == 8 * 64 * 64
    assert c["dense"] == 4 * c["plain"]
    assert c["lowrank"] / c["plain"] == pytest.approx(2 + 4 * 4 / 64)


def test_prop5_nand_truth_table_and_margin():
    layer = nand_qbnn_layer()
    x = torch.tensor([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
    assert layer(x).squeeze(-1).tolist() == [1.0, 1.0, 1.0, 0.0]
    pre = layer.W(x).squeeze(-1)
    assert pre.abs().min().item() == pytest.approx(0.5)


def test_table3_four_bit_threshold_circuit():
    results = verify_threshold_calculator(4)
    assert results == {
        "add": (256, 0),
        "sub": (256, 0),
        "mul": (256, 0),
        "div": (240, 0),
        "div_zero_flags": (16, 0),
    }


def test_prop4_lambda_zero_with_nonzero_J_is_plain_layer():
    torch.manual_seed(0)
    layer = QBNNLayerV2(5, 7, lambda_r_init=0.0, lambda_q_init=0.0).double()
    with torch.no_grad():
        layer.J_r.normal_()
        layer.J_q.normal_()
    h = torch.randn(3, 5, dtype=torch.float64)
    assert torch.equal(layer(h), torch.tanh(layer.W(h)))


def test_eq15_J_gradient_and_zero_P_gradient_at_init():
    torch.manual_seed(0)
    layer = QBNNLayerV2(5, 7).double()  # J = 0, lambda = 0.1 (Sec. 5.2)
    h = torch.randn(3, 5, dtype=torch.float64)
    out = layer(h)
    assert torch.equal(out, torch.tanh(layer.W(h)))  # Prop. 4 with J = 0
    g = torch.randn_like(out)  # dL/dh'
    (out * g).sum().backward()
    u = layer.W(h).detach()
    upstream = u * (1 - torch.tanh(u) ** 2) * g  # u_j sigma'(u_j) dL/dh'_j
    r, q = layer.last_r.detach(), layer.last_q.detach()
    assert_max_abs(layer.J_r.grad, layer.lambda_r.detach() * torch.einsum("bi,bj->ij", r, upstream))
    assert_max_abs(layer.J_q.grad, layer.lambda_q.detach() * torch.einsum("bi,bj->ij", q, upstream))
    # While J = 0 the correlation path gives P no task gradient.
    assert torch.count_nonzero(layer.P.weight.grad) == 0
    assert torch.count_nonzero(layer.P.bias.grad) == 0


def test_calculator_rejects_invalid_operands_and_widths():
    calc = ThresholdCircuitCalculator(4)
    assert calc.add(3, torch.arange(4)).tolist() == [3, 4, 5, 6]
    assert calc.add(np.uint64(3), np.uint64(5)).item() == 8
    for a, b in [(3.7, 1), (-1, 1), (16, 0)]:
        with pytest.raises((TypeError, ValueError)):
            calc.add(a, b)
    for width in (0, 32, True, 2.0):
        with pytest.raises((TypeError, ValueError)):
            ThresholdCircuitCalculator(width)
    assert ThresholdCircuitCalculator(31).mul(2 ** 31 - 1, 2 ** 31 - 1).item() == (2 ** 31 - 1) ** 2
