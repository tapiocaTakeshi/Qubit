"""Checks for the square-root amplitude paper (draft v0.1, 2026-10-17): Prop. 1-2,
Eq. (1)-(12), Tables 1-2 (10,000 random states), plus the [supplement] functions
of apqb_sqrt_amplitude.py (general rotation rule, decoherence channels)."""

import math

import pytest
import torch

import apqb_sqrt_amplitude as sa
from apqb_qbnn_v2 import APQBv2

TOL = 1e-12  # Table 2: every check is at rounding-error level
N = 10_000


def assert_max_abs(a, b, tol=TOL):
    err = (torch.as_tensor(a) - torch.as_tensor(b)).abs().max().item()
    assert err < tol, f"max abs error {err:.3e} >= {tol:.0e}"


@pytest.fixture(scope="module")
def samples():
    """Sec. 6.1 distributions: n ~ U(0,4), phi ~ U(0,2pi), kappa ~ U(0,1). The
    torch generator draws different values than the paper's NumPy script."""
    g = torch.Generator().manual_seed(20261017)
    n = 4 * torch.rand(N, generator=g, dtype=torch.float64)
    phi = 2 * math.pi * torch.rand(N, generator=g, dtype=torch.float64)
    kappa = torch.rand(N, generator=g, dtype=torch.float64)
    return n, phi, kappa


def test_pure_state_normalization_probabilities_and_chord_distance(samples):
    n, phi, _ = samples
    psi = sa.state(n, phi)
    assert_max_abs((psi.abs() ** 2).sum(-1), 1.0)
    p0, p1 = sa.probabilities(n)
    assert_max_abs(psi[..., 1].abs() ** 2, p1)
    assert_max_abs(p0 + p1, 1.0)
    b = sa.bloch(n, phi)
    assert_max_abs(b.norm(dim=-1), 1.0)
    assert_max_abs(sa.chord_distance_sq(b), n)  # Prop. 1
    assert_max_abs(4 * torch.sin(sa.polar_angle(n) / 2) ** 2, n)
    # Eq. (3) agrees with the revised paper's theta <-> (r, q) map at phi = 0.
    r, eta = sa.n_to_r_eta(n)
    theta = APQBv2.r_to_theta(r)
    assert_max_abs(APQBv2.theta_to_q(theta), eta)
    assert_max_abs(4 * torch.sin(theta) ** 2, n)


def test_density_matrix_is_valid_and_matches_bloch_form(samples):
    n, phi, kappa = samples
    rho = sa.density_matrix(n, phi, kappa)
    assert_max_abs(rho, sa.rho_from_bloch(sa.bloch(n, phi, kappa)))           # table 2 (1)
    assert_max_abs(rho, rho.conj().transpose(-1, -2))
    assert_max_abs(torch.einsum('...ii->...', rho).real, 1.0)
    assert torch.linalg.eigvalsh(rho).min() >= 0                                # table 2 (4)
    assert_max_abs(torch.linalg.det(rho).real, n * (4 - n) * (1 - kappa ** 2) / 16)  # Eq. (7)
    assert_max_abs(torch.einsum('...ij,...ji->...', rho, rho).real,
                   sa.purity(n, kappa))                                          # Eq. (8)
    assert_max_abs(2 * rho[..., 0, 1].abs(), sa.l1_coherence(n, kappa))
    # kappa = 1 is the projector onto Eq. (1).
    psi = sa.state(n, phi)
    assert_max_abs(sa.density_matrix(n, phi), psi[..., :, None] * psi[..., None, :].conj())


def test_prop2_inverse_round_trips_and_handles_degenerate_points(samples):
    n, phi, kappa = samples
    rho = sa.density_matrix(n, phi, kappa)
    n2, phi2, kappa2 = sa.from_bloch(sa.bloch_from_rho(rho))
    assert_max_abs(sa.density_matrix(n2, phi2, kappa2), rho)                     # Prop. 2 inverse
    assert (kappa2 <= 1 + TOL).all()
    # z axis and poles: phi (and at the poles kappa) are not identifiable.
    axis = torch.tensor([[0.0, 0.0, 1.0], [0.0, 0.0, -1.0], [0.0, 0.0, 0.3]], dtype=torch.float64)
    n3, phi3, kappa3 = sa.from_bloch(axis)
    assert_max_abs(n3, torch.tensor([0.0, 4.0, 1.4], dtype=torch.float64))
    assert torch.equal(phi3, torch.zeros(3, dtype=torch.float64))
    assert torch.equal(kappa3, torch.zeros(3, dtype=torch.float64))
    assert_max_abs(sa.rho_from_bloch(axis), sa.density_matrix(n3, phi3, kappa3))


def test_purity_is_one_exactly_for_kappa_one_or_poles():
    n = torch.tensor([0.0, 4.0, 1.0, 2.0], dtype=torch.float64)
    kappa = torch.tensor([0.0, 0.3, 1.0, 0.5], dtype=torch.float64)
    assert_max_abs(sa.purity(n, kappa)[:3], 1.0)
    assert sa.purity(n, kappa)[3] < 1


def test_gates_bloch_rotation_and_closed_forms(samples):
    n, phi, kappa = samples
    rho = sa.density_matrix(n, phi, kappa)
    b = sa.bloch(n, phi, kappa)
    table1 = {"X": lambda x, y, z: (x, -y, -z), "Y": lambda x, y, z: (-x, y, -z),
              "Z": lambda x, y, z: (-x, -y, z), "H": lambda x, y, z: (z, -y, x)}
    for name, U in sa.GATES.items():
        evolved = U @ rho @ U.conj().T
        b_new = torch.stack(table1[name](*b.unbind(-1)), -1)
        assert_max_abs(evolved, sa.rho_from_bloch(b_new))                       # table 2 (2)
        assert_max_abs(sa.rotation_matrix(U) @ b.unsqueeze(-1), b_new.unsqueeze(-1))
        if name != "H":
            assert_max_abs(evolved, sa.density_matrix(*sa.pauli_npk(name, n, phi, kappa)))
        assert_max_abs(evolved, sa.density_matrix(*sa.apply_unitary(U, n, phi, kappa)))
    with pytest.raises(ValueError):
        sa.pauli_npk("H", n, phi, kappa)
    p1 = (sa.GATES["H"] @ rho @ sa.GATES["H"].conj().T)[..., 1, 1].real
    assert_max_abs(p1, sa.hadamard_p1(n, phi, kappa))                           # Eq. (9)
    # Sec. 4.1: equal n, opposite phase -> opposite interference outcome.
    two = torch.tensor(2.0, dtype=torch.float64)
    assert sa.hadamard_p1(two, 0.0).item() == pytest.approx(0.0, abs=TOL)
    assert sa.hadamard_p1(two, math.pi).item() == pytest.approx(1.0, abs=TOL)


def test_rotation_rule_for_haar_random_unitaries(samples):
    n, phi, kappa = samples
    g = torch.Generator().manual_seed(20261018)
    Z = torch.complex(torch.randn(N, 2, 2, generator=g, dtype=torch.float64),
                      torch.randn(N, 2, 2, generator=g, dtype=torch.float64)) / 2 ** 0.5
    Q, R = torch.linalg.qr(Z)
    d = torch.diagonal(R, dim1=-2, dim2=-1)
    U = Q * (d / d.abs()).unsqueeze(-2)
    R_U = sa.rotation_matrix(U)
    eye = torch.eye(3, dtype=torch.float64)
    assert_max_abs(R_U @ R_U.transpose(-1, -2), eye)
    assert_max_abs(torch.linalg.det(R_U), 1.0)
    rho = sa.density_matrix(n, phi, kappa)
    evolved = U @ rho @ U.conj().transpose(-1, -2)
    b_new = torch.einsum('...ij,...j->...i', R_U, sa.bloch(n, phi, kappa))
    assert_max_abs(evolved, sa.rho_from_bloch(b_new))                           # supplement


@pytest.mark.parametrize("lam", [0.0, 0.7, 1.0])
def test_dephasing_closed_form(samples, lam):
    n, phi, kappa = samples
    rho = sa.density_matrix(n, phi, kappa)
    Z = sa.GATES["Z"]
    out = (1 + lam) / 2 * rho + (1 - lam) / 2 * (Z @ rho @ Z)
    assert_max_abs(out, sa.density_matrix(*sa.dephase(n, phi, kappa, lam)))      # supplement


@pytest.mark.parametrize("gamma", [0.0, 0.3, 0.9])
def test_amplitude_damping_closed_form(samples, gamma):
    n, phi, kappa = samples
    rho = sa.density_matrix(n, phi, kappa)
    K0 = torch.tensor([[1, 0], [0, math.sqrt(1 - gamma)]], dtype=torch.complex128)
    K1 = torch.tensor([[0, math.sqrt(gamma)], [0, 0]], dtype=torch.complex128)
    out = K0 @ rho @ K0.conj().T + K1 @ rho @ K1.conj().T
    n2, phi2, kappa2 = sa.amplitude_damp(n, phi, kappa, gamma)
    assert_max_abs(out, sa.density_matrix(n2, phi2, kappa2))                    # supplement
    assert (kappa2 <= kappa + TOL).all()
    pole = torch.tensor([4.0], dtype=torch.float64)
    assert torch.isfinite(sa.amplitude_damp(pole, pole * 0, pole * 0 + 0.5, 0.0)[2]).all()


def test_latent_parameterization_and_gradients():
    s = torch.linspace(-30, 30, 2001, dtype=torch.float64, requires_grad=True)
    n, r, eta = sa.from_latent(s)
    assert torch.isfinite(eta).all() and (n >= 0).all() and (n <= 4).all()
    assert_max_abs(r ** 2 + eta ** 2, 1.0)
    assert_max_abs(eta, APQBv2.from_latent(s)[1])
    assert_max_abs(eta, sa.eta_from_n(n), tol=1e-7)  # sqrt near the poles amplifies rounding
    dn, = torch.autograd.grad(n.sum(), s, retain_graph=True)
    dr, = torch.autograd.grad(r.sum(), s, retain_graph=True)
    deta, = torch.autograd.grad(eta.sum(), s)
    assert_max_abs(dn, -2 * eta ** 2)                                            # Eq. (12)
    assert_max_abs(dr, eta ** 2)
    assert_max_abs(deta, -r * eta)
