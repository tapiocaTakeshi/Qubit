#!/usr/bin/env python3
"""
APQB square-root amplitude coordinates (n, phi, kappa) -- reference
implementation of the paper "APQBにおける平方根振幅座標の定式化" (research
draft v0.1, 2026-10-17). Equation, proposition and table numbers follow that
paper. Functions marked [supplement] go beyond it: they are standard
single-qubit results expressed in (n, phi, kappa) and are verified in
tests/test_apqb_sqrt_amplitude.py like the rest.

    |psi(n, phi)> = sqrt(4-n)/2 |0> + e^{i phi} sqrt(n)/2 |1>      Eq. (1)

n = 4 P(1) in [0, 4] is the squared chord distance from the north pole |0>
(Prop. 1), phi is the relative phase and kappa in [0, 1] scales the
off-diagonal element of the density matrix (Eq. 5). Every single-qubit
density matrix has this form (Prop. 2).

Relation to apqb_qbnn_v2 (revised paper, 2026-10-09): r = 1 - n/2 = <Z>,
eta = sqrt(n(4-n))/2 = sqrt(1 - r^2) is that paper's coherence coordinate q,
and the latent map r = tanh(s), eta = sech(s) is APQBv2.from_latent (Eq. 12).

This is a re-formulation of standard single-qubit quantum information; it
makes no speed-up or quantum-advantage claim (Sec. 7). All functions are pure
tensor functions and broadcast over leading dimensions.
"""

import torch

# Pauli and Hadamard matrices (complex128), used by Table 1 and rotation_matrix.
_SX = torch.tensor([[0, 1], [1, 0]], dtype=torch.complex128)
_SY = torch.tensor([[0, -1j], [1j, 0]], dtype=torch.complex128)
_SZ = torch.tensor([[1, 0], [0, -1]], dtype=torch.complex128)
PAULI = (_SX, _SY, _SZ)
GATES = {
    "X": _SX,
    "Y": _SY,
    "Z": _SZ,
    "H": torch.tensor([[1, 1], [1, -1]], dtype=torch.complex128) / 2 ** 0.5,
}


def sech(x):
    """sech(x) without overflowing cosh(x), including in backward."""
    return 2.0 * torch.exp(-torch.logaddexp(x, -x))


def eta_from_n(n):
    """Eq. (3): eta = sqrt(n(4-n))/2, the pure-state coherence <X> at phi=0."""
    return torch.sqrt(torch.clamp(n * (4 - n), min=0.0)) / 2


# ---------------------------------------------------------------------------
# Sec. 2: pure states and geometry (Eq. 1-4, Prop. 1)
# ---------------------------------------------------------------------------

def state(n, phi):
    """Eq. (1): amplitudes [..., 2] (complex) of |psi(n, phi)>."""
    n, phi = torch.broadcast_tensors(torch.as_tensor(n), torch.as_tensor(phi))
    a0 = torch.sqrt(torch.clamp(4 - n, min=0.0)) / 2
    a1 = torch.sqrt(torch.clamp(n, min=0.0)) / 2
    return torch.stack([a0.to(torch.complex128),
                        torch.polar(a1.to(torch.float64), phi.to(torch.float64))], dim=-1)


def probabilities(n):
    """Eq. (2): (P(0), P(1)) = (1 - n/4, n/4)."""
    return 1 - n / 4, n / 4


def n_to_r_eta(n):
    """Eq. (3): r = cos 2theta = 1 - n/2 and eta = sqrt(1 - r^2)."""
    return 1 - n / 2, eta_from_n(n)


def chord_distance_sq(b):
    """Prop. 1: squared Euclidean distance from the north pole e_z; equals n
    for the Bloch vector of Eq. (4)."""
    e_z = torch.tensor([0.0, 0.0, 1.0], dtype=b.dtype, device=b.device)
    return ((b - e_z) ** 2).sum(-1)


def polar_angle(n):
    """[supplement] Sec. 2.2 notes n is not the geodesic distance; the geodesic
    angle from |0> is Theta = arccos(1 - n/2), i.e. n = 4 sin^2(Theta/2)."""
    return torch.arccos(torch.clamp(1 - n / 2, -1.0, 1.0))


# ---------------------------------------------------------------------------
# Sec. 3: mixed states (Eq. 5-8, Prop. 2)
# ---------------------------------------------------------------------------

def bloch(n, phi, kappa=1.0):
    """Eq. (6) (Eq. (4) at kappa = 1): b = (k eta cos phi, k eta sin phi, 1 - n/2)."""
    n, phi, kappa = torch.broadcast_tensors(*(torch.as_tensor(t, dtype=torch.float64)
                                             for t in (n, phi, kappa)))
    t = kappa * eta_from_n(n)
    return torch.stack([t * torch.cos(phi), t * torch.sin(phi), 1 - n / 2], dim=-1)


def density_matrix(n, phi, kappa=1.0):
    """Eq. (5): rho(n, phi, kappa) [..., 2, 2] (complex128)."""
    n, phi, kappa = torch.broadcast_tensors(*(torch.as_tensor(t, dtype=torch.float64)
                                             for t in (n, phi, kappa)))
    c = kappa * eta_from_n(n) / 2
    off = torch.polar(c, phi)  # rho_10 = c e^{i phi}
    rho00 = (1 - n / 4).to(torch.complex128)
    rho11 = (n / 4).to(torch.complex128)
    return torch.stack([torch.stack([rho00, off.conj()], -1),
                        torch.stack([off, rho11], -1)], -2)


def rho_from_bloch(b):
    """rho = (I + b . sigma) / 2 for b [..., 3]."""
    b = b.to(torch.complex128)
    eye = torch.eye(2, dtype=torch.complex128, device=b.device)
    return (eye + sum(b[..., i, None, None] * PAULI[i].to(b.device) for i in range(3))) / 2


def bloch_from_rho(rho):
    """b_i = Tr(rho sigma_i)."""
    return torch.stack([torch.einsum('...ij,ji->...', rho, s.to(rho.device)).real
                        for s in PAULI], dim=-1)


def from_bloch(b, tol=1e-12):
    """Prop. 2 inverse: Bloch vector [..., 3] -> (n, phi, kappa).

    On the z axis (kappa * eta = 0) phi is not identifiable and is returned
    as 0; at the poles (eta = 0) kappa is returned as 0 (Sec. 3.1).
    """
    n = 2 * (1 - b[..., 2])
    rxy = torch.hypot(b[..., 0], b[..., 1])
    den = 2 * eta_from_n(n)  # sqrt(n(4-n))
    defined = (den > tol) & (rxy > tol)
    phi = torch.where(defined, torch.remainder(torch.atan2(b[..., 1], b[..., 0]), 2 * torch.pi),
                      torch.zeros_like(n))
    kappa = torch.where(defined, 2 * rxy / torch.where(defined, den, torch.ones_like(den)),
                        torch.zeros_like(n))
    return n, phi, kappa


def purity(n, kappa):
    """Eq. (8): Tr rho^2 = 1 - n(4-n)(1-kappa^2)/8."""
    return 1 - n * (4 - n) * (1 - kappa ** 2) / 8


def l1_coherence(n, kappa):
    """[supplement] l1-norm coherence in the computational basis,
    C_l1(rho) = 2|rho_01| = kappa * eta: kappa is the ratio to the pure state
    with the same populations."""
    return kappa * eta_from_n(n)


# ---------------------------------------------------------------------------
# Sec. 4: gates (Eq. 9, Table 1) and decoherence [supplement]
# ---------------------------------------------------------------------------

def rotation_matrix(U):
    """[supplement] (R_U)_ij = Tr(sigma_i U sigma_j U^dagger) / 2, an SO(3) matrix
    with b' = R_U b; Table 1 lists the X, Y, Z, H cases."""
    U = U.to(torch.complex128)
    Ud = U.conj().transpose(-1, -2)
    rows = [torch.stack([torch.einsum('ij,...jk,kl,...li->...', si, U, sj, Ud).real / 2
                         for sj in PAULI], -1) for si in PAULI]
    return torch.stack(rows, -2)


def apply_unitary(U, n, phi, kappa=1.0):
    """[supplement] rho -> U rho U^dagger, carried out as b' = R_U b and mapped
    back to (n, phi, kappa) with the Prop. 2 inverse."""
    b = bloch(n, phi, kappa)
    return from_bloch(torch.einsum('...ij,...j->...i', rotation_matrix(U), b))


def pauli_npk(gate, n, phi, kappa):
    """Table 1 (n' column) extended to (n', phi', kappa') for X, Y, Z [supplement]:
    kappa is preserved and phi' is taken mod 2 pi."""
    two_pi = 2 * torch.pi
    if gate == "X":
        return 4 - n, torch.remainder(-phi, two_pi), kappa
    if gate == "Y":
        return 4 - n, torch.remainder(torch.pi - phi, two_pi), kappa
    if gate == "Z":
        return n, torch.remainder(phi + torch.pi, two_pi), kappa
    raise ValueError(f"no closed (n, phi, kappa) form for gate {gate!r}; use apply_unitary")


def hadamard_p1(n, phi, kappa=1.0):
    """Eq. (9): P(1) after a Hadamard, 1/2 - kappa sqrt(n(4-n)) cos(phi) / 4."""
    return 0.5 - kappa * eta_from_n(n) * torch.cos(torch.as_tensor(phi)) / 2


def dephase(n, phi, kappa, lam):
    """[supplement] phase damping with Kraus {sqrt((1+lam)/2) I, sqrt((1-lam)/2) Z}
    maps (n, phi, kappa) -> (n, phi, lam * kappa)."""
    return n, phi, lam * kappa


def amplitude_damp(n, phi, kappa, gamma):
    """[supplement] amplitude damping |1> -> |0> with probability gamma < 1:
    n' = (1-gamma) n, phi' = phi, kappa' = kappa sqrt((4-n)/(4-n')) <= kappa."""
    n_new = (1 - gamma) * n
    # At the |1> pole with gamma = 0 the ratio is 0/0; kappa is arbitrary there.
    den = 4 - n_new
    ratio = torch.where(den > 0, (4 - n) / torch.where(den > 0, den, torch.ones_like(den)),
                        torch.ones_like(den))
    return n_new, phi, kappa * torch.sqrt(ratio)


# ---------------------------------------------------------------------------
# Sec. 5: QBNN parameterization (Eq. 11-12)
# ---------------------------------------------------------------------------

def from_latent(s):
    """Eq. (11): unconstrained s -> (n, r, eta) with r = tanh s,
    n = 2(1 - tanh s) in (0, 4) and eta = sech s, so r^2 + eta^2 = 1 holds by
    construction. Eq. (12): dn/ds = -2 eta^2, dr/ds = eta^2 (and deta/ds = -r eta),
    so the gradient through r is attenuated by eta^2 when |s| >> 1."""
    r = torch.tanh(s)
    return 2 * (1 - r), r, sech(s)
