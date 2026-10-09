#!/usr/bin/env python3
"""
APQB / QBNN v2 -- implementation of the revised APQB/QBNN paper
(QBNN_APQB revised edition, 2026-10-09).

Scope note: this module implements the *corrected* formalization, which
deliberately narrows several claims made by an earlier draft
(qbnn_layered.py's EQBNN* classes follow that earlier, looser framing --
e.g. calling the auxiliary coordinate "temperature", or treating
cross-layer correlation as literal quantum entanglement). APQB here is a
classical, testable correlation encoding: it does not claim physical
qubits, quantum entanglement, or quantum speedup (Sec. 1, 5.3, 6.5).

Main corrections relative to the earlier draft (Appendix A, Table 4):
  * the Z component is <Z> = cos(2 theta) = r, not cos(theta) (Prop. 1);
  * the old "T = |sin 2theta|" is the coherence coordinate q, and is not
    the AI sampling temperature (Sec. 2.3); temperature is defined only
    through the calibration map of Eq. (18);
  * C_2 = |r| holds only for the explicit two-qubit family of Eq. (6),
    and tau_3 = q^2 only for the GHZ-type family of Eq. (8) (Sec. 3);
  * the NN correspondence is limited to the Boolean multilinear expansion
    (Eq. 10) and subset products of independent coordinates (Eq. 12);
  * the multiplicative correlation gate of Eq. (13)-(14) is the main QBNN
    specification;
  * an "untrained calculator" is a hand-wired threshold circuit
    (Prop. 5), not a randomly initialized QBNN (Sec. 6.2-6.4).

Section references (e.g. "Eq. (5)", "Sec. 5.3", "Prop. 2") point at the
corresponding equations/sections/propositions in the revised paper.
"""

import math
import numbers
import operator
from itertools import combinations, product

import torch
import torch.nn as nn
import torch.nn.functional as F


# ===========================================================================
# Section 2: APQB core -- correlation encoding r -> (theta, q) (Eq. 1-5)
# ===========================================================================

class APQBv2:
    """Stateless APQB math: r <-> theta <-> q, and the stable latent
    parameterization of Sec. 2.4. All methods are pure tensor functions.

    The encoding covers only a subset of single-qubit states (real, q >= 0);
    one real number r does not describe a general qubit (Sec. 2.1)."""

    @staticmethod
    def r_to_theta(r, eps=0.0):
        """Eq. (1): theta(r) = (1/2) arccos(r), 0 <= theta <= pi/2.

        With the default eps=0 the endpoints are exact: theta(1) = 0 and
        theta(-1) = pi/2. A positive eps is an explicit *numerical guard*
        that moves r = +-1 to +-(1 - eps) (finite arccos gradient); per
        Sec. 2.4 that guard is kept distinct from the mathematical map.
        """
        return 0.5 * torch.acos(torch.clamp(r, -1.0 + eps, 1.0 - eps))

    @staticmethod
    def theta_to_r(theta):
        """Prop. 1: <Z> = cos^2(theta) - sin^2(theta) = cos(2 theta) = r.
        (The earlier draft wrote cos(theta); that was incorrect.)"""
        return torch.cos(2 * theta)

    @staticmethod
    def theta_to_q(theta):
        """Prop. 1: <X> = 2 cos(theta) sin(theta) = sin(2 theta) = q >= 0
        on 0 <= theta <= pi/2 (the earlier draft's T = |sin 2theta|)."""
        return torch.sin(2 * theta)

    @staticmethod
    def theta_to_state(theta):
        """Eq. (1): |psi_r> = cos(theta)|0> + sin(theta)|1>."""
        return torch.stack([torch.cos(theta), torch.sin(theta)], dim=-1)

    @staticmethod
    def probabilities(r):
        """Eq. (2): P(0) = (1+r)/2, P(1) = (1-r)/2."""
        return (1 + r) / 2, (1 - r) / 2

    @staticmethod
    def q_from_r(r):
        """Eq. (1)-(2): q = sqrt(1 - r^2) = <X>."""
        return torch.sqrt(torch.clamp(1 - r ** 2, min=0.0))

    @staticmethod
    def joint_distribution(r):
        """Eq. (3): a +-1 pair (S, T) with Pr(S=s, T=t) = (1 + r s t)/4.

        Returns probabilities ordered (s,t) = (+1,+1), (+1,-1), (-1,+1),
        (-1,-1) on the last axis. Both marginals are uniform and
        E[ST] = r, so the Pearson correlation is r. This is an auxiliary
        classical representation, not the measurement of one APQB (Sec. 2.2).
        """
        same = (1 + r) / 4
        diff = (1 - r) / 4
        return torch.stack([same, diff, diff, same], dim=-1)

    @staticmethod
    def _sech(a):
        """sech(a) without overflowing cosh(a), including in backward."""
        return 2.0 * torch.exp(-torch.logaddexp(a, -a))

    @staticmethod
    def from_latent(a):
        """Eq. (5): unconstrained latent a -> (r, q, theta) with r^2+q^2=1
        satisfied automatically via tanh^2(a) + sech^2(a) = 1, and
        dr/da = q^2, dq/da = -r q. If a is clipped, build r and q from the
        same clipped a so the identity still holds (Sec. 2.4)."""
        r = torch.tanh(a)
        q = APQBv2._sech(a)
        # Equivalent to atan(exp(-a)), without overflowing exp(-a).
        theta = 0.5 * torch.atan2(q, r)
        return r, q, theta

    @staticmethod
    def z_from_r_q(r, q):
        """Sec. 4.2: complex coordinate z = r + iq = e^{i2theta}."""
        return torch.complex(r, q)

    @staticmethod
    def density_matrix(r):
        """Eq. (4): rho(r) = 1/2 [[1+r, q],[q, 1-r]]. Returns (rho00, rho01, rho11).
        rho(r) is pure: Tr(rho^2) = 1."""
        q = APQBv2.q_from_r(r)
        rho00 = (1 + r) / 2
        rho11 = (1 - r) / 2
        rho01 = q / 2
        return rho00, rho01, rho11

    @staticmethod
    def coherence_l1(r):
        """Eq. (4): l1-norm coherence in the computational basis,
        C_l1(rho) = 2|rho_01| = q [1]. q is a coherence coordinate, not a
        sampling temperature (Sec. 2.3)."""
        return APQBv2.q_from_r(r)

    @staticmethod
    def entropy_z(r):
        """Sec. 2.3: Shannon entropy of a Z-basis measurement,
        h_2((1+r)/2) bits.
        Note: this is measurement-basis uncertainty, not the
        von Neumann entropy of rho(r), which is 0 because rho(r) is pure.
        At deterministic endpoints, use 0*log(0)=0 and a finite log floor
        for backward; the exact entropy derivative is singular there.
        """
        p0 = torch.clamp((1 + r) / 2, min=0.0, max=1.0)
        p1 = torch.clamp((1 - r) / 2, min=0.0, max=1.0)
        tiny = torch.finfo(p0.dtype).tiny
        return -(p0 * torch.log2(p0.clamp_min(tiny))
                 + p1 * torch.log2(p1.clamp_min(tiny)))

    @staticmethod
    def constraint(r, q):
        """r^2 + q^2, which should equal 1 for a valid APQB state."""
        return r ** 2 + q ** 2

    @staticmethod
    def mixed_state_purity(r):
        """Supplementary (not part of the revised paper's main results):
        purity Tr(rho_mix^2) of the diagonal mixed-state alternative
        encoding, which shares <Z>=r but has zero coherence."""
        return (1 + r ** 2) / 2

    @staticmethod
    def phase_extended_bloch(r, phi):
        """Supplementary: phase-extended Bloch vector (q cos(phi), q sin(phi), r).
        phi is not derivable from r alone -- it must come from an external
        observation or a learned variable. Phase information is exactly
        what r discards (Sec. 3.1 counterexample)."""
        q = APQBv2.q_from_r(r)
        return q * torch.cos(phi), q * torch.sin(phi), r


# ===========================================================================
# Section 3: explicit multi-qubit state families (Eq. 6-9)
# ===========================================================================
#
# Amplitude vectors use the computational basis in lexicographic order
# (|00>, |01>, |10>, |11> and |000>, ..., |111>). These helpers verify the
# paper's restricted correspondences; they do not turn per-edge APQB
# features into a consistent n-qubit state (Sec. 3.3).

def two_qubit_correlated_state(r):
    """Eq. (6): |Psi_r> = A(|00>+|11>) + B(|01>+|10>),
    A = sqrt((1+r)/4), B = sqrt((1-r)/4). Returns amplitudes [..., 4]."""
    A = torch.sqrt(torch.clamp((1 + r) / 4, min=0.0))
    B = torch.sqrt(torch.clamp((1 - r) / 4, min=0.0))
    return torch.stack([A, B, B, A], dim=-1)


def zz_correlation(state):
    """<Z (x) Z> of a two-qubit pure state [..., 4] (Eq. 7)."""
    p = state.abs() ** 2
    return p[..., 0] - p[..., 1] - p[..., 2] + p[..., 3]


def concurrence_pure(state):
    """Concurrence of a pure two-qubit state a|00>+b|01>+c|10>+d|11>:
    C_2 = 2|ad - bc| [2]. For the family of Eq. (6), C_2 = |r| (Prop. 2).
    In general C_2 is *not* determined by r: (|00>+|01>+|10>-|11>)/2 has
    r = 0 but C_2 = 1 (Sec. 3.1)."""
    a, b, c, d = state.unbind(-1)
    return 2 * (a * d - b * c).abs()


def reduced_purity(state):
    """Purity Tr(rho_A^2) of qubit A for a pure two-qubit state [..., 4].
    For Eq. (6) this is (1 + q^2)/2 (Sec. 3.1)."""
    M = state.reshape(*state.shape[:-1], 2, 2)
    rho_a = M @ M.conj().transpose(-1, -2)
    return (rho_a @ rho_a).diagonal(dim1=-2, dim2=-1).sum(-1).real


def ghz_state(theta):
    """Eq. (8): |G_theta> = cos(theta)|000> + sin(theta)|111>. Returns [..., 8].
    Each qubit has <Z> = cos(2 theta) = r; the two-body ZZ product is
    always 1, so r is *not* a pairwise Pearson correlation here."""
    c, s = torch.cos(theta), torch.sin(theta)
    zeros = torch.zeros_like(c)
    return torch.stack([c] + [zeros] * 6 + [s], dim=-1)


def three_tangle(state):
    """Coffman-Kundu-Wootters residual three-tangle tau_3 of a pure
    three-qubit state [..., 8] (Cayley hyperdeterminant form) [3].
    For the GHZ-type family of Eq. (8), tau_3 = 4 cos^2 sin^2 = q^2, and
    r^2 + tau_3 = 1. tau_3 is non-negative, unlike the old sin(6 theta)."""
    a = lambda i, j, k: state[..., 4 * i + 2 * j + k]
    d1 = (a(0, 0, 0) ** 2 * a(1, 1, 1) ** 2 + a(0, 0, 1) ** 2 * a(1, 1, 0) ** 2
          + a(0, 1, 0) ** 2 * a(1, 0, 1) ** 2 + a(1, 0, 0) ** 2 * a(0, 1, 1) ** 2)
    d2 = (a(0, 0, 0) * a(1, 1, 1) * a(0, 1, 1) * a(1, 0, 0)
          + a(0, 0, 0) * a(1, 1, 1) * a(1, 0, 1) * a(0, 1, 0)
          + a(0, 0, 0) * a(1, 1, 1) * a(1, 1, 0) * a(0, 0, 1)
          + a(0, 1, 1) * a(1, 0, 0) * a(1, 0, 1) * a(0, 1, 0)
          + a(0, 1, 1) * a(1, 0, 0) * a(1, 1, 0) * a(0, 0, 1)
          + a(1, 0, 1) * a(0, 1, 0) * a(1, 1, 0) * a(0, 0, 1))
    d3 = (a(0, 0, 0) * a(1, 1, 0) * a(1, 0, 1) * a(0, 1, 1)
          + a(1, 1, 1) * a(0, 0, 1) * a(0, 1, 0) * a(1, 0, 0))
    return 4 * (d1 - 2 * d2 + 4 * d3).abs()


# ===========================================================================
# Section 4: Boolean expansion, harmonic basis, subset products (Eq. 10-12)
# ===========================================================================

def boolean_cube(d, dtype=torch.float64, device=None):
    """All 2^d points x in {-1,+1}^d, shape [2^d, d]."""
    return torch.tensor(list(product((-1.0, 1.0), repeat=d)), dtype=dtype, device=device)


def boolean_characters(x, subsets=None):
    """Eq. (10): chi_S(x) = prod_{i in S} x_i for every subset S (chi_{} = 1).
    x: [..., d]. Returns (subsets, chi [..., len(subsets)])."""
    d = x.shape[-1]
    if subsets is None:
        subsets = [S for k in range(d + 1) for S in combinations(range(d), k)]
    cols = [x[..., list(S)].prod(dim=-1) if S else torch.ones_like(x[..., 0])
            for S in subsets]
    return subsets, torch.stack(cols, dim=-1)


def boolean_fourier_coefficients(f_values, d):
    """Eq. (10) / Prop. 3: f_hat(S) = 2^-d sum_x f(x) chi_S(x), with
    f_values given on boolean_cube(d) order. The 2^d characters are an
    orthonormal basis, so the expansion is unique. This concerns Boolean
    inputs only; it says nothing about general NN parameter counts.
    Returns (subsets, coeffs [2^d])."""
    x = boolean_cube(d, dtype=f_values.dtype, device=f_values.device)
    subsets, chi = boolean_characters(x)
    return subsets, chi.transpose(0, 1) @ f_values / (2 ** d)


def boolean_fourier_eval(subsets, coeffs, x):
    """Evaluate f(x) = sum_S f_hat(S) chi_S(x) (Eq. 10)."""
    _, chi = boolean_characters(x, subsets)
    # Promote rather than cast to chi's dtype: integer +-1 inputs must not
    # truncate the (generally fractional) coefficients.
    dt = torch.promote_types(chi.dtype, coeffs.dtype)
    return chi.to(dt) @ coeffs.to(device=chi.device, dtype=dt)


def chebyshev_features(r, q, K):
    """Eq. (11): for k=1..K, Re(z^k) = T_k(r) and
    Im(z^k) = q * U_{k-1}(r), where T_k/U_k are Chebyshev polynomials of
    the first/second kind. Computed via the z^k = z^{k-1} * z recurrence
    since z = r + iq already lies on the unit circle by construction.

    Returns (real_feats, imag_feats), each of shape [..., K].
    """
    z = torch.complex(r, q)
    reals, imags = [], []
    zk = torch.ones_like(z)
    for _ in range(K):
        zk = zk * z
        reals.append(zk.real)
        imags.append(zk.imag)
    return torch.stack(reals, dim=-1), torch.stack(imags, dim=-1)


def subset_product_features(z_list, max_degree=None):
    """Eq. (12) / Sec. 4.3: builds Phi_S(z) = prod_{i in S} z_i for
    every subset S of {0, ..., d-1} with |S| <= max_degree (degree-K
    truncation to avoid the exponential blow-up for large d). Phi_{} = 1.
    At r_i = x_i in {-1,+1}, q_i = 0, Phi_S coincides with chi_S of Eq. (10).

    z_list: list of d complex tensors of identical shape.
    Returns: dict {subset_tuple: complex tensor}, with exactly
    num_subset_terms(d, max_degree) entries (2^d when
    max_degree == d).
    """
    d = len(z_list)
    if max_degree is None:
        max_degree = d
    feats = {(): torch.ones_like(z_list[0])}
    for k in range(1, max_degree + 1):
        for S in combinations(range(d), k):
            prod = z_list[S[0]]
            for i in S[1:]:
                prod = prod * z_list[i]
            feats[S] = prod
    return feats


def num_subset_terms(d, max_degree=None):
    """Eq. (12): N_{d,K} = sum_{k=0}^{K} C(d,k) surviving subset terms.
    Equals 2^d when max_degree == d."""
    if max_degree is None:
        max_degree = d
    return sum(math.comb(d, k) for k in range(0, max_degree + 1))


# ===========================================================================
# Section 5: QBNN layer -- classical affine path + APQB correlation gate
# ===========================================================================

class QBNNLayerV2(nn.Module):
    """Deterministic QBNN layer, Eq. (13)-(14).

        u = W h + b
        a = P h + c ; r = tanh(a) ; q = sech(a)
        c_r = J_r^T r ; c_q = J_q^T q
        h_next = activation( u . (1 + lambda_r*c_r + lambda_q*c_q) )

    J is a learnable feature-interaction matrix, not an operator that
    realizes physical entanglement (Sec. 5.1).

    Prop. 4: at lambda_r = lambda_q = 0 or J_r = J_q = 0 this reduces
    exactly to a plain affine + activation layer (the correlation path is
    still executed; it is not skipped automatically).

    By default, small nonzero lambdas let the zero-initialized J branches
    learn immediately (Eq. 15) while preserving the initial plain-layer
    output. Explicit zero lambdas with zero J disable task-loss learning in
    those branches; loading a state_dict preserves its saved lambdas.

    a is clipped to [-a_clip, a_clip] and r, q are built from the same
    clipped a, so r^2 + q^2 = 1 still holds (Sec. 2.4). The q_eps floor
    only acts when sech(a_clip) < q_eps; check constraint_error() then.
    """

    def __init__(self, in_dim, out_dim, rank=None, activation=torch.tanh,
                 lambda_r_init=0.1, lambda_q_init=0.1, a_clip=4.0, q_eps=1e-6):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.a_clip = a_clip
        self.q_eps = q_eps
        self.activation = activation

        self.W = nn.Linear(in_dim, out_dim)
        self.P = nn.Linear(in_dim, out_dim)

        self.low_rank = rank is not None
        if self.low_rank:
            # Sec. 5.3: J = V U^T (c = U (V^T x)) with rank k << m.
            # V is zero-initialized so J starts at exactly zero.
            self.U_r = nn.Parameter(torch.empty(out_dim, rank).normal_(std=0.01))
            self.V_r = nn.Parameter(torch.zeros(out_dim, rank))
            self.U_q = nn.Parameter(torch.empty(out_dim, rank).normal_(std=0.01))
            self.V_q = nn.Parameter(torch.zeros(out_dim, rank))
        else:
            self.J_r = nn.Parameter(torch.zeros(out_dim, out_dim))
            self.J_q = nn.Parameter(torch.zeros(out_dim, out_dim))

        self.lambda_r = nn.Parameter(torch.tensor(float(lambda_r_init)))
        self.lambda_q = nn.Parameter(torch.tensor(float(lambda_q_init)))

        self.last_r = None
        self.last_q = None

    def _apply_J(self, x, which):
        if not self.low_rank:
            J = self.J_r if which == 'r' else self.J_q
            return x @ J
        U, V = (self.U_r, self.V_r) if which == 'r' else (self.U_q, self.V_q)
        return (x @ V) @ U.t()

    def _rq(self, h):
        a = torch.clamp(self.P(h), -self.a_clip, self.a_clip)
        r = torch.tanh(a)
        q = torch.clamp(APQBv2._sech(a), min=self.q_eps)
        return r, q

    def forward(self, h):
        u = self.W(h)
        r, q = self._rq(h)

        c_r = self._apply_J(r, 'r')
        c_q = self._apply_J(q, 'q')

        gate = 1.0 + self.lambda_r * c_r + self.lambda_q * c_q
        h_next = self.activation(u * gate)

        self.last_r, self.last_q = r, q
        return h_next

    def constraint_error(self):
        """r^2 + q^2 - 1; should be ~0 given the tanh/sech parameterization."""
        if self.last_r is None:
            return None
        return self.last_r ** 2 + self.last_q ** 2 - 1.0

    def j_l1(self):
        """Sum |J_r| + |J_q| (or their low-rank factors), for an L1 penalty."""
        if self.low_rank:
            return (self.U_r.abs().sum() + self.V_r.abs().sum()
                    + self.U_q.abs().sum() + self.V_q.abs().sum())
        return self.J_r.abs().sum() + self.J_q.abs().sum()


class StochasticQBNNLayerV2(QBNNLayerV2):
    """Eq. (17): adds q-scaled classical noise for exploration/regularization.

        h_next = activation( u.gate + beta * (q_out . xi) ),  xi ~ N(0, I)

    This is *not* quantum measurement noise -- it is ordinary Gaussian
    noise whose amplitude is shaped by the APQB coherence coordinate q
    (Sec. 5.4). Noise is only sampled in training mode. For a fixed input
    the noise has a diagonal covariance; correlated noise would need an
    explicit covariance / mixing matrix.
    """

    def __init__(self, in_dim, out_dim, beta=0.1, **kwargs):
        super().__init__(in_dim, out_dim, **kwargs)
        self.beta = beta
        self.q_out_proj = nn.Linear(out_dim, out_dim, bias=False)

    def forward(self, h):
        u = self.W(h)
        r, q = self._rq(h)

        c_r = self._apply_J(r, 'r')
        c_q = self._apply_J(q, 'q')
        gate = 1.0 + self.lambda_r * c_r + self.lambda_q * c_q

        pre_act = u * gate
        if self.training and self.beta:
            q_out = self.q_out_proj(q)
            pre_act = pre_act + self.beta * q_out * torch.randn_like(q_out)

        h_next = self.activation(pre_act)
        self.last_r, self.last_q = r, q
        return h_next


def qbnn_regularization_loss(layer, r_target=None, alpha=1.0, beta_J=1e-4,
                              gamma=0.0, q_target=0.5):
    """Supplementary regularizer (carried over from the earlier draft):
    alpha*L_corr + beta_J*||J||_1 + gamma*(mean(q)-q_target)^2.
    Add the result to your task loss; this function does not include L_task.
    With zero lambda and zero J, check per-parameter gradients (Sec. 5.2).

    r_target: optional externally observed correlation r* for L_corr =
        ||r - r*||^2. Omit when P is fully learned.
    """
    reg = layer.last_r.new_zeros(())
    if r_target is not None and layer.last_r is not None:
        reg = reg + alpha * F.mse_loss(layer.last_r, r_target)
    reg = reg + beta_J * layer.j_l1()
    if gamma and layer.last_q is not None:
        reg = reg + gamma * (layer.last_q.mean() - q_target) ** 2
    return reg


def qbnn_mac_counts(batch, d, m, rank=None):
    """Eq. (16): multiply-accumulate counts of the matrix products only
    (activations, bias, clipping and memory traffic excluded).

        plain   = B d m
        dense   = B (2 d m + 2 m^2)
        lowrank = B (2 d m + 4 m k)

    These are operation counts, not wall-clock ratios (Sec. 5.3, 6.5).
    """
    counts = {"plain": batch * d * m, "dense": batch * (2 * d * m + 2 * m * m)}
    if rank is not None:
        counts["lowrank"] = batch * (2 * d * m + 4 * m * rank)
    return counts


# ===========================================================================
# Section 3.3: multivariate APQB / correlation graph
# ===========================================================================

def apqb_correlation_bank(R):
    """Sec. 3.3: build an APQB feature bank e_ij = [r_ij, q_ij] from a
    correlation matrix R (..., n, n). Returns (r, q) each shaped like R.

    This is a classical feature bank on a correlation graph,
    not a claim that pairwise APQBs jointly realize one consistent
    multi-qubit state (that would require solving the quantum marginal
    problem, which this module does not attempt).
    """
    r = R
    q = torch.sqrt(torch.clamp(1 - R ** 2, min=0.0))
    return r, q


def nearest_correlation_matrix(R, iters=100, tol=1e-8):
    """Higham (2002) alternating-projections algorithm: projects an
    estimated/learned matrix R onto the nearest valid correlation matrix
    (symmetric, unit diagonal, positive semi-definite), as required before
    building an APQB bank from it (Sec. 3.3)."""
    Y = R.clone()
    S = torch.zeros_like(R)
    eye_diag = torch.ones(R.shape[-1], dtype=R.dtype, device=R.device)
    for _ in range(iters):
        R_adj = Y - S
        eigvals, eigvecs = torch.linalg.eigh(R_adj)
        eigvals_clamped = torch.clamp(eigvals, min=0.0)
        X = eigvecs @ torch.diag_embed(eigvals_clamped) @ eigvecs.transpose(-1, -2)
        S = X - R_adj
        Y_new = X.clone()
        Y_new.diagonal(dim1=-2, dim2=-1).copy_(eye_diag.expand_as(Y_new.diagonal(dim1=-2, dim2=-1)))
        if torch.norm(Y_new - Y) < tol:
            Y = Y_new
            break
        Y = Y_new
    return Y


# ===========================================================================
# Section 6.1: calibrated temperature (Eq. 18)
# ===========================================================================

def calibrated_temperature(q, tau_min=0.3, tau_max=1.5, gamma=1.0):
    """Eq. (18): tau_AI(q) = tau_min + (tau_max - tau_min) * q^gamma.
    q is the APQB coherence coordinate, not itself a temperature (Sec. 2.3)
    -- tau_min, tau_max, gamma and the aggregation of q (mean, max, learned)
    must be fixed and calibrated on validation data (Sec. 6.1)."""
    q = torch.clamp(q, min=0.0, max=1.0)
    return tau_min + (tau_max - tau_min) * q.pow(gamma)


# ===========================================================================
# Section 6.2-6.4: untrained calculator as a fixed threshold circuit (Prop. 5)
# ===========================================================================
#
# "Untrained" here means hand-set weights, not random initialization: a
# random QBNN has no guarantee of computing arithmetic. The circuit below
# uses exactly one QBNN gate -- NAND -- with the correlation path switched
# off, and wires everything else from it. It demonstrates that such a
# circuit is a special case of QBNNLayerV2; it is not a claim of any
# QBNN-specific speedup, and no existing QBNN generates it automatically.

def heaviside(t):
    """H(t) = 1 for t >= 0, 0 for t < 0 (Prop. 5)."""
    return (t >= 0).to(t.dtype)


def nand_qbnn_layer(dtype=torch.float32):
    """Prop. 5 / Eq. (19): NAND(x, y) = H(3/2 - x - y) as a QBNNLayerV2
    with W = [-1, -1], b = 3/2, P = 0, J_r = J_q = 0, lambda = 0.
    The minimum threshold margin is 1/2."""
    layer = QBNNLayerV2(2, 1, activation=heaviside,
                        lambda_r_init=0.0, lambda_q_init=0.0).to(dtype)
    with torch.no_grad():
        layer.W.weight.copy_(torch.tensor([[-1.0, -1.0]]))
        layer.W.bias.fill_(1.5)
        layer.P.weight.zero_()
        layer.P.bias.zero_()
        layer.J_r.zero_()
        layer.J_q.zero_()
    layer.requires_grad_(False)
    layer.eval()
    return layer


def int_to_bits(x, width, dtype=torch.float32):
    """Non-negative integer tensor -> list of `width` bit tensors, LSB first."""
    return [((x >> i) & 1).to(dtype) for i in range(width)]


def bits_to_int(bits):
    """List of 0/1 bit tensors (LSB first) -> int64 tensor."""
    out = torch.zeros_like(bits[0], dtype=torch.int64)
    for i, b in enumerate(bits):
        out = out + (b.round().to(torch.int64) << i)
    return out


class ThresholdCircuitCalculator:
    """Fixed-weight integer calculator built only from the QBNN NAND gate
    (Sec. 6.2, Appendix B). All operations are batched over tensors.

    NOT(x) = NAND(x, x); AND = NOT(NAND); OR by De Morgan; XOR with four
    NANDs; full adder per Eq. (20). Subtraction is two's-complement
    addition (invert B, carry-in 1); multiplication accumulates shifted
    partial products; division is restoring division whose trial-subtract
    borrow, inverted, gives each quotient bit and the select signal.

    Only unsigned `width`-bit integers are specified (1 <= width <= 31, so
    the 2*width-bit product fits in int64). Signed numbers, fractions,
    arbitrary precision and expression parsing are out of scope, and
    non-integer or out-of-range operands are rejected, not truncated.
    """

    def __init__(self, width=4):
        if isinstance(width, bool):
            raise TypeError("width must be an int")
        width = operator.index(width)
        if not 1 <= width <= 31:
            raise ValueError("width must be in 1..31 (mul returns 2*width bits packed into int64)")
        self.width = width
        self.gate = nand_qbnn_layer()
        self.nand_count = 0

    # ---- gates -----------------------------------------------------------
    def NAND(self, x, y):
        self.nand_count += 1
        with torch.no_grad():
            h = torch.stack([x, y], dim=-1).to(self.gate.W.weight.device)
            return self.gate(h).squeeze(-1).to(x.device)

    def NOT(self, x):
        return self.NAND(x, x)

    def AND(self, x, y):
        return self.NOT(self.NAND(x, y))

    def OR(self, x, y):
        return self.NAND(self.NOT(x), self.NOT(y))

    def XOR(self, x, y):
        n = self.NAND(x, y)
        return self.NAND(self.NAND(x, n), self.NAND(y, n))

    def MUX(self, sel, a, b):
        """sel ? a : b."""
        return self.OR(self.AND(sel, a), self.AND(self.NOT(sel), b))

    def full_adder(self, x, y, c_in):
        """Eq. (20): t = x XOR y, s = t XOR c_in, c_out = (x AND y) OR (t AND c_in)."""
        t = self.XOR(x, y)
        s = self.XOR(t, c_in)
        c_out = self.OR(self.AND(x, y), self.AND(t, c_in))
        return s, c_out

    # ---- bit-vector ops (LSB first) --------------------------------------
    def _add_bits(self, A, B, c_in):
        out = []
        c = c_in
        for a, b in zip(A, B):
            s, c = self.full_adder(a, b, c)
            out.append(s)
        return out, c

    def _sub_bits(self, A, B):
        """A - B mod 2^n and borrow flag (1 if A < B)."""
        one = torch.ones_like(A[0])
        diff, carry = self._add_bits(A, [self.NOT(b) for b in B], one)
        return diff, self.NOT(carry)

    # ---- integer API -----------------------------------------------------
    def _operands(self, a, b):
        """Validate two unsigned `width`-bit integer operands, broadcast them
        to a common shape, and return their bit lists (LSB first)."""
        xs = []
        for x in (a, b):
            if isinstance(x, numbers.Integral) and not isinstance(x, bool):
                x = int(x)  # e.g. np.uint64 scalars, which as_tensor rejects
            x = torch.as_tensor(x)
            if x.is_floating_point() or x.is_complex() or x.dtype == torch.bool:
                raise TypeError("operands must be integer tensors or ints")
            x = x.to(torch.int64)
            if (x < 0).any() or ((x >> self.width) != 0).any():
                raise ValueError(f"inputs must be unsigned {self.width}-bit integers")
            xs.append(x)
        a, b = torch.broadcast_tensors(*xs)
        return int_to_bits(a, self.width), int_to_bits(b, self.width)

    def add(self, a, b):
        """Returns (A + B) as a (width+1)-bit integer (sum with carry)."""
        A, B = self._operands(a, b)
        s, c = self._add_bits(A, B, torch.zeros_like(A[0]))
        return bits_to_int(s + [c])

    def sub(self, a, b):
        """Returns ((A - B) mod 2^width, borrow)."""
        d, borrow = self._sub_bits(*self._operands(a, b))
        return bits_to_int(d), bits_to_int([borrow])

    def mul(self, a, b):
        """Returns A * B as a (2*width)-bit integer."""
        n = self.width
        A, B = self._operands(a, b)
        zero = torch.zeros_like(A[0])
        acc = [zero] * (2 * n)
        for i, b_i in enumerate(B):
            partial = [zero] * i + [self.AND(a_j, b_i) for a_j in A]
            partial += [zero] * (2 * n - len(partial))
            acc, _ = self._add_bits(acc, partial, zero)
        return bits_to_int(acc)

    def divmod(self, a, b):
        """Restoring division. Returns (quotient, remainder, div_by_zero).
        For B = 0 the flag is 1 and quotient/remainder are not meaningful."""
        n = self.width
        A, B = self._operands(a, b)
        zero = torch.zeros_like(A[0])
        B_ext = B + [zero]                      # (n+1)-bit divisor
        R = [zero] * (n + 1)                    # (n+1)-bit remainder
        Q = [zero] * n
        for i in reversed(range(n)):
            R = [A[i]] + R[:-1]                 # R = (R << 1) | A_i
            trial, borrow = self._sub_bits(R, B_ext)
            keep = self.NOT(borrow)             # 1 -> trial subtraction succeeded
            Q[i] = keep
            R = [self.MUX(keep, t, r) for t, r in zip(trial, R)]
        any_b = B[0]
        for b in B[1:]:
            any_b = self.OR(any_b, b)
        return bits_to_int(Q), bits_to_int(R), bits_to_int([self.NOT(any_b)])


def verify_threshold_calculator(width=4):
    """Sec. 6.4 / Table 3: exhaustively check the circuit against ordinary
    integer arithmetic for all unsigned `width`-bit pairs. Returns
    {op: (checked, mismatches)} plus 'div_zero_flags' for B = 0.

    This validates the fixed circuit of Prop. 5 only; it says nothing
    about randomly initialized or trained QBNNs, larger widths, or speed.
    """
    calc = ThresholdCircuitCalculator(width)
    n = 2 ** width
    a = torch.arange(n).repeat_interleave(n)
    b = torch.arange(n).repeat(n)
    results = {}

    results["add"] = (a.numel(), int((calc.add(a, b) != a + b).sum()))
    diff, borrow = calc.sub(a, b)
    bad = (diff != (a - b) % n) | (borrow != (a < b).long())
    results["sub"] = (a.numel(), int(bad.sum()))
    results["mul"] = (a.numel(), int((calc.mul(a, b) != a * b).sum()))

    nz = b != 0
    q, r, dz = calc.divmod(a, b)
    bad = (q[nz] != a[nz] // b[nz]) | (r[nz] != a[nz] % b[nz]) | (dz[nz] != 0)
    results["div"] = (int(nz.sum()), int(bad.sum()))
    results["div_zero_flags"] = (int((~nz).sum()), int((dz[~nz] != 1).sum()))
    return results
