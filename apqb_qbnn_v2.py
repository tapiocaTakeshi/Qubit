#!/usr/bin/env python3
"""
APQB / QBNN v2 -- implementation of the reformulated research draft
"Correlation-Encoded Neural Networks Based on Adjustable Pseudo Qubits:
Formalization, Expressivity, and a Testable QBNN Architecture" (revision 2).

Scope note: this module implements the *corrected* formalization from the
paper's revision 2, which deliberately narrows several claims made by an
earlier draft (qbnn_layered.py's EQBNN* classes follow that earlier,
looser framing -- e.g. calling the auxiliary coordinate "temperature", or
treating cross-layer correlation as literal quantum entanglement). APQB
here is a classical, testable correlation encoding: it does not claim
physical qubits, quantum entanglement, or quantum speedup (Sec. 1.2, 5.3,
5.4, 10).

Section references (e.g. "Eq. (12)", "Sec. 6.5", "Prop. 2") point at the
corresponding equations/sections/propositions in the paper.
"""

import math
import warnings
from itertools import combinations

import torch
import torch.nn as nn
import torch.nn.functional as F


# ===========================================================================
# Section 3: APQB core -- Bloch-sphere correlation encoding (Eq. 4-13)
# ===========================================================================

class APQBv2:
    """Stateless APQB math: r <-> theta <-> q, and the stable latent
    parameterization of Sec. 3.6. All methods are pure tensor functions."""

    @staticmethod
    def r_to_theta(r):
        """Eq. (4): theta(r) = (1/2) arccos(r)."""
        r = torch.clamp(r, -1.0 + 1e-7, 1.0 - 1e-7)
        return 0.5 * torch.acos(r)

    @staticmethod
    def theta_to_state(theta):
        """Eq. (2)/(5): |psi> = cos(theta)|0> + sin(theta)|1>."""
        return torch.stack([torch.cos(theta), torch.sin(theta)], dim=-1)

    @staticmethod
    def q_from_r(r):
        """q = sqrt(1 - r^2) = <X> (Eq. 8-9)."""
        return torch.sqrt(torch.clamp(1 - r ** 2, min=0.0))

    @staticmethod
    def from_latent(a):
        """Eq. (12): unconstrained latent a -> (r, q, theta) with r^2+q^2=1
        satisfied automatically via tanh^2(a) + sech^2(a) = 1."""
        r = torch.tanh(a)
        q = 1.0 / torch.cosh(a)
        theta = torch.atan(torch.exp(-a))
        return r, q, theta

    @staticmethod
    def z_from_r_q(r, q):
        """Eq. (14): complex coordinate z = r + iq = e^{i2theta}."""
        return torch.complex(r, q)

    @staticmethod
    def density_matrix(r):
        """Eq. (8): rho(r) = 1/2 [[1+r, q],[q, 1-r]]. Returns (rho00, rho01, rho11)."""
        q = APQBv2.q_from_r(r)
        rho00 = (1 + r) / 2
        rho11 = (1 - r) / 2
        rho01 = q / 2
        return rho00, rho01, rho11

    @staticmethod
    def coherence_l1(r):
        """Sec. 3.4: l1-norm coherence C_l1(rho) = 2|rho_01| = q."""
        return APQBv2.q_from_r(r)

    @staticmethod
    def entropy_z(r):
        """Eq. (10): Shannon entropy of a Z-basis measurement, H_Z(r).
        Note (Sec. 3.4): this is measurement-basis uncertainty, not the
        von Neumann entropy of rho(r), which is 0 because rho(r) is pure."""
        p0 = torch.clamp((1 + r) / 2, min=1e-12, max=1 - 1e-12)
        p1 = 1 - p0
        return -(p0 * torch.log2(p0) + p1 * torch.log2(p1))

    @staticmethod
    def constraint(r, q):
        """r^2 + q^2, which should equal 1 for a valid APQB state."""
        return r ** 2 + q ** 2

    @staticmethod
    def mixed_state_purity(r):
        """Eq. (11): purity Tr(rho_mix^2) of the diagonal mixed-state
        alternative encoding, which shares <Z>=r but has zero coherence."""
        return (1 + r ** 2) / 2

    @staticmethod
    def phase_extended_bloch(r, phi):
        """Eq. (13): phase-extended Bloch vector (q cos(phi), q sin(phi), r).
        phi is not derivable from r alone -- it must come from an external
        observation or a learned variable (Sec. 3.7)."""
        q = APQBv2.q_from_r(r)
        return q * torch.cos(phi), q * torch.sin(phi), r


# ===========================================================================
# Section 4: complex representation & Chebyshev harmonic basis (Eq. 14-18)
# ===========================================================================

def chebyshev_features(r, q, K):
    """Prop. 2 (Eq. 15-16): for k=1..K, Re(z^k) = T_k(r) and
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
    """Eq. (17)-(18) / Prop. 3: builds Phi_S(z) = prod_{i in S} z_i for
    every subset S of {0, ..., d-1} with |S| <= max_degree (Sec. 4.5
    truncation to avoid the exponential blow-up for large d). Phi_{} = 1.

    z_list: list of d complex tensors of identical shape.
    Returns: dict {subset_tuple: complex tensor}, with exactly
    num_subset_terms(d, max_degree) entries (Prop. 3's 2^d count when
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
    """Sec. 4.5: number of surviving subset terms, sum_{k=0}^{K} C(d,k).
    Equals 2^d when max_degree == d (Prop. 3)."""
    if max_degree is None:
        max_degree = d
    return sum(math.comb(d, k) for k in range(0, max_degree + 1))


# ===========================================================================
# Section 6: QBNN layer -- classical affine path + APQB correlation gate
# ===========================================================================

class QBNNLayerV2(nn.Module):
    """Deterministic QBNN layer, Eq. (20)-(23) / Appendix B pseudocode.

        u = W h + b
        a = P h + c ; r = tanh(a) ; q = sech(a)
        c_r = J_r^T r ; c_q = J_q^T q
        h_next = activation( u . (1 + lambda_r*c_r + lambda_q*c_q) )

    At lambda_r = lambda_q = 0 this reduces exactly to a plain affine +
    activation layer (Sec. 6.1 design requirement (i)).

    Initialization caveat (not stated in the paper): Appendix C recommends
    lambda in {0, 0.01} *and* a zero/small J init, but those two choices
    are not jointly safe. Because the gate enters as lambda * (J^T x), the
    gradient w.r.t. lambda is proportional to J and the gradient w.r.t. J
    is proportional to lambda -- so lambda=0 together with J=0 is a saddle
    with exactly zero gradient on both, and the correlation gate can never
    switch on. Pick exactly one of the two to be nonzero; this class warns
    when both are zero.
    """

    def __init__(self, in_dim, out_dim, rank=None, activation=torch.tanh,
                 lambda_r_init=0.0, lambda_q_init=0.0, a_clip=4.0, q_eps=1e-6,
                 use_r=True, use_q=True, learnable_J=True):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.a_clip = a_clip
        self.q_eps = q_eps
        self.activation = activation
        # Sec. 8.5 ablations: use_r/use_q select the QBNN-r / QBNN-q rows,
        # learnable_J=False gives the Random-J control (fixed random coupling).
        self.use_r = use_r
        self.use_q = use_q

        self.W = nn.Linear(in_dim, out_dim)
        self.P = nn.Linear(in_dim, out_dim)

        self.low_rank = rank is not None
        if self.low_rank:
            # Sec. 6.5: J = U V^T with rank(UV^T) = k << min(d_l, d_{l+1}).
            # V is zero-initialized so J starts at exactly zero (Appendix C).
            U_r = torch.empty(out_dim, rank).normal_(std=0.01)
            U_q = torch.empty(out_dim, rank).normal_(std=0.01)
            V_r = torch.zeros(out_dim, rank)
            V_q = torch.zeros(out_dim, rank)
            if not learnable_J:
                # Random-J: nonzero fixed factors, otherwise the gate is
                # identically 1 and the control degenerates to No-J.
                V_r = torch.empty(out_dim, rank).normal_(std=0.01)
                V_q = torch.empty(out_dim, rank).normal_(std=0.01)
            params = [U_r, V_r, U_q, V_q]
            names = ['U_r', 'V_r', 'U_q', 'V_q']
        else:
            J_r = torch.zeros(out_dim, out_dim)
            J_q = torch.zeros(out_dim, out_dim)
            if not learnable_J:
                J_r = torch.empty(out_dim, out_dim).normal_(std=0.01)
                J_q = torch.empty(out_dim, out_dim).normal_(std=0.01)
            params = [J_r, J_q]
            names = ['J_r', 'J_q']

        for name, tensor in zip(names, params):
            if learnable_J:
                self.register_parameter(name, nn.Parameter(tensor))
            else:
                self.register_buffer(name, tensor)

        self.lambda_r = nn.Parameter(torch.tensor(float(lambda_r_init)))
        self.lambda_q = nn.Parameter(torch.tensor(float(lambda_q_init)))

        if learnable_J and lambda_r_init == 0.0 and lambda_q_init == 0.0:
            warnings.warn(
                "QBNNLayerV2 initialized with lambda_r=lambda_q=0 and a "
                "zero-initialized J: the correlation gate has zero gradient "
                "on both lambda and J and can never activate. Set a small "
                "nonzero lambda (e.g. 0.01) or a small random J.",
                RuntimeWarning, stacklevel=2)

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
        q = torch.clamp(1.0 / torch.cosh(a), min=self.q_eps)
        return r, q

    def _gate(self, h):
        r, q = self._rq(h)
        gate = torch.ones_like(self.W.bias).expand(h.shape[0], -1)
        if self.use_r:
            gate = gate + self.lambda_r * self._apply_J(r, 'r')
        if self.use_q:
            gate = gate + self.lambda_q * self._apply_J(q, 'q')
        return gate, r, q

    def forward(self, h):
        u = self.W(h)
        gate, r, q = self._gate(h)
        h_next = self.activation(u * gate)

        self.last_r, self.last_q = r, q
        return h_next

    def constraint_error(self):
        """r^2 + q^2 - 1; should be ~0 given the tanh/sech parameterization."""
        if self.last_r is None:
            return None
        return self.last_r ** 2 + self.last_q ** 2 - 1.0

    def j_l1(self):
        """Sum |J_r| + |J_q| (or their low-rank factors), for Eq. (25)."""
        if self.low_rank:
            return (self.U_r.abs().sum() + self.V_r.abs().sum()
                    + self.U_q.abs().sum() + self.V_q.abs().sum())
        return self.J_r.abs().sum() + self.J_q.abs().sum()


class StochasticQBNNLayerV2(QBNNLayerV2):
    """Eq. (24): adds q-scaled classical noise for exploration/regularization.

        h_next = activation( u.gate + beta * (q_out . xi) ),  xi ~ N(0, I)

    This is *not* quantum measurement noise -- it is ordinary Gaussian
    noise whose amplitude is shaped by the APQB coherence coordinate q
    (Sec. 6.4). Noise is only sampled in training mode.
    """

    def __init__(self, in_dim, out_dim, beta=0.1, **kwargs):
        super().__init__(in_dim, out_dim, **kwargs)
        self.beta = beta
        self.q_out_proj = nn.Linear(out_dim, out_dim, bias=False)

    def forward(self, h):
        u = self.W(h)
        gate, r, q = self._gate(h)

        pre_act = u * gate
        if self.training and self.beta:
            q_out = self.q_out_proj(q)
            pre_act = pre_act + self.beta * q_out * torch.randn_like(q_out)

        h_next = self.activation(pre_act)
        self.last_r, self.last_q = r, q
        return h_next


class IndependentGateLayerV2(QBNNLayerV2):
    """Sec. 8.5 "Independent-gates" control: identical wiring and gate
    arithmetic to QBNNLayerV2, but r and q come from two *separate*
    projections and are therefore free of the r^2+q^2=1 coupling.

    This is the model hypothesis H3 is tested against (Sec. 6.7): if
    QBNN's benefit really comes from the APQB constraint rather than from
    having two multiplicative gates, QBNNLayerV2 should beat this control
    at matched capacity. It carries one extra projection's worth of
    parameters, so it is the *harder* baseline, not a handicapped one.
    """

    def __init__(self, in_dim, out_dim, **kwargs):
        super().__init__(in_dim, out_dim, **kwargs)
        self.P_q = nn.Linear(in_dim, out_dim)

    def _rq(self, h):
        a_r = torch.clamp(self.P(h), -self.a_clip, self.a_clip)
        a_q = torch.clamp(self.P_q(h), -self.a_clip, self.a_clip)
        # Same marginal ranges as APQB (r in [-1,1], q in [0,1]) so that the
        # only removed ingredient is the coupling itself.
        return torch.tanh(a_r), torch.sigmoid(a_q)


def qbnn_regularization_loss(layer, r_target=None, alpha=1.0, beta_J=1e-4,
                              gamma=0.0, q_target=0.5):
    """Eq. (25): alpha*L_corr + beta_J*||J||_1 + gamma*(mean(q)-q_target)^2.
    Add the result to your task loss; this function does not include L_task.

    r_target: optional externally observed correlation r* for L_corr =
        ||r - r*||^2 (Sec. 6.6). Omit when P is fully learned.
    """
    reg = layer.last_r.new_zeros(())
    if r_target is not None and layer.last_r is not None:
        reg = reg + alpha * F.mse_loss(layer.last_r, r_target)
    reg = reg + beta_J * layer.j_l1()
    if gamma and layer.last_q is not None:
        reg = reg + gamma * (layer.last_q.mean() - q_target) ** 2
    return reg


# ===========================================================================
# Section 5: multivariate APQB / correlation graph (Eq. 19)
# ===========================================================================

def apqb_correlation_bank(R):
    """Sec. 5.1-5.2: build an APQB feature bank e_ij = [r_ij, q_ij] from a
    correlation matrix R (..., n, n). Returns (r, q) each shaped like R.

    Per Sec. 5.1, this is a classical feature bank on a correlation graph,
    not a claim that pairwise APQBs jointly realize one consistent
    multi-qubit state (that would require solving the quantum marginal
    problem, which this module does not attempt).
    """
    r = R
    q = torch.sqrt(torch.clamp(1 - R ** 2, min=0.0))
    return r, q


def nearest_correlation_matrix(R, iters=100, tol=1e-8):
    """Higham (2002) alternating-projections algorithm [14]: projects an
    estimated/learned matrix R onto the nearest valid correlation matrix
    (symmetric, unit diagonal, positive semi-definite), as required before
    building an APQB bank from it (Sec. 5.1)."""
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
# Section 7: APQB-based probability/exploration control (Eq. 26)
# ===========================================================================

def calibrated_temperature(q, tau_min=0.3, tau_max=1.5, gamma=1.0):
    """Eq. (26): tau_AI(q) = tau_min + (tau_max - tau_min) * q^gamma.
    q is the APQB coherence coordinate, not itself a temperature (Sec. 7.1)
    -- tau_min, tau_max, gamma must be calibrated on validation data."""
    q = torch.clamp(q, min=0.0, max=1.0)
    return tau_min + (tau_max - tau_min) * q.pow(gamma)
