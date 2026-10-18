"""Checks for the complete Bloch sphere (two-sheeted hyperboloid + two half-balls,
CB1-CB7 in complete_qbnn.py) against the square-root amplitude reference
implementation, and for the complete QBNN built from it."""

import math

import pytest
import torch

import apqb_sqrt_amplitude as sa
import complete_qbnn as cq

TOL = 1e-12
N = 10_000


def assert_max_abs(a, b, tol=TOL):
    err = (torch.as_tensor(a) - torch.as_tensor(b)).abs().max().item()
    assert err < tol, f"max abs error {err:.3e} >= {tol:.0e}"


@pytest.fixture(autouse=True)
def deterministic_cpu_tests():
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    with torch.random.fork_rng():
        torch.manual_seed(20261018)
        yield
    torch.set_num_threads(threads)


@pytest.fixture(scope="module")
def latent():
    g = torch.Generator().manual_seed(20261018)
    c, u, v = (3 * torch.randn(3, N, generator=g, dtype=torch.float64)).unbind(0)
    return c, u, v


def test_sheets_project_onto_the_two_hemispheres(latent):
    _, u, v = latent
    for sheet in (1.0, -1.0):
        q = cq.hyperboloid_point(u, v, sheet)
        assert_max_abs(cq.minkowski_norm_sq(q), 1.0, 1e-9)
        b = cq.sheet_to_sphere(q)                                       # CB1
        assert_max_abs(b.norm(dim=-1), 1.0)
        assert (sheet * b[..., 2] > 0).all()
        mirror = q * torch.tensor([1.0, 1.0, -1.0], dtype=q.dtype)        # q~ = (u, v, -zeta)
        assert_max_abs(cq.sheet_to_sphere(mirror), -b)                  # antipodal sheets
    apex = cq.sheet_to_sphere(torch.tensor([[0.0, 0.0, 1.0], [0.0, 0.0, -1.0]], dtype=torch.float64))
    assert_max_abs(sa.from_bloch(apex)[0], torch.tensor([0.0, 4.0], dtype=torch.float64))  # |0>, |1>
    far = cq.sheet_to_sphere(cq.hyperboloid_point(torch.tensor(1e8, dtype=torch.float64),
                                                  torch.tensor(0.0, dtype=torch.float64)))
    assert sa.from_bloch(far)[0].item() == pytest.approx(2.0)          # infinity -> equator


def test_geodesic_coordinates_give_tanh_sech_and_paper_latent():
    g = torch.Generator().manual_seed(1)
    rho = 5 * torch.rand(N, generator=g, dtype=torch.float64) + 1e-3
    a = 2 * math.pi * torch.rand(N, generator=g, dtype=torch.float64)
    b = cq.sheet_to_sphere(cq.geodesic_coords(rho, a))                  # CB2
    n, phi, kappa = sa.from_bloch(b)
    assert_max_abs(sa.eta_from_n(n), torch.tanh(rho))                   # eta = tanh rho
    assert_max_abs(1 - n / 2, sa.sech(rho))                             # |z| = sech rho
    assert_max_abs(kappa, 1.0, 1e-9)
    assert_max_abs(torch.remainder(phi - a + math.pi, 2 * math.pi) - math.pi, 0.0, 1e-9)
    assert_max_abs(cq.hyperbolic_depth(b), rho, 1e-6)
    # CB3: the paper's latent s with z = tanh s (Eq. 11) satisfies sinh s sinh rho = 1.
    s = torch.atanh(b[..., 2])
    assert_max_abs(torch.sinh(s) * torch.sinh(rho), 1.0, 1e-9)
    assert_max_abs(sa.from_latent(s)[0], n)


def test_complete_bloch_states_are_valid_and_cover_the_ball(latent):
    c, u, v = latent
    b = cq.complete_bloch(c, u, v)                                      # CB4
    assert_max_abs(b.norm(dim=-1), torch.tanh(c).abs())
    rho = sa.rho_from_bloch(b)
    assert torch.linalg.eigvalsh(rho).min() >= -TOL
    # rho is the mixture of the two antipodal sheet states with weights (1 +- m)/2.
    q = cq.hyperboloid_point(u, v)
    m = torch.tanh(c)[..., None, None]
    mix = (1 + m) / 2 * sa.rho_from_bloch(cq.sheet_to_sphere(q)) \
        + (1 - m) / 2 * sa.rho_from_bloch(cq.sheet_to_sphere(cq.hyperboloid_point(u, v, -1.0)))
    assert_max_abs(rho, mix)
    assert_max_abs(cq.polarization(b), torch.tanh(c))
    assert ((cq.sheet_of(b) > 0) == (c > 0)).all()
    # Every Bloch vector off the equatorial plane is reached: CB6 inverts CB4.
    g = torch.Generator().manual_seed(2)
    target = torch.randn(N, 3, generator=g, dtype=torch.float64)
    target = target / target.norm(dim=-1, keepdim=True) * torch.rand(N, 1, generator=g, dtype=torch.float64)
    target = target[(target[:, 2].abs() > 1e-3) & (target.norm(dim=-1) < 1 - 1e-6)]
    assert_max_abs(cq.complete_bloch(*cq.complete_coords(target)), target, 1e-9)
    n, phi, kappa = cq.to_npk(target)
    assert_max_abs(sa.bloch(n, phi, kappa), target, 1e-9)


def test_hyperbolic_distance_matches_fidelity_formula(latent):
    _, u, v = latent
    q1 = cq.hyperboloid_point(u, v)
    q2 = cq.hyperboloid_point(v.flip(0), u.flip(0))
    b1, b2 = cq.sheet_to_sphere(q1), cq.sheet_to_sphere(q2)
    d = cq.hyperbolic_distance(q1, q2)
    fidelity = (1 + (b1 * b2).sum(-1)) / 2
    expected = 1 + 2 * (1 - fidelity) / (b1[..., 2] * b2[..., 2])     # CB7
    assert torch.allclose(torch.cosh(d), expected, rtol=1e-9, atol=1e-9)
    # Distance from the apex |0> is the depth rho with sech rho = 1 - n/2.
    apex = cq.hyperboloid_point(torch.zeros(N, dtype=torch.float64), torch.zeros(N, dtype=torch.float64))
    n, _, _ = sa.from_bloch(b1)
    assert torch.allclose(1 / torch.cosh(cq.hyperbolic_distance(apex, q1)), 1 - n / 2,
                          rtol=1e-9, atol=1e-12)


def test_measurements_follow_born_rule_and_are_unbiased(latent):
    c, u, v = latent
    b = cq.complete_bloch(c[:200], u[:200], v[:200])
    g = torch.Generator().manual_seed(3)
    reps = 4000
    sheet = cq.measure_sheet(b.expand(reps, -1, -1), g)                 # CB5
    assert_max_abs(sheet.norm(dim=-1), 1.0)
    axis = b / b.norm(dim=-1, keepdim=True)
    assert_max_abs(sheet.abs(), axis.abs().expand(reps, -1, -1))        # collapses onto +-P(q)
    assert (sheet.mean(0) - b).abs().max() < 0.06
    z = cq.measure_z(b.expand(reps, -1, -1), g)                         # paper Eq. (17)
    assert (z[..., :2] == 0).all() and (z[..., 2].abs() == 1).all()
    assert (z[..., 2].mean(0) - b[..., 2]).abs().max() < 0.06
    centre = cq.measure_sheet(torch.zeros(reps, 3, dtype=torch.float64), g)
    assert_max_abs(centre.norm(dim=-1), 1.0)


def test_axis_angle_gates_are_rotations_and_match_unitaries():
    g = torch.Generator().manual_seed(4)
    w = torch.randn(500, 3, generator=g, dtype=torch.float64)
    w[:5] *= 1e-6
    R = cq.axis_angle_rotation(w)
    assert_max_abs(R @ R.transpose(-1, -2), torch.eye(3, dtype=torch.float64).expand(500, 3, 3), 1e-12)
    assert_max_abs(torch.linalg.det(R), 1.0, 1e-12)
    assert_max_abs(cq.axis_angle_rotation(torch.zeros(3, dtype=torch.float64)), torch.eye(3, dtype=torch.float64))
    # exp(-i theta/2 n.sigma) acts on Bloch vectors as the rotation by theta about n (Eq. 8).
    theta = w.norm(dim=-1)
    axis = w / theta[:, None]
    gen = sum(axis[:, i, None, None] * sa.PAULI[i] for i in range(3))
    U = torch.linalg.matrix_exp(-0.5j * theta[:, None, None] * gen)
    assert_max_abs(sa.rotation_matrix(U), R, 1e-10)


def test_layer_states_stay_in_the_ball_and_gradients_flow():
    layer = cq.CompleteBlochLayer(7, 11)
    with torch.no_grad():
        layer.gate.normal_()
    x = torch.randn(5, 7) * 10
    b = layer(x)
    assert b.shape == (5, 11, 3)
    assert (b.norm(dim=-1) <= 1 + 1e-6).all()
    b.sum().backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in layer.parameters())
    # Measured forward: pure collapsed states, straight-through gradient of the unmeasured b.
    layer.zero_grad()
    bm = layer(x, measure=1.0, generator=torch.Generator().manual_seed(0))
    assert torch.allclose(bm.norm(dim=-1), torch.ones(5, 11), atol=1e-5)
    bm.sum().backward()
    grads = [p.grad.clone() for p in layer.parameters()]
    layer.zero_grad()
    layer(x).sum().backward()
    for g1, p in zip(grads, layer.parameters()):
        torch.testing.assert_close(g1, p.grad)


def test_complete_qbnn_learns_xor_and_lm_overfits_short_text():
    x = torch.tensor([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]]).repeat(16, 1)
    y = (x[:, 0] != x[:, 1]).long()
    net = cq.CompleteQBNN(2, [8], 2)
    opt = torch.optim.Adam(net.parameters(), lr=3e-2)
    for _ in range(300):
        loss = torch.nn.functional.cross_entropy(net(x), y)
        opt.zero_grad()
        loss.backward()
        opt.step()
    assert (net(x).argmax(-1) == y).all()
    diag = net.diagnostics()[0]
    assert 0 <= diag['n_mean'] <= 4 and 0 <= diag['upper_sheet_frac'] <= 1

    text = "完全ブロッホ球は二葉双曲面と二つの半球体でできている。" * 4
    ai = cq.CompleteQBNNAI.from_text(text, embed_dim=16, neurons=32, layers=1)
    losses = ai.train(text, steps=150, batch=8, seq_len=16, lr=1e-2, log_every=0)
    assert losses[-1] < 0.5 * losses[0]
    out = ai.generate("完全", max_new=10, temperature=0)
    assert out.startswith("完全") and len(out) == 12
    assert len(ai.generate("完全", max_new=5, measure=1.0, seed=0)) == 7


def test_save_and_load_round_trip(tmp_path):
    text = "量子ビットの状態は双曲面の葉の上を動く。"
    ai = cq.CompleteQBNNAI.from_text(text, embed_dim=8, neurons=4, layers=2)
    path = tmp_path / "lm.pt"
    ai.save(str(path))
    ai2 = cq.CompleteQBNNAI.load(str(path))
    assert ai2.chars == ai.chars
    tokens = ai.encode(text)[None]
    torch.testing.assert_close(ai.model(tokens)[0], ai2.model(tokens)[0])


# ---------------------------------------------------------------------------
# Entanglement (CE1-CE4)
# ---------------------------------------------------------------------------

def random_pure_pairs(n, seed):
    g = torch.Generator().manual_seed(seed)
    psi = torch.complex(torch.randn(n, 4, generator=g, dtype=torch.float64),
                        torch.randn(n, 4, generator=g, dtype=torch.float64))
    return psi / psi.norm(dim=-1, keepdim=True)


def test_local_polarization_fixes_the_concurrence():
    psi = random_pure_pairs(N, 5)
    b_a, b_b = cq.reduced_bloch(psi)
    assert_max_abs(b_a.norm(dim=-1), b_b.norm(dim=-1))                  # shared Schmidt spectrum
    C = cq.concurrence(psi)
    assert_max_abs(b_a.norm(dim=-1) ** 2 + C ** 2, 1.0)
    c, *_ = cq.complete_coords(b_a)
    assert_max_abs(cq.concurrence_from_polarization(c), C, 1e-9)        # CE1: C = sech c
    coef = psi.reshape(-1, 2, 2)
    schmidt = torch.linalg.svdvals(coef) ** 2
    p = schmidt.clamp_min(1e-300)
    assert_max_abs(cq.entanglement_entropy_from_polarization(c), -(p * p.log()).sum(-1), 1e-9)
    assert cq.entanglement_entropy_from_polarization(0.0).item() == pytest.approx(math.log(2))


def test_two_complete_bloch_spheres_parameterize_every_pure_pair(latent):
    c, u, v = latent
    g = torch.Generator().manual_seed(6)
    u_b, v_b = (2 * torch.randn(2, N, generator=g, dtype=torch.float64)).unbind(0)
    chi = 2 * math.pi * torch.rand(N, generator=g, dtype=torch.float64)
    sheet_b = torch.where(torch.rand(N, generator=g) < 0.5, -1.0, 1.0).double()
    psi = cq.entangled_pair(c, u, v, u_b, v_b, chi, sheet_b)            # CE2
    assert_max_abs(psi.norm(dim=-1), 1.0)
    b_a, b_b = cq.reduced_bloch(psi)
    assert_max_abs(b_a, cq.complete_bloch(c, u, v))
    assert_max_abs(b_b, cq.complete_bloch(sheet_b * c, u_b, v_b))
    assert_max_abs(cq.concurrence(psi), cq.concurrence_from_polarization(c), 1e-9)
    # Inverse: any pure two-qubit state (off the centre and the equator) is recovered up to phase.
    target = random_pure_pairs(N, 7)
    coords = cq.pair_coords(target)
    rebuilt = cq.entangled_pair(*coords)
    fidelity = (rebuilt.conj() * target).sum(-1).abs()
    assert_max_abs(fidelity, 1.0, 1e-9)


def test_sheet_measurement_on_one_neuron_collapses_the_partner():
    psi = cq.entangled_pair(torch.full((3000,), 0.4, dtype=torch.float64),
                            0.7, -0.3, -1.2, 0.5, 1.0)
    outcome, b_b = cq.measure_pair_sheets(psi, torch.Generator().manual_seed(8))   # CE3
    d_b = cq.sheet_to_sphere(cq.hyperboloid_point(torch.tensor(-1.2, dtype=torch.float64),
                                                  torch.tensor(0.5, dtype=torch.float64)))
    assert_max_abs(b_b, outcome[:, None] * d_b, 1e-9)                   # perfectly correlated sheets
    assert abs((outcome > 0).double().mean().item() - (1 + math.tanh(0.4)) / 2) < 0.03


def test_local_filters_move_the_entanglement_hyperboloid_by_lorentz_maps():
    psi = random_pure_pairs(500, 9)
    g = torch.Generator().manual_seed(10)
    A = torch.complex(torch.randn(500, 2, 2, generator=g, dtype=torch.float64),
                      torch.randn(500, 2, 2, generator=g, dtype=torch.float64))
    A = A / torch.sqrt(torch.linalg.det(A))[:, None, None]             # SL(2, C)
    L = cq.lorentz_from_filter(A)
    eta = torch.diag(torch.tensor([1.0, -1.0, -1.0, -1.0], dtype=torch.float64))
    assert_max_abs(L.transpose(-1, -2) @ eta @ L, eta.expand(500, 4, 4), 1e-8)
    filtered = torch.einsum('nij,njk->nik', A, psi.reshape(-1, 2, 2)).reshape(-1, 4)
    filtered = filtered / filtered.norm(dim=-1, keepdim=True)          # (A x I)|psi>, renormalized
    h_before = cq.entanglement_hyperboloid(cq.reduced_bloch(psi)[0])
    h_after = cq.entanglement_hyperboloid(cq.reduced_bloch(filtered)[0])
    assert_max_abs(h_before[:, 0] ** 2 - (h_before[:, 1:] ** 2).sum(-1), 1.0, 1e-9)
    assert torch.allclose(torch.einsum('nij,nj->ni', L, h_before), h_after, rtol=1e-7, atol=1e-7)
    bell = cq.entanglement_hyperboloid(torch.zeros(3, dtype=torch.float64))
    assert_max_abs(bell, torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float64))
