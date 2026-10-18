"""NeuroQuantum with complete Bloch sphere neurons (QBNNLayer(bloch_mode='complete'),
NeuroQuantumConfig(qbnn_bloch_mode='complete')): inline formulas against
complete_qbnn.py, exact compatibility with 1D / standard 3D checkpoints, gradients,
and sheet-axis measurement sampling."""

import pytest
import torch

import apqb_sqrt_amplitude as sa
import complete_qbnn as cq
import neuroquantum_layered as nq
from neuroquantum_layered import NeuroQuantum, NeuroQuantumConfig, QBNNLayer


@pytest.fixture(autouse=True)
def deterministic_cpu_tests():
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    with torch.random.fork_rng():
        torch.manual_seed(23)
        yield
    torch.set_num_threads(threads)


def randomize_complete_state(layer, std=0.5):
    with torch.no_grad():
        layer.bloch_uv.normal_(std=std)
        layer.bloch_uv_bias.normal_(std=std)
        layer.bloch_gate.normal_(std=std)


def test_inline_formulas_match_complete_qbnn():
    g = torch.Generator().manual_seed(0)
    c, u, v = (2 * torch.randn(3, 1000, generator=g, dtype=torch.float64)).unbind(0)
    torch.testing.assert_close(nq.complete_bloch_vectors(torch.tanh(c), u, v),
                               cq.complete_bloch(c, u, v), rtol=0, atol=1e-15)
    w = torch.randn(1000, 3, generator=g, dtype=torch.float64)
    w[:10] *= 1e-6
    torch.testing.assert_close(nq.axis_angle_rotation(w), cq.axis_angle_rotation(w), rtol=0, atol=1e-15)
    assert torch.equal(nq.axis_angle_rotation(torch.zeros(4, 3)), torch.eye(3).expand(4, 3, 3))


@pytest.mark.parametrize("use_core", [False, True])
def test_complete_layer_reproduces_1d_layer_exactly(use_core):
    flat = QBNNLayer(8, 12, use_qbnn_layered=use_core, bloch=False).eval()
    complete = QBNNLayer(8, 12, use_qbnn_layered=use_core, bloch_mode='complete').eval()
    randomize_complete_state(complete)  # the load hook must put every neuron back on an apex
    complete.load_state_dict(flat.state_dict(), strict=True)
    assert torch.count_nonzero(complete.bloch_uv) == 0
    assert torch.count_nonzero(complete.bloch_gate) == 0
    assert torch.count_nonzero(complete.J_xy_in) > 0  # random, so (u, v) get gradients
    h = torch.randn(3, 4, 8)
    torch.testing.assert_close(complete(h), flat(h), rtol=0, atol=0)
    assert set(complete.state_dict()) - set(flat.state_dict()) == {
        "bloch_proj.weight", "bloch_uv", "bloch_uv_bias", "bloch_gate",
        "J_xy_in", "J_xy_out"}
    # A fresh complete layer starts on the apexes too: b = (0, 0, tanh h̃).
    fresh = QBNNLayer(8, 12, use_qbnn_layered=use_core, bloch_mode='complete').eval()
    flat.load_state_dict({k: v for k, v in fresh.state_dict().items() if k in flat.state_dict()})
    torch.testing.assert_close(fresh(h), flat(h), rtol=0, atol=0)


def test_standard_and_complete_checkpoints_load_into_each_other():
    standard = QBNNLayer(8, 12, use_qbnn_layered=False).eval()
    complete = QBNNLayer(8, 12, use_qbnn_layered=False, bloch_mode='complete').eval()
    randomize_complete_state(complete)
    complete.load_state_dict(standard.state_dict(), strict=True)
    torch.testing.assert_close(complete.J_xy_in, standard.J_xy_in)
    assert torch.count_nonzero(complete.bloch_uv) == 0
    randomize_complete_state(complete)
    standard.load_state_dict(complete.state_dict(), strict=True)
    torch.testing.assert_close(standard.J_xy_out, complete.J_xy_out)
    with pytest.raises(ValueError):
        QBNNLayer(8, 12, bloch_mode='hyperbolic')


def test_complete_neurons_are_valid_states_in_paper_coordinates():
    layer = QBNNLayer(8, 12, use_qbnn_layered=False, bloch_mode='complete').double()
    randomize_complete_state(layer, std=2.0)
    h = torch.randn(5, 8, dtype=torch.float64) * 3
    h_tilde = layer.W(h)
    b, kappa = layer.complete_vectors(h, torch.tanh(h_tilde))
    uv = layer.bloch_proj(h) @ layer.bloch_uv + layer.bloch_uv_bias
    u, v = uv.unflatten(-1, (12, 2)).unbind(-1)
    R = cq.axis_angle_rotation(layer.bloch_gate)
    expected = torch.einsum('oij,...oj->...oi', R, cq.complete_bloch(h_tilde, u, v))
    torch.testing.assert_close(b, expected, rtol=0, atol=1e-12)
    assert (b.norm(dim=-1) <= 1 + 1e-12).all()
    assert torch.linalg.eigvalsh(sa.rho_from_bloch(b)).min() >= -1e-12
    n, _, kappa_ref = sa.from_bloch(b)
    torch.testing.assert_close(kappa.double(), kappa_ref, rtol=0, atol=1e-5)
    layer(h)
    info = layer.get_quantum_info()
    assert info["bloch_mode"] == "complete"
    assert 0 < info["bloch_upper_sheet_frac"] < 1
    assert info["bloch_depth_median"] > 0


def test_hyperboloid_coordinates_and_gates_learn_from_the_apex():
    layer = QBNNLayer(8, 12, use_qbnn_layered=False, bloch_mode='complete')
    h = torch.randn(4, 3, 8)
    layer(h).pow(2).sum().backward()
    # At the apexes x = y = 0, but the random J_x, J_y give (u, v) and the gates a gradient.
    for name in ("bloch_uv", "bloch_uv_bias", "bloch_gate"):
        assert layer.get_parameter(name).grad.abs().sum() > 0, name
    opt = torch.optim.SGD(layer.parameters(), lr=1.0)
    opt.step()
    layer.zero_grad()
    layer(h).pow(2).sum().backward()
    for name in ("bloch_proj.weight", "J_xy_in", "J_xy_out"):
        assert layer.get_parameter(name).grad.abs().sum() > 0, name


def test_sheet_measurement_collapses_and_is_unbiased():
    layer = QBNNLayer(6, 50, use_qbnn_layered=False, bloch_mode='complete').double()
    randomize_complete_state(layer)
    h = torch.randn(1, 6, dtype=torch.float64)
    b, _ = layer.complete_vectors(h, torch.tanh(layer.W(h)))
    layer.measurement_rate = 1.0
    samples = torch.stack([layer.measure_sheet(b) for _ in range(4000)])
    torch.testing.assert_close(samples.norm(dim=-1), torch.ones_like(samples[..., 0]))
    torch.testing.assert_close(samples.abs(), (b / b.norm(dim=-1, keepdim=True)).abs().expand_as(samples))
    assert (samples.mean(0) - b).abs().max() < 0.06
    # Straight-through: same gradient as the unmeasured state.
    bb = b.detach().clone().requires_grad_(True)
    layer.measure_sheet(bb).sum().backward()
    torch.testing.assert_close(bb.grad, torch.ones_like(b))
    layer.measurement_rate = 0.0
    assert layer.measure_sheet(b) is b


def test_neuroquantum_model_with_complete_bloch_neurons():
    kwargs = dict(vocab_size=32, embed_dim=16, hidden_dim=24, num_heads=2, num_layers=2,
                  max_seq_len=32, attention_window=4, dropout=0.0, gradient_checkpointing=False)
    flat = NeuroQuantum(NeuroQuantumConfig(qbnn_bloch=False, **kwargs)).eval()
    complete = NeuroQuantum(NeuroQuantumConfig(qbnn_bloch_mode='complete', **kwargs)).eval()
    layers = [m for m in complete.modules() if isinstance(m, QBNNLayer)]
    # NeuroQuantum._init_weights re-initializes every nn.Linear; the apex must survive it,
    # including when missing keys are filled from the fresh model (migrate_legacy_state_dict).
    assert all(torch.count_nonzero(l.bloch_uv) == 0 and torch.count_nonzero(l.bloch_uv_bias) == 0
               for l in layers)
    tokens = torch.randint(0, 32, (2, 10))
    complete.load_state_dict(nq.migrate_legacy_state_dict(flat.state_dict(), complete), strict=True)
    torch.testing.assert_close(complete(tokens), flat(tokens), rtol=0, atol=0)
    complete.load_state_dict(flat.state_dict(), strict=True)
    torch.testing.assert_close(complete(tokens), flat(tokens), rtol=0, atol=0)
    assert len(layers) == 4 and all(l.bloch_mode == 'complete' for l in layers)
    for layer in layers:
        randomize_complete_state(layer, std=0.3)
    with complete.measurement(1.0):
        measured = complete(tokens)
    assert torch.isfinite(measured).all()
    info = complete.get_quantum_info()
    assert all('bloch_upper_sheet_frac' in block for block in info)
