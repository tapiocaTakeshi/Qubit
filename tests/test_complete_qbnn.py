"""Complete Bloch neurons (complete QBNN): sphere 0 <= n <= 4 plus the two-sheeted
hyperboloid n < 0, n > 4 of the square-root amplitude paper v0.4, Section 6."""

import pytest
import torch

from neuroquantum_layered import NeuroQuantum, NeuroQuantumConfig, QBNNLayer


@pytest.fixture(autouse=True)
def deterministic_cpu_tests():
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    with torch.random.fork_rng():
        torch.manual_seed(5)
        yield
    torch.set_num_threads(threads)


def test_rejects_unknown_neuron_kind():
    with pytest.raises(ValueError):
        QBNNLayer(4, 6, neuron="torus")


def test_beta_zero_is_the_sphere_neuron():
    sphere = QBNNLayer(8, 12, use_qbnn_layered=False).eval()
    complete = QBNNLayer(8, 12, use_qbnn_layered=False, neuron="complete").eval()
    complete.load_state_dict(sphere.state_dict(), strict=True)  # beta is filled with 0
    assert complete.complete_beta.item() == 0.0
    h = 3 * torch.randn(3, 5, 8)
    torch.testing.assert_close(complete(h), sphere(h), rtol=0, atol=0)  # J_x = J_y = 0
    with torch.no_grad():
        for layer in (sphere, complete):
            layer.J_xy_in.copy_(torch.linspace(-0.5, 0.5, layer.J_xy_in.numel()).view_as(layer.J_xy_in))
    # rho = sqrt(1 - tanh^2) equals sech up to rounding.
    torch.testing.assert_close(complete(h), sphere(h), rtol=0, atol=1e-5)
    assert complete.get_quantum_info()["region_sphere"] == 1.0


def test_complete_coordinates_follow_sphere_and_hyperboloid():
    layer = QBNNLayer(4, 6, use_qbnn_layered=False, neuron="complete").double()
    with torch.no_grad():
        layer.complete_beta.fill_(1.0)
    h = torch.linspace(-3, 3, 601, dtype=torch.float64)
    z, rho, sigma = layer.complete_coords(h)
    torch.testing.assert_close(z, h)  # beta = 1: z is the pre-activation itself
    sphere, hyper = sigma > 0, sigma < 0
    assert torch.equal(sphere, h.abs() <= 1) and torch.equal(hyper, h.abs() > 1)
    # Sphere: z^2 + rho^2 = 1.  Hyperboloid: z^2 - rho^2 = 1 (Section 6.1).
    torch.testing.assert_close((z ** 2 + rho ** 2)[sphere], torch.ones(int(sphere.sum()), dtype=torch.float64))
    torch.testing.assert_close((z ** 2 - rho ** 2)[hyper], torch.ones(int(hyper.sum()), dtype=torch.float64))
    # Proposition 3: Minkowski interval to the north pole equals n = 2(1 - z).
    n = 2 * (1 - z)
    torch.testing.assert_close(((z - 1) ** 2 - rho ** 2)[hyper], n[hyper])


def test_gradients_stay_finite_on_the_boundary():
    layer = QBNNLayer(4, 6, use_qbnn_layered=False, neuron="complete")
    with torch.no_grad():
        layer.complete_beta.fill_(1.0)
    h = torch.tensor([-1.0, -0.999, 1.0, 1.001, 0.0], requires_grad=True)
    z, rho, sigma = layer.complete_coords(h)
    (rho.sum() + z.sum()).backward()
    assert torch.isfinite(h.grad).all()
    assert h.grad.abs().max() <= 1 + 1 / (2 * layer.complete_eps) * 2 + 1e-6


def test_beta_learns_and_layer_trains_end_to_end():
    layer = QBNNLayer(8, 12, use_qbnn_layered=False, neuron="complete")
    with torch.no_grad():
        layer.J_xy_in.normal_(std=0.1)
    out = layer(2 * torch.randn(4, 6, 8))
    out.pow(2).mean().backward()
    assert layer.complete_beta.grad is not None and layer.complete_beta.grad.abs() > 0
    assert all(torch.isfinite(p.grad).all() for p in layer.parameters() if p.grad is not None)


def test_measurement_and_floor_only_use_probabilities_on_the_sphere():
    layer = QBNNLayer(4, 6, use_qbnn_layered=False, neuron="complete").double()
    with torch.no_grad():
        layer.complete_beta.fill_(1.0)
    layer.measurement_rate = 1.0
    z = torch.tensor([-2.5, -0.3, 0.4, 3.0], dtype=torch.float64).repeat(200, 1)
    out, measured = layer.measure_z(z, z.abs() <= 1)
    assert not measured[:, [0, 3]].any() and measured[:, [1, 2]].all()
    torch.testing.assert_close(out[:, [0, 3]], z[:, [0, 3]])  # hyperboloid: no probability
    assert set(out[:, [1, 2]].unique().tolist()) <= {-1.0, 1.0}
    # Floor on the whole real line: n = 2(1 - z) -> floor(n), z on the half-integer grid.
    layer.measurement_rate = 0.0
    layer.n_floor = True
    h = torch.tensor([-2.3, -0.3, 0.4, 2.9], dtype=torch.float64)
    zf, _, _ = layer.complete_coords(h)
    torch.testing.assert_close(zf, 1 - torch.floor(2 * (1 - h)) / 2)
    assert (zf.abs() > 1).any()


def test_complete_neuroquantum_model_loads_sphere_checkpoints_and_reports_regions():
    common = dict(vocab_size=40, embed_dim=16, hidden_dim=24, num_heads=2, num_layers=2,
                  max_seq_len=32, attention_window=8, dropout=0.0, gradient_checkpointing=False)
    sphere = NeuroQuantum(NeuroQuantumConfig(**common)).eval()
    complete = NeuroQuantum(NeuroQuantumConfig(**common, qbnn_neuron="complete")).eval()
    complete.load_state_dict(sphere.state_dict(), strict=True)
    tokens = torch.randint(0, 40, (2, 10))
    with torch.no_grad():
        torch.testing.assert_close(complete(tokens), sphere(tokens), rtol=0, atol=0)
        for m in complete.modules():
            if isinstance(m, QBNNLayer):
                m.complete_beta.fill_(1.0)
                m.W.weight.mul_(100.0)  # trained pre-activations are far larger than at init
        complete(tokens)
    info = complete.get_quantum_info()[0]["qbnn_layers"][0]
    assert info["neuron"] == "complete" and info["complete_beta"] == 1.0
    assert abs(info["region_upper"] + info["region_sphere"] + info["region_lower"] - 1) < 1e-6
    assert info["region_upper"] + info["region_lower"] > 0  # some neurons left the sphere


def test_sphere_model_refuses_a_trained_complete_checkpoint():
    complete = QBNNLayer(4, 6, use_qbnn_layered=False, neuron="complete")
    sphere = QBNNLayer(4, 6, use_qbnn_layered=False)
    sphere.load_state_dict(complete.state_dict(), strict=True)  # beta = 0: identical, accepted
    with torch.no_grad():
        complete.complete_beta.fill_(0.4)
    with pytest.raises(ValueError, match="qbnn_neuron='complete'"):
        sphere.load_state_dict(complete.state_dict(), strict=True)


def test_scale_gamma_multiplies_the_pre_activation_and_is_saved():
    with pytest.raises(ValueError):
        QBNNLayer(4, 6, neuron="complete", complete_scale=0.0)
    layer = QBNNLayer(4, 6, use_qbnn_layered=False, neuron="complete", complete_scale=10.0).double()
    h = torch.linspace(-0.5, 0.5, 101, dtype=torch.float64)
    with torch.no_grad():
        layer.complete_beta.fill_(1.0)
    z, _, sigma = layer.complete_coords(h)
    torch.testing.assert_close(z, 10 * h)  # beta = 1: z = gamma * h
    assert (sigma < 0).float().mean() > 0.7  # |10 h| > 1 for most of [-0.5, 0.5]
    with torch.no_grad():
        layer.complete_beta.fill_(0.0)
    torch.testing.assert_close(layer.complete_coords(h)[0], torch.tanh(10 * h))
    # gamma travels with the checkpoint; old checkpoints without it were trained with gamma = 1.
    other = QBNNLayer(4, 6, use_qbnn_layered=False, neuron="complete", complete_scale=3.0).double()
    other.load_state_dict(layer.state_dict(), strict=True)
    assert other.complete_scale.item() == 10.0
    legacy = {k: v for k, v in layer.state_dict().items() if k != "complete_scale"}
    other.load_state_dict(legacy, strict=True)
    assert other.complete_scale.item() == 1.0
    # A sphere model must not silently drop gamma != 1, even with beta = 0.
    with pytest.raises(ValueError, match="qbnn_neuron='complete'"):
        QBNNLayer(4, 6, use_qbnn_layered=False).double().load_state_dict(layer.state_dict(), strict=True)


def test_config_scale_reaches_every_layer():
    cfg = NeuroQuantumConfig(vocab_size=40, embed_dim=16, hidden_dim=24, num_heads=2, num_layers=2,
                             max_seq_len=32, dropout=0.0, gradient_checkpointing=False,
                             qbnn_neuron="complete", qbnn_complete_scale=25.0)
    model = NeuroQuantum(cfg)
    scales = {m.complete_scale.item() for m in model.modules() if isinstance(m, QBNNLayer)}
    assert scales == {25.0}
