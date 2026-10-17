"""NeuroQuantum's use of the square-root amplitude paper (draft v0.1): APQB
diagnostics of QBNNLayer (Eq. 11-12), the opt-in coherence-calibrated sampling temperature
(revised paper Eq. 18), and quantum-info reporting after training."""

from unittest.mock import patch

import pytest
import torch

import apqb_sqrt_amplitude as sa
import neuroquantum_layered as nq
from apqb_qbnn_v2 import calibrated_temperature
from neuroquantum_layered import NeuroQuantumAI, QBNNLayer


@pytest.fixture(autouse=True)
def deterministic_cpu_tests():
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    with torch.random.fork_rng():
        torch.manual_seed(17)
        yield
    torch.set_num_threads(threads)


def test_inline_formulas_match_reference_implementations():
    s = torch.linspace(-50, 50, 4001, dtype=torch.float64)
    n, eta = nq.apqb_latent_coords(s)
    n_ref, _, eta_ref = sa.from_latent(s)
    torch.testing.assert_close(n, n_ref, rtol=0, atol=1e-15)
    torch.testing.assert_close(eta, eta_ref, rtol=0, atol=1e-15)
    huge = torch.tensor([-1e4, 1e4])  # float32: cosh would overflow
    assert torch.isfinite(nq.apqb_latent_coords(huge)[1]).all()
    for q in (0.0, 0.25, 0.9, 1.0):
        for gamma in (0.5, 1.0, 2.0):
            ref = calibrated_temperature(torch.tensor(q), tau_min=0.4, tau_max=0.8, gamma=gamma)
            assert nq.apqb_calibrated_temperature(q, 0.4, 0.8, gamma) == pytest.approx(ref.item())
    assert nq.apqb_calibrated_temperature(1.7, 0.4, 0.8) == pytest.approx(0.8)


@pytest.mark.parametrize("use_core", [False, True])
def test_qbnn_layer_records_apqb_coordinates_without_changing_checkpoints(use_core):
    layer = QBNNLayer(8, 12, use_qbnn_layered=use_core).eval()
    keys_before = set(layer.state_dict())
    h = torch.randn(3, 5, 8)
    layer(h)
    h_tilde = layer.W(h)
    n, eta = sa.from_latent(h_tilde.double())[0], sa.sech(h_tilde.double())
    info = layer.get_quantum_info()
    assert info["apqb_n_mean"] == pytest.approx(n.mean().item(), abs=1e-6)
    assert info["apqb_p1_mean"] == pytest.approx(n.mean().item() / 4, abs=1e-6)
    assert info["apqb_eta_mean"] == pytest.approx(eta.mean().item(), abs=1e-6)
    assert info["apqb_grad_gain"] == pytest.approx((eta ** 2).mean().item(), abs=1e-6)
    torch.testing.assert_close(layer.last_eta_token, eta.mean(-1).float(), rtol=0, atol=1e-6)
    # Diagnostics are non-persistent: checkpoints keep exactly the same keys.
    assert set(layer.state_dict()) == keys_before
    restored = QBNNLayer(8, 12, use_qbnn_layered=use_core).eval()
    restored.load_state_dict(layer.state_dict(), strict=True)
    torch.testing.assert_close(restored(h), layer(h), rtol=0, atol=0)


def test_grad_gain_is_the_tanh_path_derivative_and_saturation_is_detected():
    layer = QBNNLayer(4, 6, use_qbnn_layered=False)
    h_tilde = torch.randn(2, 6, dtype=torch.float64, requires_grad=True)
    grad, = torch.autograd.grad(torch.tanh(h_tilde).sum(), h_tilde)
    torch.testing.assert_close(grad, nq.apqb_latent_coords(h_tilde.detach())[1] ** 2)  # Eq. (12)
    with torch.no_grad():
        layer.W.weight.fill_(10.0)
        layer.W.bias.zero_()
    layer(torch.ones(2, 3, 4))  # every h̃ = 40: fully saturated tanh path
    info = layer.get_quantum_info()
    assert info["apqb_saturation"] == pytest.approx(1.0)
    assert info["apqb_grad_gain"] < 1e-30
    assert info["apqb_n_mean"] == pytest.approx(0.0, abs=1e-6)


@pytest.fixture(scope="module")
def trained_ai():
    ai = NeuroQuantumAI(embed_dim=16, hidden_dim=24, num_heads=2, num_layers=2,
                        max_seq_len=32, dropout=0.0)
    texts = ["量子ビットの状態は確率と位相とコヒーレンスで決まる。"] * 4
    # Character fallback tokenizer whether or not sentencepiece is installed
    # (this corpus is far too small for a 32k SentencePiece vocabulary).
    with patch.object(nq, "SENTENCEPIECE_AVAILABLE", False):
        # Regression: train() used to end with KeyError('attn_lambda').
        ai.train(texts, epochs=1, batch_size=2, seq_len=16)
    return ai


def test_model_quantum_info_reports_every_block(trained_ai):
    info = trained_ai.model.get_quantum_info()
    assert [b["block"] for b in info] == [0, 1]
    for block in info:
        assert len(block["qbnn_layers"]) == 2
        assert 0 <= block["apqb_n_mean"] <= 4
        assert 0 < block["apqb_eta_mean"] <= 1
        assert 0 <= block["apqb_saturation"] <= 1
    trained_ai._print_quantum_info()
    details = trained_ai.model.forward_with_details(torch.tensor([[1, 2, 3]]))
    assert len(details["layers"][0]["qbnn"]) == 2


def test_relative_coherence_normalizes_within_the_context():
    eta = torch.tensor([0.80, 0.82, 0.84, 0.86, 0.88, 0.90])
    assert nq.apqb_relative_coherence(eta) == pytest.approx(1.0)       # highest in context
    assert nq.apqb_relative_coherence(eta.flip(0)) == pytest.approx(0.0)
    mid = torch.tensor([0.80, 0.90, 0.85])
    assert 0.0 < nq.apqb_relative_coherence(mid) < 1.0
    assert nq.apqb_relative_coherence(torch.tensor([0.8, 0.9])) == 0.5  # too few tokens
    assert nq.apqb_relative_coherence(torch.full((5,), 0.85)) == 0.5    # no spread


def test_apqb_temperature_rises_when_coherence_is_low(trained_ai):
    """High eta-bar goes with low predictive entropy, so the temperature uses
    u = 1 - (relative coherence of the last token)."""
    seen = []
    real = nq.apqb_calibrated_temperature

    def spy(u, tau_min, tau_max, gamma):
        expected = 1.0 - nq.apqb_relative_coherence(trained_ai.model.apqb_coherence()[0])
        seen.append((u, expected, tau_min, tau_max, gamma))
        return real(u, tau_min, tau_max, gamma)

    with patch.object(nq, "apqb_calibrated_temperature", side_effect=spy):
        trained_ai.generate("量子", max_length=5, temp_min=0.3, temp_max=0.9,
                            temperature_mode="apqb", temp_gamma=2.0)
    assert seen
    for u, expected, tau_min, tau_max, gamma in seen:
        assert u == pytest.approx(expected) and 0.0 <= u <= 1.0
        assert (tau_min, tau_max, gamma) == (0.3, 0.9, 2.0)
    assert nq.apqb_calibrated_temperature(0.0, 0.3, 0.9) == pytest.approx(0.3)  # confident
    assert nq.apqb_calibrated_temperature(1.0, 0.3, 0.9) == pytest.approx(0.9)  # unsure

    with patch.object(nq, "apqb_calibrated_temperature", side_effect=spy) as default_call:
        trained_ai.generate("量子", max_length=3)
    default_call.assert_not_called()  # the default sine schedule is unchanged

    with pytest.raises(ValueError):
        trained_ai.generate("量子", temperature_mode="cosine")
    with pytest.raises(ValueError):
        trained_ai.generate("量子", temperature_mode="apqb", temp_gamma=0.0)


def test_generate_prompt_matches_training_format_without_trailing_eos(trained_ai):
    """Training sequences are [BOF, BOS] + text + [EOS, EOF]; a prompt ending in
    EOS made the model predict EOF immediately, so generation returned nothing."""
    tok = trained_ai.tokenizer
    seen = []
    real_forward = trained_ai.model.forward

    def spy(token_ids, *args, **kwargs):
        seen.append(token_ids[0].tolist())
        return real_forward(token_ids, *args, **kwargs)

    with patch.object(trained_ai.model, "forward", side_effect=spy):
        trained_ai.generate("量子ビット", max_length=2)
    content = tok.encode("量子ビット", add_special=False)
    assert seen[0] == [tok.bof_id, tok.bos_id] + content


def test_bloch_neurons_are_valid_density_matrices_in_paper_coordinates():
    layer = QBNNLayer(8, 12, use_qbnn_layered=False).double()
    with torch.no_grad():
        layer.W.weight.mul_(3.0)  # spread h̃ so z covers most of (-1, 1)
    h = torch.randn(4, 5, 8, dtype=torch.float64)
    h_tilde = layer.W(h)
    b, kappa = layer.bloch_vectors(h, h_tilde)
    assert b.shape == (4, 5, 12, 3)
    # Eq. (6): b = (k eta cos phi, k eta sin phi, 1 - n/2) with n = 2(1 - tanh h̃).
    n = 2 * (1 - torch.tanh(h_tilde))
    phi = layer.bloch_phi(layer.bloch_proj(h))
    torch.testing.assert_close(b, sa.bloch(n, phi, kappa), rtol=0, atol=1e-12)
    assert (b.norm(dim=-1) <= 1 + 1e-12).all()
    rho = sa.rho_from_bloch(b)
    assert torch.linalg.eigvalsh(rho).min() >= -1e-12  # Prop. 2
    n2, _, kappa2 = sa.from_bloch(b)
    torch.testing.assert_close(n2, n, rtol=0, atol=1e-12)
    torch.testing.assert_close(kappa2, kappa, rtol=0, atol=1e-9)


@pytest.mark.parametrize("use_core", [False, True])
def test_bloch_layer_reproduces_1d_layer_from_a_1d_checkpoint(use_core):
    flat = QBNNLayer(8, 12, use_qbnn_layered=use_core, bloch=False).eval()
    bloch = QBNNLayer(8, 12, use_qbnn_layered=use_core, bloch=True).eval()
    with torch.no_grad():
        bloch.J_xy_in.normal_()  # must be reset to zero by the load hook
    bloch.load_state_dict(flat.state_dict(), strict=True)
    assert torch.count_nonzero(bloch.J_xy_in) == 0
    h = torch.randn(3, 4, 8)
    torch.testing.assert_close(bloch(h), flat(h), rtol=0, atol=0)
    assert set(bloch.state_dict()) - set(flat.state_dict()) == {
        "bloch_proj.weight", "bloch_phi.weight", "bloch_phi.bias",
        "bloch_kappa.weight", "bloch_kappa.bias", "J_xy_in", "J_xy_out"}


def test_bloch_neurons_carry_phase_that_the_z_coordinate_cannot():
    """Eq. (9): states with equal n but different phase interfere differently.
    Changing only phi leaves z = tanh h̃ untouched, so a 1D neuron cannot see it."""
    layer = QBNNLayer(8, 12, use_qbnn_layered=False).eval()
    h = torch.randn(2, 3, 8)
    at_init = layer(h)
    with torch.no_grad():
        layer.bloch_phi.bias.add_(torch.pi)  # phi -> phi + pi
    torch.testing.assert_close(layer(h), at_init, rtol=0, atol=0)  # J_x = J_y = 0 at init
    with torch.no_grad():
        layer.J_xy_in.normal_(std=0.5)
        layer.lambda_base.fill_(5.0)
    before = layer(h)
    with torch.no_grad():
        layer.bloch_phi.bias.add_(torch.pi)
    after = layer(h)
    assert (after - before).abs().max() > 1e-3


def test_bloch_parameters_receive_gradients():
    layer = QBNNLayer(8, 12, use_qbnn_layered=False)
    h = torch.randn(2, 3, 8)
    layer(h).pow(2).sum().backward()
    # At init only J_xy_in can move (it gates the x, y path).
    assert layer.J_xy_in.grad.abs().sum() > 0
    layer.zero_grad()
    with torch.no_grad():
        layer.J_xy_in.normal_(std=0.1)
    layer(h).pow(2).sum().backward()
    for name in ("bloch_proj.weight", "bloch_phi.weight", "bloch_kappa.weight", "J_xy_out"):
        assert layer.get_parameter(name).grad.abs().sum() > 0, name
    info = layer.get_quantum_info()
    assert 0 < info["bloch_kappa_mean"] < 1
    assert 0 < info["bloch_norm_mean"] <= 1
    assert 0.5 <= info["bloch_purity_mean"] <= 1
    assert info["J_xy_norm"] > 0


def test_measurement_is_an_unbiased_born_rule_sample():
    layer = QBNNLayer(4, 6, use_qbnn_layered=False)
    z = torch.tensor([1.0, 0.5, 0.0, -0.5, -1.0], dtype=torch.float64).repeat(40_000, 1)
    assert layer.measure_z(z) == (z, None)  # rate 0: unchanged
    layer.measurement_rate = 1.0
    out, measured = layer.measure_z(z)
    assert measured.all()
    assert set(out.unique().tolist()) <= {-1.0, 1.0}
    # Eq. (2): P(1) = n/4 = (1 - z)/2, so E[outcome] = z; the poles are deterministic.
    torch.testing.assert_close(out.mean(0), z[0], rtol=0, atol=0.02)
    assert (out[:, 0] == 1).all() and (out[:, -1] == -1).all()
    layer.measurement_rate = 0.3
    _, measured = layer.measure_z(z)
    assert measured.float().mean().item() == pytest.approx(0.3, abs=0.01)
    # Straight-through: the gradient is that of the expectation z.
    zg = torch.full((3, 6), 0.2, requires_grad=True)
    layer.measure_z(zg)[0].sum().backward()
    torch.testing.assert_close(zg.grad, torch.ones_like(zg))


def test_measured_neurons_lose_phase():
    layer = QBNNLayer(8, 12, use_qbnn_layered=False).eval()
    with torch.no_grad():
        layer.J_xy_in.normal_(std=0.5)
        layer.lambda_base.fill_(5.0)
    h = torch.randn(2, 3, 8)
    layer.measurement_rate = 1.0

    def run():
        torch.manual_seed(0)
        return layer(h)

    before = run()
    with torch.no_grad():
        layer.bloch_phi.bias.add_(torch.pi)
    torch.testing.assert_close(run(), before, rtol=0, atol=0)  # x = y = 0 after measurement
    layer.measurement_rate = 0.0
    unmeasured = layer(h)
    with torch.no_grad():
        layer.bloch_phi.bias.add_(torch.pi)
    assert (layer(h) - unmeasured).abs().max() > 1e-3


def test_measurement_context_restores_rate_and_validates(trained_ai):
    model = trained_ai.model
    layers = [m for m in model.modules() if isinstance(m, QBNNLayer)]
    with model.measurement(0.5):
        assert all(layer.measurement_rate == 0.5 for layer in layers)
    assert all(layer.measurement_rate == 0.0 for layer in layers)
    with pytest.raises(ValueError):
        model.set_measurement_rate(1.5)
    with pytest.raises(ValueError):
        trained_ai.generate("量子", measurement_rate=-0.1)
    tokens = torch.tensor([[1, 2, 3, 4]])
    with torch.no_grad():
        plain = model(tokens)
        torch.manual_seed(1)
        with model.measurement(1.0):
            measured = model(tokens)
    assert not torch.equal(plain, measured)
    torch.testing.assert_close(model(tokens), plain, rtol=0, atol=0)  # rate restored


def test_measurement_uncertainty_reports_ensemble_disagreement(trained_ai):
    torch.manual_seed(3)
    result = trained_ai.measurement_uncertainty("量子", shots=4, max_length=3)
    nt = result["next_token"]
    assert nt["mutual_information"] >= 0
    assert nt["entropy"] >= nt["expected_entropy"] - 1e-6
    assert 0.25 <= nt["top1_agreement"] <= 1
    assert len(result["answers"]) == 4
    assert 0.25 <= result["answer_agreement"] <= 1
    assert 0 <= result["answer_similarity"] <= 1
    assert all(layer.measurement_rate == 0.0
               for layer in trained_ai.model.modules() if isinstance(layer, QBNNLayer))
    with pytest.raises(ValueError):
        trained_ai.measurement_uncertainty("量子", shots=1)
    with pytest.raises(ValueError):
        trained_ai.measurement_uncertainty("量子", measurement_rate=0.0)


def test_n_floor_quantizes_to_the_five_integer_levels():
    layer = QBNNLayer(4, 6, use_qbnn_layered=False)
    h = torch.linspace(-12, 12, 2001, dtype=torch.float64, requires_grad=True)
    z, eta = layer.z_eta(h)  # default: continuous
    torch.testing.assert_close(z, torch.tanh(h))
    torch.testing.assert_close(eta, sa.sech(h))
    layer.n_floor = True
    z, eta = layer.z_eta(h)
    n = 2 * (1 - torch.tanh(h.detach()))
    n_int = torch.floor(n)
    assert set(n_int.unique().tolist()) <= {0.0, 1.0, 2.0, 3.0, 4.0}
    torch.testing.assert_close(z.detach(), 1 - n_int / 2, rtol=0, atol=0)
    torch.testing.assert_close(eta.detach(), torch.sqrt(n_int * (4 - n_int)) / 2, rtol=0, atol=0)
    torch.testing.assert_close(z.detach() ** 2 + eta.detach() ** 2, torch.ones_like(z), rtol=0, atol=1e-12)
    assert ((n - n_int) >= 0).all() and ((n - n_int) < 1).all()  # floor, not rounding
    # Straight-through: gradients are those of tanh and sech.
    grad, = torch.autograd.grad(z.sum() + eta.sum(), h)
    torch.testing.assert_close(grad, sa.sech(h.detach()) ** 2 - torch.tanh(h.detach()) * sa.sech(h.detach()))


def test_n_floor_keeps_bloch_vectors_valid_and_combines_with_measurement():
    layer = QBNNLayer(8, 12, use_qbnn_layered=False).double()
    layer.n_floor = True
    with torch.no_grad():
        layer.W.weight.mul_(3.0)
    h = torch.randn(4, 5, 8, dtype=torch.float64)
    h_tilde = layer.W(h)
    b, _ = layer.bloch_vectors(h, h_tilde)
    assert (b.norm(dim=-1) <= 1 + 1e-12).all()
    assert set(b[..., 2].unique().tolist()) <= {1.0, 0.5, 0.0, -0.5, -1.0}
    # Measuring a floored z: P(1) = floor(n)/4, so floor(n) = 0 always gives +1.
    layer.measurement_rate = 1.0
    z = torch.ones(1000, dtype=torch.float64)
    assert (layer.measure_z(z)[0] == 1).all()


def test_model_n_floor_context_and_generate(trained_ai):
    model = trained_ai.model
    layers = [m for m in model.modules() if isinstance(m, QBNNLayer)]
    tokens = torch.tensor([[1, 2, 3, 4, 5]])
    with torch.no_grad():
        plain = model(tokens)
        with model.n_floor():
            assert all(layer.n_floor for layer in layers)
            floored = model(tokens)
    assert not any(layer.n_floor for layer in layers)
    assert not torch.equal(plain, floored)
    torch.testing.assert_close(model(tokens), plain, rtol=0, atol=0)
    seen = []
    real_forward = model.forward

    def spy(*args, **kwargs):
        seen.append(all(layer.n_floor for layer in layers))
        return real_forward(*args, **kwargs)

    with patch.object(model, "forward", side_effect=spy):
        trained_ai.generate("量子", max_length=3, n_floor=True)
    assert seen and all(seen)
    assert not any(layer.n_floor for layer in layers)
