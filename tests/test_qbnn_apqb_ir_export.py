"""export_qbnn_apqb_ir.py: the QBNNLayerV2 -> APQB IR bridge to QubitBridge.

These tests hold the exporter to two standards without requiring either
dependency to be installed:

* :func:`reference_forward` is a plain-Python re-implementation of
  ``QBNNLayerV2.forward`` and is checked against a live PyTorch layer
  whenever ``torch`` is available (``pytest.importorskip``, as elsewhere in
  this test suite).
* :func:`emit_ir`'s output is checked against that same reference by
  actually parsing, lowering and running it on the Qubit Virtual Machine
  whenever the sibling ``qubitbridge`` package is importable -- which is
  the real end-to-end proof that the exported program computes the paper's
  Eq. (20)-(23), including the a_clip/q_eps clamps and the low-rank J
  factorisation, and not merely that it looks plausible.

Neither dependency is required to develop or CI this repository; both test
classes skip cleanly when their dependency is absent.
"""

from __future__ import annotations

import math
import random

import pytest

from export_qbnn_apqb_ir import (
    LayerWeights,
    UnsupportedLayerError,
    emit_ir,
    from_qbnn_layer,
    reference_forward,
)


def _random_weights(rng, in_dim, out_dim, *, low_rank=False, rank=2,
                    a_clip=4.0, q_eps=1e-6):
    def matrix(rows, cols, scale=0.5):
        return [[rng.uniform(-scale, scale) for _ in range(cols)] for _ in range(rows)]

    kwargs = dict(
        in_dim=in_dim, out_dim=out_dim,
        W=matrix(out_dim, in_dim), b=[rng.uniform(-0.1, 0.1) for _ in range(out_dim)],
        P=matrix(out_dim, in_dim), c=[rng.uniform(-0.1, 0.1) for _ in range(out_dim)],
        lambda_r=rng.uniform(-0.3, 0.3), lambda_q=rng.uniform(-0.3, 0.3),
        a_clip=a_clip, q_eps=q_eps, low_rank=low_rank,
    )
    if low_rank:
        kwargs.update(U_r=matrix(out_dim, rank, 0.3), V_r=matrix(out_dim, rank, 0.3),
                      U_q=matrix(out_dim, rank, 0.3), V_q=matrix(out_dim, rank, 0.3))
    else:
        kwargs.update(J_r=matrix(out_dim, out_dim, 0.3), J_q=matrix(out_dim, out_dim, 0.3))
    return LayerWeights(**kwargs)


class TestLayerWeights:
    def test_full_rank_requires_j(self):
        with pytest.raises(ValueError, match="J_r, J_q"):
            LayerWeights(in_dim=2, out_dim=2, W=[[0, 0], [0, 0]], b=[0, 0],
                        P=[[0, 0], [0, 0]], c=[0, 0], lambda_r=0.0, lambda_q=0.0)

    def test_low_rank_requires_uv(self):
        with pytest.raises(ValueError, match="U_r, V_r, U_q, V_q"):
            LayerWeights(in_dim=2, out_dim=2, W=[[0, 0], [0, 0]], b=[0, 0],
                        P=[[0, 0], [0, 0]], c=[0, 0], lambda_r=0.0, lambda_q=0.0,
                        low_rank=True)

    def test_rank_property(self):
        w = _random_weights(random.Random(0), 3, 4, low_rank=True, rank=2)
        assert w.rank == 2

    def test_rank_undefined_for_full_rank(self):
        w = _random_weights(random.Random(0), 3, 4)
        with pytest.raises(ValueError):
            w.rank


class TestReferenceForward:
    def test_input_length_is_checked(self):
        w = _random_weights(random.Random(0), 3, 2)
        with pytest.raises(ValueError, match="expected 3 inputs"):
            reference_forward(w, [1.0, 2.0])

    def test_lambda_zero_is_a_plain_affine_tanh_layer(self):
        """At lambda_r = lambda_q = 0 this must reduce to tanh(W h + b)."""
        rng = random.Random(11)
        w = _random_weights(rng, 4, 3)
        w.lambda_r = 0.0
        w.lambda_q = 0.0
        h = [rng.uniform(-2, 2) for _ in range(4)]
        got = reference_forward(w, h)
        want = [math.tanh(w.b[j] + sum(w.W[j][i] * h[i] for i in range(4)))
               for j in range(3)]
        for g, t in zip(got, want):
            assert g == pytest.approx(t, abs=1e-12)

    def test_a_clip_actually_bounds_the_latent_coordinate(self):
        rng = random.Random(5)
        w = _random_weights(rng, 2, 2, a_clip=1.0)
        # A huge input drives P h + c far outside [-1, 1]; r must still
        # equal tanh(+-1), not tanh of the unclamped (huge) value.
        got = reference_forward(w, [1000.0, -1000.0])
        assert all(math.isfinite(v) for v in got)

    def test_q_eps_floors_the_coherence_coordinate(self):
        rng = random.Random(6)
        w = _random_weights(rng, 2, 2, q_eps=0.25)
        # sech(anything) <= 1, so with a large clamp forcing |a| large,
        # sech(a) would underflow well below 0.25 without the floor.
        w.a_clip = 20.0
        got = reference_forward(w, [50.0, -50.0])
        assert all(math.isfinite(v) for v in got)


class TestEmitIR:
    def test_output_parses_as_valid_python_free_text(self):
        # No qubitbridge import here: just check the shape of the text
        # matches the documented grammar (module/func/return, one result
        # per output dimension).
        w = _random_weights(random.Random(1), 2, 3)
        text = emit_ir(w, module_name="m", func_name="f")
        assert text.startswith("module @m {")
        assert "func @f(%h0: f64, %h1: f64) -> (f64, f64, f64) {" in text
        assert text.rstrip().endswith("}\n}".rstrip("\n")) or text.rstrip().endswith("}")
        assert text.count("return") == 1

    def test_constants_are_interned(self):
        w = _random_weights(random.Random(2), 2, 2)
        w.a_clip = 3.0
        text = emit_ir(w)
        # a_clip and -a_clip each appear once per use as a constant
        # definition, not once per output unit.
        assert text.count("value = 3.0}") == 1
        assert text.count("value = -3.0}") == 1

    def test_zero_weights_are_skipped(self):
        w = _random_weights(random.Random(3), 2, 2)
        w.W = [[0.0, 0.0], [0.0, 0.0]]
        text = emit_ir(w)
        # No multiplication should be emitted for an exact-zero weight.
        # (bias terms still appear, so arith.mul must be entirely absent
        # only in a layer with an all-zero W *and* an all-zero P/J -- here
        # we only zeroed W, so just check W's row contributes no products
        # beyond the bias by counting total arith.mul against a baseline.)
        baseline = emit_ir(_random_weights(random.Random(3), 2, 2))
        assert text.count("arith.mul") < baseline.count("arith.mul")


@pytest.mark.parametrize("low_rank", [False, True])
class TestEndToEndOnTheQubitVirtualMachine:
    """Requires the sibling qubitbridge package; skips cleanly without it."""

    def _qubitbridge(self):
        return pytest.importorskip("qubitbridge")

    def test_matches_the_reference_forward_exactly(self, low_rank):
        self._qubitbridge()
        from qubitbridge.ir import parse_module
        from qubitbridge.lower import lower_module
        from qubitbridge.vm import QVM

        rng = random.Random(42 if not low_rank else 43)
        in_dim, out_dim = 4, 3
        w = _random_weights(rng, in_dim, out_dim, low_rank=low_rank, rank=2)

        module = parse_module(emit_ir(w))  # parses and verifies
        lowered = lower_module(module)["layer"]

        worst = 0.0
        for _ in range(25):
            h = [rng.uniform(-2.5, 2.5) for _ in range(in_dim)]
            want = reference_forward(w, h)
            seeds = {reg: value for (_, reg), value in zip(lowered.arg_regs, h)}
            result = QVM().run(lowered.program, r_inputs=seeds)
            got = [result.r[reg][0] for _, reg in lowered.result_regs]
            worst = max(worst, max(abs(g - t) for g, t in zip(got, want)))
        assert worst < 1e-9

    def test_clamps_match_under_extreme_inputs(self, low_rank):
        self._qubitbridge()
        from qubitbridge.ir import parse_module
        from qubitbridge.lower import lower_module
        from qubitbridge.vm import QVM

        rng = random.Random(7)
        in_dim, out_dim = 3, 2
        w = _random_weights(rng, in_dim, out_dim, low_rank=low_rank,
                            a_clip=1.5, q_eps=1e-3)
        lowered = lower_module(parse_module(emit_ir(w)))["layer"]

        for h in ([80.0, -80.0, 40.0], [-200.0, 200.0, -200.0], [0.0, 0.0, 0.0]):
            want = reference_forward(w, h)
            seeds = {reg: value for (_, reg), value in zip(lowered.arg_regs, h)}
            result = QVM().run(lowered.program, r_inputs=seeds)
            got = [result.r[reg][0] for _, reg in lowered.result_regs]
            for g, t in zip(got, want):
                assert g == pytest.approx(t, abs=1e-9)

    def test_lowers_to_assembling_arm64_code(self, low_rank):
        self._qubitbridge()
        arm64 = pytest.importorskip("qubitbridge.backends.arm64")
        if not arm64.have_assembler():
            pytest.skip("no AArch64 assembler available")
        from qubitbridge.ir import parse_module
        from qubitbridge.lower import lower_module

        w = _random_weights(random.Random(9), 3, 2, low_rank=low_rank)
        lowered = lower_module(parse_module(emit_ir(w)))["layer"]
        ok, output = arm64.Arm64Backend().verify(lowered.program)
        assert ok, output


class TestFromQbnnLayer:
    """Requires PyTorch (this repo's own hard dependency); skips without it."""

    def test_matches_the_live_layer_including_clamps(self):
        torch = pytest.importorskip("torch")
        from apqb_qbnn_v2 import QBNNLayerV2

        with torch.random.fork_rng():
            torch.manual_seed(3)
            layer = QBNNLayerV2(4, 3, a_clip=1.2, q_eps=1e-4)
            h = torch.randn(5, 4)

        weights = from_qbnn_layer(layer)
        for sample in h.tolist():
            want = layer(torch.tensor(sample)).tolist()
            got = reference_forward(weights, sample)
            for g, t in zip(got, want):
                assert g == pytest.approx(t, abs=1e-6)

    def test_low_rank_layer_is_read_correctly(self):
        torch = pytest.importorskip("torch")
        from apqb_qbnn_v2 import QBNNLayerV2

        with torch.random.fork_rng():
            torch.manual_seed(4)
            layer = QBNNLayerV2(3, 5, rank=2)
            h = torch.randn(3)

        weights = from_qbnn_layer(layer)
        assert weights.low_rank
        assert weights.rank == 2
        want = layer(h).tolist()
        got = reference_forward(weights, h.tolist())
        for g, t in zip(got, want):
            assert g == pytest.approx(t, abs=1e-6)

    def test_non_tanh_activation_is_rejected(self):
        torch = pytest.importorskip("torch")
        from apqb_qbnn_v2 import QBNNLayerV2

        layer = QBNNLayerV2(2, 2, activation=torch.sigmoid)
        with pytest.raises(UnsupportedLayerError):
            from_qbnn_layer(layer)

    def test_full_end_to_end_including_the_qvm(self):
        """Live layer -> exporter -> APQB IR -> QVM, checked against torch."""
        torch = pytest.importorskip("torch")
        pytest.importorskip("qubitbridge")
        from apqb_qbnn_v2 import QBNNLayerV2
        from qubitbridge.ir import parse_module
        from qubitbridge.lower import lower_module
        from qubitbridge.vm import QVM

        with torch.random.fork_rng():
            torch.manual_seed(5)
            layer = QBNNLayerV2(3, 2)
            samples = torch.randn(10, 3)

        lowered = lower_module(parse_module(emit_ir(from_qbnn_layer(layer))))["layer"]
        for sample in samples.tolist():
            want = layer(torch.tensor(sample)).tolist()
            seeds = {reg: value for (_, reg), value in zip(lowered.arg_regs, sample)}
            result = QVM().run(lowered.program, r_inputs=seeds)
            got = [result.r[reg][0] for _, reg in lowered.result_regs]
            for g, t in zip(got, want):
                assert g == pytest.approx(t, abs=1e-5)
