#!/usr/bin/env python3
"""Export a QBNNLayerV2 (apqb_qbnn_v2.py) to APQB IR text for QubitBridge.

QubitBridge (github.com/tapiocaTakeshi/QubitBridge) is the systems half of
this project: an APQB IR -> Qubit ISA -> {portable, numpy, AArch64/NEON}
toolchain. This module is the bridge from the *research* formalisation here
to that toolchain -- it turns a trained ``QBNNLayerV2`` into the textual APQB
IR that ``qubitbridge.parse_module`` accepts, so a layer's inference can run
on the Qubit Virtual Machine (and, from there, be lowered to AArch64/NEON)
instead of PyTorch.

The exported program is not an approximation of Eq. (20)-(23): it reproduces
QBNNLayerV2.forward exactly, including the ``a_clip`` clamp on the latent
coordinate and the ``q_eps`` floor on the coherence coordinate, and the
low-rank ``J = U V^T`` factorisation of Sec. 6.5 when the layer uses one.

    u      = W h + b
    a      = clamp(P h + c, -a_clip, a_clip)
    r, q   = tanh(a), max(sech(a), q_eps)
    c_r    = J_r^T r ,  c_q = J_q^T q          (or the low-rank U/V form)
    h_next = tanh( u . (1 + lambda_r*c_r + lambda_q*c_q) )

One IR op does double duty here: ``apqb.encode {mode = "latent"}`` computes
``tanh(a)`` and ``sech(a)`` together as the ``r``/``eta`` of a single pseudo
qubit state (Eq. 12) -- which is not merely convenient but *necessary* here,
because the classical arithmetic ops (add/sub/mul/div/neg/tanh/atanh/min/max)
have no ``sech`` or ``exp`` primitive of their own. Every other operation
(the affine paths, the gate, the clamps) stays on the classical side, exactly
as QBNNLayerV2 computes it -- this exporter does not invent an APQB encoding
for values the paper's own model treats classically.

This module has no hard dependency on either PyTorch or QubitBridge:

* :func:`from_qbnn_layer` imports ``torch`` lazily, only when called, so the
  exporter and its pure-Python reference forward pass are testable without it.
* :func:`emit_ir` is a self-contained text generator; it does not import
  ``qubitbridge``. Its output is plain text meant to be handed to
  ``qubitbridge.parse_module`` / ``qvm compile`` by whoever consumes it.

Only ``activation=torch.tanh`` (the module's default) is supported, since
that is the only nonlinearity the APQB IR's ``arith.tanh`` op provides; a
layer built with a different activation is rejected with a clear error
rather than silently exported wrong.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

__all__ = [
    "LayerWeights",
    "from_qbnn_layer",
    "emit_ir",
    "reference_forward",
    "UnsupportedLayerError",
]


class UnsupportedLayerError(ValueError):
    """The layer uses a feature this exporter does not (yet) translate."""


@dataclass
class LayerWeights:
    """Plain-Python snapshot of a QBNNLayerV2's parameters.

    Shapes follow ``nn.Linear``'s own convention: ``W`` and ``P`` are
    ``[out_dim][in_dim]``, everything else that multiplies ``r``/``q``
    (``[out_dim]``-shaped) is ``[out_dim][out_dim]`` (full rank) or
    ``[out_dim][rank]`` (``U``, low rank) / ``[out_dim][rank]`` (``V``).
    """

    in_dim: int
    out_dim: int
    W: list[list[float]]
    b: list[float]
    P: list[list[float]]
    c: list[float]
    lambda_r: float
    lambda_q: float
    a_clip: float = 4.0
    q_eps: float = 1e-6
    low_rank: bool = False
    # Full-rank coupling (low_rank=False).
    J_r: list[list[float]] | None = None
    J_q: list[list[float]] | None = None
    # Low-rank coupling J = V @ U^T (low_rank=True); rank = len(U[0]).
    U_r: list[list[float]] | None = None
    V_r: list[list[float]] | None = None
    U_q: list[list[float]] | None = None
    V_q: list[list[float]] | None = None

    def __post_init__(self) -> None:
        if self.low_rank:
            missing = [name for name in ("U_r", "V_r", "U_q", "V_q")
                      if getattr(self, name) is None]
            if missing:
                raise ValueError(f"low_rank=True needs {', '.join(missing)}")
        else:
            missing = [name for name in ("J_r", "J_q")
                      if getattr(self, name) is None]
            if missing:
                raise ValueError(f"low_rank=False needs {', '.join(missing)}")

    @property
    def rank(self) -> int:
        if not self.low_rank:
            raise ValueError("rank is only defined for a low-rank layer")
        return len(self.U_r[0]) if self.U_r else 0


def from_qbnn_layer(layer, *, activation: str = "tanh") -> LayerWeights:
    """Snapshot a live ``QBNNLayerV2`` / ``StochasticQBNNLayerV2`` instance.

    Only ``activation="tanh"`` is supported (the module's default and the
    only nonlinearity ``apqb.encode``'s ``arith.tanh`` companion provides).
    """
    if activation != "tanh":
        raise UnsupportedLayerError(
            f"only activation='tanh' is exportable, got {activation!r}")
    if getattr(layer, "activation", None) is not None:
        import torch
        if layer.activation is not torch.tanh:
            raise UnsupportedLayerError(
                "the layer's activation is not torch.tanh; the APQB IR's "
                "arith.tanh is the only nonlinearity this exporter emits")

    def rows(linear) -> list[list[float]]:
        return linear.weight.detach().cpu().tolist()

    def vec(tensor) -> list[float]:
        return tensor.detach().cpu().tolist()

    kwargs = dict(
        in_dim=layer.in_dim, out_dim=layer.out_dim,
        W=rows(layer.W), b=vec(layer.W.bias), P=rows(layer.P), c=vec(layer.P.bias),
        lambda_r=float(layer.lambda_r.item()), lambda_q=float(layer.lambda_q.item()),
        a_clip=float(layer.a_clip), q_eps=float(layer.q_eps),
        low_rank=bool(layer.low_rank),
    )
    if layer.low_rank:
        kwargs.update(U_r=vec(layer.U_r), V_r=vec(layer.V_r),
                      U_q=vec(layer.U_q), V_q=vec(layer.V_q))
    else:
        kwargs.update(J_r=vec(layer.J_r), J_q=vec(layer.J_q))
    return LayerWeights(**kwargs)


# ===========================================================================
# Pure-Python reference forward pass -- the same math QBNNLayerV2.forward
# computes, usable as a test oracle without PyTorch installed.
# ===========================================================================

def _sech(a: float) -> float:
    """sech(a) without overflowing cosh(a), matching APQBv2._sech."""
    m = abs(a)
    e = math.exp(-m)
    return 2.0 * e / (1.0 + e * e)


def reference_forward(w: LayerWeights, h: list[float]) -> list[float]:
    """``QBNNLayerV2.forward`` on a single sample, in plain Python."""
    if len(h) != w.in_dim:
        raise ValueError(f"expected {w.in_dim} inputs, got {len(h)}")

    u = [w.b[j] + sum(w.W[j][i] * h[i] for i in range(w.in_dim))
         for j in range(w.out_dim)]

    a = [w.c[j] + sum(w.P[j][i] * h[i] for i in range(w.in_dim))
         for j in range(w.out_dim)]
    a = [max(-w.a_clip, min(w.a_clip, aj)) for aj in a]
    r = [math.tanh(aj) for aj in a]
    q = [max(w.q_eps, _sech(aj)) for aj in a]

    if w.low_rank:
        rank = w.rank
        t_r = [sum(r[i] * w.V_r[i][k] for i in range(w.out_dim))
              for k in range(rank)]
        t_q = [sum(q[i] * w.V_q[i][k] for i in range(w.out_dim))
              for k in range(rank)]
        c_r = [sum(t_r[k] * w.U_r[j][k] for k in range(rank))
              for j in range(w.out_dim)]
        c_q = [sum(t_q[k] * w.U_q[j][k] for k in range(rank))
              for j in range(w.out_dim)]
    else:
        c_r = [sum(r[i] * w.J_r[i][j] for i in range(w.out_dim))
              for j in range(w.out_dim)]
        c_q = [sum(q[i] * w.J_q[i][j] for i in range(w.out_dim))
              for j in range(w.out_dim)]

    gate = [1.0 + w.lambda_r * c_r[j] + w.lambda_q * c_q[j]
           for j in range(w.out_dim)]
    return [math.tanh(u[j] * gate[j]) for j in range(w.out_dim)]


# ===========================================================================
# APQB IR text generation
# ===========================================================================

class _IRFunc:
    """A minimal SSA text builder for one APQB IR function body.

    This deliberately does not import ``qubitbridge``: it is a plain string
    generator whose output follows the grammar in QubitBridge's
    ``docs/apqb-ir.md`` exactly, so it is decoupled from that package's
    internals and only coupled to its documented, versioned text format.
    """

    def __init__(self):
        self.args: list[str] = []
        self.lines: list[str] = []
        self._n = 0
        self._const_cache: dict[str, str] = {}

    def _fresh(self, hint: str) -> str:
        self._n += 1
        return f"%{hint}{self._n}"

    def arg(self, name: str) -> str:
        ref = f"%{name}"
        self.args.append(f"{ref}: f64")
        return ref

    def const(self, value: float) -> str:
        # repr() round-trips exactly, and interning keeps a literal like
        # a_clip or 0.0 from being re-materialised at every use site.
        key = repr(float(value))
        if key not in self._const_cache:
            ref = self._fresh("k")
            self.lines.append(f'{ref} = arith.const {{value = {key}}} : f64')
            self._const_cache[key] = ref
        return self._const_cache[key]

    def binop(self, op: str, a: str, b: str) -> str:
        ref = self._fresh("v")
        self.lines.append(f"{ref} = arith.{op} {a}, {b} : f64")
        return ref

    def add(self, a: str, b: str) -> str: return self.binop("add", a, b)
    def mul(self, a: str, b: str) -> str: return self.binop("mul", a, b)
    def min_(self, a: str, b: str) -> str: return self.binop("min", a, b)
    def max_(self, a: str, b: str) -> str: return self.binop("max", a, b)

    def tanh(self, a: str) -> str:
        ref = self._fresh("v")
        self.lines.append(f"{ref} = arith.tanh {a} : f64")
        return ref

    def encode_latent(self, a: str) -> str:
        ref = self._fresh("q")
        self.lines.append(f'{ref} = apqb.encode {a} {{mode = "latent"}} : !apqb.state')
        return ref

    def decode_linear(self, q: str) -> str:
        ref = self._fresh("v")
        self.lines.append(f'{ref} = apqb.decode {q} {{mode = "linear"}} : f64')
        return ref

    def uncertainty(self, q: str) -> str:
        ref = self._fresh("v")
        self.lines.append(f"{ref} = apqb.uncertainty {q} : f64")
        return ref

    def sum_(self, terms: list[str]) -> str:
        """Left fold add over a nonempty list of value refs."""
        acc = terms[0]
        for term in terms[1:]:
            acc = self.add(acc, term)
        return acc

    def dot_const(self, weights: list[float], values: list[str],
                 bias: float | None = None) -> str:
        """``bias + sum(weight_i * value_i)``, skipping exact-zero weights."""
        terms = [self.const(bias)] if bias is not None else []
        for weight, value in zip(weights, values):
            if weight == 0.0:
                continue
            terms.append(self.mul(self.const(weight), value))
        return self.sum_(terms) if terms else self.const(0.0)

    def render(self, module_name: str, func_name: str, results: list[str]) -> str:
        header = f"  func @{func_name}({', '.join(self.args)}) -> ({', '.join(['f64'] * len(results))}) {{"
        body = "\n".join(f"    {line}" for line in self.lines)
        ret = f"    return {', '.join(results)}"
        return f"module @{module_name} {{\n{header}\n{body}\n{ret}\n  }}\n}}\n"


def emit_ir(w: LayerWeights, *, module_name: str = "qbnn",
           func_name: str = "layer") -> str:
    """Emit APQB IR text for one QBNNLayerV2 forward pass.

    The result parses and verifies under ``qubitbridge.parse_module`` and
    lowers to a runnable QVM program under ``qubitbridge.lower_module``.
    """
    fn = _IRFunc()
    h = [fn.arg(f"h{i}") for i in range(w.in_dim)]

    # u = W h + b  (classical affine path, Eq. 20).
    u = [fn.dot_const(w.W[j], h, bias=w.b[j]) for j in range(w.out_dim)]

    # a = clamp(P h + c, -a_clip, a_clip); (r, q) from one apqb.encode each.
    a_clip = fn.const(w.a_clip)
    neg_a_clip = fn.const(-w.a_clip)
    q_eps = fn.const(w.q_eps)
    r, q = [], []
    for j in range(w.out_dim):
        raw = fn.dot_const(w.P[j], h, bias=w.c[j])
        clamped = fn.min_(fn.max_(raw, neg_a_clip), a_clip)
        state = fn.encode_latent(clamped)
        r.append(fn.decode_linear(state))
        q.append(fn.max_(fn.uncertainty(state), q_eps))

    # c_r, c_q: the QBNN coupling, Eq. 21-22 (full rank or J = V U^T).
    if w.low_rank:
        rank = w.rank
        t_r = [fn.dot_const([w.V_r[i][k] for i in range(w.out_dim)], r)
              for k in range(rank)]
        t_q = [fn.dot_const([w.V_q[i][k] for i in range(w.out_dim)], q)
              for k in range(rank)]
        c_r = [fn.dot_const(w.U_r[j], t_r) for j in range(w.out_dim)]
        c_q = [fn.dot_const(w.U_q[j], t_q) for j in range(w.out_dim)]
    else:
        c_r = [fn.dot_const([w.J_r[i][j] for i in range(w.out_dim)], r)
              for j in range(w.out_dim)]
        c_q = [fn.dot_const([w.J_q[i][j] for i in range(w.out_dim)], q)
              for j in range(w.out_dim)]

    # gate = 1 + lambda_r*c_r + lambda_q*c_q; h_next = tanh(u * gate).
    lambda_r, lambda_q, one = fn.const(w.lambda_r), fn.const(w.lambda_q), fn.const(1.0)
    outputs = []
    for j in range(w.out_dim):
        gate = fn.add(one, fn.add(fn.mul(lambda_r, c_r[j]), fn.mul(lambda_q, c_q[j])))
        outputs.append(fn.tanh(fn.mul(u[j], gate)))

    return fn.render(module_name, func_name, outputs)


def main() -> None:  # pragma: no cover - a runnable, dependency-free demo
    """Build a tiny random layer with no PyTorch, and print its IR."""
    import random
    rng = random.Random(0)
    in_dim, out_dim = 3, 2

    def matrix(rows, cols, scale=0.5):
        return [[rng.uniform(-scale, scale) for _ in range(cols)] for _ in range(rows)]

    weights = LayerWeights(
        in_dim=in_dim, out_dim=out_dim,
        W=matrix(out_dim, in_dim), b=[rng.uniform(-0.1, 0.1) for _ in range(out_dim)],
        P=matrix(out_dim, in_dim), c=[rng.uniform(-0.1, 0.1) for _ in range(out_dim)],
        J_r=matrix(out_dim, out_dim, 0.3), J_q=matrix(out_dim, out_dim, 0.3),
        lambda_r=0.1, lambda_q=0.1,
    )
    print(emit_ir(weights))

    h = [0.4, -0.9, 0.2]
    print(f"reference_forward({h}) = {reference_forward(weights, h)}")


if __name__ == "__main__":  # pragma: no cover
    main()
