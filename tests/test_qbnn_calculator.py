"""Tests for the constructed (untrained) QBNN calculator and its gradients."""

import numpy as np
import pytest
import torch

from qbnn_diff_calculator import DifferentiableQBNNCalculator


@pytest.fixture(scope="module")
def calc():
    return DifferentiableQBNNCalculator(value_max=1e4)


@pytest.mark.parametrize("op", ["+", "-", "*", "/"])
def test_binary_ops_match_float64(calc, op):
    rng = np.random.default_rng(0)
    a = rng.uniform(-1e4, 1e4, 500)
    b = rng.uniform(1e-3, 1e4, 500) * rng.choice([-1, 1], 500)
    ref = {"+": a + b, "-": a - b, "*": a * b, "/": a / b}[op]
    out = calc.binary(op, a, b).numpy()
    np.testing.assert_allclose(out, ref, rtol=1e-12, atol=1e-12)


def test_expression_evaluation(calc):
    assert calc.evaluate("123456*789") == 97406784.0
    assert calc.evaluate("(12+7)*3-45/9") == pytest.approx(52.0, rel=1e-14)
    assert calc.evaluate("-(8-15)*(2.5+0.5)/7") == pytest.approx(3.0, rel=1e-14)


def test_gradients_match_analytic(calc):
    a, b = np.array([3.0, -40.0, 0.5]), np.array([7.0, 2.5, -9.0])
    _, g = calc.grad("a * b", a=a, b=b)
    np.testing.assert_allclose(g["a"].numpy(), b, rtol=1e-13)
    np.testing.assert_allclose(g["b"].numpy(), a, rtol=1e-13)
    _, g = calc.grad("a / b", a=a, b=b)
    np.testing.assert_allclose(g["a"].numpy(), 1 / b, rtol=1e-13)
    np.testing.assert_allclose(g["b"].numpy(), -a / b ** 2, rtol=1e-13)


@pytest.mark.parametrize("expr,target,x0,true", [
    ("x*x", 2.0, 1.0, 2 ** 0.5),
    ("x*x + 3*x", 10.0, 1.0, 2.0),
    ("x/(x+1)", 0.75, 1.0, 3.0),
    ("1/x", 7.0, 0.1, 1 / 7),
])
def test_solve_inverts_expression(calc, expr, target, x0, true):
    x, _ = calc.solve(expr, target, x=x0)
    assert float(x) == pytest.approx(true, rel=1e-12)


def test_solve_system(calc):
    sol, _ = calc.solve_system(["a*b", "a+b"], [12, 7], a=1.0, b=5.0)
    assert sol["a"] == pytest.approx(3.0, rel=1e-12)
    assert sol["b"] == pytest.approx(4.0, rel=1e-12)


def test_batched_solve(calc):
    ks = np.arange(1, 101, dtype=np.float64)
    x, _ = calc.solve("x*x", ks, x=np.full(100, 50.0))
    np.testing.assert_allclose(x.numpy(), np.sqrt(ks), rtol=1e-13)
