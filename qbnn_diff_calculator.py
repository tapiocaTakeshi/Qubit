#!/usr/bin/env python3
"""
微分可能な QBNN 計算機 (Differentiable untrained-QBNN calculator)

qbnn_calculator_benchmark.py の構成的 QBNN セル (学習なしで重みを解析的に
設定した QBNNLayerV2) は、すべて微分可能な演算 (線形層・tanh・APQB 相関
ゲート) だけでできている。したがって計算結果を入力で自動微分でき、
その勾配を使って「答えから入力を逆算する」= 方程式を解くことができる。

    calc = DifferentiableQBNNCalculator()
    calc.grad("x*x + 3*x", x=2.0)          # 値と df/dx (どちらも QBNN 経由)
    calc.solve("x*x + 3*x", 10, x=1.0)      # f(x) = 10 を Newton 法で解く
    calc.solve_system(["a*b", "a+b"], [12, 7], a=1.0, b=5.0)  # 連立方程式

使い方:
    python qbnn_diff_calculator.py            # 勾配精度・逆算・速度のテスト
    python qbnn_diff_calculator.py --quick
"""

import argparse
import ast
import time

import numpy as np
import torch

from qbnn_calculator_benchmark import DTYPE, ConstructedQBNNCalculator, timeit

_BIN_OPS = {ast.Add: "+", ast.Sub: "-", ast.Mult: "*", ast.Div: "/"}


class DifferentiableQBNNCalculator(ConstructedQBNNCalculator):
    """変数付きの式を QBNN セルで評価し、autograd で微分・逆算する計算機。

    注意: 乗算セルの右オペランドと除算の除数は |値| <= value_max が前提
    (APQB ゲートの tanh 飽和 / Newton 逆数の収束条件)。
    """

    def apply(self, op, a, b):
        """binary() と同じだが inference_mode を使わず、勾配が流れる。"""
        cell = {"+": self.add, "-": self.sub, "*": self.mul, "/": self.div}[op]
        a, b = torch.as_tensor(a, dtype=DTYPE), torch.as_tensor(b, dtype=DTYPE)
        a, b = torch.broadcast_tensors(a, b)
        return cell(a, b)

    def compile(self, expr):
        """式文字列 → (変数 dict を受け取り tensor を返す関数)."""
        tree = ast.parse(expr, mode="eval").body

        def ev(node, env):
            if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
                return torch.tensor(float(node.value), dtype=DTYPE)
            if isinstance(node, ast.Name):
                if node.id not in env:
                    raise NameError(f"variable '{node.id}' is not given")
                return env[node.id]
            if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
                v = ev(node.operand, env)
                return self.apply("-", 0.0, v) if isinstance(node.op, ast.USub) else v
            if isinstance(node, ast.BinOp) and type(node.op) in _BIN_OPS:
                return self.apply(_BIN_OPS[type(node.op)], ev(node.left, env), ev(node.right, env))
            raise ValueError(f"unsupported expression element: {ast.dump(node)}")

        return lambda env: ev(tree, env)

    @staticmethod
    def _vars(values):
        return {k: torch.as_tensor(v, dtype=DTYPE).clone().requires_grad_(True)
                for k, v in values.items()}

    def grad(self, expr, **values):
        """f(変数) とその勾配 ∂f/∂変数 を返す (すべて QBNN の順伝播 + 逆伝播)."""
        env = self._vars(values)
        out = self.compile(expr)(env)
        grads = torch.autograd.grad(out.sum(), list(env.values()))
        return out.detach(), {k: g for k, g in zip(env, grads)}

    def solve(self, expr, target, max_iter=50, tol=1e-12, **x0):
        """1 変数方程式 f(x) = target を Newton 法で解く。f'(x) は QBNN の勾配。

        x0 には配列も渡せる (複数の初期値・複数の target を並列に解く)。
        """
        (name, start), = x0.items()
        f = self.compile(expr)
        x = torch.as_tensor(start, dtype=DTYPE).clone()
        target = torch.as_tensor(target, dtype=DTYPE)
        for it in range(1, max_iter + 1):
            xv = x.clone().requires_grad_(True)
            res = f({name: xv}) - target
            (d,) = torch.autograd.grad(res.sum(), [xv])
            step = res.detach() / d
            x = x - step
            if step.abs().max() < tol * max(1.0, float(x.abs().max())):
                break
        return x, it

    def solve_system(self, exprs, targets, max_iter=50, tol=1e-12, **x0):
        """連立方程式 f_i(vars) = target_i を Newton 法 (ヤコビアンは QBNN の勾配) で解く。"""
        names = list(x0)
        fs = [self.compile(e) for e in exprs]
        x = torch.tensor([float(x0[n]) for n in names], dtype=DTYPE)
        t = torch.tensor([float(v) for v in targets], dtype=DTYPE)

        def F(vec):
            env = {n: vec[i] for i, n in enumerate(names)}
            return torch.stack([f(env) for f in fs]) - t

        for it in range(1, max_iter + 1):
            J = torch.autograd.functional.jacobian(F, x)
            step = torch.linalg.lstsq(J, F(x).detach().unsqueeze(-1)).solution.squeeze(-1)
            x = x - step
            if step.abs().max() < tol * max(1.0, float(x.abs().max())):
                break
        return dict(zip(names, x.tolist())), it


# ===========================================================================
# Tests / report
# ===========================================================================

def gradient_accuracy(calc, n, rng, hi):
    """各演算の ∂/∂a, ∂/∂b を解析解と比較 (最大相対誤差)."""
    a = rng.uniform(-hi, hi, n)
    b = rng.uniform(1, hi, n)
    exact = {
        "+": (np.ones(n), np.ones(n)),
        "-": (np.ones(n), -np.ones(n)),
        "*": (b, a),
        "/": (1 / b, -a / b ** 2),
    }
    rows = {}
    for op, (da_ref, db_ref) in exact.items():
        _, g = calc.grad(f"a {op} b", a=a, b=b)
        rel = lambda g_, ref: float(np.max(np.abs(g_.numpy() - ref) / np.maximum(np.abs(ref), 1e-12)))
        rows[op] = (rel(g["a"], da_ref), rel(g["b"], db_ref))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    rng = np.random.default_rng(0)
    n = 2000 if args.quick else 10000

    print(f"torch {torch.__version__}, threads={torch.get_num_threads()}, dtype=float64")

    print("\n### 1. 勾配の精度 (QBNN の自動微分 vs 解析解、最大相対誤差)\n")
    print("| 範囲 | 演算 | ∂f/∂a | ∂f/∂b |")
    print("|---|---|---|---|")
    for hi in (100.0, 1e4):
        calc = DifferentiableQBNNCalculator(value_max=hi)
        for op, (ea, eb) in gradient_accuracy(calc, n, rng, hi).items():
            print(f"| ±{hi:g} | `{op}` | {ea:.2e} | {eb:.2e} |")

    calc = DifferentiableQBNNCalculator(value_max=1e4)

    print("\n### 2. 式の微分\n")
    print("| 式 | 点 | 値 | QBNN 勾配 | 解析解 |")
    print("|---|---|---|---|---|")
    cases = [
        ("x*x*x - 2*x", {"x": 3.0}, {"x": 3 * 9 - 2}),
        ("(x+1)/(x-1)", {"x": 3.0}, {"x": -2 / 4}),
        ("a*b/(a+b)", {"a": 2.0, "b": 6.0}, {"a": 36 / 64, "b": 4 / 64}),
    ]
    for expr, pt, ref in cases:
        v, g = calc.grad(expr, **pt)
        pts = ", ".join(f"{k}={x:g}" for k, x in pt.items())
        gs = ", ".join(f"{k}: {float(g[k])!r}" for k in pt)
        rs = ", ".join(f"{k}: {r!r}" for k, r in ref.items())
        print(f"| `{expr}` | {pts} | {float(v)!r} | {gs} | {rs} |")

    print("\n### 3. 逆算 (方程式を解く) — Newton 法、勾配は QBNN から\n")
    print("| 問題 | 初期値 | QBNN 解 | 反復 | 真の解 |")
    print("|---|---|---|---|---|")
    problems = [
        ("x*x = 2  (√2)", "x*x", 2.0, 1.0, 2 ** 0.5),
        ("x*x + 3*x = 10", "x*x + 3*x", 10.0, 1.0, 2.0),
        ("x/(x+1) = 0.75", "x/(x+1)", 0.75, 1.0, 3.0),
        ("x*x*x = 1000", "x*x*x", 1000.0, 5.0, 10.0),
        ("1/x = 7  (逆数)", "1/x", 7.0, 0.1, 1 / 7),
    ]
    for label, expr, target, x0, true in problems:
        x, it = calc.solve(expr, target, x=x0)
        print(f"| {label} | {x0:g} | {float(x)!r} | {it} | {true!r} |")
    sol, it = calc.solve_system(["a*b", "a+b"], [12, 7], a=1.0, b=5.0)
    print(f"| a·b=12, a+b=7 | a=1, b=5 | a={sol['a']!r}, b={sol['b']!r} | {it} | a=3, b=4 |")
    sol, it = calc.solve_system(["x+y+z", "x*y - z", "x/y + z"], [6, -1, 3.5],
                                x=1.0, y=1.5, z=2.0)
    print(f"| x+y+z=6, xy-z=-1, x/y+z=3.5 | (1, 1.5, 2) | "
          + ", ".join(f"{k}={v!r}" for k, v in sol.items()) + f" | {it} | x=1, y=2, z=3 |")

    # 並列逆算: √k を k=1..N についてまとめて解く
    N = 1000 if args.quick else 10000
    ks = np.arange(1, N + 1, dtype=np.float64)
    t0 = time.perf_counter()
    x, it = calc.solve("x*x", ks, x=np.full(N, 50.0))
    el = time.perf_counter() - t0
    err = float(np.max(np.abs(x.numpy() - np.sqrt(ks)) / np.sqrt(ks)))
    print(f"\n並列逆算: √1..√{N} を一括で解く → 最大相対誤差 {err:.2e}, {it} 反復, "
          f"{el * 1e3:.1f} ms (1 問あたり {el / N * 1e6:.2f} µs)")

    print("\n### 4. 速度 (順伝播のみ vs 順伝播+逆伝播、1 演算あたり µs)\n")
    print("| 演算 | batch | 順伝播 | 順+逆伝播 | 比 |")
    print("|---|---|---|---|---|")
    for op in ("+", "*", "/"):
        for bs in (1, 1000, 100000) if not args.quick else (1, 1000):
            a = torch.tensor(rng.uniform(1, 9999, bs), dtype=DTYPE)
            b = torch.tensor(rng.uniform(1, 9999, bs), dtype=DTYPE)

            def fwd():
                with torch.no_grad():
                    calc.apply(op, a, b)

            def fwd_bwd():
                aa, bb = a.clone().requires_grad_(True), b.clone().requires_grad_(True)
                torch.autograd.grad(calc.apply(op, aa, bb).sum(), [aa, bb])

            tf, tb = timeit(fwd), timeit(fwd_bwd)
            print(f"| `{op}` | {bs} | {tf / bs * 1e6:.3g} | {tb / bs * 1e6:.3g} | {tb / tf:.1f}x |")


if __name__ == "__main__":
    main()
