#!/usr/bin/env python3
"""
未学習QBNN計算機ベンチマーク (Untrained-QBNN calculator benchmark)

勾配学習を一切行わない QBNN で四則演算がどこまでできるか、そして速度を測る。

3 つの方式を比較する:

  A. ランダム初期化 QBNN をそのまま使う (完全に未学習)
       - QBNNLayerV2 (apqb_qbnn_v2.py)  … J はゼロ初期化なので実質 tanh MLP
       - QBNNLayerV2 + ランダム J        … 相関ゲートを有効化
       - EQBNNLayer (qbnn_layered.py)    … 層間エンタングルメント付き
  B. リザバー方式: 上記のランダム QBNN を固定し、読み出し線形層だけを
     最小二乗 (閉形式・逆伝播なし) で解く
  C. 構成的 QBNN: QBNNLayerV2 の重みを解析的に設定して演算回路にする
       - 加減算: λ=0 の QBNN 層 (= 恒等活性の線形層)
       - 乗算:   APQB 相関ゲート  u * (1 + λ J^T tanh(P h)) を利用し、
                 tanh(εy)/ε ≈ y を Richardson 外挿で高精度化
       - 除算:   乗算セルを積んだ Newton-Raphson 逆数 x ← x(2 - d x)

使い方:
    python qbnn_calculator_benchmark.py            # 全テスト
    python qbnn_calculator_benchmark.py --quick    # 短縮版
    python qbnn_calculator_benchmark.py --expr "(12+7)*3-45/9"
"""

import argparse
import ast
import contextlib
import io
import json
import math
import operator
import time

import numpy as np
import torch
import torch.nn as nn

from apqb_qbnn_v2 import QBNNLayerV2

with contextlib.redirect_stdout(io.StringIO()):  # qbnn_layered prints a banner on import
    from qbnn_layered import EQBNNLayer

DTYPE = torch.float64
OPS = ["+", "-", "*", "/"]
REF = {"+": np.add, "-": np.subtract, "*": np.multiply, "/": np.divide}


def identity(x):
    return x


# ===========================================================================
# Datasets
# ===========================================================================

def make_operands(op, n, lo, hi, rng):
    a = rng.integers(lo, hi + 1, size=n).astype(np.float64)
    b_lo = max(lo, 1) if op == "/" else lo
    b = rng.integers(b_lo, hi + 1, size=n).astype(np.float64)
    return a, b


def score(op, pred, true):
    """+,-,* は整数への丸めで完全一致、/ は小数第2位まで一致 (|err|<0.005)。"""
    err = np.abs(pred - true)
    if op == "/":
        correct = err < 5e-3
    else:
        correct = np.round(pred) == true
    rel = err / np.maximum(np.abs(true), 1.0)
    return {
        "acc": float(correct.mean()),
        "mae": float(err.mean()),
        "max_abs_err": float(err.max()),
        "max_rel_err": float(rel.max()),
    }


# ===========================================================================
# A/B. Random (untrained) QBNN networks
# ===========================================================================

class RandomQBNNV2(nn.Module):
    """Stack of default-initialized QBNNLayerV2 + random linear readout."""

    def __init__(self, in_dim, hidden=256, depth=3, random_J=False, seed=0):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        torch.manual_seed(seed)
        dims = [in_dim] + [hidden] * depth
        self.layers = nn.ModuleList(QBNNLayerV2(dims[i], dims[i + 1]) for i in range(depth))
        if random_J:
            with torch.no_grad():
                for layer in self.layers:
                    d = layer.out_dim
                    layer.J_r.copy_(torch.randn(d, d, generator=g) / math.sqrt(d))
                    layer.J_q.copy_(torch.randn(d, d, generator=g) / math.sqrt(d))
                    layer.lambda_r.fill_(0.5)
                    layer.lambda_q.fill_(0.5)
        self.readout = nn.Linear(hidden, 1)

    def features(self, x):
        h = x
        for layer in self.layers:
            h = layer(h)
        return h

    def forward(self, x):
        return self.readout(self.features(x)).squeeze(-1)


class RandomEQBNN(nn.Module):
    """Stack of default-initialized EQBNNLayer (inter-layer entanglement)."""

    def __init__(self, in_dim, hidden=256, depth=3, seed=0):
        super().__init__()
        torch.manual_seed(seed)
        self.layers = nn.ModuleList()
        prev_q = in_dim
        dims = [in_dim] + [hidden] * depth
        for i in range(depth):
            self.layers.append(EQBNNLayer(dims[i], dims[i + 1], prev_q))
            prev_q = dims[i + 1]
        self.readout = nn.Linear(hidden, 1)

    def features(self, x):
        h, q = x, None
        for layer in self.layers:
            h, q = layer(h, q)
        return h

    def forward(self, x):
        return self.readout(self.features(x)).squeeze(-1)


def encode(op, a, b, scale):
    """入力 = [a/scale, b/scale, one-hot(op)]; 出力は y/out_scale で回帰。"""
    onehot = np.zeros((len(a), len(OPS)))
    onehot[:, OPS.index(op)] = 1.0
    x = np.column_stack([a / scale, b / scale, onehot])
    return torch.tensor(x, dtype=DTYPE)


OUT_SCALE = {"+": 200.0, "-": 100.0, "*": 10000.0, "/": 100.0}


def random_models(hidden):
    return {
        "QBNN-v2 (J=0 init)": RandomQBNNV2(6, hidden, random_J=False, seed=1).to(DTYPE),
        "QBNN-v2 (random J)": RandomQBNNV2(6, hidden, random_J=True, seed=2).to(DTYPE),
        "E-QBNN (entangled)": RandomEQBNN(6, hidden, seed=3).to(DTYPE),
    }


@torch.inference_mode()
def eval_untrained(models, n_test, rng, lo=0, hi=99):
    """方式 A: 未学習のまま出力をそのまま答えとみなす。"""
    results = {}
    for name, model in models.items():
        model.eval()
        results[name] = {}
        for op in OPS:
            a, b = make_operands(op, n_test, lo, hi, rng)
            pred = model(encode(op, a, b, hi)).numpy() * OUT_SCALE[op]
            results[name][op] = score(op, pred, REF[op](a, b))
    return results


@torch.inference_mode()
def eval_reservoir(models, n_fit, n_test, rng, lo=0, hi=99, ridge=1e-8):
    """方式 B: 隠れ層は未学習のまま固定し、読み出しのみ閉形式で解く。

    1 つの読み出しで 4 演算すべてを扱う (演算子は one-hot で入力)。
    検証は学習に使っていない新しいオペランドで行う。
    """
    results = {}
    for name, model in models.items():
        model.eval()
        feats, targets = [], []
        for op in OPS:
            a, b = make_operands(op, n_fit, lo, hi, rng)
            feats.append(model.features(encode(op, a, b, hi)))
            targets.append(torch.tensor(REF[op](a, b), dtype=DTYPE))
        H = torch.cat(feats)
        H = torch.cat([H, torch.ones(len(H), 1, dtype=DTYPE)], dim=1)
        y = torch.cat(targets)
        t0 = time.perf_counter()
        A = H.T @ H + ridge * torch.eye(H.shape[1], dtype=DTYPE)
        w = torch.linalg.solve(A, H.T @ y)
        solve_s = time.perf_counter() - t0
        results[name] = {"readout_solve_s": solve_s}
        for op in OPS:
            a, b = make_operands(op, n_test, lo, hi, rng)
            Ht = model.features(encode(op, a, b, hi))
            Ht = torch.cat([Ht, torch.ones(len(Ht), 1, dtype=DTYPE)], dim=1)
            pred = (Ht @ w).numpy()
            results[name][op] = score(op, pred, REF[op](a, b))
    return results


# ===========================================================================
# C. Constructed (analytically wired, never trained) QBNN calculator
# ===========================================================================

def _layer(in_dim, out_dim):
    layer = QBNNLayerV2(in_dim, out_dim, activation=identity).to(DTYPE)
    layer.requires_grad_(False)
    with torch.no_grad():
        for p in layer.parameters():
            p.zero_()
    return layer


class QBNNAddSub(nn.Module):
    """λ_r = λ_q = 0 の QBNN 層 = 恒等活性の線形層 (Sec. 6.1 (i))."""

    def __init__(self, sign):
        super().__init__()
        self.layer = _layer(2, 1)
        with torch.no_grad():
            self.layer.W.weight.copy_(torch.tensor([[1.0, float(sign)]], dtype=DTYPE))

    def forward(self, x, y):
        return self.layer(torch.stack([x, y], -1)).squeeze(-1)


class QBNNMul(nn.Module):
    """APQB 相関ゲートによる乗算セル (1 個の QBNNLayerV2 + 線形読み出し).

    ニューロン i (i=1..4):  u_i = x,  a_i = s_i ε y / y_max,
        out_i = x * (1 + λ_r K_i tanh(a_i))      (J_r = diag(K_i))
    (out_1 - out_2) / (2K)        = x tanh(εy') / ε  ≈ x y' (1 - ε²y'²/3)
    (out_3 - out_4) / (2K·2) [2ε] = 同じく 2ε 版
    Richardson 外挿 (4 f(ε) - f(2ε)) / 3 で ε² 項を消去 → 誤差 O(ε⁴)。
    y は |y| <= y_max を仮定 (tanh の飽和を避けるための入力正規化)。
    """

    def __init__(self, y_max=1.0, eps=1e-4, K=1e6):
        super().__init__()
        self.y_max = y_max
        self.eps = eps
        self.K = K
        self.layer = _layer(2, 4)
        e = eps / y_max
        with torch.no_grad():
            self.layer.W.weight[:, 0] = 1.0                      # u_i = x
            self.layer.P.weight[:, 1] = torch.tensor([e, -e, 2 * e, -2 * e], dtype=DTYPE)
            self.layer.J_r.copy_(torch.diag(torch.full((4,), K / eps, dtype=DTYPE)))
            self.layer.lambda_r.fill_(1.0)
            self.layer.a_clip = 4.0
        y_max_t = y_max / (2 * K)
        # 読み出し: y_max * [4/3 * (o1-o2)/(2K)  -  1/3 * (o3-o4)/(2K*2)]
        self.readout = torch.tensor(
            [4 / 3, -4 / 3, -1 / 6, 1 / 6], dtype=DTYPE) * y_max_t

    def forward(self, x, y):
        out = self.layer(torch.stack([x, y], -1))
        return out @ self.readout


class QBNNDiv(nn.Module):
    """Newton-Raphson 逆数 (乗算セルを 2n 段積む) + 最後の乗算.

    負の除数は tanh 符号ニューロン s = sign(d) で 1/d = s / |d| に帰着する。
    d' = |d| / d_max ∈ (0, 1],  r_0 = 1,  r_{k+1} = r_k (2 - d' r_k)
    → 1/d = r / d_max。 収束に必要な段数 n ≈ log2(40 d_max / d_min).
    """

    def __init__(self, d_max, a_max, d_min=1e-3, iters=None):
        super().__init__()
        self.d_max = float(d_max)
        # (1 - d_min/d_max)^(2^n) < 1e-16 となる段数 (d_min <= |d| <= d_max で収束)
        self.iters = iters or int(math.ceil(math.log2(40 * d_max / d_min))) + 1
        # 符号ニューロン: s = tanh(K d) = ±1 (|d| >= 1e-9 で厳密に ±1、勾配 0)
        self.sign = QBNNLayerV2(1, 1, activation=torch.tanh).to(DTYPE)
        self.sign.requires_grad_(False)
        with torch.no_grad():
            for p in self.sign.parameters():
                p.zero_()
            self.sign.W.weight.fill_(1e12)
        self.mul_s = QBNNMul(y_max=1.0)          # |d| = d * s,  r * s
        self.mul_d = QBNNMul(y_max=1.0)          # d' * r   (d' <= 1)
        self.mul_r = QBNNMul(y_max=2.0)          # r * (2 - d' r)  (0 < . <= 2)
        self.mul_a = QBNNMul(y_max=float(a_max))  # (r/d_max) * a
        self.sub = QBNNAddSub(-1)

    def forward(self, a, d):
        s = self.sign(d.unsqueeze(-1)).squeeze(-1)
        ds = self.mul_s(d, s) / self.d_max       # |d| / d_max ∈ (0, 1]
        r = torch.ones_like(d)
        two = torch.full_like(d, 2.0)
        for _ in range(self.iters):
            r = self.mul_r(r, self.sub(two, self.mul_d(r, ds)))
        return self.mul_a(self.mul_s(r, s) / self.d_max, a)

    @property
    def n_layers(self):
        return 3 * self.iters + 4


class ConstructedQBNNCalculator:
    """学習なしで配線した QBNN セルで四則演算 / 式評価を行う計算機。"""

    def __init__(self, value_max=1e6):
        self.value_max = value_max
        self.add = QBNNAddSub(+1)
        self.sub = QBNNAddSub(-1)
        self.mul = QBNNMul(y_max=value_max)
        self.div = QBNNDiv(d_max=value_max, a_max=value_max)

    @torch.inference_mode()
    def binary(self, op, a, b):
        cell = {"+": self.add, "-": self.sub, "*": self.mul, "/": self.div}[op]
        return cell(torch.as_tensor(a, dtype=DTYPE), torch.as_tensor(b, dtype=DTYPE))

    def evaluate(self, expr):
        """四則演算の式文字列を QBNN セルだけで評価する。"""
        bin_ops = {ast.Add: "+", ast.Sub: "-", ast.Mult: "*", ast.Div: "/"}

        def ev(node):
            if isinstance(node, ast.Expression):
                return ev(node.body)
            if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
                return torch.tensor(float(node.value), dtype=DTYPE)
            if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
                v = ev(node.operand)
                return self.binary("-", 0.0, v) if isinstance(node.op, ast.USub) else v
            if isinstance(node, ast.BinOp) and type(node.op) in bin_ops:
                return self.binary(bin_ops[type(node.op)], ev(node.left), ev(node.right))
            raise ValueError(f"unsupported expression element: {ast.dump(node)}")

        return float(ev(ast.parse(expr, mode="eval")))


def eval_constructed(n_test, rng, ranges):
    results = {}
    for lo, hi in ranges:
        calc = ConstructedQBNNCalculator(value_max=float(hi))
        key = f"[{lo}, {hi}]"
        results[key] = {}
        for op in OPS:
            a, b = make_operands(op, n_test, lo, hi, rng)
            pred = calc.binary(op, a, b).numpy()
            results[key][op] = score(op, pred, REF[op](a, b))
        results[key]["div_layers"] = calc.div.n_layers
    return results


# ===========================================================================
# Speed
# ===========================================================================

def timeit(fn, min_time=0.3, max_reps=10000):
    fn()  # warm-up
    reps, t0 = 0, time.perf_counter()
    while True:
        fn()
        reps += 1
        el = time.perf_counter() - t0
        if el >= min_time or reps >= max_reps:
            return el / reps


def speed_benchmark(batches, hidden, rng):
    calc = ConstructedQBNNCalculator(value_max=1e4)
    rand_v2 = RandomQBNNV2(6, hidden, random_J=True, seed=2).to(DTYPE).eval()
    eqbnn = RandomEQBNN(6, hidden, seed=3).to(DTYPE).eval()
    rows = []
    for n in batches:
        for op in OPS:
            a, b = make_operands(op, n, 1, 9999, rng)
            ta, tb = torch.tensor(a, dtype=DTYPE), torch.tensor(b, dtype=DTYPE)
            x = encode(op, a, b, 9999)
            al, bl = a.tolist(), b.tolist()
            pyop = {"+": operator.add, "-": operator.sub,
                    "*": operator.mul, "/": operator.truediv}[op]
            with torch.inference_mode():
                t = {
                    "Python float (loop)": timeit(lambda: [pyop(p, q) for p, q in zip(al, bl)]),
                    "NumPy (vectorized)": timeit(lambda: REF[op](a, b)),
                    "Constructed QBNN": timeit(lambda: calc.binary(op, ta, tb)),
                    f"Random QBNN-v2 h={hidden}": timeit(lambda: rand_v2(x)),
                    f"Random E-QBNN h={hidden}": timeit(lambda: eqbnn(x)),
                }
            for method, sec in t.items():
                rows.append({"batch": n, "op": op, "method": method,
                             "sec_per_call": sec, "ops_per_sec": n / sec,
                             "us_per_op": sec / n * 1e6})
    return rows


# ===========================================================================
# Reporting
# ===========================================================================

def print_acc_table(title, results):
    print(f"\n### {title}\n")
    print("| モデル | " + " | ".join(f"`{o}` 正答率" for o in OPS) + " | `*` 最大相対誤差 | `/` 最大相対誤差 |")
    print("|---|" + "---|" * (len(OPS) + 2))
    for name, r in results.items():
        accs = " | ".join(f"{r[o]['acc'] * 100:.1f}%" for o in OPS)
        print(f"| {name} | {accs} | {r['*']['max_rel_err']:.2e} | {r['/']['max_rel_err']:.2e} |")


def print_speed_table(rows):
    print("\n### 速度 (CPU, float64, 1 演算あたり µs / 毎秒演算数)\n")
    batches = sorted({r["batch"] for r in rows})
    methods = list(dict.fromkeys(r["method"] for r in rows))
    for op in OPS:
        print(f"\n**演算 `{op}`**\n")
        print("| 方式 | " + " | ".join(f"batch={b}" for b in batches) + " |")
        print("|---|" + "---|" * len(batches))
        for m in methods:
            cells = []
            for b in batches:
                r = next(x for x in rows if x["op"] == op and x["method"] == m and x["batch"] == b)
                cells.append(f"{r['us_per_op']:.3g} µs ({r['ops_per_sec']:.3g}/s)")
            print(f"| {m} | " + " | ".join(cells) + " |")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--expr", type=str, default=None)
    ap.add_argument("--json", type=str, default=None, help="save raw results")
    ap.add_argument("--threads", type=int, default=None)
    args = ap.parse_args()

    if args.threads:
        torch.set_num_threads(args.threads)

    if args.expr:
        calc = ConstructedQBNNCalculator()
        val = calc.evaluate(args.expr)
        print(f"{args.expr} = {val!r}   (Python: {eval(args.expr, {'__builtins__': {}})!r})")
        return

    rng = np.random.default_rng(0)
    n_test = 2000 if args.quick else 10000
    hidden = 256

    print(f"torch {torch.__version__}, threads={torch.get_num_threads()}, dtype=float64")

    models = random_models(hidden)
    res_a = eval_untrained(models, n_test, rng)
    print_acc_table("A. ランダム初期化 QBNN (未学習・そのまま) — オペランド 0..99", res_a)

    res_b = eval_reservoir(models, 2000, n_test, rng)
    print_acc_table("B. リザバー方式 (隠れ層未学習 + 閉形式読み出し) — 0..99", res_b)

    ranges = [(0, 99), (0, 9999), (0, 999999)]
    res_c = eval_constructed(n_test, rng, ranges)
    print_acc_table("C. 構成的 QBNN 計算機 (重みを解析的に設定・学習なし)",
                    {k: v for k, v in res_c.items()})
    for k, v in res_c.items():
        print(f"- 範囲 {k}: 除算の QBNN 層数 = {v['div_layers']}")

    calc = ConstructedQBNNCalculator()
    exprs = ["(12+7)*3-45/9", "123456*789", "1/3", "-(8-15)*(2.5+0.5)/7", "99999/7*7"]
    print("\n### 式評価 (構成的 QBNN)\n")
    print("| 式 | QBNN | Python | 絶対誤差 |")
    print("|---|---|---|---|")
    expr_rows = []
    for e in exprs:
        v, ref = calc.evaluate(e), eval(e, {"__builtins__": {}})
        expr_rows.append({"expr": e, "qbnn": v, "python": ref})
        print(f"| `{e}` | {v!r} | {ref!r} | {abs(v - ref):.2e} |")

    batches = [1, 1000, 100000] if not args.quick else [1, 1000]
    rows = speed_benchmark(batches, hidden, rng)
    print_speed_table(rows)

    if args.json:
        with open(args.json, "w") as f:
            json.dump({"untrained": res_a, "reservoir": res_b, "constructed": res_c,
                       "expressions": expr_rows, "speed": rows}, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
