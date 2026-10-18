#!/usr/bin/env python3
"""
完全ブロッホ球 (Complete Bloch Sphere) と、それをニューロンとする完全QBNN.

完全ブロッホ球は、二葉双曲面と2つの半球体の組である。

    H = {q = (u, v, zeta) in R^3 : zeta^2 - u^2 - v^2 = 1}
      = H+ (zeta >= 1, 上の葉)  U  H- (zeta <= -1, 下の葉)

    P(q) = (u, v, 1) / zeta                                         (CB1)

P は上の葉 H+ を北半球 (z > 0, |0> 側) に、下の葉 H- を南半球 (z < 0, |1> 側)
に移す. 頂点 (0, 0, +-1) は両極 |0>, |1> に、葉の無限遠は赤道 (n = 2) に対応し,
q = (u, v, zeta) と同じ (u, v) を持つ反対の葉の点 q~ = (u, v, -zeta) は対蹠点
P(q~) = -P(q) に写る (下の葉は上の葉の対蹠像). 葉の上の測地座標
(u, v) = sinh(rho) (cos a, sin a), zeta = +-cosh(rho) では

    b = +-(tanh(rho) cos a, tanh(rho) sin a, sech(rho)),              (CB2)

すなわち η = tanh(rho), |1 - n/2| = sech(rho) となる. 双曲距離 rho は極から
の「コヒーレンスの深さ」であり, 平方根振幅論文の潜在変数 s (z = tanh s) と
sinh(s) sinh(rho) = 1 で結ばれる (CB3).

各ニューロンの状態は葉の上の点 q と偏極 m = tanh(c) in (-1, 1) で,

    b = m P(q),   rho = (1 + m)/2 |P(q)><P(q)| + (1 - m)/2 |P(q~)><P(q~)|  (CB4)

m > 0 なら q の葉の半球体, m < 0 なら反対側の葉の半球体に入り, 2つの半球体で
ブロッホ球体全体 (任意の1量子ビット密度行列, 赤道面は極限として) を覆う. 学習座標で葉を
またぐ連続な経路は球の中心 (最大混合状態) か赤道 (無限遠) を通る. 葉軸 P(q) で
測ると Born 則で +-1 が確率 (1 +- m)/2 で出て, 状態は P(q) か P(q~) の純粋状態, つまり
どちらかの葉に収縮する (CB5). 逆写像は b -> (u, v) = (b_x, b_y)/b_z (極からの
中心射影), m = sign(b_z) |b| である (CB6). 同じ葉の2点の双曲距離は

    cosh d(q1, q2) = 1 + (1 - b1.b2) / (z1 z2)                       (CB7)

で, 純粋状態の忠実度 F = (1 + b1.b2)/2 を用いれば 1 + 2(1 - F)/(z1 z2).

これは単一量子ビットの標準的な量子情報の再表現であり, ネットワークのニューロ
ンは古典計算機上の状態座標である. 量子優位性や高速化は主張しない. 式の検証は
tests/test_complete_qbnn.py にある.

使い方:
    python complete_qbnn.py classify-demo
    python complete_qbnn.py train --text README.md docs/*.md --out complete_qbnn_lm.pt
    python complete_qbnn.py generate --ckpt complete_qbnn_lm.pt --prompt "量子" --measure 1.0
"""

import argparse
import math
from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

import apqb_sqrt_amplitude as sa


# ---------------------------------------------------------------------------
# 完全ブロッホ球の幾何 (CB1-CB7)
# ---------------------------------------------------------------------------

def hyperboloid_point(u, v, sheet=1.0):
    """(u, v) を葉 sheet = +-1 に持ち上げた q = (u, v, zeta), zeta^2 - u^2 - v^2 = 1."""
    zeta = torch.sqrt(1 + u * u + v * v)
    return torch.stack(torch.broadcast_tensors(u, v, sheet * zeta), dim=-1)


def minkowski_norm_sq(q):
    """zeta^2 - u^2 - v^2 (双曲面上では 1)."""
    return q[..., 2] ** 2 - q[..., 0] ** 2 - q[..., 1] ** 2


def sheet_to_sphere(q):
    """(CB1) P(q) = (u, v, 1) / zeta: 上の葉 -> 北半球, 下の葉 -> 南半球."""
    one = torch.ones_like(q[..., 2])
    return torch.stack([q[..., 0], q[..., 1], one], dim=-1) / q[..., 2:3]


def geodesic_coords(rho, a, sheet=1.0):
    """(CB2) 葉の測地座標 (rho, a) -> q. rho は頂点 (極) からの双曲距離."""
    s = torch.sinh(rho)
    return torch.stack(torch.broadcast_tensors(s * torch.cos(a), s * torch.sin(a),
                                               sheet * torch.cosh(rho)), dim=-1)


def complete_bloch(c, u, v):
    """(CB4) ニューロンの学習座標 (c, u, v) in R^3 -> ブロッホベクトル b = tanh(c) P(q).

    q = (u, v, sqrt(1 + u^2 + v^2)) は上の葉の点; tanh(c) < 0 は下の葉の q~ = (u, v, -zeta)
    の側 (南の半球体) を選ぶ.
    |b| = |tanh c| <= 1 なので b は常に妥当な密度行列 (命題2) に対応する.
    """
    inv_zeta = torch.rsqrt(1 + u * u + v * v)
    m = torch.tanh(c)
    return torch.stack([m * u * inv_zeta, m * v * inv_zeta, m * inv_zeta], dim=-1)


def complete_coords(b, eps=1e-12):
    """(CB6) b -> (c, u, v) の逆写像. b_z = 0 (赤道面) は無限遠なので eps で丸める."""
    bz = b[..., 2]
    sign = torch.where(bz < 0, -torch.ones_like(bz), torch.ones_like(bz))
    bz_safe = sign * torch.clamp(bz.abs(), min=eps)
    m = sign * torch.linalg.vector_norm(b, dim=-1)
    c = torch.atanh(torch.clamp(m, -1 + eps, 1 - eps))
    return c, b[..., 0] / bz_safe, b[..., 1] / bz_safe


def polarization(b):
    """符号付き偏極 m = sign(b_z) |b|: 正なら上の葉の半球体, 負なら下の葉の半球体."""
    sign = torch.where(b[..., 2] < 0, -torch.ones_like(b[..., 2]), torch.ones_like(b[..., 2]))
    return sign * torch.linalg.vector_norm(b, dim=-1)


def sheet_of(b):
    """b が属する半球体を与える葉 (+1: H+ / |0> 側, -1: H- / |1> 側)."""
    return torch.where(b[..., 2] < 0, -torch.ones_like(b[..., 2]), torch.ones_like(b[..., 2]))


def hyperbolic_depth(b, eps=1e-12):
    """(CB2) 葉軸 P(q) = b/|b| の頂点からの双曲距離 rho = arccosh(1/|cos Theta|)."""
    cos_t = b[..., 2].abs() / torch.clamp(torch.linalg.vector_norm(b, dim=-1), min=eps)
    return torch.acosh(1 / torch.clamp(cos_t, min=eps))


def hyperbolic_distance(q1, q2):
    """同じ葉の2点の双曲距離 arccosh(<q1, q2>_L), <q1,q2>_L = zeta1 zeta2 - u1 u2 - v1 v2."""
    lorentz = q1[..., 2] * q2[..., 2] - q1[..., 0] * q2[..., 0] - q1[..., 1] * q2[..., 1]
    return torch.acosh(torch.clamp(lorentz, min=1.0))


def measure_sheet(b, generator=None):
    """(CB5) 葉軸 b/|b| での射影測定: +-1 を確率 (1 +- |b|)/2 で得て P(+-q) に収縮.
    戻り値は収縮後の純粋状態 +-P(q) (|b'| = 1) で E[b'] = b."""
    r = torch.linalg.vector_norm(b, dim=-1, keepdim=True)
    axis = b / torch.clamp(r, min=1e-12)
    # 中心 (r = 0) では軸が定まらないので z 軸を使う (どの軸でも E[b'] = 0).
    axis = torch.where(r > 1e-12, axis, torch.tensor([0.0, 0.0, 1.0], dtype=b.dtype, device=b.device))
    p_plus = (1 + r) / 2
    outcome = torch.where(torch.rand(p_plus.shape, generator=generator, dtype=b.dtype,
                                     device=b.device) < p_plus, 1.0, -1.0)
    return outcome * axis


def measure_z(b, generator=None):
    """平方根振幅論文 式(17): 計算基底で測定し z -> +-1 (確率 (1 +- z)/2), x = y = 0."""
    p0 = (1 + b[..., 2:3]) / 2
    z = torch.where(torch.rand(p0.shape, generator=generator, dtype=b.dtype,
                               device=b.device) < p0, 1.0, -1.0)
    return torch.cat([torch.zeros_like(b[..., :2]), z], dim=-1)


def to_npk(b):
    """完全ブロッホ球の状態を平方根振幅座標 (n, phi, kappa) に戻す (命題2の逆写像)."""
    return sa.from_bloch(b.to(torch.float64))


def describe(b) -> Dict[str, float]:
    """ニューロン群の状態の要約 (n, κ, 純度, 上の葉の割合, 双曲深さ, 偏極)."""
    b = b.detach().to(torch.float64).reshape(-1, 3)
    n, _, kappa = to_npk(b)
    r2 = (b ** 2).sum(-1)
    return {
        'n_mean': n.mean().item(),
        'kappa_mean': kappa.mean().item(),
        'purity_mean': ((1 + r2) / 2).mean().item(),
        'upper_sheet_frac': (sheet_of(b) > 0).double().mean().item(),
        'depth_median': hyperbolic_depth(b).median().item(),
        'polarization_abs_mean': r2.sqrt().mean().item(),
    }


def axis_angle_rotation(w):
    """軸角ベクトル w [..., 3] -> SO(3) 回転 (ロドリゲス). 1量子ビットゲートのブロッホ表現
    (平方根振幅論文 式(8)) を学習可能にしたもの. w = 0 で恒等."""
    theta2 = (w * w).sum(-1)[..., None, None]
    theta = torch.sqrt(theta2 + 1e-24)
    small = theta2 < 1e-8
    a = torch.where(small, 1 - theta2 / 6, torch.sin(theta) / theta)
    bcoef = torch.where(small, 0.5 - theta2 / 24, (1 - torch.cos(theta)) / torch.clamp(theta2, min=1e-24))
    zero = torch.zeros_like(w[..., 0])
    K = torch.stack([torch.stack([zero, -w[..., 2], w[..., 1]], -1),
                     torch.stack([w[..., 2], zero, -w[..., 0]], -1),
                     torch.stack([-w[..., 1], w[..., 0], zero], -1)], -2)
    eye = torch.eye(3, dtype=w.dtype, device=w.device)
    return eye + a * K + bcoef * (K @ K)


# ---------------------------------------------------------------------------
# 完全QBNN
# ---------------------------------------------------------------------------

class CompleteBlochLayer(nn.Module):
    """完全ブロッホ球ニューロンの層.

    入力 x から各ニューロンの学習座標 (c, u, v) を線形に作り (CB4) で b を得る.
    gates=True なら各ニューロンに学習可能な1量子ビットゲート R_j in SO(3) を掛ける
    (恒等で初期化). 出力は [..., neurons, 3] のブロッホベクトルで, 次の層はこれを
    3 * neurons 個の実数として読む (読み出しの重みが古典的な相互作用 J に当たる).

    measure > 0 のとき各ニューロンを確率 measure で測定し (CB5 または式(17)),
    勾配は測定前の b に流す (straight-through, E[測定値] = b).
    """

    def __init__(self, in_features: int, neurons: int, gates: bool = True):
        super().__init__()
        self.neurons = neurons
        self.proj = nn.Linear(in_features, 3 * neurons)
        self.gate = nn.Parameter(torch.zeros(neurons, 3)) if gates else None
        self.last_state: Optional[torch.Tensor] = None

    def forward(self, x, measure: float = 0.0, basis: str = 'sheet', generator=None):
        c, u, v = self.proj(x).unflatten(-1, (self.neurons, 3)).unbind(-1)
        b = complete_bloch(c, u, v)
        if self.gate is not None:
            b = torch.einsum('nij,...nj->...ni', axis_angle_rotation(self.gate), b)
        if measure > 0:
            collapsed = (measure_sheet if basis == 'sheet' else measure_z)(b, generator)
            pick = torch.rand(b.shape[:-1] + (1,), generator=generator, dtype=b.dtype,
                              device=b.device) < measure
            b = b + (torch.where(pick, collapsed, b) - b).detach()
        self.last_state = b.detach()
        return b


class CompleteQBNN(nn.Module):
    """完全ブロッホ球ニューロンだけで隠れ層を作る順伝播ネットワーク (分類・回帰用)."""

    def __init__(self, in_features: int, hidden: List[int], out_features: int, gates: bool = True):
        super().__init__()
        dims = [in_features] + [3 * h for h in hidden]
        self.layers = nn.ModuleList(CompleteBlochLayer(d, h, gates) for d, h in zip(dims, hidden))
        self.readout = nn.Linear(dims[-1], out_features)

    def forward(self, x, measure: float = 0.0, basis: str = 'sheet', generator=None):
        for layer in self.layers:
            x = layer(x, measure, basis, generator).flatten(-2)
        return self.readout(x)

    def diagnostics(self) -> List[Dict[str, float]]:
        return [describe(l.last_state) for l in self.layers if l.last_state is not None]


class CompleteQBNNLM(nn.Module):
    """完全ブロッホ球ニューロンを記憶とする再帰型の文字言語モデル.

    各層の隠れ状態は neurons 個の完全ブロッホ球 b_t で, 次の時刻の状態は
    b_t = CompleteBlochLayer([入力; b_{t-1}]) で決まる. 初期状態は最大混合状態 b = 0.
    """

    def __init__(self, vocab_size: int, embed_dim: int = 64, neurons: int = 128,
                 layers: int = 2, gates: bool = True):
        super().__init__()
        self.config = dict(vocab_size=vocab_size, embed_dim=embed_dim, neurons=neurons,
                           layers=layers, gates=gates)
        self.neurons = neurons
        self.embed = nn.Embedding(vocab_size, embed_dim)
        dims = [embed_dim] + [3 * neurons] * (layers - 1)
        self.cells = nn.ModuleList(CompleteBlochLayer(d + 3 * neurons, neurons, gates) for d in dims)
        self.readout = nn.Linear(3 * neurons, vocab_size)

    def init_state(self, batch: int, device=None):
        return [torch.zeros(batch, self.neurons, 3, device=device) for _ in self.cells]

    def step(self, token, state, measure=0.0, basis='sheet', generator=None):
        x = self.embed(token)
        new_state = []
        for cell, b_prev in zip(self.cells, state):
            b = cell(torch.cat([x, b_prev.flatten(-2)], -1), measure, basis, generator)
            new_state.append(b)
            x = b.flatten(-2)
        return self.readout(x), new_state

    def forward(self, tokens, state=None, measure=0.0, basis='sheet', generator=None):
        """tokens [B, T] -> logits [B, T, V], 最終状態."""
        if state is None:
            state = self.init_state(tokens.shape[0], tokens.device)
        logits = []
        for t in range(tokens.shape[1]):
            out, state = self.step(tokens[:, t], state, measure, basis, generator)
            logits.append(out)
        return torch.stack(logits, 1), state

    def diagnostics(self) -> List[Dict[str, float]]:
        return [describe(c.last_state) for c in self.cells if c.last_state is not None]


class CompleteQBNNAI:
    """文字単位の語彙と CompleteQBNNLM をまとめた学習・生成用のラッパー."""

    def __init__(self, chars: List[str], **model_kwargs):
        self.chars = chars
        self.index = {ch: i for i, ch in enumerate(chars)}
        self.model = CompleteQBNNLM(len(chars), **model_kwargs)

    @classmethod
    def from_text(cls, text: str, **model_kwargs):
        return cls(sorted(set(text)), **model_kwargs)

    def encode(self, text: str) -> torch.Tensor:
        return torch.tensor([self.index[ch] for ch in text if ch in self.index], dtype=torch.long)

    def decode(self, ids) -> str:
        return ''.join(self.chars[i] for i in ids)

    def train(self, text: str, steps: int = 1000, batch: int = 32, seq_len: int = 64,
              lr: float = 3e-3, seed: int = 0, log_every: int = 100) -> List[float]:
        data = self.encode(text)
        if len(data) <= seq_len + 1:
            raise ValueError(f"text too short: {len(data)} tokens for seq_len {seq_len}")
        g = torch.Generator().manual_seed(seed)
        opt = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=0.01)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
        self.model.train()
        losses = []
        for step in range(1, steps + 1):
            start = torch.randint(0, len(data) - seq_len - 1, (batch,), generator=g)
            idx = start[:, None] + torch.arange(seq_len + 1)
            chunk = data[idx]
            logits, _ = self.model(chunk[:, :-1])
            loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), chunk[:, 1:].reshape(-1))
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            opt.step()
            sched.step()
            losses.append(loss.item())
            if log_every and step % log_every == 0:
                avg = sum(losses[-log_every:]) / len(losses[-log_every:])
                print(f"step {step:5d}  loss {avg:.3f}  ppl {math.exp(avg):.1f}")
        return losses

    @torch.no_grad()
    def generate(self, prompt: str, max_new: int = 80, temperature: float = 0.8,
                 measure: float = 0.0, basis: str = 'sheet', seed: Optional[int] = None) -> str:
        """temperature = 0 は貪欲法. measure > 0 なら各ニューロンを確率 measure で測定する
        (平方根振幅論文 第7.6節と同じ「測定サンプリング」)."""
        self.model.eval()
        g = torch.Generator().manual_seed(seed) if seed is not None else None
        ids = self.encode(prompt).tolist() or [0]
        state = self.model.init_state(1)
        for tok in ids[:-1]:
            _, state = self.model.step(torch.tensor([tok]), state, measure, basis, g)
        out = list(ids)
        tok = ids[-1]
        for _ in range(max_new):
            logits, state = self.model.step(torch.tensor([tok]), state, measure, basis, g)
            if temperature <= 0:
                tok = int(logits.argmax(-1))
            else:
                probs = torch.softmax(logits[0] / temperature, -1)
                tok = int(torch.multinomial(probs, 1, generator=g))
            out.append(tok)
        return self.decode(out)

    def save(self, path: str):
        torch.save({'chars': self.chars, 'config': self.model.config,
                    'state_dict': self.model.state_dict()}, path)

    @classmethod
    def load(cls, path: str):
        ckpt = torch.load(path, map_location='cpu', weights_only=True)
        cfg = dict(ckpt['config'])
        cfg.pop('vocab_size')
        ai = cls(ckpt['chars'], **cfg)
        ai.model.load_state_dict(ckpt['state_dict'])
        return ai


# ---------------------------------------------------------------------------
# デモ
# ---------------------------------------------------------------------------

def two_spirals(n: int, noise: float = 0.05, seed: int = 0):
    g = torch.Generator().manual_seed(seed)
    t = torch.sqrt(torch.rand(n, generator=g)) * 3 * math.pi
    label = torch.randint(0, 2, (n,), generator=g)
    sign = 1 - 2 * label.float()
    x = torch.stack([sign * t * torch.cos(t), sign * t * torch.sin(t)], -1) / (3 * math.pi)
    return x + noise * torch.randn(n, 2, generator=g), label


def classify_demo(steps: int = 2000, seed: int = 0):
    torch.manual_seed(seed)
    x, y = two_spirals(2000, seed=seed)
    xt, yt = two_spirals(1000, seed=seed + 1)
    net = CompleteQBNN(2, [32, 32], 2)
    opt = torch.optim.Adam(net.parameters(), lr=1e-2)
    for step in range(1, steps + 1):
        loss = F.cross_entropy(net(x), y)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step % 500 == 0:
            print(f"step {step:5d}  loss {loss.item():.4f}")
    with torch.no_grad():
        acc = (net(xt).argmax(-1) == yt).float().mean().item()
        print(f"two spirals test accuracy: {acc:.3f}")
        for i, d in enumerate(net.diagnostics()):
            print(f"  layer {i}: " + ", ".join(f"{k}={v:.3f}" for k, v in d.items()))
        g = torch.Generator().manual_seed(seed)
        votes = torch.stack([net(xt, measure=1.0, generator=g).argmax(-1) for _ in range(16)])
        agree = (votes == votes.mode(0).values).float().mean(0)
        correct = votes.mode(0).values == yt
        print(f"measurement sampling (16 runs, all neurons measured): "
              f"agreement {agree.mean():.3f}, majority accuracy {correct.float().mean():.3f}, "
              f"agreement on wrong answers {agree[~correct].mean() if (~correct).any() else float('nan'):.3f}")
    return acc


def main():
    ap = argparse.ArgumentParser(description="完全ブロッホ球ニューロンの AI (完全QBNN)")
    sub = ap.add_subparsers(dest='cmd', required=True)
    d = sub.add_parser('classify-demo', help='二重らせんの分類で完全QBNNを学習する')
    d.add_argument('--steps', type=int, default=2000)
    t = sub.add_parser('train', help='テキストで文字言語モデルを学習する')
    t.add_argument('--text', nargs='+', required=True)
    t.add_argument('--out', default='complete_qbnn_lm.pt')
    t.add_argument('--steps', type=int, default=2000)
    t.add_argument('--neurons', type=int, default=128)
    t.add_argument('--layers', type=int, default=2)
    t.add_argument('--seq-len', type=int, default=64)
    t.add_argument('--batch', type=int, default=32)
    t.add_argument('--lr', type=float, default=3e-3)
    gen = sub.add_parser('generate', help='学習済みモデルで文章を生成する')
    gen.add_argument('--ckpt', default='complete_qbnn_lm.pt')
    gen.add_argument('--prompt', default='')
    gen.add_argument('--max-new', type=int, default=120)
    gen.add_argument('--temperature', type=float, default=0.8)
    gen.add_argument('--measure', type=float, default=0.0)
    gen.add_argument('--samples', type=int, default=1)
    gen.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()

    if args.cmd == 'classify-demo':
        classify_demo(args.steps)
    elif args.cmd == 'train':
        text = '\n'.join(open(p, encoding='utf-8').read() for p in args.text)
        ai = CompleteQBNNAI.from_text(text, neurons=args.neurons, layers=args.layers)
        n_params = sum(p.numel() for p in ai.model.parameters())
        print(f"{len(text):,} chars, vocab {len(ai.chars)}, {n_params:,} parameters, "
              f"{args.layers} x {args.neurons} complete Bloch neurons")
        ai.train(text, steps=args.steps, batch=args.batch, seq_len=args.seq_len, lr=args.lr)
        for i, dgn in enumerate(ai.model.diagnostics()):
            print(f"  layer {i}: " + ", ".join(f"{k}={v:.3f}" for k, v in dgn.items()))
        ai.save(args.out)
        print(f"saved {args.out}")
    else:
        ai = CompleteQBNNAI.load(args.ckpt)
        for i in range(args.samples):
            print(f"--- sample {i + 1}")
            print(ai.generate(args.prompt, args.max_new, args.temperature, args.measure,
                              seed=args.seed + i))


if __name__ == '__main__':
    main()
