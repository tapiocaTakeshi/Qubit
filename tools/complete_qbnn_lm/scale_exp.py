"""Scaling comparison: tiny char-level GPT whose FFN neurons are GELU, tanh or complete Bloch spheres.

Same width h (a multiple of 3), same parameter count up to the 3h gate parameters of the complete
variant, same data order, same optimizer. Reports held-out loss in bits per character.
    python scale_exp.py SIZE NEURON STEPS SEED
"""
import json, math, os, sys, time
import torch, torch.nn as nn, torch.nn.functional as F
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..')))   # リポジトリ直下（complete_qbnn など）
import complete_qbnn as cq

SIZES = {'XS': (64, 2, 2), 'S': (128, 4, 4), 'M': (256, 4, 8), 'L': (384, 6, 8), 'XL': (512, 8, 8)}   # d_model, layers, heads
CTX, BATCH = 128, 32
D = os.environ.get('QBNN_LM_DATA', os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data'))   # 学習データとモデルを置く場所


def load_data():
    text = open(f'{D}/jawiki.txt', encoding='utf-8').read()
    train, val = text[:-1_000_000], text[-1_000_000:]
    try:
        vocab = json.load(open(f'{D}/vocab.json', encoding='utf-8'))
    except FileNotFoundError:
        from collections import Counter
        vocab = ['<unk>'] + [c for c, _ in Counter(train).most_common(3999)]
        json.dump(vocab, open(f'{D}/vocab.json', 'w', encoding='utf-8'), ensure_ascii=False)
    idx = {c: i for i, c in enumerate(vocab)}
    enc = lambda s: torch.tensor([idx.get(c, 0) for c in s], dtype=torch.long)
    return enc(train), enc(val), vocab


class NQFFN(nn.Module):
    """NeuroQuantum 型: 0.7 × 標準 FFN + 0.3 × (QBNNLayer → QBNNLayer)。neuroquantum_layered の QBNNLayer を
    そのまま使う（3次元ブロッホニューロン, λ_entangle = 0.5）。隠れ幅は両方 h/3 にしてパラメータ数を揃える。"""
    def __init__(self, d, h):
        super().__init__()
        from neuroquantum_layered import QBNNLayer
        kw = dict(lambda_min=0.25, lambda_max=0.75, use_qbnn_layered=False, bloch=True, bloch_rank=16)
        cnt = lambda m: sum(p.numel() for p in m.parameters())
        size = lambda hq: 2 * d * hq + hq + d + cnt(QBNNLayer(d, hq, **kw)) + cnt(QBNNLayer(hq, d, **kw))
        hq = min(range(8, h // 3 + 1), key=lambda k: abs(size(k) - (2 * d * h + h + d)))   # GELU の FFN と同じパラメータ数
        self.std = nn.Sequential(nn.Linear(d, hq), nn.GELU(), nn.Linear(hq, d))
        self.q1, self.q2 = QBNNLayer(d, hq, **kw), QBNNLayer(hq, d, **kw)

    def forward(self, x):
        return 0.7 * self.std(x) + 0.3 * self.q2(self.q1(x))


class FFN(nn.Module):
    def __new__(cls, d, h, kind):
        if kind == 'nq':
            return NQFFN(d, h)
        return super().__new__(cls)

    def __init__(self, d, h, kind):
        super().__init__()
        if kind == 'cmix':   # RWKV-7 の channel mixing: token shift → ReLU² → 線形（バイアスなし）
            self.mix_k = nn.Parameter(torch.full((d,), 0.5))
        if kind == 'rnn':   # 再帰の重み h×h の分だけ幅を減らし、GELU の FFN とパラメータ数を揃える
            h = 3 * int((-(2 * d + 1) + math.sqrt((2 * d + 1) ** 2 + 4 * (2 * d * h + h))) / 6)
        self.kind, self.h = kind, h
        self.up, self.down = nn.Linear(d, h), nn.Linear(h, d)
        if kind in ('complete', 'rnn'):
            self.gate = nn.Parameter(torch.zeros(h // 3, 3))
        if kind == 'rnn':   # b_{t-1} -> 次の時刻の (c, u, v)。ゼロで初期化すると最初は 'complete' と同じ
            self.rec = nn.Linear(h, h, bias=False)
            nn.init.zeros_(self.rec.weight)

    def bloch(self, a):
        c, u, v = a.unflatten(-1, (self.h // 3, 3)).unbind(-1)
        b = cq.complete_bloch(c, u, v)
        return torch.einsum('nij,...nj->...ni', cq.axis_angle_rotation(self.gate), b).flatten(-2)

    def forward(self, x, state=None):
        """kind='rnn' のときだけ state（直前の b、[B, h]）を受け取り (出力, 最後の b) を返せる。"""
        if self.kind == 'cmix':
            xk = x + (F.pad(x, (0, 0, 1, -1)) - x) * self.mix_k
            return self.down(torch.relu(self.up(xk)) ** 2)
        a = self.up(x)
        if self.kind == 'rnn':
            R = cq.axis_angle_rotation(self.gate)
            b = torch.zeros_like(a[:, 0]) if state is None else state
            outs = []
            for t in range(a.shape[1]):   # 完全ブロッホ球ニューロンの状態 b_t を記憶として次の時刻へ渡す
                c, u, v = (a[:, t] + self.rec(b)).unflatten(-1, (self.h // 3, 3)).unbind(-1)
                b = torch.einsum('nij,bnj->bni', R, cq.complete_bloch(c, u, v)).flatten(-2)
                outs.append(b)
            self.last_state = b
            return self.down(torch.stack(outs, 1))
        if self.kind == 'gelu':
            a = F.gelu(a)
        elif self.kind == 'tanh':
            a = torch.tanh(a)
        else:
            c, u, v = a.unflatten(-1, (self.h // 3, 3)).unbind(-1)
            b = cq.complete_bloch(c, u, v)
            b = torch.einsum('nij,...nj->...ni', cq.axis_angle_rotation(self.gate), b)
            a = b.flatten(-2)
        return self.down(a)


class RWKV7(nn.Module):
    """Attention の代わりの RWKV-7 time mixing（一般化デルタ則）。ヘッドごとの状態 S は 値 N × 鍵 nk の行列:
        S_t = S_{t-1} (diag(w_t) − κ̂_tᵀ (a_t ⊙ κ̂_t)) + v_tᵀ k_t,   y_t = S_t r_t
    w_t（減衰）と a_t（書き込みの強さ）は入力ごとに変わる。v-first の残差は省いた。
    bloch=True なら受容 r と鍵 k を完全ブロッホ球ニューロン（ヘッドあたり N//3 個、ゲート R_j つき）の出力にする。
    apqb='k' / 'ka' なら、apqb_neurons 個の完全ブロッホ球ニューロンの状態 b_t から鍵（と書き込みの強さ）に足し込む:
        k_t' = k_t + γ_k W_B b_t,   a_t' = sigmoid(u_a,t + γ_a W_A b_t)
    γ はゼロで初期化するので、最初は標準の RWKV-7 と同じ出力になる。"""
    def __init__(self, d, heads, bloch=False, lora=16, apqb=None, apqb_neurons=16):
        super().__init__()
        self.apqb = apqb
        if apqb:
            self.apqb_in = nn.Linear(d, 3 * apqb_neurons)
            self.apqb_gate = nn.Parameter(torch.zeros(apqb_neurons, 3))
            self.W_B = nn.Linear(3 * apqb_neurons, d // heads * heads, bias=False)
            self.gamma_k = nn.Parameter(torch.zeros(()))
            if apqb == 'ka':
                self.W_A = nn.Linear(3 * apqb_neurons, d // heads * heads, bias=False)
                self.gamma_a = nn.Parameter(torch.zeros(()))
        H, N = heads, d // heads
        nk = 3 * (N // 3) if bloch else N
        self.H, self.N, self.nk, self.bloch = H, N, nk, bloch
        self.mix = nn.Parameter(torch.full((6, d), 0.5))   # token shift: r, w, k, v, a, g
        self.r, self.k = nn.Linear(d, H * nk, bias=False), nn.Linear(d, H * nk, bias=False)
        self.v, self.o = nn.Linear(d, d, bias=False), nn.Linear(d, d, bias=False)
        lo = lambda n_out, act: nn.Sequential(nn.Linear(d, lora, bias=False), act, nn.Linear(lora, n_out, bias=False))
        self.w_lora, self.a_lora, self.g_lora = lo(H * nk, nn.Tanh()), lo(H * nk, nn.Identity()), lo(d, nn.Sigmoid())
        for m in (self.w_lora, self.a_lora):
            nn.init.zeros_(m[2].weight)
        self.w0 = nn.Parameter(torch.linspace(-4, 1, nk).repeat(H))
        self.a0 = nn.Parameter(torch.zeros(H * nk))
        self.k_k, self.k_a = nn.Parameter(torch.full((H * nk,), 0.85)), nn.Parameter(torch.ones(H * nk))
        self.r_k = nn.Parameter(torch.zeros(H, nk))
        self.ln_x = nn.GroupNorm(H, d, eps=64e-5)
        if bloch:
            self.gate_r, self.gate_k = nn.Parameter(torch.zeros(H * nk // 3, 3)), nn.Parameter(torch.zeros(H * nk // 3, 3))

    def to_bloch(self, a, gate):
        c, u, v = a.unflatten(-1, (-1, 3)).unbind(-1)
        return torch.einsum('nij,...nj->...ni', cq.axis_angle_rotation(gate), cq.complete_bloch(c, u, v)).flatten(-2)

    def forward(self, x):
        B, T, C = x.shape
        H, N, nk = self.H, self.N, self.nk
        xx = F.pad(x, (0, 0, 1, -1)) - x
        xr, xw, xk, xv, xa, xg = (x + xx * m for m in self.mix)
        r, k = self.r(xr), self.k(xk)
        if self.bloch:
            r, k = self.to_bloch(r, self.gate_r), self.to_bloch(k, self.gate_k)
        v, g = self.v(xv), self.g_lora(xg)
        w = torch.exp(-0.606531 * torch.sigmoid(self.w0 + self.w_lora(xw)))   # (0.545, 1)
        u_a = self.a0 + self.a_lora(xa)
        if self.apqb:   # 完全APQB の状態 b_t（位相・コヒーレンス・偏極）が鍵と更新率を動かす
            bq = self.to_bloch(self.apqb_in(xk), self.apqb_gate)
            self.last_apqb = bq.detach()
            k = k + self.gamma_k * self.W_B(bq)
            if self.apqb == 'ka':
                u_a = u_a + self.gamma_a * self.W_A(bq)
        a = torch.sigmoid(u_a)
        kk = F.normalize((k * self.k_k).view(B, T, H, nk), dim=-1)
        k = k * (1 + (a - 1) * self.k_a)
        r, k, w, a = (t.view(B, T, H, nk) for t in (r, k, w, a))
        v = v.view(B, T, H, N)
        S = x.new_zeros(B, H, N, nk)
        ys = []
        for t in range(T):
            kt = kk[:, t]
            S = S * w[:, t, :, None, :] - (S @ kt[..., None]) @ (kt * a[:, t])[..., None, :] \
                + v[:, t, :, :, None] @ k[:, t, :, None, :]
            ys.append((S @ r[:, t, :, :, None]).squeeze(-1))
        y = self.ln_x(torch.stack(ys, 1).reshape(B * T, C)).view(B, T, H, N)
        y = y + (r * k * self.r_k).sum(-1, keepdim=True) * v
        return self.o(y.reshape(B, T, C) * g)


class Block(nn.Module):
    def __init__(self, d, heads, h, kind):
        super().__init__()
        self.ln1, self.ln2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.heads = heads
        if kind.startswith('rwkv7'):   # Attention を RWKV-7 にする
            # rwkv7a: 標準 RWKV-7（channel mixing は ReLU²）/ rwkv7: FFN を完全ブロッホ球に / rwkv7b: r, k をブロッホ球に置換
            # rwkv7c: rwkv7 + APQB 由来の鍵 / rwkv7d: rwkv7c + APQB 由来の更新率 a
            self.tmix = RWKV7(d, heads, bloch=kind == 'rwkv7b', apqb={'rwkv7c': 'k', 'rwkv7d': 'ka'}.get(kind))
            kind = 'cmix' if kind == 'rwkv7a' else 'complete'
        else:
            self.qkv, self.proj = nn.Linear(d, 3 * d), nn.Linear(d, d)
        self.ffn = FFN(d, h, kind)

    def forward(self, x):
        if hasattr(self, 'tmix'):
            x = x + self.tmix(self.ln1(x))
            return x + self.ffn(self.ln2(x))
        B, T, C = x.shape
        q, k, v = self.qkv(self.ln1(x)).split(C, -1)
        q, k, v = (t.view(B, T, self.heads, C // self.heads).transpose(1, 2) for t in (q, k, v))
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True).transpose(1, 2).reshape(B, T, C)
        x = x + self.proj(y)
        return x + self.ffn(self.ln2(x))


class GPT(nn.Module):
    def __init__(self, vocab, d, layers, heads, kind):
        super().__init__()
        h = 3 * int(4 * d / 3)
        self.tok, self.pos = nn.Embedding(vocab, d), nn.Embedding(CTX, d)
        self.blocks = nn.ModuleList(Block(d, heads, h, kind) for _ in range(layers))
        self.ln = nn.LayerNorm(d)

    def forward(self, idx):
        x = self.tok(idx) + self.pos(torch.arange(idx.shape[1]))
        for blk in self.blocks:
            x = blk(x)
        return self.ln(x) @ self.tok.weight.T   # tied output


def batches(data, g, n):
    for _ in range(n):
        s = torch.randint(0, len(data) - CTX - 1, (BATCH,), generator=g)
        chunk = data[s[:, None] + torch.arange(CTX + 1)]
        yield chunk[:, :-1], chunk[:, 1:]


@torch.no_grad()
def evaluate(model, val, n=100):
    model.eval()
    losses = [F.cross_entropy(model(x).flatten(0, 1), y.flatten()).item()
              for x, y in batches(val, torch.Generator().manual_seed(1234), n)]
    model.train()
    return sum(losses) / len(losses) / math.log(2)   # bits per char


def main():
    size, kind, steps, seed = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
    torch.set_num_threads(int(sys.argv[5]) if len(sys.argv) > 5 else 1)
    train, val, vocab = load_data()
    torch.manual_seed(seed)
    d, layers, heads = SIZES[size]
    model = GPT(len(vocab), d, layers, heads, kind)
    n_params = sum(p.numel() for p in model.parameters())
    n_body = n_params - model.tok.weight.numel() - model.pos.weight.numel()
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3 if size in ('XS', 'S') else 1e-3, betas=(0.9, 0.95), weight_decay=0.1)
    warm = max(steps // 20, 1)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min((s + 1) / warm, 0.5 * (1 + math.cos(math.pi * min(s, steps) / steps))))
    g = torch.Generator().manual_seed(seed)    # same data order for every neuron type
    import os
    ck_path = f'{D}/resume_{size}_{kind}_{steps}.pt'
    start, log = 1, []
    if os.path.exists(ck_path):   # コンテナが再起動しても途中から再開する
        ck = torch.load(ck_path, weights_only=False)
        model.load_state_dict(ck['model']); opt.load_state_dict(ck['opt']); sched.load_state_dict(ck['sched'])
        g.set_state(ck['gen']); start, log = ck['step'] + 1, ck['log']
        print(f'{size} {kind} resumed at step {start}', flush=True)
    t0 = time.time()
    for step, (x, y) in enumerate(batches(train, g, steps - start + 1), start):
        loss = F.cross_entropy(model(x).flatten(0, 1), y.flatten())
        opt.zero_grad(); loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sched.step()
        if step % max(steps // 5, 1) == 0 or step == steps:
            log.append((step, evaluate(model, val, 40), time.time() - t0))
            print(f'{size} {kind} seed{seed} step {step} val {log[-1][1]:.3f} bpc ({log[-1][2]:.0f}s)', flush=True)
        if step % 600 == 0 and step < steps:
            torch.save({'model': model.state_dict(), 'opt': opt.state_dict(), 'sched': sched.state_dict(),
                        'gen': g.get_state(), 'step': step, 'log': log}, ck_path + '.tmp')
            os.replace(ck_path + '.tmp', ck_path)
    result = dict(size=size, kind=kind, seed=seed, steps=steps, params=n_params, body_params=n_body,
                  train_chars=steps * BATCH * CTX, val_bpc=evaluate(model, val, 200), log=log,
                  seconds=time.time() - t0)
    torch.save(model.state_dict(), f'{D}/model_{size}_{kind}_{steps}.pt')
    with open(f'{D}/results_long.jsonl', 'a') as f:
        f.write(json.dumps(result) + '\n')
    print('RESULT', json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
