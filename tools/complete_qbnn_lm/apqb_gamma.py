"""APQB 由来の鍵・更新率（rwkv7c / rwkv7d / rwkv7e）がどれだけ使われているかを調べる。

    python apqb_gamma.py SIZE 重みファイル [重みファイル ...]

重みファイルは model_{SIZE}_{kind}_{steps}.pt か、学習途中の resume_*.pt（kind はファイル名から読む）。
層ごとに次を出す:
- γ_k, γ_a: 学習された混ぜ具合（ゼロから始まる）
- |γ_k W_B b| / |k|: APQB の項が鍵 k をどれだけ動かしているか（検証データでの RMS の比）
- |γ_a W_A b| / |u_a|: 同じく更新率の sigmoid の手前の値 u_a に対して
そのうえで、全層の γ をゼロにした（APQB を切った）ときの検証損失の変化（bits/文字）を出す。
"""
import os, re, sys, torch
sys.argv, ARGS = sys.argv[:1], sys.argv[1:]
exec(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'scale_exp.py'), encoding='utf-8').read()
     .replace("if __name__ == '__main__':\n    main()", ''))

size, paths = ARGS[0], ARGS[1:]
torch.set_num_threads(int(os.environ.get('THREADS', '1')))
train, val, vocab = load_data()
rms = lambda t: t.pow(2).mean().sqrt().item()


@torch.no_grad()
def contributions(model, n=10):
    """検証データ n バッチで、層ごとに APQB の項と元の値の RMS を測る。"""
    stats = {i: {'k': [], 'bk': [], 'ua': [], 'ba': []} for i in range(len(model.blocks))}
    hooks = []
    for i, blk in enumerate(model.blocks):
        def pre(mod, inp, i=i):
            x = inp[0]
            xx = F.pad(x, (0, 0, 1, -1)) - x
            xk, xa = x + xx * mod.mix[2], x + xx * mod.mix[4]
            bq = mod.to_bloch(mod.apqb_in(xk), mod.apqb_gate)
            s = stats[i]
            if hasattr(mod, 'W_B'):
                s['k'].append(rms(mod.k(xk))); s['bk'].append(rms(mod.gamma_k * mod.W_B(bq)))
            if hasattr(mod, 'W_A'):
                s['ua'].append(rms(mod.a0 + mod.a_lora(xa))); s['ba'].append(rms(mod.gamma_a * mod.W_A(bq)))
        hooks.append(blk.tmix.register_forward_pre_hook(pre))
    for x, _ in batches(val, torch.Generator().manual_seed(1234), n):
        model(x)
    for h in hooks:
        h.remove()
    mean = lambda v: sum(v) / len(v) if v else float('nan')
    return {i: {k: mean(v) for k, v in s.items()} for i, s in stats.items()}


for path in paths:
    name = os.path.basename(path)
    kind = re.match(rf'(?:model|resume)_{size}_(rwkv7[a-z]?)_', name).group(1)
    ck = torch.load(path, weights_only=False)
    ck = ck.get('model', ck)
    model = GPT(len(vocab), *SIZES[size], kind)
    model.load_state_dict(ck)
    model.eval()
    if not model.blocks[0].tmix.apqb:
        print(f'{name}: APQB なし（{kind}）'); continue
    print(f'\n## {name}')
    print('層  γ_k      |γ_k W_B b|/|k|   γ_a      |γ_a W_A b|/|u_a|')
    fmt = lambda t, g: f'{getattr(t, g).item():+.4f}' if hasattr(t, g) else '   -   '
    ratio = lambda a, b: '  -   ' if a != a else (f'{a / b:.4f}' if b else '  inf ')
    for i, s in contributions(model).items():
        t = model.blocks[i].tmix
        print(f'{i}   {fmt(t, "gamma_k")}  {ratio(s["bk"], s["k"])}            {fmt(t, "gamma_a")}  {ratio(s["ba"], s["ua"])}')
    full = evaluate(model, val, int(os.environ.get("NVAL", "100")))
    gammas = [getattr(b.tmix, g) for b in model.blocks for g in ('gamma_k', 'gamma_a') if hasattr(b.tmix, g)]
    with torch.no_grad():
        saved = [g.clone() for g in gammas]
        for g in gammas:
            g.zero_()
        off = evaluate(model, val, int(os.environ.get("NVAL", "100")))
        for g, v in zip(gammas, saved):
            g.copy_(v)
    print(f'検証損失: APQB あり {full:.3f} / γ=0 で切る {off:.3f}（差 {off - full:+.3f} bits/文字）')
