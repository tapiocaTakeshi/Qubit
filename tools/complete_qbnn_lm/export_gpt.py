"""Export a scale_exp GPT(kind='complete') checkpoint for engine_gpt.js, plus a PyTorch reference."""
import base64, json, sys, numpy as np, torch
sys.argv_real = list(sys.argv); sys.argv = ['x']
exec(open(__import__('os').path.join(__import__('os').path.dirname(__import__('os').path.abspath(__file__)), 'scale_exp.py')).read().replace("if __name__ == '__main__':\n    main()", ''))
import complete_qbnn as cq
torch.set_num_threads(1)


def export(ckpt, size, out_dir, sft_meta=None):
    global CTX
    special = None
    if sft_meta:   # 指示学習したモデル: 特殊トークン入りの語彙と文脈 256
        sm = json.load(open(sft_meta, encoding='utf-8'))
        vocab, special, CTX = sm['vocab'], sm['special'], sm['ctx']
    else:
        vocab = json.load(open(f'{D}/vocab.json', encoding='utf-8'))
    d, layers, heads = SIZES[size]
    model = GPT(len(vocab), d, layers, heads, 'complete')
    model.load_state_dict(torch.load(ckpt)); model.eval()
    sd = model.state_dict()
    parts = [('tok', sd['tok.weight']), ('pos', sd['pos.weight'])]
    for i in range(layers):
        p = f'blocks.{i}.'
        parts += [(f'b{i}.ln1.w', sd[p + 'ln1.weight']), (f'b{i}.ln1.b', sd[p + 'ln1.bias']),
                  (f'b{i}.qkv.w', sd[p + 'qkv.weight']), (f'b{i}.qkv.b', sd[p + 'qkv.bias']),
                  (f'b{i}.proj.w', sd[p + 'proj.weight']), (f'b{i}.proj.b', sd[p + 'proj.bias']),
                  (f'b{i}.ln2.w', sd[p + 'ln2.weight']), (f'b{i}.ln2.b', sd[p + 'ln2.bias']),
                  (f'b{i}.up.w', sd[p + 'ffn.up.weight']), (f'b{i}.up.b', sd[p + 'ffn.up.bias']),
                  (f'b{i}.down.w', sd[p + 'ffn.down.weight']), (f'b{i}.down.b', sd[p + 'ffn.down.bias']),
                  (f'b{i}.R', cq.axis_angle_rotation(sd[p + 'ffn.gate'].double()))]
    parts += [('ln.w', sd['ln.weight']), ('ln.b', sd['ln.bias'])]
    blobs, layout, off = [], {}, 0
    for name, t in parts:
        a = t.float().numpy().astype(np.float16).ravel(); layout[name] = [off, list(t.shape)]; off += a.size; blobs.append(a)
    h = 3 * int(4 * d / 3)
    meta = {'config': {'d': d, 'layers': layers, 'heads': heads, 'h': h, 'ctx': CTX}, 'chars': vocab, 'layout': layout}
    if special:
        meta['special'] = special
    json.dump(meta, open(f'{out_dir}/meta_gpt.json', 'w', encoding='utf-8'), ensure_ascii=False)
    raw = np.concatenate(blobs).tobytes()
    open(f'{out_dir}/weights_gpt.b64', 'w').write(base64.b64encode(raw).decode())
    part = 12_000_000                                   # 1ファイル 15MB の上限に収まる大きさで分ける
    for k in range(0, len(raw), part):
        open(f'{out_dir}/weights_{k // part}.wasm', 'wb').write(raw[k:k + part])
    # reference with fp16-rounded weights and the same sliding-window policy as engine_gpt.js
    model.load_state_dict({k: v.half().float() for k, v in sd.items()})
    idx = {c: i for i, c in enumerate(vocab)}
    prompt = '東京都は日本の首都'
    ids = [idx.get(c, 0) for c in prompt]
    if special:
        prompt = '日本の首都はどこですか？'
        ids = [idx['<bos>']] + [idx.get(c, 0) for c in prompt] + [idx['<sep>']]
    def push(window, t):
        if len(window) >= CTX:
            window = window[-CTX // 2:]
        return window + [t]
    with torch.no_grad():
        window = []
        for t in ids:
            window = push(window, t)
        logits_first = model(torch.tensor([window]))[0, -1].tolist()
        window, out, tok = [], [], ids[-1]
        for t in ids[:-1]:
            window = push(window, t)
        for _ in range(200):
            window = push(window, tok)
            tok = int(model(torch.tensor([window]))[0, -1].argmax())
            out.append(tok)
    json.dump({'prompt': prompt, 'logits_first': logits_first, 'greedy': ''.join(vocab[i] for i in out)},
              open(f'{out_dir}/ref_gpt.json', 'w', encoding='utf-8'), ensure_ascii=False)
    print('floats', off, 'base64 MB', round(len(base64.b64encode(np.concatenate(blobs).tobytes())) / 1e6, 2))


if __name__ == '__main__':
    export(*sys.argv_real[1:5])
