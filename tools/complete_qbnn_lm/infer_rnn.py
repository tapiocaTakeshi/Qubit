"""FFN を 'complete'（位置ごと）と 'rnn'（完全ブロッホ球の状態を次の時刻へ渡す）にしたモデルを比べる。

    python infer_rnn.py SIZE STEPS [プロンプト ...]

$QBNN_LM_DATA/model_{SIZE}_{kind}_{STEPS}.pt を読み、次を出す。
- 検証損失（bits/文字）。'rnn' は再帰の重みをゼロにしたとき（記憶を切ったとき）の損失も出す
- 同じプロンプト・同じ乱数での生成
"""
import os, sys, torch, torch.nn.functional as F
sys.argv, ARGS = sys.argv[:1], sys.argv[1:]
exec(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'scale_exp.py'), encoding='utf-8').read()
     .replace("if __name__ == '__main__':\n    main()", ''))

size, steps = ARGS[0], int(ARGS[1])
prompts = ARGS[2:] or ['日本の首都は', '量子コンピュータとは', '第二次世界大戦', '東京大学']
torch.set_num_threads(4)
train, val, vocab = load_data()
idx = {c: i for i, c in enumerate(vocab)}


@torch.no_grad()
def generate(model, prompt, n=80, temp=0.7, seed=0):
    g = torch.Generator().manual_seed(seed)
    ids = [idx.get(c, 0) for c in prompt]
    for _ in range(n):
        p = F.softmax(model(torch.tensor([ids[-CTX:]]))[0, -1] / temp, -1)
        ids.append(int(torch.multinomial(p, 1, generator=g)))
    return ''.join(vocab[i] for i in ids)


models = {}
for kind in ('complete', 'rnn'):
    m = GPT(len(vocab), *SIZES[size], kind)
    m.load_state_dict(torch.load(f'{D}/model_{size}_{kind}_{steps}.pt'))
    m.eval()
    models[kind] = m
    print(f'{kind}: params {sum(p.numel() for p in m.parameters()):,}, val {evaluate(m, val, 100):.3f} bpc')

rnn = models['rnn']
print('rnn 再帰の重みのノルム（層ごと）:', [round(b.ffn.rec.weight.norm().item(), 2) for b in rnn.blocks])
saved = [b.ffn.rec.weight.clone() for b in rnn.blocks]
for b in rnn.blocks:
    b.ffn.rec.weight.zero_()
print(f'rnn 再帰を切ったとき: val {evaluate(rnn, val, 100):.3f} bpc')
for b, w in zip(rnn.blocks, saved):
    b.ffn.rec.weight.copy_(w)

for p in prompts:
    print(f'\n### {p}')
    for kind, m in models.items():
        print(f'[{kind}] {generate(m, p)}')
