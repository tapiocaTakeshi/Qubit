"""50,000 ステップのモデルと追加学習中のモデルで、同じ質問・同じ乱数で応答を比べる。"""
import json, sys, torch, torch.nn.functional as F
sys.argv = ['x']
exec(open(__import__('os').path.join(__import__('os').path.dirname(__import__('os').path.abspath(__file__)), 'scale_exp.py')).read()
     .replace("if __name__ == '__main__':\n    main()", ''))
CTX = 256
torch.set_num_threads(1)
meta = json.load(open(f'{D}/sft_meta.json', encoding='utf-8'))
vocab = meta['vocab']; idx = {c: i for i, c in enumerate(vocab)}
BOS, SEP, EOS, PAD = (idx[s] for s in meta['special'])
enc = lambda s: [idx.get(c, 0) for c in s]


@torch.no_grad()
def reply(model, history, msg, seed, T=0.6, pen=1.3):
    g = torch.Generator().manual_seed(seed)
    ids = [t for u, r in history for t in [BOS] + enc(u) + [SEP] + enc(r) + [EOS]] + [BOS] + enc(msg) + [SEP]
    gen = []
    for _ in range(120):
        lg = model(torch.tensor([ids[-CTX:]]))[0, -1].clone()
        for t in set(gen[-48:]):
            lg[t] = lg[t] / pen if lg[t] > 0 else lg[t] * pen
        t = int(torch.multinomial(F.softmax(lg / T, -1), 1, generator=g))
        if t == EOS:
            return ''.join(vocab[i] for i in gen)
        gen.append(t); ids.append(t)
    return ''.join(vocab[i] for i in gen) + '…'


models = {}
for name, path, size in [('XL10万', 'model_M_complete_cont_XL260000.pt', 'XL'), ('XL14万', 'model_M_complete_cont_XL340000.pt', 'XL')]:
    m = GPT(len(vocab), *SIZES[size], 'complete')
    sd = torch.load(f'{D}/{path}', weights_only=False)
    if 'model' in sd:
        print(f'{name}: step {sd["step"]}'); sd = sd['model']
    m.load_state_dict(sd); m.eval(); models[name] = m
qs = ['こんにちは！', 'あなたは誰ですか？', '日本の首都はどこですか？', '日本で一番高い山は？', '猫と犬の違いを教えてください。',
      'おすすめの勉強方法を教えてください。', '短い詩を書いてください。', 'Pythonで1から10までの合計を求めるコードを書いてください。',
      '量子コンピュータとは何ですか？', '富士山について教えてください。', 'ドラえもんの作者は誰でしょう？', '夏目漱石の代表作は何でしょう？', 'アメリカの首都はどこでしょう？', 'ありがとう！']
for q in qs:
    print(f'\n【あなた】{q}', flush=True)
    for name, m in models.items():
        for seed in (1, 2):
            print(f'【{name}・{seed}】{reply(m, [], q, seed).replace(chr(10), "⏎")}', flush=True)
for name, m in models.items():
    h = []; print(f'\n--- 会話（{name}）')
    for q in ['こんにちは！', '今日は何をしましたか？', 'おすすめの本はありますか？']:
        r = reply(m, h, q, 5); print(f'【あなた】{q}\n【モデル】{r.replace(chr(10), "⏎")}'); h.append((q, r))
