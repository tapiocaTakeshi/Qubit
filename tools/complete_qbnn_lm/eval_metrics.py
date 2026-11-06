"""Compare saved scaling-experiment models on metrics other than validation loss.
    python eval_metrics.py NAME=SIZE:KIND:CKPT ...
"""
import json, math, random, sys, time
import torch, torch.nn.functional as F
argv = sys.argv[1:]; sys.argv = ['x']
exec(open(__import__('os').path.join(__import__('os').path.dirname(__import__('os').path.abspath(__file__)), 'scale_exp.py')).read()
     .replace("if __name__ == '__main__':\n    main()", ''))
torch.set_num_threads(2)
train, val, vocab = load_data()
corpus = open(f'{D}/jawiki.txt', encoding='utf-8').read()[:-1_000_000]
idx = {c: i for i, c in enumerate(vocab)}
code = open(f'{D}/code.txt', encoding='utf-8').read().replace('\r', '')[-1_000_000:]
code_ids = torch.tensor([idx.get(c, 0) for c in code], dtype=torch.long)


def batches_fixed(data, n, seed):
    g = torch.Generator().manual_seed(seed)
    return list(batches(data, g, n))


VAL, TRAIN, CODE = batches_fixed(val, 60, 1234), batches_fixed(train, 30, 99), batches_fixed(code_ids, 30, 7)


@torch.no_grad()
def metrics(model):
    model.eval()
    r = {}
    nll, top1, top5, n = 0.0, 0, 0, 0
    conf_bins = torch.zeros(15); acc_bins = torch.zeros(15); cnt_bins = torch.zeros(15)
    pos_loss = torch.zeros(CTX)
    for x, y in VAL:
        logits = model(x)
        logp = F.log_softmax(logits, -1)
        l = -logp.gather(-1, y[..., None])[..., 0]
        nll += l.sum().item(); n += y.numel()
        pos_loss += l.sum(0)
        top = logits.topk(5, -1).indices
        top1 += (top[..., 0] == y).sum().item(); top5 += (top == y[..., None]).any(-1).sum().item()
        conf = logp.exp().max(-1).values.flatten(); correct = (top[..., 0] == y).flatten().float()
        b = (conf * 15).clamp(max=14).long()
        conf_bins.index_add_(0, b, conf); acc_bins.index_add_(0, b, correct); cnt_bins.index_add_(0, b, torch.ones_like(conf))
    r['val_bpc'] = nll / n / math.log(2)
    r['top1'] = top1 / n; r['top5'] = top5 / n
    r['ece'] = ((conf_bins - acc_bins).abs().sum() / cnt_bins.sum()).item()   # = Σ_bin |平均確信度 − 正解率| × 割合
    pos = pos_loss / len(VAL) / BATCH / math.log(2)
    r['bpc_pos_0_15'] = pos[:16].mean().item(); r['bpc_pos_96_127'] = pos[96:].mean().item()
    tl = sum(F.cross_entropy(model(x).flatten(0, 1), y.flatten()).item() for x, y in TRAIN) / len(TRAIN) / math.log(2)
    r['gap'] = r['val_bpc'] - tl
    r['code_bpc'] = sum(F.cross_entropy(model(x).flatten(0, 1), y.flatten()).item() for x, y in CODE) / len(CODE) / math.log(2)
    x1 = VAL[0][0][:1]
    t0 = time.time()
    for _ in range(20):
        model(x1)
    r['chars_per_s'] = 20 * CTX / (time.time() - t0)
    # generation: 6 samples x 200 chars at T = 0.8
    g = torch.Generator().manual_seed(0)
    texts = []
    for p in ['日本の首都は', '東京大学は', '富士山は', '第二次世界大戦', 'この映画は', '経済学では']:
        ids = [idx.get(c, 0) for c in p]
        for _ in range(200):
            lg = model(torch.tensor([ids[-CTX:]]))[0, -1]
            ids.append(int(torch.multinomial(F.softmax(lg / 0.8, -1), 1, generator=g)))
        texts.append(''.join(vocab[i] for i in ids[len(p):]))
    grams = lambda t, k: [t[i:i + k] for i in range(len(t) - k + 1)]
    all2 = [g2 for t in texts for g2 in grams(t, 2)]
    r['distinct2'] = len(set(all2)) / len(all2)
    rep = [sum(c - 1 for c in __import__('collections').Counter(grams(t, 4)).values()) / max(len(grams(t, 4)), 1) for t in texts]
    r['repeat4'] = sum(rep) / len(rep)
    g16 = [g_ for t in texts for g_ in grams(t, 16)[::4]]
    r['copy16'] = sum(g_ in corpus for g_ in g16) / len(g16)
    return r


out = {}
for spec in argv:
    name, rest = spec.split('=')
    size, kind, ckpt = rest.split(':')
    model = GPT(len(vocab), *SIZES[size], kind)
    model.load_state_dict(torch.load(f'{D}/{ckpt}'))
    out[name] = metrics(model)
    print(name, json.dumps({k: round(v, 4) for k, v in out[name].items()}), flush=True)
prev = {}
try:
    prev = json.load(open(f'{D}/metrics.json'))
except FileNotFoundError:
    pass
prev.update(out)
json.dump(prev, open(f'{D}/metrics.json', 'w'), indent=1)
