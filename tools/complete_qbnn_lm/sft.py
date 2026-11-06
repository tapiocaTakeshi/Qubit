"""Instruction tuning of the M complete-Bloch GPT: <bos>instruction<sep>response<eos>, context 128 -> 256.

Loss only on the response and <eos>. Samples are packed whole into 256-character rows (padding is masked).
Checkpoints every 300 steps so a container restart resumes where it stopped.
    python sft.py STEPS
"""
import json, math, os, random, sys, time
import torch, torch.nn.functional as F
sys.argv_real = list(sys.argv); sys.argv = ['x']
exec(open(__import__('os').path.join(__import__('os').path.dirname(__import__('os').path.abspath(__file__)), 'scale_exp.py')).read()
     .replace("if __name__ == '__main__':\n    main()", ''))

STEPS = int(sys.argv_real[1]) if len(sys.argv_real) > 1 else 4000
CTX = 256                      # GPT() reads the global CTX for its position table
ROWS = 16                      # 16 x 256 = 4096 characters per step, as in pretraining
SPECIAL = ['<bos>', '<sep>', '<eos>', '<pad>']
torch.set_num_threads(4)
random.seed(0)

base_vocab = json.load(open(f'{D}/vocab.json', encoding='utf-8'))
vocab = base_vocab + SPECIAL
idx = {c: i for i, c in enumerate(vocab)}
BOS, SEP, EOS, PAD = (idx[s] for s in SPECIAL)

raw = json.load(open(f'{D}/instruct_raw.json', encoding='utf-8'))
samples = []
for name, rows in raw.items():
    for r in rows:
        ins = (r.get('instruction') or '').strip()
        extra = (r.get('input') or '').strip()
        if extra:
            ins += '\n' + extra
        out = (r.get('output') or '').strip()
        if not ins or not out or len(ins) + len(out) + 3 > CTX:
            continue
        ids = [BOS] + [idx.get(c, 0) for c in ins] + [SEP] + [idx.get(c, 0) for c in out] + [EOS]
        mask = [0] * (len(ins) + 2) + [1] * (len(out) + 1)     # 応答と <eos> だけで損失を取る
        samples += [(ids, mask)] * (3 if 'code' in name else 1)
random.shuffle(samples)
val, train = samples[:600], samples[600:]
print(f'{len(train)} train / {len(val)} val samples', flush=True)


def pack(pool, rng, rows=ROWS):
    """Whole samples packed into rows of CTX+1 tokens; targets masked outside responses."""
    X = torch.full((rows, CTX + 1), PAD, dtype=torch.long)
    M = torch.zeros((rows, CTX + 1), dtype=torch.bool)
    for r in range(rows):
        pos = 0
        while True:
            ids, mask = pool[rng.randrange(len(pool))]
            if pos + len(ids) > CTX + 1:
                break
            X[r, pos:pos + len(ids)] = torch.tensor(ids)
            M[r, pos:pos + len(ids)] = torch.tensor(mask, dtype=torch.bool)
            pos += len(ids)
    return X[:, :-1], X[:, 1:], M[:, 1:]


def masked_loss(model, x, y, m):
    logits = model(x)
    loss = F.cross_entropy(logits.flatten(0, 1), y.flatten(), reduction='none')
    return (loss * m.flatten()).sum() / m.sum()


@torch.no_grad()
def evaluate_sft(model):
    model.eval()
    rng = random.Random(123)
    tot = n = 0.0
    for _ in range(20):
        x, y, m = pack(val, rng)
        logits = model(x)
        loss = F.cross_entropy(logits.flatten(0, 1), y.flatten(), reduction='none')
        tot += (loss * m.flatten()).sum().item(); n += m.sum().item()
    model.train()
    return tot / n / math.log(2)


model = GPT(len(vocab), *SIZES['M'], 'complete')
CK = f'{D}/sft_resume.pt'
opt = torch.optim.AdamW(model.parameters(), lr=5e-4, betas=(0.9, 0.95), weight_decay=0.1)
warm = 100
sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min((s + 1) / warm, 0.5 * (1 + math.cos(math.pi * min(s, STEPS) / STEPS))))
rng = random.Random(7)
if os.path.exists(CK):
    ck = torch.load(CK, weights_only=False)
    model.load_state_dict(ck['model']); opt.load_state_dict(ck['opt']); sched.load_state_dict(ck['sched'])
    rng.setstate(ck['rng']); start, log = ck['step'] + 1, ck['log']
    print(f'resumed at step {start}', flush=True)
else:
    base = torch.load(f'{D}/model_M_complete_6000_code3000.pt')
    sd = model.state_dict()
    for k, v in base.items():
        if k == 'tok.weight':          # 既存の文字はそのまま、特殊トークンは平均＋小さな乱数
            sd[k][:len(base_vocab)] = v
            sd[k][len(base_vocab):] = v.mean(0) + 0.02 * torch.randn(len(SPECIAL), v.shape[1])
        elif k == 'pos.weight':        # 位置 128..255 は 0..127 を繰り返して初期化
            sd[k][:v.shape[0]] = v
            sd[k][v.shape[0]:] = v[:CTX - v.shape[0]]
        else:
            sd[k] = v
    model.load_state_dict(sd)
    start, log = 1, [(0, evaluate_sft(model))]
    print(f'before: response loss {log[0][1]:.3f} bpc', flush=True)

t0 = time.time()
for step in range(start, STEPS + 1):
    x, y, m = pack(train, rng)
    loss = masked_loss(model, x, y, m)
    opt.zero_grad(); loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step(); sched.step()
    if step % (STEPS // 8) == 0:
        log.append((step, evaluate_sft(model)))
        print(f'step {step}: response loss {log[-1][1]:.3f} bpc ({time.time() - t0:.0f}s)', flush=True)
    if step % 300 == 0 and step < STEPS:
        torch.save({'model': model.state_dict(), 'opt': opt.state_dict(), 'sched': sched.state_dict(),
                    'rng': rng.getstate(), 'step': step, 'log': log}, CK + '.tmp')
        os.replace(CK + '.tmp', CK)
torch.save(model.state_dict(), f'{D}/model_M_complete_sft{STEPS}.pt')
json.dump({'vocab': vocab, 'special': SPECIAL, 'ctx': CTX, 'log': log}, open(f'{D}/sft_meta.json', 'w', encoding='utf-8'), ensure_ascii=False)
print(f'FINAL response loss {evaluate_sft(model):.3f} bpc', flush=True)
