"""Stage 2: conversation tuning after the instruction model (sft.py).

Conversations (oasst1/oasst2 ja, multi-turn when it fits 256 chars) mixed 60/40 with the instruction data.
Loss on assistant replies and <eos> only. Resumable every 300 steps.
    python sft_chat.py STEPS
"""
import json, math, os, random, sys, time
import torch, torch.nn.functional as F
sys.argv_real = list(sys.argv); sys.argv = ['x']
exec(open(__import__('os').path.join(__import__('os').path.dirname(__import__('os').path.abspath(__file__)), 'scale_exp.py')).read()
     .replace("if __name__ == '__main__':\n    main()", ''))
sys.path.insert(0, D)
from chat_data import build

STEPS = int(sys.argv_real[1]) if len(sys.argv_real) > 1 else 3000
CTX, ROWS, P_CHAT = 256, 16, 0.6
torch.set_num_threads(4)
random.seed(1)
meta = json.load(open(f'{D}/sft_meta.json', encoding='utf-8'))
vocab = meta['vocab']
idx = {c: i for i, c in enumerate(vocab)}
BOS, SEP, EOS, PAD = (idx[s] for s in meta['special'])
encode = lambda s: [idx.get(c, 0) for c in s]

chat, _ = build([f'{D}/oasst1-chat-44k-ja.json', f'{D}/oasst2-chat-68k-ja.json'], encode, BOS, SEP, EOS)
chat = list({tuple(i): (i, m) for i, m in chat}.values())       # oasst2 は oasst1 を含むので重複を除く
random.shuffle(chat)
chat_val, chat_train = chat[:600], chat[600:]

instr = []
for name, rows in json.load(open(f'{D}/instruct_raw.json', encoding='utf-8')).items():
    for r in rows:
        ins = (r.get('instruction') or '').strip() + (('\n' + r['input'].strip()) if (r.get('input') or '').strip() else '')
        out = (r.get('output') or '').strip()
        if ins and out and len(ins) + len(out) + 3 <= CTX:
            instr += [([BOS] + encode(ins) + [SEP] + encode(out) + [EOS], [0] * (len(ins) + 2) + [1] * (len(out) + 1))] * (3 if 'code' in name else 1)
random.shuffle(instr)
instr_val, instr_train = instr[:600], instr[600:]
print(f'chat {len(chat_train)} / instruction {len(instr_train)} train samples', flush=True)


def pack(pools, probs, rng, rows=ROWS):
    X = torch.full((rows, CTX + 1), PAD, dtype=torch.long)
    M = torch.zeros((rows, CTX + 1), dtype=torch.bool)
    for r in range(rows):
        pos = 0
        while True:
            pool = pools[0] if rng.random() < probs[0] else pools[1]
            ids, mask = pool[rng.randrange(len(pool))]
            if pos + len(ids) > CTX + 1:
                break
            X[r, pos:pos + len(ids)] = torch.tensor(ids)
            M[r, pos:pos + len(ids)] = torch.tensor(mask, dtype=torch.bool)
            pos += len(ids)
    return X[:, :-1], X[:, 1:], M[:, 1:]


@torch.no_grad()
def evaluate_pool(model, pool):
    model.eval()
    rng, tot, n = random.Random(123), 0.0, 0.0
    for _ in range(15):
        x, y, m = pack([pool, pool], [1.0, 0.0], rng)
        loss = F.cross_entropy(model(x).flatten(0, 1), y.flatten(), reduction='none')
        tot += (loss * m.flatten()).sum().item(); n += m.sum().item()
    model.train()
    return tot / n / math.log(2)


model = GPT(len(vocab), *SIZES['M'], 'complete')
CK = f'{D}/sft_chat_resume.pt'
opt = torch.optim.AdamW(model.parameters(), lr=3e-4, betas=(0.9, 0.95), weight_decay=0.1)
warm = 100
sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min((s + 1) / warm, 0.5 * (1 + math.cos(math.pi * min(s, STEPS) / STEPS))))
rng = random.Random(11)
if os.path.exists(CK):
    ck = torch.load(CK, weights_only=False)
    model.load_state_dict(ck['model']); opt.load_state_dict(ck['opt']); sched.load_state_dict(ck['sched'])
    rng.setstate(ck['rng']); start, log = ck['step'] + 1, ck['log']
    print(f'resumed at step {start}', flush=True)
else:
    model.load_state_dict(torch.load(f'{D}/model_M_complete_sft4000.pt'))
    start, log = 1, [(0, evaluate_pool(model, chat_val), evaluate_pool(model, instr_val))]
    print(f'before: chat {log[0][1]:.3f} bpc, instruction {log[0][2]:.3f} bpc', flush=True)

t0 = time.time()
for step in range(start, STEPS + 1):
    x, y, m = pack([chat_train, instr_train], [P_CHAT, 1 - P_CHAT], rng)
    loss = F.cross_entropy(model(x).flatten(0, 1), y.flatten(), reduction='none')
    loss = (loss * m.flatten()).sum() / m.sum()
    opt.zero_grad(); loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step(); sched.step()
    if step % (STEPS // 6) == 0:
        log.append((step, evaluate_pool(model, chat_val), evaluate_pool(model, instr_val)))
        print(f'step {step}: chat {log[-1][1]:.3f} bpc, instruction {log[-1][2]:.3f} bpc ({time.time() - t0:.0f}s)', flush=True)
    if step % 300 == 0 and step < STEPS:
        torch.save({'model': model.state_dict(), 'opt': opt.state_dict(), 'sched': sched.state_dict(),
                    'rng': rng.getstate(), 'step': step, 'log': log}, CK + '.tmp')
        os.replace(CK + '.tmp', CK)
torch.save(model.state_dict(), f'{D}/model_M_complete_chat{STEPS}.pt')
json.dump({'log': log}, open(f'{D}/sft_chat_log.json', 'w'))
print('CHAT_FINAL', json.dumps(log[-1]), flush=True)
