"""(cont2) 新しいデータ（青空文庫・多段会話・クイズ）を加えた追加学習。元: Continued training of the chat model: half plain text (new Wikipedia + code), half conversation/instruction.

Plain-text rows: loss on every character. Chat rows: loss on replies and <eos> only.
Evaluates on held-out Wikipedia, code, conversation and instruction data. Resumable every 300 steps.
    python cont.py STEPS
"""
import json, math, os, random, sys, time
import torch, torch.nn.functional as F
sys.argv_real = list(sys.argv); sys.argv = ['x']
exec(open(__import__('os').path.join(__import__('os').path.dirname(__import__('os').path.abspath(__file__)), 'scale_exp.py')).read()
     .replace("if __name__ == '__main__':\n    main()", ''))
sys.path.insert(0, D)
from chat_data import build

# 使い方: python cont.py STEPS [INIT_CKPT LM_ROWS CHAT_ROWS LR TAG]（省略時はフェーズ1の設定）
args = sys.argv_real[1:] + [None] * 9
STEPS = int(args[0] or 8000)
INIT = args[1] or 'model_M_complete_chat3000.pt'
LM_ROWS, CHAT_ROWS = int(args[2] or 8), int(args[3] or 8)
LR = float(args[4] or 4e-4)
TAG = args[5] or ''
WIKI = args[6] or 'jawiki2.txt'
SIZE = args[7] or 'M'                 # モデルの大きさ（L は 1,065 万パラメータ）
WARM = int(args[8] or 150)      # 普通の文章として読む Wikipedia（フェーズ4以降は新しい記事）
CTX = 256
torch.set_num_threads(4)
random.seed(2)
meta = json.load(open(f'{D}/sft_meta.json', encoding='utf-8'))
vocab = meta['vocab']
idx = {c: i for i, c in enumerate(vocab)}
BOS, SEP, EOS, PAD = (idx[s] for s in meta['special'])
encode = lambda s: [idx.get(c, 0) for c in s]
as_ids = lambda s: torch.tensor(encode(s), dtype=torch.long)

# WIKI は「a.txt+b.txt」のように + で複数指定でき、普通の文章の行を均等に分ける
text_srcs = [as_ids(open(f'{D}/{f}', encoding='utf-8').read()) for f in WIKI.split('+')]
aoz_all = as_ids(open(f'{D}/aozora.txt', encoding='utf-8').read())       # 新: 青空文庫（最後の 100 万文字は検証用）
text_srcs.append(aoz_all[:-1_000_000]); aoz_val = aoz_all[-1_000_000:]
wiki_new = text_srcs[0]
wiki_val = as_ids(open(f'{D}/jawiki.txt', encoding='utf-8').read()[-1_000_000:])
web_val = as_ids(open(f'{D}/fineweb_val.txt', encoding='utf-8').read()) if os.path.exists(f'{D}/fineweb_val.txt') else None
code_all = as_ids(open(f'{D}/code.txt', encoding='utf-8').read().replace('\r', ''))
code_train, code_val = code_all[:-1_000_000], code_all[-1_000_000:]

chat, _ = build([f'{D}/oasst1-chat-44k-ja.json', f'{D}/oasst2-chat-68k-ja.json'], encode, BOS, SEP, EOS)
chat = list({tuple(i): (i, m) for i, m in chat}.values())
random.Random(1).shuffle(chat)                     # sft_chat.py と同じ分け方（検証用 600 件）
chat_val, chat_train = chat[:600], chat[600:]
cw, _ = build([f'{D}/conv_wiki.json'], encode, BOS, SEP, EOS)             # 新: Wikipedia 話題の多段会話
random.Random(3).shuffle(cw)
cw_val = cw[:600]; chat_train = chat_train + cw[600:]
instr = []
for name, rows in json.load(open(f'{D}/instruct_raw.json', encoding='utf-8')).items():
    for r in rows:
        ins = (r.get('instruction') or '').strip() + (('\n' + r['input'].strip()) if (r.get('input') or '').strip() else '')
        out = (r.get('output') or '').strip()
        if ins and out and len(ins) + len(out) + 3 <= CTX:
            instr += [([BOS] + encode(ins) + [SEP] + encode(out) + [EOS], [0] * (len(ins) + 2) + [1] * (len(out) + 1))] * (3 if 'code' in name else 1)
random.Random(1).shuffle(instr)
instr_val, instr_train = instr[:600], instr[600:]
lj = [([BOS] + encode(r['instruction']) + [SEP] + encode(r['output']) + [EOS], [0] * (len(r['instruction']) + 2) + [1] * (len(r['output']) + 1))
      for r in json.load(open(f'{D}/llmjp_mix.json', encoding='utf-8'))]   # 新: クイズ・用語説明など
lj_val = lj[:600]; instr_train = instr_train + lj[600:]
print(f'text {" + ".join(f"{len(t):,}" for t in text_srcs)} chars, code {len(code_train):,}, chat {len(chat_train)}, instruction {len(instr_train)}', flush=True)


def lm_rows(src, n, rng):
    s = [rng.randrange(len(src) - CTX - 1) for _ in range(n)]
    chunk = torch.stack([src[i:i + CTX + 1] for i in s])
    return chunk[:, :-1], chunk[:, 1:], torch.ones(n, CTX, dtype=torch.bool)


def chat_rows(pools, probs, n, rng):
    X = torch.full((n, CTX + 1), PAD, dtype=torch.long)
    M = torch.zeros((n, CTX + 1), dtype=torch.bool)
    for r in range(n):
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


def batch(rng):
    n_code = LM_ROWS // 4
    n_text = LM_ROWS - n_code
    parts = [lm_rows(src, n_text // len(text_srcs) + (k < n_text % len(text_srcs)), rng) for k, src in enumerate(text_srcs)]
    parts += [lm_rows(code_train, n_code, rng), chat_rows([chat_train, instr_train], [0.5, 0.5], CHAT_ROWS, rng)]
    return tuple(torch.cat(t) for t in zip(*parts))


@torch.no_grad()
def evaluate_all(model):
    model.eval()
    res = {}
    extra = [('web', lambda r: lm_rows(web_val, 16, r))] if web_val is not None else []
    for name, make in extra + [('wiki', lambda r: lm_rows(wiki_val, 16, r)), ('code', lambda r: lm_rows(code_val, 16, r)),
                       ('chat', lambda r: chat_rows([chat_val, chat_val], [1, 0], 16, r)),
                       ('instruction', lambda r: chat_rows([instr_val, instr_val], [1, 0], 16, r)),
                       ('aozora', lambda r: lm_rows(aoz_val, 16, r)), ('conv_wiki', lambda r: chat_rows([cw_val, cw_val], [1, 0], 16, r)),
                       ('quiz', lambda r: chat_rows([lj_val, lj_val], [1, 0], 16, r))]:
        rng, tot, n = random.Random(123), 0.0, 0.0
        for _ in range(12):
            x, y, m = make(rng)
            loss = F.cross_entropy(model(x).flatten(0, 1), y.flatten(), reduction='none')
            tot += (loss * m.flatten()).sum().item(); n += m.sum().item()
        res[name] = round(tot / n / math.log(2), 3)
    model.train()
    return res


model = GPT(len(vocab), *SIZES[SIZE], 'complete')
print(f'{SIZE}: {sum(p.numel() for p in model.parameters()):,} parameters', flush=True)
CK = f'{D}/cont{TAG}_resume.pt'
opt = torch.optim.AdamW(model.parameters(), lr=LR, betas=(0.9, 0.95), weight_decay=0.1)
warm = WARM
sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min((s + 1) / warm, 0.1 + 0.45 * (1 + math.cos(math.pi * min(s, STEPS) / STEPS))))
rng = random.Random(21)
if os.path.exists(CK):
    ck = torch.load(CK, weights_only=False)
    model.load_state_dict(ck['model']); opt.load_state_dict(ck['opt']); sched.load_state_dict(ck['sched'])
    rng.setstate(ck['rng']); start, log = ck['step'] + 1, ck['log']
    print(f'resumed at step {start}', flush=True)
else:
    if INIT != 'scratch':            # scratch なら乱数の重みから学習する
        model.load_state_dict(torch.load(f'{D}/{INIT}'))
    start, log = 1, [(0, evaluate_all(model))]
    print('before:', log[0][1], flush=True)

t0 = time.time()
for step in range(start, STEPS + 1):
    x, y, m = batch(rng)
    loss = F.cross_entropy(model(x).flatten(0, 1), y.flatten(), reduction='none')
    loss = (loss * m.flatten()).sum() / m.sum()
    opt.zero_grad(); loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step(); sched.step()
    if step % 1000 == 0 or step == STEPS:
        log.append((step, evaluate_all(model)))
        print(f'step {step}:', log[-1][1], f'({time.time() - t0:.0f}s)', flush=True)
    if step % 300 == 0 and step < STEPS:
        torch.save({'model': model.state_dict(), 'opt': opt.state_dict(), 'sched': sched.state_dict(),
                    'rng': rng.getstate(), 'step': step, 'log': log}, CK + '.tmp')
        os.replace(CK + '.tmp', CK)
torch.save(model.state_dict(), f'{D}/model_M_complete_cont{TAG}{STEPS}.pt')
json.dump({'log': log}, open(f'{D}/cont{TAG}_log.json', 'w'))
print('CONT_DONE', flush=True)
