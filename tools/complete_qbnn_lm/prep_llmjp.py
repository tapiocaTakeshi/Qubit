"""llm-japanese-dataset から種類ごとに上限を決めて抜き出す（翻訳ばかりにならないように）。"""
import json, random, collections
import os
DATA = os.environ.get('QBNN_LM_DATA', os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data'))
DL = os.environ.get('QBNN_LM_DOWNLOADS', os.path.join(DATA, 'downloads'))   # Hugging Face から落としたファイル
vocab = set(json.load(open(f'{DATA}/vocab.json', encoding='utf-8')))
CAP = {'translate': 25_000, 'explain': 120_000, 'fix': 20_000, 'easy': 20_000, 'quiz': 10**9, 'other': 60_000}
def kind(ins):
    head = ins.split('\n')[0]
    if '翻訳' in head: return 'translate'
    if head.startswith('入力されたワードを説明'): return 'explain'
    if head.startswith('次の日本語には間違っている'): return 'fix'
    if 'やさしい日本語' in head: return 'easy'
    if '\n' not in ins: return 'quiz'
    return 'other'
pools = collections.defaultdict(list)
rng = random.Random(0)
seen = collections.Counter()
with open(f'{DL}/llmjp.jsonl', encoding='utf-8') as f:
    for line in f:
        r = json.loads(line)
        ins = (r.get('instruction') or '').strip() + (('\n' + r['input'].strip()) if (r.get('input') or '').strip() else '')
        out = (r.get('output') or '').strip()
        if not ins or not out or len(ins) + len(out) + 3 > 256 or any(c not in vocab for c in ins + out):
            continue
        k = kind(ins); seen[k] += 1; p = pools[k]
        if len(p) < CAP[k]: p.append({'instruction': ins, 'output': out})
        else:                                   # 貯水池抽出で全体から均等に選ぶ
            j = rng.randrange(seen[k])
            if j < CAP[k]: p[j] = {'instruction': ins, 'output': out}
rows = [r for p in pools.values() for r in p]
rng.shuffle(rows)
json.dump(rows, open(f'{DATA}/llmjp_mix.json', 'w', encoding='utf-8'), ensure_ascii=False)
print({k: (len(v), seen[k]) for k, v in pools.items()}, 'total', len(rows))
for r in pools['quiz'][:6] + pools['explain'][:3]:
    print('  ', r['instruction'][:50], '→', r['output'][:50].replace('\n', ' '))
