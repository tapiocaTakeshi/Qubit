"""新しい学習データの準備: 青空文庫（文章）、Wikipedia 話題の多段会話、llm-japanese-dataset（クイズ・指示）。"""
import gzip, json, random, sys
import os
DATA = os.environ.get('QBNN_LM_DATA', os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data'))
DL = os.environ.get('QBNN_LM_DOWNLOADS', os.path.join(DATA, 'downloads'))   # Hugging Face から落としたファイル
vocab = set(json.load(open(f'{DATA}/vocab.json', encoding='utf-8')))
unk = lambda t: sum(c not in vocab for c in t) / max(len(t), 1)

# 青空文庫: 語彙外の文字が多い作品（旧字体など）は除く
out, n, skipped = [], 0, 0
with gzip.open(f'{DL}/aozora.jsonl.gz', 'rt', encoding='utf-8') as f:
    for line in f:
        t = json.loads(line)['text'].strip()
        if unk(t) > 0.01:
            skipped += 1; continue
        out.append(t); n += len(t)
        if n > 150_000_000:
            break
random.Random(0).shuffle(out)
open(f'{DATA}/aozora.txt', 'w', encoding='utf-8').write('\n\n'.join(out))
print(f'aozora: {len(out)} works, {n:,} chars, skipped {skipped}', flush=True)

# Wikipedia 話題の多段会話: そのまま oasst と同じ形式
conv = json.load(open(f'{DL}/conv_wiki.json', encoding='utf-8'))
print('conv_wiki:', len(conv), 'conversations', flush=True)

# llm-japanese-dataset: 短い組だけ、語彙内の文字だけのものから 25 万件
rows = []
with open(f'{DL}/llmjp.jsonl', encoding='utf-8') as f:
    for line in f:
        r = json.loads(line)
        ins = (r.get('instruction') or '').strip() + (('\n' + r['input'].strip()) if (r.get('input') or '').strip() else '')
        out_ = (r.get('output') or '').strip()
        if ins and out_ and len(ins) + len(out_) + 3 <= 256 and unk(ins + out_) == 0:
            rows.append({'instruction': ins, 'output': out_})
print('llmjp usable rows:', len(rows), flush=True)
random.Random(0).shuffle(rows)
json.dump(rows[:250_000], open(f'{DATA}/llmjp_250k.json', 'w', encoding='utf-8'), ensure_ascii=False)
for r in rows[:8]:
    print('  ', r['instruction'][:60].replace('\n', ' '), '→', r['output'][:40].replace('\n', ' '))
