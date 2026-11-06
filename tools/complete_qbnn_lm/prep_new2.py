"""新しい学習データ（第3弾）: hh-rlhf 日本語・Magpie-Tanuki・Nemotron 多段会話・JaQuAD（閉じた質問応答）・Wikinews。"""
import glob, json, random, sys
import pandas as pd
DATA, DL = sys.argv[1], sys.argv[2]   # 学習データの場所, ダウンロードの場所
vocab = set(json.load(open(f'{DATA}/vocab.json', encoding='utf-8')))
ok = lambda t: all(c in vocab for c in t)

def conv(msgs):          # role/content → oasst と同じ from/value 形式
    out = []
    for m in msgs:
        r = m['role']
        if r == 'system':
            continue
        out.append({'from': 'human' if r == 'user' else 'gpt', 'value': m['content']})
    return out

# 会話: Magpie-Tanuki（主に1往復）＋ Nemotron（多段）
chats = []
for p in sorted(glob.glob(f'{DL}/tanuki*.parquet')):
    for msgs in pd.read_parquet(p)['messages']:
        c = conv(list(msgs))
        if len(c) >= 2 and ok(''.join(x['value'] for x in c[:2])):
            chats.append({'conversations': c})
n_t = len(chats)
for line in open(f'{DL}/nemo.jsonl', encoding='utf-8'):
    c = conv(json.loads(line)['messages'])
    if len(c) >= 2 and ok(c[0]['value'] + c[1]['value']):
        chats.append({'conversations': c})
print('chat: tanuki', n_t, 'nemotron', len(chats) - n_t)
random.Random(0).shuffle(chats)
json.dump(chats, open(f'{DATA}/chat_new3.json', 'w', encoding='utf-8'), ensure_ascii=False)

# 指示: hh-rlhf 日本語（短いものだけ）＋ JaQuAD の質問 → 答え（文脈なしで答えさせる）
instr = []
for r in json.load(open(f'{DL}/hh.json', encoding='utf-8')):
    ins, out = (r.get('instruction') or '').strip(), (r.get('output') or '').strip()
    if ins and out and len(ins) + len(out) + 3 <= 256 and ok(ins + out):
        instr.append({'instruction': ins, 'output': out, 'src': 'hh'})
n_h = len(instr)
def jaquad(files, tag):
    rows = []
    for f in files:
        for art in json.load(open(f, encoding='utf-8'))['data']:
            for para in art['paragraphs']:
                for qa in para['qas']:
                    q, a = qa['question'].strip(), qa['answers'][0]['text'].strip()
                    if q and a and ok(q + a):
                        rows.append({'instruction': q, 'output': a, 'src': tag})
    return rows
instr += jaquad(sorted(glob.glob(f'{DL}/jaquad_train_*.json')), 'jaquad')
print('instruction: hh', n_h, 'jaquad', len(instr) - n_h)
random.Random(0).shuffle(instr)
json.dump(instr, open(f'{DATA}/instr_new3.json', 'w', encoding='utf-8'), ensure_ascii=False)
jv = jaquad(sorted(glob.glob(f'{DL}/jaquad_dev_*.json')), 'jaquad')
random.Random(0).shuffle(jv)
json.dump(jv[:600], open(f'{DATA}/jaquad_val.json', 'w', encoding='utf-8'), ensure_ascii=False)
print('jaquad val', len(jv))

# 文章: Wikinews を青空文庫の先頭に足す（青空文庫の最後 100 万文字は検証用のまま）
news = '\n\n'.join(t.strip() for t in pd.read_parquet(f'{DL}/news.parquet')['text'])
aoz = open(f'{DATA}/aozora.txt', encoding='utf-8').read()
open(f'{DATA}/news_aozora.txt', 'w', encoding='utf-8').write(news + '\n\n' + aoz)
print('news chars', len(news))
for r in instr[:4] + [x for x in instr if x['src'] == 'jaquad'][:4]:
    print('  ', r['instruction'][:50].replace('\n', ' '), '→', r['output'][:40].replace('\n', ' '))
