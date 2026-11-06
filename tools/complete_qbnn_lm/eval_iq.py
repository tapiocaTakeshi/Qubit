"""「賢さ」のテスト: 選択肢ごとの尤度で答えを選ばせる（ゼロショット）。
    python eval_iq.py NAME=SIZE:KIND:CKPT ...
"""
import json, math, random, re, sys
import torch, torch.nn.functional as F
argv = sys.argv[1:]; sys.argv = ['x']
exec(open(__import__('os').path.join(__import__('os').path.dirname(__import__('os').path.abspath(__file__)), 'scale_exp.py')).read()
     .replace("if __name__ == '__main__':\n    main()", ''))
torch.set_num_threads(4)
vocab = json.load(open(f'{D}/vocab.json', encoding='utf-8'))
idx = {c: i for i, c in enumerate(vocab)}
enc = lambda s: [idx.get(c, 0) for c in s]
rng = random.Random(0)

CAPITALS = {'日本': '東京', 'アメリカ合衆国': 'ワシントン', 'イギリス': 'ロンドン', 'フランス': 'パリ', 'ドイツ': 'ベルリン',
            'イタリア': 'ローマ', 'スペイン': 'マドリード', 'ロシア': 'モスクワ', '中国': '北京', '韓国': 'ソウル',
            'インド': 'ニューデリー', 'エジプト': 'カイロ', 'カナダ': 'オタワ', 'オーストラリア': 'キャンベラ', 'ブラジル': 'ブラジリア',
            'アルゼンチン': 'ブエノスアイレス', 'メキシコ': 'メキシコシティ', 'タイ': 'バンコク', 'ベトナム': 'ハノイ', 'インドネシア': 'ジャカルタ',
            'フィリピン': 'マニラ', 'トルコ': 'アンカラ', 'ギリシャ': 'アテネ', 'ポルトガル': 'リスボン', 'オランダ': 'アムステルダム',
            'ベルギー': 'ブリュッセル', 'スイス': 'ベルン', 'オーストリア': 'ウィーン', 'スウェーデン': 'ストックホルム', 'ノルウェー': 'オスロ',
            'フィンランド': 'ヘルシンキ', 'デンマーク': 'コペンハーゲン', 'ポーランド': 'ワルシャワ', 'ハンガリー': 'ブダペスト', 'チェコ': 'プラハ',
            'イラン': 'テヘラン', 'イラク': 'バグダード', 'ケニア': 'ナイロビ', 'ペルー': 'リマ', 'チリ': 'サンティアゴ'}
PREF = {'北海道': '札幌市', '青森県': '青森市', '岩手県': '盛岡市', '宮城県': '仙台市', '秋田県': '秋田市', '山形県': '山形市',
        '福島県': '福島市', '茨城県': '水戸市', '栃木県': '宇都宮市', '群馬県': '前橋市', '埼玉県': 'さいたま市', '千葉県': '千葉市',
        '東京都': '新宿区', '神奈川県': '横浜市', '新潟県': '新潟市', '富山県': '富山市', '石川県': '金沢市', '福井県': '福井市',
        '山梨県': '甲府市', '長野県': '長野市', '岐阜県': '岐阜市', '静岡県': '静岡市', '愛知県': '名古屋市', '三重県': '津市',
        '滋賀県': '大津市', '京都府': '京都市', '大阪府': '大阪市', '兵庫県': '神戸市', '奈良県': '奈良市', '和歌山県': '和歌山市',
        '鳥取県': '鳥取市', '島根県': '松江市', '岡山県': '岡山市', '広島県': '広島市', '山口県': '山口市', '徳島県': '徳島市',
        '香川県': '高松市', '愛媛県': '松山市', '高知県': '高知市', '福岡県': '福岡市', '佐賀県': '佐賀市', '長崎県': '長崎市',
        '熊本県': '熊本市', '大分県': '大分市', '宮崎県': '宮崎市', '鹿児島県': '鹿児島市', '沖縄県': '那覇市'}


def mc(table, template):
    keys, items = list(table), []
    for k in keys:
        wrong = rng.sample([table[o] for o in keys if o != k], 4)
        choices = [table[k]] + wrong
        rng.shuffle(choices)
        items.append((template.format(k), choices, choices.index(table[k])))
    return items


jcqa = [(r['question'] + '答えは', [r[f'choice{i}'] for i in range(5)], r['label'])
        for r in json.load(open(f'{D}/jcqa.json', encoding='utf-8'))]
TESTS = {
    'JCommonsenseQA': jcqa,
    '首都': mc(CAPITALS, '{}の首都は'),
    '県庁所在地': mc(PREF, '{}の県庁所在地は'),
    '足し算': [(f'{a}+{b}=', [str(c) for c in range(19)], a + b) for a, b in
             [(rng.randrange(10), rng.randrange(10)) for _ in range(100)]],
}
# 助詞: 検証用の文章で助詞の位置を見つけ、7つの候補から選ばせる
val_text = open(f'{D}/jawiki.txt', encoding='utf-8').read()[-1_000_000:]
PARTICLES = ['は', 'が', 'を', 'に', 'で', 'と', 'の']
part = []
for m in re.finditer(r'[一-龯ァ-ヶー]{2,}([はがをにでとの])[一-龯ァ-ヶぁ-ん]', val_text):
    start = max(0, m.start(1) - 60)
    part.append((val_text[start:m.start(1)], PARTICLES, PARTICLES.index(m.group(1)), val_text[m.start(1) + 1:m.start(1) + 6]))
    if len(part) >= 400:
        break
TESTS['助詞'] = part
CHANCE = {'JCommonsenseQA': 0.2, '首都': 0.2, '県庁所在地': 0.2, '足し算': 1 / 19, '助詞': 1 / 7}


@torch.no_grad()
def score(model, ctx, cont):
    """log P(cont | ctx) の1文字あたり平均。"""
    ids = enc(ctx + cont)[-CTX:]
    n = len(enc(cont))
    logp = F.log_softmax(model(torch.tensor([ids[:-1]]))[0], -1)
    tgt = torch.tensor(ids[1:])
    lp = logp.gather(-1, tgt[:, None])[:, 0]
    return lp[-n:].mean().item()


def run(model):
    res = {}
    for name, items in TESTS.items():
        ok = 0
        for item in items:
            ctx, choices, ans = item[:3]
            tail = item[3] if len(item) > 3 else ''
            s = [score(model, ctx, c + tail) for c in choices]       # 助詞は直後の5文字まで含めて比べる
            ok += max(range(len(choices)), key=lambda i: s[i]) == ans
        res[name] = ok / len(items)
    return res


out = {}
for spec in argv:
    name, rest = spec.split('=')
    size, kind, ckpt = rest.split(':')[:3]
    if rest.endswith(':sft'):        # 会話モデル: 特殊トークン入りの語彙（文字の番号は同じ）と文脈 256
        CTX = 256
        model = GPT(len(vocab) + 4, *SIZES[size], kind)
    else:
        CTX = 128
        model = GPT(len(vocab), *SIZES[size], kind)
    model.load_state_dict(torch.load(f'{D}/{ckpt}')); model.eval()
    out[name] = run(model)
    print(name, json.dumps({k: round(v, 3) for k, v in out[name].items()}, ensure_ascii=False), flush=True)
print('chance', json.dumps({k: round(v, 3) for k, v in CHANCE.items()}, ensure_ascii=False))
prev = {}
try:
    prev = json.load(open(f'{D}/iq.json'))
except FileNotFoundError:
    pass
prev.update(out)
json.dump(prev, open(f'{D}/iq.json', 'w'), indent=1, ensure_ascii=False)
