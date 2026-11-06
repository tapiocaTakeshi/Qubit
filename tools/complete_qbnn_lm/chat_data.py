"""会話データ (oasst1/oasst2 日本語) を <bos>ユーザー<sep>アシスタント<eos> の列にする（文脈 256 文字に収める）."""
import json

CTX = 256


def cut(text, budget):
    """budget 文字以内に収まる最後の「。」で切る。収まらなければ None."""
    if len(text) <= budget:
        return text
    head = text[:budget]
    k = head.rfind('。')
    return head[:k + 1] if k >= 20 else None


def conversations(paths):
    for p in paths:
        for row in json.load(open(p, encoding='utf-8')):
            yield row['conversations']


def build(paths, encode, BOS, SEP, EOS):
    samples, turns_hist = [], []
    for conv in conversations(paths):
        ids, mask = [], []
        for i in range(0, len(conv) - 1, 2):
            u, a = conv[i], conv[i + 1]
            if u['from'] != 'human' or a['from'] != 'gpt':
                break
            user = u['value'].strip()
            room = CTX - len(ids) - len(user) - 3
            if len(user) > 150 or room < 20:
                break
            reply = cut(a['value'].strip(), room)
            if not reply:
                break
            ids += [BOS] + encode(user) + [SEP] + encode(reply) + [EOS]
            mask += [0] * (len(user) + 2) + [1] * (len(reply) + 1)
            if len(reply) < len(a['value'].strip()):
                break                  # 切った応答の後には続けない
        if ids:
            samples.append((ids, mask))
            turns_hist.append(ids.count(BOS))
    return samples, turns_hist


if __name__ == '__main__':
    from collections import Counter
    s, t = build(['oasst1-chat-44k-ja.json', 'oasst2-chat-68k-ja.json'], lambda x: list(x), 'B', 'S', 'E')
    print(len(s), 'conversations; turns:', sorted(Counter(t).items())[:5], '; mean length', sum(len(i) for i, _ in s) / len(s))
