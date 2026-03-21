#!/usr/bin/env python3
"""数学の集中学習スクリプト - データ量・エポック数を大幅増加。"""
import os, sys, json, time, random, math, gc
import requests
import torch
import torch.nn.functional as F
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(__file__))
from neuroquantum_layered import NeuroQuantum, NeuroQuantumConfig, NeuroQuantumTokenizer

CKPT_PATH = os.path.join(os.path.dirname(__file__), "neuroq_checkpoint.pt")
TOKENIZER_PATH = os.path.join(os.path.dirname(__file__), "neuroq_tokenizer.model")
ENDPOINT_URL = "https://vvcci2ps4y3wfx7m.us-east4.gcp.endpoints.huggingface.cloud"

BATCH_SIZE = 4
GRAD_ACCUM = 4
LR = 5e-5
WARMUP_STEPS = 30
MIN_LR_RATIO = 0.05
EPOCHS = 10

# 大量の数学プロンプト（具体的な説明を引き出すように設計）
PROMPTS = [
    # === 算術・基礎 ===
    "1+1の答えは2です。足し算は",
    "2×3=6です。掛け算とは",
    "10÷2=5です。割り算の基本は",
    "分数1/2+1/3の計算方法を詳しく説明すると",
    "小数0.5×0.3の計算手順は",
    "百分率の計算で、50%は半分を意味し",
    "最大公約数の求め方をユークリッドの互除法で説明すると",
    "最小公倍数は2つの数の共通の倍数のうち最小のもので",
    "素数とは1と自分自身以外に約数を持たない自然数で、2, 3, 5, 7, 11などが",
    "因数分解とは多項式を因数の積に分解することで",
    "絶対値|x|とはxの原点からの距離で",
    "四則演算の優先順位は掛け算割り算が先で",
    "負の数同士の掛け算は正になり、例えば(-2)×(-3)=6で",
    "累乗とは同じ数を何回も掛けることで、2の3乗は8で",
    "平方根√4=2で、平方根とは",

    # === 代数 ===
    "一次方程式2x+3=7の解き方は、まず3を移項して2x=4、両辺を2で割ってx=2で",
    "二次方程式ax²+bx+c=0の解の公式はx=(-b±√(b²-4ac))/2aで",
    "連立方程式の加減法では、一方の式を足したり引いたりして変数を消去し",
    "不等式2x>6を解くと、両辺を2で割ってx>3で",
    "二次関数y=ax²+bx+cのグラフは放物線で",
    "指数法則でa^m × a^n = a^(m+n)で",
    "対数log_a(b)とは、a を何乗すると b になるかを表す数で",
    "等差数列の一般項は a_n = a_1 + (n-1)d で、和は S_n = n(a_1 + a_n)/2 で",
    "等比数列の一般項は a_n = a_1 × r^(n-1) で、和は S_n = a_1(r^n - 1)/(r-1) で",
    "二項定理 (a+b)^n = Σ C(n,k) a^(n-k) b^k で",
    "因数定理とは、f(a)=0ならばf(x)は(x-a)で割り切れるという定理で",
    "剰余の定理とは、f(x)を(x-a)で割った余りはf(a)であるという定理で",
    "三次方程式の解法には、因数分解やカルダノの公式があり",
    "複素数z=a+biで、iは虚数単位でi²=-1で",
    "複素数の絶対値|z|=√(a²+b²)で",

    # === 幾何学 ===
    "三角形の面積は底辺×高さ÷2で計算でき",
    "円の面積はπr²で、円周は2πrで",
    "ピタゴラスの定理は直角三角形でa²+b²=c²（cは斜辺）で",
    "三角比でsinθ=対辺/斜辺、cosθ=隣辺/斜辺、tanθ=対辺/隣辺で",
    "sin30°=1/2、cos30°=√3/2、tan30°=1/√3で",
    "sin60°=√3/2、cos60°=1/2、tan60°=√3で",
    "sin90°=1、cos90°=0で",
    "余弦定理はa²=b²+c²-2bc cosAで",
    "正弦定理はa/sinA=b/sinB=c/sinC=2Rで",
    "ベクトルの内積a·b=|a||b|cosθで",
    "台形の面積は(上底+下底)×高さ÷2で",
    "球の体積は(4/3)πr³で、表面積は4πr²で",
    "円錐の体積は(1/3)πr²hで",
    "平行四辺形の面積は底辺×高さで",
    "正三角形の面積は(√3/4)a²で",

    # === 微分積分 ===
    "微分とは関数の変化率を求めることで、f'(x)=lim[h→0](f(x+h)-f(x))/hで",
    "x^nの微分はnx^(n-1)で、例えばx³の微分は3x²で",
    "sinxの微分はcosxで、cosxの微分は-sinxで",
    "e^xの微分はe^xで、自然対数lnxの微分は1/xで",
    "積の微分法則は(fg)'=f'g+fg'で",
    "商の微分法則は(f/g)'=(f'g-fg')/g²で",
    "合成関数の微分は連鎖律で、(f(g(x)))'=f'(g(x))·g'(x)で",
    "不定積分∫x^n dx = x^(n+1)/(n+1) + C (n≠-1)で",
    "∫sinx dx = -cosx + Cで、∫cosx dx = sinx + Cで",
    "∫e^x dx = e^x + Cで、∫1/x dx = ln|x| + Cで",
    "定積分∫[a,b]f(x)dxは、関数f(x)のaからbまでの面積を表し",
    "部分積分法は∫f·g'dx = fg - ∫f'·gdxで",
    "置換積分法はu=g(x)と置いて∫f(g(x))g'(x)dx=∫f(u)duで",
    "テイラー展開はf(x)=f(a)+f'(a)(x-a)+f''(a)(x-a)²/2!+...で",
    "マクローリン展開はa=0のテイラー展開で、e^x=1+x+x²/2!+x³/3!+...で",

    # === 線形代数 ===
    "行列の掛け算では、左の行列の行と右の行列の列の内積を計算し",
    "2×2行列[[a,b],[c,d]]の行列式はad-bcで",
    "逆行列は元の行列と掛けると単位行列になる行列で",
    "固有値λと固有ベクトルvはAv=λvを満たし",
    "単位行列Eは対角成分が全て1で、それ以外が0の行列で",
    "転置行列は行と列を入れ替えた行列で",
    "連立一次方程式は行列を使ってAx=bと表現でき",
    "行列のランクは行列の独立な行（または列）の最大数で",

    # === 確率・統計 ===
    "確率とはある事象が起こる可能性の度合いで、0から1の値を取り",
    "サイコロで1が出る確率は1/6で",
    "条件付き確率P(A|B)=P(A∩B)/P(B)で",
    "ベイズの定理はP(A|B)=P(B|A)P(A)/P(B)で",
    "期待値E(X)=Σx_i·P(x_i)で、確率変数の平均的な値を表し",
    "分散V(X)=E((X-μ)²)=E(X²)-(E(X))²で",
    "標準偏差σ=√V(X)で、データのばらつきの度合いを表し",
    "正規分布N(μ,σ²)は平均μ、分散σ²の釣鐘型の分布で",
    "二項分布B(n,p)はn回の試行で成功確率pの分布で",
    "ポアソン分布はまれな事象の発生回数の分布で",

    # === 数論・その他 ===
    "フィボナッチ数列は1,1,2,3,5,8,13,...で、前の2項の和が次の項になり",
    "オイラーの公式e^(iπ)+1=0は数学で最も美しい等式と言われ",
    "順列P(n,r)=n!/(n-r)!で、組み合わせC(n,r)=n!/(r!(n-r)!)で",
    "数学的帰納法は、n=1で成立を示し、n=kで成立を仮定してn=k+1でも成立を示す証明法で",
    "極限lim[x→0](sinx/x)=1は重要な極限値で",
    "無限級数Σ(1/n²)=π²/6はバーゼル問題の解で",
    "ガウス積分∫[-∞,∞]e^(-x²)dx=√πで",
    "黄金比φ=(1+√5)/2≈1.618で",
    "ネイピア数e≈2.71828で、自然対数の底で",
    "πは円周率で約3.14159265358979で",

    # === Q&A形式（推論力強化） ===
    "質問: 1+1は？ 回答: 2です。足し算は数を合わせる演算で",
    "質問: 2×3は？ 回答: 6です。掛け算は同じ数を繰り返し足す演算で",
    "質問: 10÷2は？ 回答: 5です。割り算は等分する演算で",
    "質問: √9は？ 回答: 3です。平方根は二乗してその数になる値で",
    "質問: 三角形の面積の公式は？ 回答: 底辺×高さ÷2で",
    "質問: 円の面積の公式は？ 回答: πr²で、rは半径で",
    "質問: ピタゴラスの定理とは？ 回答: 直角三角形でa²+b²=c²（cは斜辺）で",
    "質問: 微分とは？ 回答: 関数の変化率を求めることで、接線の傾きを表し",
    "質問: 積分とは？ 回答: 微分の逆演算で、面積を求めることに使われ",
    "質問: 確率とは？ 回答: ある事象が起こる可能性の度合いで、0から1の値を取り",
    "質問: 素数とは？ 回答: 1と自分自身以外に約数を持たない自然数で、2,3,5,7,11などが",
    "質問: 行列とは？ 回答: 数を長方形に並べたもので、線形変換や連立方程式の表現に使われ",
    "質問: ベクトルとは？ 回答: 大きさと向きを持つ量で、矢印で表現され",
    "質問: 対数とは？ 回答: ある数を底の何乗で表せるかを示す値で、log₂8=3は2³=8を意味し",
    "質問: フィボナッチ数列とは？ 回答: 前の2つの数を足して次の数を作る数列で、1,1,2,3,5,8,13,...で",
    "質問: sin(90度)の値は？ 回答: 1です。sinは直角三角形の対辺と斜辺の比で",
    "質問: 正規分布とは？ 回答: 平均を中心に左右対称の釣鐘型の確率分布で",
    "質問: 二次方程式の解の公式は？ 回答: x=(-b±√(b²-4ac))/2aで、ax²+bx+c=0の解を求められ",
    "質問: 期待値とは？ 回答: 確率変数の平均的な値で、各値にその確率を掛けて足し合わせたもので",
    "質問: 行列式とは？ 回答: 正方行列に対して定義される値で、2×2行列では ad-bc で",
]


def fetch_texts(prompts, max_new_tokens=400):
    texts = []
    for i, prompt in enumerate(prompts):
        try:
            resp = requests.post(
                ENDPOINT_URL,
                json={"inputs": prompt, "parameters": {"max_new_tokens": max_new_tokens, "temperature": 0.7, "top_p": 0.9}},
                headers={"Content-Type": "application/json"},
                timeout=30,
            )
            if resp.status_code == 200:
                data = resp.json()
                if isinstance(data, list) and data:
                    text = data[0].get("generated_text", "")
                    full_text = prompt + text
                    if len(full_text) > 20:
                        texts.append(full_text)
                        if (i + 1) % 20 == 0:
                            print(f"  取得済み: {len(texts)}/{i+1}")
            else:
                print(f"  HTTP {resp.status_code} for prompt {i+1}")
            time.sleep(0.2)
        except Exception as e:
            print(f"  Error on prompt {i+1}: {e}")
            time.sleep(1)
    return texts


def tokenize_texts(texts, tok, max_seq_len):
    sequences = []
    for t in texts:
        ids = tok.encode(t, add_special=True)
        if len(ids) <= max_seq_len:
            if len(ids) >= 4:
                sequences.append(ids)
        else:
            stride = max(max_seq_len // 2, 1)
            for start in range(0, len(ids) - max_seq_len + 1, stride):
                sequences.append(ids[start:start + max_seq_len])
    # データ増強: 各シーケンスを2回ずつ追加（ドロップアウトによる正則化で効果的）
    augmented = sequences + sequences
    return augmented


def train(model, tokenizer, cfg, device, sequences):
    if not sequences:
        print("No sequences!")
        return None

    max_seq_len = cfg["max_seq_len"]
    total_steps = (len(sequences) // BATCH_SIZE * EPOCHS) // GRAD_ACCUM
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)

    model.train()
    global_step = 0
    best_loss = float('inf')

    for epoch in range(EPOCHS):
        random.shuffle(sequences)
        total_loss = 0
        n_batches = 0
        optimizer.zero_grad()

        for i in range(0, len(sequences), BATCH_SIZE):
            batch_seqs = sequences[i:i + BATCH_SIZE]
            if not batch_seqs:
                continue

            max_len = min(max(len(s) for s in batch_seqs), max_seq_len)
            input_ids = []
            labels = []
            for s in batch_seqs:
                ids = s[:max_len]
                pad_len = max_len - len(ids)
                input_ids.append(ids + [tokenizer.pad_id] * pad_len)
                labels.append(ids + [-100] * pad_len)

            input_ids_t = torch.tensor(input_ids, dtype=torch.long, device=device)
            labels_t = torch.tensor(labels, dtype=torch.long, device=device)

            logits = model(input_ids_t)
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels_t[..., 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, cfg["vocab_size"]),
                shift_labels.view(-1),
                ignore_index=-100
            )
            loss = loss / GRAD_ACCUM
            loss.backward()

            total_loss += loss.item() * GRAD_ACCUM
            n_batches += 1

            if n_batches % GRAD_ACCUM == 0:
                if global_step < WARMUP_STEPS:
                    cur_lr = LR * (global_step + 1) / max(WARMUP_STEPS, 1)
                else:
                    progress = (global_step - WARMUP_STEPS) / max(total_steps - WARMUP_STEPS, 1)
                    cosine_decay = 0.5 * (1 + math.cos(math.pi * min(progress, 1.0)))
                    cur_lr = LR * (MIN_LR_RATIO + (1 - MIN_LR_RATIO) * cosine_decay)
                for pg in optimizer.param_groups:
                    pg['lr'] = cur_lr
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()
                global_step += 1

        if n_batches % GRAD_ACCUM != 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()
            global_step += 1

        avg_loss = total_loss / max(n_batches, 1)
        print(f"  Epoch {epoch+1}/{EPOCHS} | Loss: {avg_loss:.4f} | Steps: {global_step}")
        if avg_loss < best_loss:
            best_loss = avg_loss

    del optimizer
    gc.collect()
    return best_loss


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    checkpoint = torch.load(CKPT_PATH, map_location="cpu")
    cfg = checkpoint["config"]
    tokenizer = NeuroQuantumTokenizer(vocab_size=cfg["vocab_size"], model_file=TOKENIZER_PATH)

    nq_config = NeuroQuantumConfig(
        vocab_size=cfg["vocab_size"], embed_dim=cfg["embed_dim"],
        hidden_dim=cfg.get("hidden_dim", cfg["embed_dim"] * 2),
        num_heads=cfg["num_heads"], num_layers=cfg["num_layers"],
        max_seq_len=cfg["max_seq_len"], dropout=cfg.get("dropout", 0.1),
        lambda_entangle=cfg.get("entangle_strength", 0.5),
    )
    model = NeuroQuantum(config=nq_config).to(device)
    model.load_state_dict(checkpoint["model_state"])
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {n_params:,} params")

    # Phase 1: データ取得
    print(f"\n=== Phase 1: 数学データ大量取得 ({len(PROMPTS)} prompts) ===")
    texts = fetch_texts(PROMPTS, max_new_tokens=400)
    print(f"取得テキスト数: {len(texts)}")

    if not texts:
        print("No data, aborting.")
        return

    # Phase 2: トークナイズ + 増強
    print(f"\n=== Phase 2: トークナイズ + データ増強 ===")
    sequences = tokenize_texts(texts, tokenizer, cfg["max_seq_len"])
    print(f"シーケンス数: {len(sequences)} (増強後)")
    del texts
    gc.collect()

    # Phase 3: 集中学習
    print(f"\n=== Phase 3: 集中学習 ({EPOCHS} epochs) ===")
    best_loss = train(model, tokenizer, cfg, device, sequences)
    print(f"\nBest Loss: {best_loss:.4f}")
    del sequences
    gc.collect()

    # Save
    model.eval()
    prev_log = checkpoint.get("training_log", [])
    new_checkpoint = {
        "model_state": model.state_dict(),
        "config": cfg,
        "training_log": prev_log + [{"type": "math-intensive", "info": f"loss={best_loss:.4f}, prompts={len(PROMPTS)}, epochs={EPOCHS}", "endpoint": ENDPOINT_URL}],
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "datasets": list(set(checkpoint.get("datasets", []) + ["math-intensive"])),
        "qa_training": checkpoint.get("qa_training", False),
    }
    torch.save(new_checkpoint, CKPT_PATH)
    print(f"Checkpoint saved!")

    # テスト推論
    print(f"\n=== テスト推論 ===")
    import api
    api.model = model
    api.tokenizer = tokenizer
    api.config = cfg
    api.device = device
    from api import generate_text

    test_prompts = [
        "質問: 1+1は？ 回答:",
        "質問: 三角形の面積の公式は？ 回答:",
        "質問: 円の面積の公式は？ 回答:",
        "質問: 円周率とは？ 回答:",
        "質問: 二次方程式の解の公式は？ 回答:",
        "質問: ピタゴラスの定理とは？ 回答:",
        "質問: 微分とは？ 回答:",
        "質問: 積分とは？ 回答:",
        "質問: 確率とは？ 回答:",
        "質問: 素数とは？ 回答:",
        "質問: 行列とは？ 回答:",
        "質問: ベクトルとは？ 回答:",
        "質問: 対数とは？ 回答:",
        "質問: フィボナッチ数列とは？ 回答:",
        "質問: sin(90度)の値は？ 回答:",
        "質問: 正規分布とは？ 回答:",
        "質問: 期待値とは？ 回答:",
        "質問: 行列式とは？ 回答:",
    ]
    for prompt in test_prompts:
        result = generate_text(prompt, max_new_tokens=150, temperature=0.7, top_k=40, top_p=0.9, repetition_penalty=1.3)
        q = prompt.replace(" 回答:", "")
        print(f"\n{q}")
        print(f"  → {result.strip()[:250]}")

    print(f"\n=== 数学集中学習完了! ===")


if __name__ == "__main__":
    main()
