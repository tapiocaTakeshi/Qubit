#!/usr/bin/env python3
"""Enhanced training: HF endpoint + expanded crafted QA + HF datasets."""
import os
import sys
import torch
import torch.nn.functional as F
import math
import random
import gc
import requests
import time
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(__file__))
from neuroquantum_layered import NeuroQuantum, NeuroQuantumConfig, NeuroQuantumTokenizer

CKPT_PATH = os.path.join(os.path.dirname(__file__), "neuroq_checkpoint.pt")
TOKENIZER_PATH = os.path.join(os.path.dirname(__file__), "neuroq_tokenizer.model")
ENDPOINT_URL = "https://vvcci2ps4y3wfx7m.us-east4.gcp.endpoints.huggingface.cloud"

# Training params
EPOCHS = 5
BATCH_SIZE = 4
GRAD_ACCUM = 2
LR = 5e-5
WARMUP_STEPS = 30

# Expanded high-quality QA pairs (the core training signal)
CRAFTED_QA = [
    # 一般知識
    "質問: 日本の首都はどこですか？\n回答: 日本の首都は東京です。",
    "質問: 富士山の高さは？\n回答: 富士山の高さは3776メートルです。日本で最も高い山です。",
    "質問: 太陽系で一番大きい惑星は？\n回答: 木星が太陽系で最も大きい惑星です。",
    "質問: 水の化学式は？\n回答: 水の化学式はH2Oです。水素2つと酸素1つから成ります。",
    "質問: 光の速さは？\n回答: 光の速さは秒速約30万キロメートル（299,792,458m/s）です。",
    "質問: 日本で一番長い川は？\n回答: 信濃川が日本で最も長い川で、全長367kmです。",
    "質問: 地球の年齢はどれくらい？\n回答: 地球の年齢は約46億年です。",
    "質問: 人間の体で一番大きい臓器は？\n回答: 皮膚が人間の体で最も大きい臓器です。",
    # 歴史
    "質問: 明治維新は何年に起きましたか？\n回答: 明治維新は1868年に起きました。日本が近代国家へと変わる大きな転換点でした。",
    "質問: 第二次世界大戦はいつ終わりましたか？\n回答: 第二次世界大戦は1945年に終わりました。",
    "質問: 源頼朝は何をした人ですか？\n回答: 源頼朝は鎌倉幕府を開いた人物で、日本初の武家政権を樹立しました。",
    "質問: 江戸時代はいつからいつまでですか？\n回答: 江戸時代は1603年から1868年までの約265年間です。",
    # 科学
    "質問: DNAとは何ですか？\n回答: DNAはデオキシリボ核酸の略で、遺伝情報を保持する分子です。二重らせん構造をしています。",
    "質問: 重力とは何ですか？\n回答: 重力とは質量を持つ物体同士が引き合う力です。地球上では約9.8m/s²の加速度を生みます。",
    "質問: 光合成とは何ですか？\n回答: 光合成は植物が太陽光のエネルギーを使って二酸化炭素と水から酸素とブドウ糖を作る反応です。",
    "質問: 原子とは何ですか？\n回答: 原子は物質を構成する最小の粒子で、原子核と電子から成ります。",
    # 数学
    "質問: 円周率とは何ですか？\n回答: 円周率は円の周の長さと直径の比で、約3.14159です。πで表されます。",
    "質問: 三角形の内角の和は？\n回答: 三角形の内角の和は180度です。",
    "質問: 素数とは何ですか？\n回答: 素数は1と自分自身でしか割り切れない2以上の自然数です。例えば2、3、5、7、11などです。",
    "質問: ピタゴラスの定理とは何ですか？\n回答: ピタゴラスの定理は直角三角形において、斜辺の2乗が他の2辺の2乗の和に等しいという定理です。a²+b²=c²で表されます。",
    # 技術
    "質問: 人工知能とは何ですか？\n回答: 人工知能（AI）は、人間の知的能力をコンピュータで実現する技術です。機械学習や深層学習が代表的な手法です。",
    "質問: 量子コンピュータとは？\n回答: 量子コンピュータは量子力学の原理を利用して計算を行うコンピュータです。従来のコンピュータでは困難な問題を高速に解ける可能性があります。",
    "質問: インターネットとは何ですか？\n回答: インターネットは世界中のコンピュータネットワークを相互に接続した通信基盤です。",
    "質問: プログラミングとは何ですか？\n回答: プログラミングとはコンピュータに実行させたい処理を記述する作業です。様々なプログラミング言語を使います。",
    # 文化
    "質問: 俳句とは何ですか？\n回答: 俳句は五七五の17音で構成される日本の伝統的な短い詩です。季語を含むのが特徴です。",
    "質問: 歌舞伎とは何ですか？\n回答: 歌舞伎は日本の伝統的な演劇で、華やかな衣装と独特の演技が特徴です。",
    "質問: 茶道について教えてください。\n回答: 茶道は抹茶を点てて客人にもてなす日本の伝統文化です。「わび・さび」の精神を大切にします。",
    "質問: 日本の国花は何ですか？\n回答: 日本の国花は桜と菊です。",
    # 地理
    "質問: 琵琶湖はどこにありますか？\n回答: 琵琶湖は滋賀県にある日本最大の湖です。面積は約670平方キロメートルです。",
    "質問: 北海道の県庁所在地はどこですか？\n回答: 北海道の道庁所在地は札幌市です。",
    "質問: 日本の人口はどれくらいですか？\n回答: 日本の人口は約1億2500万人です。",
    "質問: 東京タワーの高さは何メートルですか？\n回答: 東京タワーの高さは333メートルです。1958年に完成しました。",
    # 追加
    "質問: 月の直径はどれくらいですか？\n回答: 月の直径は約3474キロメートルです。",
    "質問: 酸素の元素記号は何ですか？\n回答: 酸素の元素記号はOです。原子番号は8です。",
    "質問: 日本の国歌は何ですか？\n回答: 日本の国歌は「君が代」です。",
    "質問: ノーベル賞とは何ですか？\n回答: ノーベル賞はアルフレッド・ノーベルの遺言で創設された国際的な賞で、物理学・化学・生理学医学・文学・平和・経済学の6分野があります。",
    # さらに追加（混同防止用）
    "質問: 太陽の表面温度は？\n回答: 太陽の表面温度は約5500度です。",
    "質問: 日本の通貨は何ですか？\n回答: 日本の通貨は円です。",
    "質問: 酸素と水素が化合すると何ができますか？\n回答: 酸素と水素が化合すると水（H2O）ができます。",
    "質問: 地球から太陽までの距離は？\n回答: 地球から太陽までの距離は約1億5000万キロメートルです。",
    "質問: 日本語の文字の種類は？\n回答: 日本語にはひらがな、カタカナ、漢字の3種類の文字があります。",
    "質問: 1年は何日ですか？\n回答: 1年は365日です。うるう年は366日になります。",
    "質問: 地球の直径は？\n回答: 地球の直径は約12,742キロメートルです。",
    "質問: 日本で一番高い建物は？\n回答: 日本で一番高い建物は東京スカイツリーで、高さ634メートルです。",
    "質問: 二酸化炭素の化学式は？\n回答: 二酸化炭素の化学式はCO2です。炭素1つと酸素2つから成ります。",
    "質問: 日本の面積はどれくらいですか？\n回答: 日本の面積は約37万8000平方キロメートルです。",
]


def query_endpoint(prompt, max_new_tokens=100, temperature=0.5):
    """Query the HF inference endpoint."""
    payload = {
        "inputs": f"質問: {prompt}\n回答:",
        "parameters": {
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
            "do_sample": True,
            "top_p": 0.9,
            "repetition_penalty": 1.3,
        }
    }
    for attempt in range(3):
        try:
            resp = requests.post(
                ENDPOINT_URL,
                headers={"Content-Type": "application/json"},
                json=payload,
                timeout=30,
            )
            if resp.status_code == 200:
                result = resp.json()
                if isinstance(result, list) and result:
                    text = result[0].get("generated_text", "")
                elif isinstance(result, dict):
                    text = result.get("generated_text", "")
                else:
                    text = ""
                # Filter low quality: too short, too repetitive
                if text and len(text) > 5:
                    # Check for excessive repetition
                    words = text.split()
                    if len(words) > 3:
                        unique_ratio = len(set(words)) / len(words)
                        if unique_ratio < 0.3:
                            return ""  # Too repetitive
                    return text
            elif resp.status_code == 503:
                time.sleep(5)
                continue
        except Exception as e:
            if attempt < 2:
                time.sleep(2 * (attempt + 1))
    return ""


def generate_endpoint_data():
    """Generate data from HF endpoint with quality filtering."""
    prompts = [
        "日本の首都はどこですか？",
        "富士山の高さは何メートルですか？",
        "太陽系で一番大きい惑星は何ですか？",
        "水の化学式は何ですか？",
        "光の速さはどれくらいですか？",
        "地球の年齢はどれくらいですか？",
        "人間の体で一番大きい臓器は何ですか？",
        "明治維新は何年に起きましたか？",
        "DNAとは何ですか？",
        "重力とは何ですか？",
        "光合成とは何ですか？",
        "円周率とは何ですか？",
        "三角形の内角の和は何度ですか？",
        "人工知能とは何ですか？",
        "量子コンピュータとは何ですか？",
        "インターネットとは何ですか？",
        "俳句とは何ですか？",
        "茶道について教えてください。",
        "健康のために大切なことは何ですか？",
        "読書のメリットは何ですか？",
    ]

    print(f"  Querying endpoint with {len(prompts)} prompts...")
    texts = []
    for i, prompt in enumerate(prompts):
        text = query_endpoint(prompt)
        if text:
            full = f"質問: {prompt}\n回答:{text}" if not text.startswith("質問:") else text
            texts.append(full)
        time.sleep(0.3)
        if (i + 1) % 10 == 0:
            print(f"    {i+1}/{len(prompts)} done ({len(texts)} valid)")

    print(f"  Endpoint: {len(texts)} valid samples from {len(prompts)} queries")
    return texts


def load_hf_datasets(max_per_dataset=300):
    """Load HF datasets for additional training data."""
    texts = []
    try:
        import signal
        from datasets import load_dataset

        class DatasetTimeout(Exception):
            pass

        def handler(signum, frame):
            raise DatasetTimeout()

        datasets_info = [
            {"id": "fujiki/japanese_alpaca_data", "format": "alpaca"},
            {"id": "kunishou/oasst1-chat-44k-ja", "format": "conversations"},
        ]

        for ds_info in datasets_info:
            ds_id = ds_info["id"]
            fmt = ds_info["format"]
            try:
                print(f"  Loading {ds_id} (max={max_per_dataset})...")
                signal.signal(signal.SIGALRM, handler)
                signal.alarm(60)
                ds = load_dataset(ds_id, split="train", streaming=True)
                count = 0
                for row in ds:
                    if fmt == "alpaca":
                        inst = row.get("instruction", "")
                        out = row.get("output", "")
                        inp = row.get("input", "")
                        if inst and out:
                            q = f"{inst}\n{inp}".strip() if inp else inst
                            text = f"質問: {q}\n回答: {out}"
                            if len(text) > 15:
                                texts.append(text)
                                count += 1
                    elif fmt == "conversations":
                        convs = row.get("conversations", [])
                        if convs and len(convs) >= 2:
                            parts = []
                            for c in convs:
                                role = c.get("from", "")
                                val = c.get("value", "")
                                if role == "human":
                                    parts.append(f"質問: {val}")
                                elif role == "gpt":
                                    parts.append(f"回答: {val}")
                            if parts:
                                text = "\n".join(parts)
                                if len(text) > 15:
                                    texts.append(text)
                                    count += 1
                    if count >= max_per_dataset:
                        break
                signal.alarm(0)
                print(f"    -> {count} samples")
            except DatasetTimeout:
                signal.alarm(0)
                print(f"    -> Timed out")
            except Exception as e:
                signal.alarm(0)
                print(f"    -> Failed: {e}")
            gc.collect()
    except ImportError:
        print("  datasets not available")

    return texts


def tokenize_texts(texts, tokenizer, max_seq_len):
    sequences = []
    for t in texts:
        ids = tokenizer.encode(t)
        if len(ids) > 3:
            sequences.append(ids[:max_seq_len])
    return sequences


def train(model, config, tokenizer, all_texts, device):
    """Train the model."""
    max_seq_len = config["max_seq_len"]
    sequences = tokenize_texts(all_texts, tokenizer, max_seq_len)
    random.shuffle(sequences)

    print(f"  Sequences: {len(sequences)}")
    if not sequences:
        print("  No sequences!")
        return

    steps_per_epoch = len(sequences) // BATCH_SIZE
    total_steps = (steps_per_epoch * EPOCHS) // GRAD_ACCUM
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)

    model.train()
    global_step = 0
    training_log = []

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
                shift_logits.view(-1, config["vocab_size"]),
                shift_labels.view(-1),
                ignore_index=-100
            )
            loss = loss / GRAD_ACCUM
            loss.backward()

            total_loss += loss.item() * GRAD_ACCUM
            n_batches += 1

            if n_batches % GRAD_ACCUM == 0:
                if global_step < WARMUP_STEPS:
                    lr = LR * (global_step + 1) / max(WARMUP_STEPS, 1)
                else:
                    progress = (global_step - WARMUP_STEPS) / max(total_steps - WARMUP_STEPS, 1)
                    cosine_decay = 0.5 * (1 + math.cos(math.pi * min(progress, 1.0)))
                    lr = LR * (0.1 + 0.9 * cosine_decay)
                for pg in optimizer.param_groups:
                    pg['lr'] = lr
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
        training_log.append({"epoch": epoch + 1, "loss": round(avg_loss, 4)})
        print(f"  Epoch {epoch+1}/{EPOCHS} | Loss: {avg_loss:.4f} | Steps: {global_step}")

    del optimizer
    gc.collect()
    return training_log


def main():
    print("=" * 60)
    print("  Enhanced API Training")
    print("=" * 60)

    # Load model
    print("\n[1/4] Loading model...")
    device = torch.device("cpu")
    ckpt = torch.load(CKPT_PATH, map_location=device, weights_only=False)
    cfg = ckpt["config"]
    model_config = NeuroQuantumConfig(
        vocab_size=cfg["vocab_size"],
        embed_dim=cfg["embed_dim"],
        hidden_dim=cfg["hidden_dim"],
        num_heads=cfg["num_heads"],
        num_layers=cfg["num_layers"],
        max_seq_len=cfg["max_seq_len"],
        lambda_entangle=cfg.get("entangle_strength", cfg.get("lambda_entangle", 0.5)),
        dropout=cfg.get("dropout", 0.1),
    )
    model = NeuroQuantum(model_config)
    model.load_state_dict(ckpt.get("model_state_dict") or ckpt["model_state"])
    model.to(device)
    params = sum(p.numel() for p in model.parameters())
    print(f"  Model: {params:,} params")

    tokenizer = NeuroQuantumTokenizer(model_file=TOKENIZER_PATH)
    del ckpt
    gc.collect()

    # Collect data
    print("\n[2/4] Collecting data from HF endpoint...")
    endpoint_texts = generate_endpoint_data()

    print("\n[3/4] Loading HF datasets...")
    hf_texts = load_hf_datasets(max_per_dataset=300)

    # Build training set: crafted QA heavily repeated + endpoint + HF data
    crafted_repeats = 40
    all_texts = []
    for _ in range(crafted_repeats):
        all_texts.extend(CRAFTED_QA)
    all_texts.extend(endpoint_texts)
    all_texts.extend(hf_texts)

    print(f"\n  Training data breakdown:")
    print(f"    Crafted QA: {len(CRAFTED_QA)} x {crafted_repeats} = {len(CRAFTED_QA) * crafted_repeats}")
    print(f"    Endpoint:   {len(endpoint_texts)}")
    print(f"    HF datasets: {len(hf_texts)}")
    print(f"    Total:      {len(all_texts)}")

    # Train
    print("\n[4/4] Training...")
    training_log = train(model, cfg, tokenizer, all_texts, device)

    # Save
    ckpt = {
        "model_state": model.state_dict(),
        "config": cfg,
        "training_log": training_log,
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "data_source": "api_enhanced",
        "data_stats": {
            "crafted_qa": len(CRAFTED_QA) * crafted_repeats,
            "endpoint": len(endpoint_texts),
            "hf_datasets": len(hf_texts),
        },
    }
    torch.save(ckpt, CKPT_PATH)
    size_mb = os.path.getsize(CKPT_PATH) / (1024 * 1024)
    print(f"\n  Checkpoint saved: {size_mb:.1f}MB")

    # Inference test
    print("\n" + "=" * 60)
    print("  Inference Test")
    print("=" * 60)
    model.eval()
    test_prompts = [
        "質問: 日本の首都はどこですか？\n回答:",
        "質問: 富士山の高さは？\n回答:",
        "質問: 水の化学式は？\n回答:",
        "質問: 光の速さは？\n回答:",
        "質問: 人工知能とは何ですか？\n回答:",
        "質問: 量子コンピュータとは？\n回答:",
        "質問: 地球の年齢はどれくらい？\n回答:",
        "質問: 三角形の内角の和は？\n回答:",
    ]
    for prompt in test_prompts:
        ids = tokenizer.encode(prompt)
        input_t = torch.tensor([ids], dtype=torch.long, device=device)
        with torch.no_grad():
            for _ in range(60):
                logits = model(input_t)
                next_logits = logits[0, -1, :] / 0.5
                probs = torch.softmax(next_logits, dim=-1)
                next_id = torch.multinomial(probs, 1).item()
                input_t = torch.cat([input_t, torch.tensor([[next_id]], device=device)], dim=1)
                if next_id == tokenizer.eos_id:
                    break
        output = tokenizer.decode(input_t[0].tolist())
        print(f"  {output.strip()}")


if __name__ == "__main__":
    main()
