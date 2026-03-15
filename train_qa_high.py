#!/usr/bin/env python3
"""
QA形式の日本語データで高エポック数ファインチューニング。
質問→回答のパターンをより深く学習させる。
CPU環境対応: データ数を最適化しつつ、エポック数を増やして品質向上。
"""
import os
import sys
import torch
import torch.nn.functional as F
from datasets import load_dataset
from datetime import datetime, timezone
import json
import random
import math

sys.path.insert(0, os.path.dirname(__file__))
from neuroquantum_layered import NeuroQuantum, NeuroQuantumConfig, NeuroQuantumTokenizer

CKPT_PATH = os.path.join(os.path.dirname(__file__), "neuroq_checkpoint.pt")

# High-epoch training hyperparameters (CPU-optimized)
EPOCHS = 40
LR = 3e-5          # Lower LR for longer training stability
BATCH_SIZE = 4
GRAD_ACCUM_STEPS = 4
WARMUP_STEPS = 30
GRAD_CLIP = 1.0
MIN_LR_RATIO = 0.1  # Minimum LR as fraction of max LR

# QA datasets
QA_DATASETS = [
    {
        "id": "fujiki/japanese_alpaca_data",
        "format": "alpaca",
    },
    {
        "id": "FreedomIntelligence/alpaca-gpt4-japanese",
        "format": "conversations",
    },
    {
        "id": "kunishou/oasst1-chat-44k-ja",
        "format": "conversations",
    },
    {
        "id": "izumi-lab/llm-japanese-dataset",
        "format": "izumi",
    },
]


def format_qa_alpaca(row):
    instruction = row.get("instruction", "").strip()
    inp = row.get("input", "").strip()
    output = row.get("output", "").strip()
    if not instruction or not output:
        return None
    if inp:
        q = f"{instruction}\n{inp}"
    else:
        q = instruction
    return f"質問: {q}\n回答: {output}"


def format_qa_conversations(row):
    convs = row.get("conversations", [])
    if not convs:
        return None
    pairs = []
    i = 0
    while i < len(convs) - 1:
        turn = convs[i]
        next_turn = convs[i + 1]
        q_text = ""
        a_text = ""
        if isinstance(turn, dict):
            q_text = turn.get("value", turn.get("content", "")).strip()
        elif isinstance(turn, str):
            q_text = turn.strip()
        if isinstance(next_turn, dict):
            a_text = next_turn.get("value", next_turn.get("content", "")).strip()
        elif isinstance(next_turn, str):
            a_text = next_turn.strip()
        if q_text and a_text:
            pairs.append(f"質問: {q_text}\n回答: {a_text}")
        i += 2
    if pairs:
        return "\n\n".join(pairs)
    return None


def format_qa_izumi(row):
    output = row.get("output", "").strip()
    instruction = row.get("input", row.get("instruction", "")).strip() if isinstance(row.get("input", ""), str) else ""
    if not output:
        return None
    if instruction:
        return f"質問: {instruction}\n回答: {output}"
    return f"回答: {output}"


def load_qa_data():
    all_qa = []

    for ds_info in QA_DATASETS:
        ds_id = ds_info["id"]
        fmt = ds_info["format"]
        print(f"  Loading {ds_id}...")

        try:
            ds = load_dataset(ds_id, split="train", trust_remote_code=True)
            n = len(ds)
            count = 0

            for row in ds.select(range(n)):
                if fmt == "alpaca":
                    text = format_qa_alpaca(row)
                elif fmt == "conversations":
                    text = format_qa_conversations(row)
                elif fmt == "izumi":
                    text = format_qa_izumi(row)
                else:
                    continue

                if text and len(text) > 10:
                    all_qa.append(text)
                    count += 1

            print(f"    -> {count} QA samples")
        except Exception as e:
            print(f"    -> ERROR: {e}")

    # 手作りQA例（拡張版）- パターン強化用
    crafted_qa = [
        "質問: 日本の首都はどこですか？\n回答: 日本の首都は東京です。",
        "質問: 富士山の高さはどのくらいですか？\n回答: 富士山の高さは3,776メートルです。日本で最も高い山です。",
        "質問: プログラミングとは何ですか？\n回答: プログラミングとは、コンピュータに実行させる命令を書くことです。Python、Java、C++などのプログラミング言語を使って、ソフトウェアやアプリケーションを作成します。",
        "質問: 人工知能とは何ですか？\n回答: 人工知能（AI）は、人間の知能を模倣するコンピュータシステムです。機械学習、深層学習、自然言語処理などの技術を含みます。",
        "質問: 量子コンピュータとは何ですか？\n回答: 量子コンピュータは、量子力学の原理を利用して計算を行うコンピュータです。量子ビット（キュービット）を使い、従来のコンピュータでは困難な問題を解くことができます。",
        "質問: 機械学習とは何ですか？\n回答: 機械学習は、データからパターンを学習し、予測や判断を行う人工知能の一分野です。教師あり学習、教師なし学習、強化学習などの手法があります。",
        "質問: Pythonとは何ですか？\n回答: Pythonは、読みやすく書きやすい汎用プログラミング言語です。データ分析、AI開発、Web開発など幅広い分野で使われています。",
        "質問: インターネットとは何ですか？\n回答: インターネットは、世界中のコンピュータネットワークを相互に接続した通信網です。Webサイトの閲覧、メール、動画配信などのサービスを支えています。",
        "質問: 太陽系にはいくつの惑星がありますか？\n回答: 太陽系には8つの惑星があります。水星、金星、地球、火星、木星、土星、天王星、海王星です。",
        "質問: 光の速さはどのくらいですか？\n回答: 光の速さは秒速約30万キロメートル（299,792,458 m/s）です。これは宇宙で最も速い速度です。",
        "質問: 日本語の文字体系について教えてください。\n回答: 日本語には、ひらがな、カタカナ、漢字の3種類の文字体系があります。ひらがなは46文字、カタカナも46文字あり、漢字は数千字が日常的に使われています。",
        "質問: 地球の年齢はどのくらいですか？\n回答: 地球の年齢は約46億年です。太陽系の形成とほぼ同時期に誕生しました。",
        "質問: DNAとは何ですか？\n回答: DNAはデオキシリボ核酸の略で、生物の遺伝情報を保持する分子です。二重らせん構造を持ち、アデニン、チミン、グアニン、シトシンの4つの塩基から構成されています。",
        "質問: 民主主義とは何ですか？\n回答: 民主主義は、国民が主権を持ち、選挙などを通じて政治に参加する政治体制です。言論の自由、法の支配、人権の尊重が基本原則です。",
        "質問: 相対性理論とは何ですか？\n回答: 相対性理論はアインシュタインが提唱した物理学の理論です。特殊相対性理論と一般相対性理論があり、時間と空間が相対的であること、重力が時空の歪みであることを示しました。",
        # 追加の手作りQA（幅広いトピック）
        "質問: 北海道の特徴を教えてください。\n回答: 北海道は日本最北の島で、面積は日本最大です。冬は寒く雪が多く、夏は涼しいです。農業、酪農、漁業が盛んで、ラベンダー畑や温泉など観光地も多いです。",
        "質問: 地震はなぜ起きるのですか？\n回答: 地震は、地球の表面を覆うプレートが動くことで起きます。プレート同士がぶつかったり、すれ違ったりする際にエネルギーが蓄積され、それが解放されるときに地震が発生します。",
        "質問: 水は何度で沸騰しますか？\n回答: 水は標準気圧（1気圧）のもとで100度（摂氏）で沸騰します。気圧が低い高山では、沸点はこれより低くなります。",
        "質問: 日本の四季について教えてください。\n回答: 日本には春、夏、秋、冬の四季があります。春は桜が咲き、夏は暑く湿度が高いです。秋は紅葉が美しく、冬は寒く雪が降る地域もあります。四季の変化は日本文化に大きな影響を与えています。",
        "質問: ニューラルネットワークとは何ですか？\n回答: ニューラルネットワークは、人間の脳の神経回路を模倣した機械学習モデルです。入力層、隠れ層、出力層からなり、重みを学習することでパターン認識や予測を行います。深層学習の基盤技術です。",
        "質問: 酸素はどのような元素ですか？\n回答: 酸素は原子番号8の元素で、元素記号はOです。空気中に約21%含まれ、生物の呼吸に不可欠です。また、物質の燃焼にも必要で、水（H2O）の構成元素でもあります。",
        "質問: 東京タワーについて教えてください。\n回答: 東京タワーは1958年に完成した電波塔で、高さは333メートルです。東京都港区芝公園にあり、赤と白の鮮やかな外観が特徴的です。展望台からは東京の景色を一望できます。",
        "質問: 円周率とは何ですか？\n回答: 円周率（π）は、円の直径に対する円周の比率です。約3.14159で、無限に続く無理数です。数学、物理学、工学など多くの分野で重要な定数です。",
        "質問: 織田信長はどんな人物ですか？\n回答: 織田信長は戦国時代の武将で、天下統一を目指しました。鉄砲を活用した長篠の戦いや、楽市楽座などの経済政策で知られます。1582年の本能寺の変で明智光秀に討たれました。",
        "質問: クラウドコンピューティングとは何ですか？\n回答: クラウドコンピューティングは、インターネット経由でサーバー、ストレージ、データベースなどのコンピューティングリソースを利用するサービスです。AWS、Azure、GCPなどが代表的なプロバイダーです。",
    ]
    # 高エポック学習なので手作りQAの繰り返しを増やす
    for _ in range(40):
        all_qa.extend(crafted_qa)
    print(f"  + {len(crafted_qa) * 40} crafted QA samples")

    return all_qa


def tokenize_texts(texts, tokenizer, max_seq_len):
    sequences = []
    for t in texts:
        ids = tokenizer.encode(t, add_special=True)
        if len(ids) <= max_seq_len:
            if len(ids) >= 4:
                sequences.append(ids)
        else:
            stride = max(max_seq_len // 2, 1)
            for start in range(0, len(ids) - max_seq_len + 1, stride):
                sequences.append(ids[start:start + max_seq_len])
    return sequences


def get_lr(step, total_steps, warmup_steps, max_lr):
    """Learning rate with linear warmup and cosine decay (with minimum LR)."""
    if step < warmup_steps:
        return max_lr * step / max(warmup_steps, 1)
    progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
    cosine_decay = 0.5 * (1 + math.cos(math.pi * progress))
    return max_lr * (MIN_LR_RATIO + (1 - MIN_LR_RATIO) * cosine_decay)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load checkpoint
    print("=== Loading checkpoint ===")
    checkpoint = torch.load(CKPT_PATH, map_location="cpu")
    config = checkpoint["config"]
    prev_log = checkpoint.get("training_log", [])
    print(f"Config: embed_dim={config['embed_dim']}, layers={config['num_layers']}, "
          f"vocab={config['vocab_size']}")
    if prev_log:
        print(f"Previous training: {len(prev_log)} epochs, last loss: {prev_log[-1]['loss']:.4f}")

    # Load tokenizer
    tokenizer_path = os.path.join(os.path.dirname(__file__), "neuroq_tokenizer.model")
    tokenizer = NeuroQuantumTokenizer(vocab_size=config["vocab_size"], model_file=tokenizer_path)

    # Build model
    max_seq_len = config["max_seq_len"]
    nq_config = NeuroQuantumConfig(
        vocab_size=config["vocab_size"],
        embed_dim=config["embed_dim"],
        hidden_dim=config.get("hidden_dim", config["embed_dim"] * 2),
        num_heads=config["num_heads"],
        num_layers=config["num_layers"],
        max_seq_len=max_seq_len,
        dropout=config.get("dropout", 0.1),
        lambda_entangle=config.get("entangle_strength", 0.5),
    )
    model = NeuroQuantum(config=nq_config).to(device)
    model.load_state_dict(checkpoint["model_state"])
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model loaded: {n_params:,} parameters")

    # Load QA data
    print("\n=== Loading QA datasets ===")
    qa_texts = load_qa_data()
    print(f"\nTotal QA texts: {len(qa_texts)}")

    # Tokenize
    print("\n=== Tokenizing ===")
    sequences = tokenize_texts(qa_texts, tokenizer, max_seq_len)
    print(f"Training sequences: {len(sequences)}")

    # Calculate steps
    steps_per_epoch = len(sequences) // BATCH_SIZE
    total_steps = (steps_per_epoch * EPOCHS) // GRAD_ACCUM_STEPS
    print(f"Steps per epoch: {steps_per_epoch}")
    print(f"Total optimization steps: {total_steps}")
    print(f"Effective batch size: {BATCH_SIZE * GRAD_ACCUM_STEPS}")
    print(f"Epochs: {EPOCHS}, LR: {LR}, Min LR ratio: {MIN_LR_RATIO}")

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)

    # Training loop
    print(f"\n=== Training for {EPOCHS} epochs (QA high-epoch) ===")
    model.train()
    training_log = []
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

            input_ids = torch.tensor(input_ids, dtype=torch.long, device=device)
            labels_t = torch.tensor(labels, dtype=torch.long, device=device)

            logits = model(input_ids)
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels_t[..., 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, nq_config.vocab_size),
                shift_labels.view(-1),
                ignore_index=-100
            )
            loss = loss / GRAD_ACCUM_STEPS
            loss.backward()

            total_loss += loss.item() * GRAD_ACCUM_STEPS
            n_batches += 1

            if n_batches % GRAD_ACCUM_STEPS == 0:
                lr = get_lr(global_step, total_steps, WARMUP_STEPS, LR)
                for pg in optimizer.param_groups:
                    pg['lr'] = lr
                torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
                optimizer.step()
                optimizer.zero_grad()
                global_step += 1

            if n_batches % 300 == 0:
                avg = total_loss / n_batches
                lr = get_lr(global_step, total_steps, WARMUP_STEPS, LR)
                print(f"  Epoch {epoch+1}/{EPOCHS} | Batch {n_batches} | "
                      f"Loss: {avg:.4f} | LR: {lr:.2e} | Step: {global_step}")

        # Handle remaining gradients
        if n_batches % GRAD_ACCUM_STEPS != 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimizer.step()
            optimizer.zero_grad()
            global_step += 1

        avg_loss = total_loss / max(n_batches, 1)
        print(f"Epoch {epoch+1}/{EPOCHS} | Avg Loss: {avg_loss:.6f}")
        training_log.append({"epoch": len(prev_log) + epoch + 1, "loss": avg_loss})

        # Save best checkpoint
        if avg_loss < best_loss:
            best_loss = avg_loss
            print(f"  New best loss: {best_loss:.4f}, saving checkpoint...")
            save_checkpoint(model, config, prev_log + training_log, checkpoint)

        # Save every 5 epochs as safety
        if (epoch + 1) % 5 == 0:
            print(f"  Periodic save at epoch {epoch+1}...")
            save_checkpoint(model, config, prev_log + training_log, checkpoint)

    # Final save
    save_checkpoint(model, config, prev_log + training_log, checkpoint)

    # QA Inference test
    print("\n=== QA Inference test ===")
    model.eval()
    test_qa = [
        "質問: 日本の首都はどこですか？\n回答:",
        "質問: プログラミングとは何ですか？\n回答:",
        "質問: AIとは何ですか？\n回答:",
        "質問: 量子コンピュータとは何ですか？\n回答:",
        "質問: 太陽系の惑星を教えてください。\n回答:",
        "質問: 富士山について教えてください。\n回答:",
        "質問: 機械学習とは何ですか？\n回答:",
        "質問: 北海道の特徴を教えてください。\n回答:",
        "質問: ニューラルネットワークとは何ですか？\n回答:",
        "質問: 織田信長はどんな人物ですか？\n回答:",
    ]
    for prompt in test_qa:
        tokens = tokenizer.encode(prompt, add_special=True)
        input_tensor = torch.tensor([tokens], dtype=torch.long, device=device)
        generated = list(tokens)

        with torch.no_grad():
            for _ in range(120):
                seq = input_tensor[:, -max_seq_len:]
                logits = model(seq)[:, -1, :] / 0.7
                topk_vals = torch.topk(logits, 40)[0]
                logits[logits < topk_vals[:, -1:]] = float('-inf')
                for prev in set(generated[-50:]):
                    if prev < logits.size(-1):
                        logits[0, prev] /= 1.3
                probs = F.softmax(logits, dim=-1)
                nxt = torch.multinomial(probs, 1)
                nxt_id = nxt.item()
                if nxt_id == tokenizer.eos_id:
                    break
                if nxt_id == tokenizer.pad_id:
                    continue
                generated.append(nxt_id)
                input_tensor = torch.cat([input_tensor, nxt], dim=1)
                partial = tokenizer.decode(generated[len(tokens):], skip_special=True)
                if "質問:" in partial:
                    break

        generated_text = tokenizer.decode(generated[len(tokens):], skip_special=True)
        if "質問:" in generated_text:
            generated_text = generated_text[:generated_text.index("質問:")].strip()
        print(f'  {prompt.strip()} {generated_text}')

    print("\nDone!")


def save_checkpoint(model, config, training_log, original_ckpt):
    qa_datasets = [d["id"] for d in QA_DATASETS]
    all_datasets = list(set(
        original_ckpt.get("datasets", []) + qa_datasets
    ))
    new_checkpoint = {
        "model_state": model.state_dict(),
        "config": config,
        "training_log": training_log,
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "datasets": all_datasets,
        "qa_training": True,
        "qa_high_epoch": True,
    }
    torch.save(new_checkpoint, CKPT_PATH)
    size_mb = os.path.getsize(CKPT_PATH) / 1024 / 1024
    print(f"  Saved: {CKPT_PATH} ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
