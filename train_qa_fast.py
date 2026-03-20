#!/usr/bin/env python3
"""
QA形式の日本語データでファインチューニング（軽量版）。
CPU環境対応: エポック数・データ数を最適化。
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
from neuroquantum_layered import NeuroQuantum, NeuroQuantumConfig, NeuroQuantumTokenizer, migrate_legacy_state_dict

CKPT_PATH = os.path.join(os.path.dirname(__file__), "neuroq_checkpoint.pt")

# CPU-optimized hyperparameters
EPOCHS = 3
LR = 5e-5
BATCH_SIZE = 4
GRAD_ACCUM_STEPS = 4
WARMUP_STEPS = 20
GRAD_CLIP = 1.0
MAX_SAMPLES = 2000  # Per dataset


def format_qa_alpaca(row):
    instruction = row.get("instruction", "").strip()
    inp = row.get("input", "").strip()
    output = row.get("output", "").strip()
    if not instruction or not output:
        return None
    q = f"{instruction}\n{inp}" if inp else instruction
    return f"質問: {q}\n回答: {output}"


def format_qa_conversations(row):
    convs = row.get("conversations", [])
    if not convs:
        return None
    pairs = []
    i = 0
    while i < len(convs) - 1:
        turn, next_turn = convs[i], convs[i + 1]
        q = turn.get("value", turn.get("content", "")).strip() if isinstance(turn, dict) else str(turn).strip()
        a = next_turn.get("value", next_turn.get("content", "")).strip() if isinstance(next_turn, dict) else str(next_turn).strip()
        if q and a:
            pairs.append(f"質問: {q}\n回答: {a}")
        i += 2
    return "\n\n".join(pairs) if pairs else None


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load checkpoint
    print("=== Loading checkpoint ===")
    checkpoint = torch.load(CKPT_PATH, map_location="cpu")
    config = checkpoint["config"]
    prev_log = checkpoint.get("training_log", [])
    print(f"Config: embed_dim={config['embed_dim']}, layers={config['num_layers']}")

    tokenizer_path = os.path.join(os.path.dirname(__file__), "neuroq_tokenizer.model")
    tokenizer = NeuroQuantumTokenizer(vocab_size=config["vocab_size"], model_file=tokenizer_path)

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
    migrated = migrate_legacy_state_dict(checkpoint["model_state"], model)
    model.load_state_dict(migrated)
    print(f"Model: {sum(p.numel() for p in model.parameters()):,} params")

    # Load QA data
    print("\n=== Loading QA datasets ===")
    qa_texts = []

    # Dataset 1: japanese_alpaca_data (alpaca format)
    print("  Loading fujiki/japanese_alpaca_data...")
    try:
        ds = load_dataset("fujiki/japanese_alpaca_data", split="train")
        n = min(MAX_SAMPLES, len(ds))
        count = 0
        for row in ds.select(range(n)):
            text = format_qa_alpaca(row)
            if text and len(text) > 10:
                qa_texts.append(text)
                count += 1
        print(f"    -> {count} QA pairs")
    except Exception as e:
        print(f"    -> ERROR: {e}")

    # Dataset 2: alpaca-gpt4-japanese (conversations)
    print("  Loading FreedomIntelligence/alpaca-gpt4-japanese...")
    try:
        ds = load_dataset("FreedomIntelligence/alpaca-gpt4-japanese", split="train")
        n = min(MAX_SAMPLES, len(ds))
        count = 0
        for row in ds.select(range(n)):
            text = format_qa_conversations(row)
            if text and len(text) > 10:
                qa_texts.append(text)
                count += 1
        print(f"    -> {count} QA pairs")
    except Exception as e:
        print(f"    -> ERROR: {e}")

    # Dataset 3: oasst1 (conversations)
    print("  Loading kunishou/oasst1-chat-44k-ja...")
    try:
        ds = load_dataset("kunishou/oasst1-chat-44k-ja", split="train")
        n = min(MAX_SAMPLES, len(ds))
        count = 0
        for row in ds.select(range(n)):
            text = format_qa_conversations(row)
            if text and len(text) > 10:
                qa_texts.append(text)
                count += 1
        print(f"    -> {count} QA pairs")
    except Exception as e:
        print(f"    -> ERROR: {e}")

    # Hand-crafted QA for pattern reinforcement
    crafted = [
        "質問: 日本の首都はどこですか？\n回答: 日本の首都は東京です。",
        "質問: 富士山の高さは？\n回答: 富士山の高さは3,776メートルです。日本一高い山です。",
        "質問: プログラミングとは何ですか？\n回答: プログラミングとは、コンピュータに命令を書くことです。Python、Java、C++などの言語を使います。",
        "質問: 人工知能とは？\n回答: 人工知能（AI）は人間の知能を模倣するコンピュータシステムです。機械学習や深層学習などの技術を含みます。",
        "質問: 量子コンピュータとは？\n回答: 量子コンピュータは量子力学の原理で計算を行うコンピュータです。量子ビットを使い従来困難な問題を解けます。",
        "質問: 機械学習とは？\n回答: 機械学習はデータからパターンを学習し予測を行うAIの一分野です。教師あり学習と教師なし学習があります。",
        "質問: Pythonとは？\n回答: Pythonは読みやすく書きやすい汎用プログラミング言語です。AI開発やWeb開発で広く使われています。",
        "質問: インターネットとは？\n回答: インターネットは世界中のコンピュータを接続した通信網です。Webやメールを支えています。",
        "質問: 太陽系の惑星は？\n回答: 太陽系には8つの惑星があります。水星、金星、地球、火星、木星、土星、天王星、海王星です。",
        "質問: 光の速さは？\n回答: 光の速さは秒速約30万キロメートルです。宇宙で最速です。",
        "質問: 日本語の文字は？\n回答: 日本語にはひらがな、カタカナ、漢字の3種類の文字体系があります。",
        "質問: 地球の年齢は？\n回答: 地球の年齢は約46億年です。",
        "質問: DNAとは？\n回答: DNAは生物の遺伝情報を保持する分子で、二重らせん構造を持ちます。",
        "質問: 相対性理論とは？\n回答: アインシュタインが提唱した物理学の理論で、時間と空間が相対的であることを示しました。",
        "質問: こんにちは\n回答: こんにちは！何かお手伝いできることはありますか？",
        "質問: 今日の天気は？\n回答: 申し訳ありませんが、リアルタイムの天気情報にはアクセスできません。天気予報サイトをご確認ください。",
        "質問: 1+1は？\n回答: 1+1は2です。",
        "質問: 日本で一番長い川は？\n回答: 日本で一番長い川は信濃川で、全長367キロメートルです。",
        "質問: 水の化学式は？\n回答: 水の化学式はH2Oです。水素原子2つと酸素原子1つからなります。",
        "質問: 東京タワーの高さは？\n回答: 東京タワーの高さは333メートルです。1958年に完成しました。",
    ]
    for _ in range(50):
        qa_texts.extend(crafted)
    print(f"  + {len(crafted) * 50} crafted QA samples")
    print(f"\nTotal QA texts: {len(qa_texts)}")

    # Tokenize
    print("\n=== Tokenizing ===")
    sequences = []
    for t in qa_texts:
        ids = tokenizer.encode(t, add_special=True)
        if len(ids) <= max_seq_len and len(ids) >= 4:
            sequences.append(ids)
        elif len(ids) > max_seq_len:
            stride = max(max_seq_len // 2, 1)
            for start in range(0, len(ids) - max_seq_len + 1, stride):
                sequences.append(ids[start:start + max_seq_len])
    print(f"Training sequences: {len(sequences)}")

    # Training
    steps_per_epoch = len(sequences) // BATCH_SIZE
    total_steps = (steps_per_epoch * EPOCHS) // GRAD_ACCUM_STEPS
    print(f"Steps/epoch: {steps_per_epoch}, Total opt steps: {total_steps}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
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
                lr_now = LR
                if global_step < WARMUP_STEPS:
                    lr_now = LR * global_step / max(WARMUP_STEPS, 1)
                else:
                    progress = (global_step - WARMUP_STEPS) / max(total_steps - WARMUP_STEPS, 1)
                    lr_now = LR * 0.5 * (1 + math.cos(math.pi * progress))
                for pg in optimizer.param_groups:
                    pg['lr'] = lr_now
                torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
                optimizer.step()
                optimizer.zero_grad()
                global_step += 1

            if n_batches % 500 == 0:
                avg = total_loss / n_batches
                print(f"  Epoch {epoch+1}/{EPOCHS} | Batch {n_batches}/{steps_per_epoch} | Loss: {avg:.4f} | Step: {global_step}")

        if n_batches % GRAD_ACCUM_STEPS != 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimizer.step()
            optimizer.zero_grad()
            global_step += 1

        avg_loss = total_loss / max(n_batches, 1)
        print(f"Epoch {epoch+1}/{EPOCHS} | Avg Loss: {avg_loss:.6f}")
        training_log.append({"epoch": len(prev_log) + epoch + 1, "loss": avg_loss})

        if avg_loss < best_loss:
            best_loss = avg_loss
            print(f"  Best loss: {best_loss:.4f}, saving...")
            save_ckpt(model, config, prev_log + training_log, checkpoint)

    save_ckpt(model, config, prev_log + training_log, checkpoint)

    # QA Inference test
    print("\n=== QA Inference test ===")
    model.eval()
    test_prompts = [
        "質問: 日本の首都はどこですか？\n回答:",
        "質問: プログラミングとは何ですか？\n回答:",
        "質問: AIとは何ですか？\n回答:",
        "質問: 量子コンピュータとは？\n回答:",
        "質問: 富士山について教えてください。\n回答:",
    ]
    for prompt in test_prompts:
        tokens = tokenizer.encode(prompt, add_special=True)
        input_tensor = torch.tensor([tokens], dtype=torch.long, device=device)
        generated = list(tokens)
        with torch.no_grad():
            for _ in range(80):
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
        text = tokenizer.decode(generated[len(tokens):], skip_special=True)
        if "質問:" in text:
            text = text[:text.index("質問:")].strip()
        print(f'  {prompt.strip()} {text}')

    print("\nDone!")


def save_ckpt(model, config, training_log, original_ckpt):
    qa_ds = ["fujiki/japanese_alpaca_data", "FreedomIntelligence/alpaca-gpt4-japanese",
             "kunishou/oasst1-chat-44k-ja"]
    all_ds = list(set(original_ckpt.get("datasets", []) + qa_ds))
    ckpt = {
        "model_state": model.state_dict(),
        "config": config,
        "training_log": training_log,
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "datasets": all_ds,
        "qa_training": True,
    }
    torch.save(ckpt, CKPT_PATH)
    print(f"  Saved: {os.path.getsize(CKPT_PATH) / 1024 / 1024:.1f} MB")


if __name__ == "__main__":
    main()
