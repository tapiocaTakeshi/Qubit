#!/usr/bin/env python3
"""
QA形式の日本語データでファインチューニング。
質問→回答のパターンを学習させる。
"""
import os
import sys
import torch
import torch.nn.functional as F
from dataset_utils import safe_load_dataset
from datetime import datetime, timezone
import json
import random
import math

sys.path.insert(0, os.path.dirname(__file__))
from neuroquantum_layered import NeuroQuantum, NeuroQuantumConfig, NeuroQuantumTokenizer, migrate_legacy_state_dict

CKPT_PATH = os.path.join(os.path.dirname(__file__), "neuroq_checkpoint.pt")

# Training hyperparameters
EPOCHS = 10
LR = 5e-5
BATCH_SIZE = 4
GRAD_ACCUM_STEPS = 8
WARMUP_STEPS = 50
GRAD_CLIP = 1.0

# QA datasets
QA_DATASETS = [
    {
        "id": "fujiki/japanese_alpaca_data",
        "max_samples": 8000,
        "format": "alpaca",  # instruction, input, output
    },
    {
        "id": "FreedomIntelligence/alpaca-gpt4-japanese",
        "max_samples": 8000,
        "format": "conversations",  # list of turns
    },
    {
        "id": "kunishou/oasst1-chat-44k-ja",
        "max_samples": 8000,
        "format": "conversations",
    },
    {
        "id": "izumi-lab/llm-japanese-dataset",
        "max_samples": 3000,
        "format": "izumi",  # output field with Q&A
    },
]


def format_qa_alpaca(row):
    """Format alpaca-style data as QA."""
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
    """Format conversation-style data as QA pairs."""
    convs = row.get("conversations", [])
    if not convs:
        return None
    pairs = []
    i = 0
    while i < len(convs) - 1:
        turn = convs[i]
        next_turn = convs[i + 1]
        # Extract text from turn
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
    """Format izumi-lab data as QA."""
    output = row.get("output", "").strip()
    instruction = row.get("input", row.get("instruction", "")).strip() if isinstance(row.get("input", ""), str) else ""
    if not output:
        return None
    if instruction:
        return f"質問: {instruction}\n回答: {output}"
    return f"回答: {output}"


def load_qa_data():
    """Load and format all QA datasets."""
    all_qa = []

    for ds_info in QA_DATASETS:
        ds_id = ds_info["id"]
        fmt = ds_info["format"]
        max_samples = ds_info["max_samples"]
        print(f"  Loading {ds_id}...")

        try:
            ds = safe_load_dataset(ds_id, split="train")
            n = min(max_samples, len(ds))
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

    # Add hand-crafted QA examples for pattern reinforcement
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
    ]
    # Repeat crafted examples to reinforce QA pattern
    for _ in range(20):
        all_qa.extend(crafted_qa)
    print(f"  + {len(crafted_qa) * 20} crafted QA samples")

    return all_qa


def tokenize_texts(texts, tokenizer, max_seq_len):
    """Tokenize texts into training sequences with BOF/EOF boundary markers."""
    sequences = []
    for t in texts:
        ids = tokenizer.encode(t, add_special=True, add_boundary=True)
        if len(ids) <= max_seq_len:
            if len(ids) >= 4:
                sequences.append(ids)
        else:
            stride = max(max_seq_len // 2, 1)
            chunks = list(range(0, len(ids) - max_seq_len + 1, stride))
            for idx, start in enumerate(chunks):
                chunk = ids[start:start + max_seq_len]
                # 先頭チャンク以外はBOFを除去、末尾チャンク以外はEOFを除去
                if idx > 0 and chunk[0] == tokenizer.bof_id:
                    chunk = chunk[1:]
                if idx < len(chunks) - 1 and chunk[-1] == tokenizer.eof_id:
                    chunk = chunk[:-1]
                sequences.append(chunk)
    return sequences


def get_lr(step, total_steps, warmup_steps, max_lr):
    """Learning rate with linear warmup and cosine decay."""
    if step < warmup_steps:
        return max_lr * step / max(warmup_steps, 1)
    progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
    return max_lr * 0.5 * (1 + math.cos(math.pi * progress))


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
    migrated = migrate_legacy_state_dict(checkpoint["model_state"], model)
    model.load_state_dict(migrated)
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

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)

    # Training loop
    print(f"\n=== Training for {EPOCHS} epochs (QA format) ===")
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

            if n_batches % 200 == 0:
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
    ]
    for prompt in test_qa:
        tokens = tokenizer.encode(prompt, add_special=True)
        input_tensor = torch.tensor([tokens], dtype=torch.long, device=device)
        generated = list(tokens)

        with torch.no_grad():
            for _ in range(100):
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
                if nxt_id in (tokenizer.eos_id, tokenizer.eof_id):
                    break
                if nxt_id in (tokenizer.pad_id, tokenizer.bof_id):
                    continue
                # Stop if model starts new question
                generated.append(nxt_id)
                input_tensor = torch.cat([input_tensor, nxt], dim=1)
                # Check for new 質問: marker
                partial = tokenizer.decode(generated[len(tokens):], skip_special=True)
                if "質問:" in partial:
                    break

        generated_text = tokenizer.decode(generated[len(tokens):], skip_special=True)
        # Trim if it started a new question
        if "質問:" in generated_text:
            generated_text = generated_text[:generated_text.index("質問:")].strip()
        print(f'  {prompt.strip()} {generated_text}')

    print("\nDone!")


def save_checkpoint(model, config, training_log, original_ckpt):
    """Save checkpoint."""
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
    }
    torch.save(new_checkpoint, CKPT_PATH)
    size_mb = os.path.getsize(CKPT_PATH) / 1024 / 1024
    print(f"  Saved: {CKPT_PATH} ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
