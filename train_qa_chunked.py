#!/usr/bin/env python3
"""
QA形式の日本語データで分割ファインチューニング。
メモリ節約のため、データをチャンクに分けて順次学習。
各チャンク学習後にメモリ解放＆チェックポイント保存。
"""
import os
import sys
import gc
import torch
import torch.nn.functional as F
from datasets import load_dataset
from datetime import datetime, timezone
import random
import math

sys.path.insert(0, os.path.dirname(__file__))
from neuroquantum_layered import NeuroQuantum, NeuroQuantumConfig, NeuroQuantumTokenizer

CKPT_PATH = os.path.join(os.path.dirname(__file__), "neuroq_checkpoint.pt")

# Chunked training config
CHUNK_SIZE = 30000       # sequences per chunk
EPOCHS_PER_CHUNK = 3     # epochs per chunk
LR = 5e-5
BATCH_SIZE = 4
GRAD_ACCUM_STEPS = 4
WARMUP_STEPS = 20
GRAD_CLIP = 1.0


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


def format_qa_izumi(row):
    output = row.get("output", "").strip()
    instruction = row.get("input", row.get("instruction", "")).strip() if isinstance(row.get("input", ""), str) else ""
    if not output:
        return None
    if instruction:
        return f"質問: {instruction}\n回答: {output}"
    return f"回答: {output}"


def get_crafted_qa():
    """手作りQAデータ"""
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
        "質問: 1+1は？\n回答: 1+1は2です。",
        "質問: 日本で一番長い川は？\n回答: 日本で一番長い川は信濃川で、全長367キロメートルです。",
        "質問: 水の化学式は？\n回答: 水の化学式はH2Oです。水素原子2つと酸素原子1つからなります。",
        "質問: 東京タワーの高さは？\n回答: 東京タワーの高さは333メートルです。1958年に完成しました。",
        "質問: 日本の人口は？\n回答: 日本の人口は約1億2500万人です。",
    ]
    # 50回繰り返してパターン強化
    result = []
    for _ in range(50):
        result.extend(crafted)
    return result


def tokenize_texts(texts, tokenizer, max_seq_len):
    """テキストをトークン化（メモリ効率重視）"""
    sequences = []
    for t in texts:
        ids = tokenizer.encode(t, add_special=True)
        if len(ids) <= max_seq_len and len(ids) >= 4:
            sequences.append(ids)
        elif len(ids) > max_seq_len:
            stride = max(max_seq_len // 2, 1)
            for start in range(0, len(ids) - max_seq_len + 1, stride):
                sequences.append(ids[start:start + max_seq_len])
    return sequences


def get_lr(step, total_steps, warmup_steps, max_lr):
    if step < warmup_steps:
        return max_lr * step / max(warmup_steps, 1)
    progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
    return max_lr * 0.5 * (1 + math.cos(math.pi * progress))


def train_on_sequences(model, sequences, tokenizer, nq_config, device,
                       epochs, global_step, total_steps, chunk_name=""):
    """指定されたシーケンスで学習"""
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    model.train()
    max_seq_len = nq_config.max_seq_len
    best_loss = float('inf')

    for epoch in range(epochs):
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
                lr_now = get_lr(global_step, total_steps, WARMUP_STEPS, LR)
                for pg in optimizer.param_groups:
                    pg['lr'] = lr_now
                torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
                optimizer.step()
                optimizer.zero_grad()
                global_step += 1

            if n_batches % 500 == 0:
                avg = total_loss / n_batches
                print(f"  [{chunk_name}] Epoch {epoch+1}/{epochs} | Batch {n_batches} | Loss: {avg:.4f} | Step: {global_step}")

        if n_batches % GRAD_ACCUM_STEPS != 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimizer.step()
            optimizer.zero_grad()
            global_step += 1

        avg_loss = total_loss / max(n_batches, 1)
        print(f"  [{chunk_name}] Epoch {epoch+1}/{epochs} | Avg Loss: {avg_loss:.6f}")
        if avg_loss < best_loss:
            best_loss = avg_loss

    # Optimizer解放
    del optimizer
    gc.collect()

    return global_step, best_loss


def save_checkpoint(model, config, training_log, original_ckpt, datasets_used):
    all_ds = list(set(original_ckpt.get("datasets", []) + datasets_used))
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


def run_inference_test(model, tokenizer, max_seq_len, device):
    """QA推論テスト"""
    print("\n=== QA Inference test ===")
    model.eval()
    test_prompts = [
        "質問: 日本の首都はどこですか？\n回答:",
        "質問: プログラミングとは何ですか？\n回答:",
        "質問: AIとは何ですか？\n回答:",
        "質問: 量子コンピュータとは？\n回答:",
        "質問: 富士山について教えてください。\n回答:",
        "質問: 機械学習とは何ですか？\n回答:",
        "質問: 太陽系の惑星を教えてください。\n回答:",
    ]
    for prompt in test_prompts:
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
    model.train()


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load checkpoint
    print("=== Loading checkpoint ===")
    checkpoint = torch.load(CKPT_PATH, map_location="cpu")
    config = checkpoint["config"]
    prev_log = checkpoint.get("training_log", [])
    print(f"Config: embed_dim={config['embed_dim']}, layers={config['num_layers']}, vocab={config['vocab_size']}")
    if prev_log:
        print(f"Previous training: {len(prev_log)} epochs, last loss: {prev_log[-1]['loss']:.4f}")

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
    model.load_state_dict(checkpoint["model_state"])
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model loaded: {n_params:,} parameters")

    gc.collect()

    training_log = list(prev_log)
    global_step = 0
    datasets_used = []
    chunk_idx = 0

    # ========== Chunk 0: 手作りQA（パターン強化） ==========
    chunk_idx += 1
    print(f"\n{'='*60}")
    print(f"=== Chunk {chunk_idx}: 手作りQAデータ ===")
    print(f"{'='*60}")
    crafted_texts = get_crafted_qa()
    sequences = tokenize_texts(crafted_texts, tokenizer, max_seq_len)
    print(f"Sequences: {len(sequences)}")
    del crafted_texts
    gc.collect()

    total_steps_est = 50000  # rough estimate for LR scheduling
    global_step, loss = train_on_sequences(
        model, sequences, tokenizer, nq_config, device,
        epochs=5, global_step=global_step, total_steps=total_steps_est,
        chunk_name=f"Chunk{chunk_idx}-Crafted"
    )
    training_log.append({"epoch": len(training_log) + 1, "loss": loss, "chunk": "crafted_qa"})
    del sequences
    gc.collect()
    save_checkpoint(model, config, training_log, checkpoint, datasets_used)
    print(f"  Chunk {chunk_idx} done. Step: {global_step}")

    # ========== Chunk 1: fujiki/japanese_alpaca_data ==========
    chunk_idx += 1
    print(f"\n{'='*60}")
    print(f"=== Chunk {chunk_idx}: fujiki/japanese_alpaca_data ===")
    print(f"{'='*60}")
    try:
        ds = load_dataset("fujiki/japanese_alpaca_data", split="train")
        texts = []
        for row in ds:
            text = format_qa_alpaca(row)
            if text and len(text) > 10:
                texts.append(text)
        print(f"  Loaded {len(texts)} QA pairs")
        del ds
        gc.collect()

        # Tokenize and split into sub-chunks if needed
        sequences = tokenize_texts(texts, tokenizer, max_seq_len)
        del texts
        gc.collect()
        print(f"  Sequences: {len(sequences)}")

        if len(sequences) > CHUNK_SIZE:
            random.shuffle(sequences)
            sub_chunks = [sequences[i:i+CHUNK_SIZE] for i in range(0, len(sequences), CHUNK_SIZE)]
            del sequences
            gc.collect()
            for si, sub in enumerate(sub_chunks):
                print(f"  Sub-chunk {si+1}/{len(sub_chunks)}: {len(sub)} sequences")
                global_step, loss = train_on_sequences(
                    model, sub, tokenizer, nq_config, device,
                    epochs=EPOCHS_PER_CHUNK, global_step=global_step,
                    total_steps=total_steps_est,
                    chunk_name=f"Chunk{chunk_idx}-alpaca-{si+1}"
                )
                del sub
                gc.collect()
            del sub_chunks
        else:
            global_step, loss = train_on_sequences(
                model, sequences, tokenizer, nq_config, device,
                epochs=EPOCHS_PER_CHUNK, global_step=global_step,
                total_steps=total_steps_est,
                chunk_name=f"Chunk{chunk_idx}-alpaca"
            )
            del sequences

        gc.collect()
        datasets_used.append("fujiki/japanese_alpaca_data")
        training_log.append({"epoch": len(training_log) + 1, "loss": loss, "chunk": "alpaca"})
        save_checkpoint(model, config, training_log, checkpoint, datasets_used)
        print(f"  Chunk {chunk_idx} done. Step: {global_step}")
    except Exception as e:
        print(f"  ERROR: {e}")

    # ========== Chunk 2: FreedomIntelligence/alpaca-gpt4-japanese ==========
    chunk_idx += 1
    print(f"\n{'='*60}")
    print(f"=== Chunk {chunk_idx}: FreedomIntelligence/alpaca-gpt4-japanese ===")
    print(f"{'='*60}")
    try:
        ds = load_dataset("FreedomIntelligence/alpaca-gpt4-japanese", split="train")
        texts = []
        for row in ds:
            text = format_qa_conversations(row)
            if text and len(text) > 10:
                texts.append(text)
        print(f"  Loaded {len(texts)} QA pairs")
        del ds
        gc.collect()

        sequences = tokenize_texts(texts, tokenizer, max_seq_len)
        del texts
        gc.collect()
        print(f"  Sequences: {len(sequences)}")

        if len(sequences) > CHUNK_SIZE:
            random.shuffle(sequences)
            sub_chunks = [sequences[i:i+CHUNK_SIZE] for i in range(0, len(sequences), CHUNK_SIZE)]
            del sequences
            gc.collect()
            for si, sub in enumerate(sub_chunks):
                print(f"  Sub-chunk {si+1}/{len(sub_chunks)}: {len(sub)} sequences")
                global_step, loss = train_on_sequences(
                    model, sub, tokenizer, nq_config, device,
                    epochs=EPOCHS_PER_CHUNK, global_step=global_step,
                    total_steps=total_steps_est,
                    chunk_name=f"Chunk{chunk_idx}-gpt4ja-{si+1}"
                )
                del sub
                gc.collect()
            del sub_chunks
        else:
            global_step, loss = train_on_sequences(
                model, sequences, tokenizer, nq_config, device,
                epochs=EPOCHS_PER_CHUNK, global_step=global_step,
                total_steps=total_steps_est,
                chunk_name=f"Chunk{chunk_idx}-gpt4ja"
            )
            del sequences

        gc.collect()
        datasets_used.append("FreedomIntelligence/alpaca-gpt4-japanese")
        training_log.append({"epoch": len(training_log) + 1, "loss": loss, "chunk": "gpt4-japanese"})
        save_checkpoint(model, config, training_log, checkpoint, datasets_used)
        print(f"  Chunk {chunk_idx} done. Step: {global_step}")
    except Exception as e:
        print(f"  ERROR: {e}")

    # ========== Chunk 3: kunishou/oasst1-chat-44k-ja ==========
    chunk_idx += 1
    print(f"\n{'='*60}")
    print(f"=== Chunk {chunk_idx}: kunishou/oasst1-chat-44k-ja ===")
    print(f"{'='*60}")
    try:
        ds = load_dataset("kunishou/oasst1-chat-44k-ja", split="train")
        texts = []
        for row in ds:
            text = format_qa_conversations(row)
            if text and len(text) > 10:
                texts.append(text)
        print(f"  Loaded {len(texts)} QA pairs")
        del ds
        gc.collect()

        sequences = tokenize_texts(texts, tokenizer, max_seq_len)
        del texts
        gc.collect()
        print(f"  Sequences: {len(sequences)}")

        if len(sequences) > CHUNK_SIZE:
            random.shuffle(sequences)
            sub_chunks = [sequences[i:i+CHUNK_SIZE] for i in range(0, len(sequences), CHUNK_SIZE)]
            del sequences
            gc.collect()
            for si, sub in enumerate(sub_chunks):
                print(f"  Sub-chunk {si+1}/{len(sub_chunks)}: {len(sub)} sequences")
                global_step, loss = train_on_sequences(
                    model, sub, tokenizer, nq_config, device,
                    epochs=EPOCHS_PER_CHUNK, global_step=global_step,
                    total_steps=total_steps_est,
                    chunk_name=f"Chunk{chunk_idx}-oasst-{si+1}"
                )
                del sub
                gc.collect()
            del sub_chunks
        else:
            global_step, loss = train_on_sequences(
                model, sequences, tokenizer, nq_config, device,
                epochs=EPOCHS_PER_CHUNK, global_step=global_step,
                total_steps=total_steps_est,
                chunk_name=f"Chunk{chunk_idx}-oasst"
            )
            del sequences

        gc.collect()
        datasets_used.append("kunishou/oasst1-chat-44k-ja")
        training_log.append({"epoch": len(training_log) + 1, "loss": loss, "chunk": "oasst1"})
        save_checkpoint(model, config, training_log, checkpoint, datasets_used)
        print(f"  Chunk {chunk_idx} done. Step: {global_step}")
    except Exception as e:
        print(f"  ERROR: {e}")

    # ========== Chunk 4: izumi-lab (streaming, limited) ==========
    chunk_idx += 1
    print(f"\n{'='*60}")
    print(f"=== Chunk {chunk_idx}: izumi-lab/llm-japanese-dataset (streaming, max 30k) ===")
    print(f"{'='*60}")
    try:
        ds = load_dataset("izumi-lab/llm-japanese-dataset", split="train", streaming=True)
        texts = []
        count = 0
        max_izumi = 50000  # テキスト数上限
        for row in ds:
            text = format_qa_izumi(row)
            if text and len(text) > 10:
                texts.append(text)
                count += 1
                if count >= max_izumi:
                    break
            if count % 10000 == 0 and count > 0:
                print(f"    ... {count} samples loaded")
        print(f"  Loaded {len(texts)} samples")
        del ds
        gc.collect()

        sequences = tokenize_texts(texts, tokenizer, max_seq_len)
        del texts
        gc.collect()
        print(f"  Sequences: {len(sequences)}")

        if len(sequences) > CHUNK_SIZE:
            random.shuffle(sequences)
            sub_chunks = [sequences[i:i+CHUNK_SIZE] for i in range(0, len(sequences), CHUNK_SIZE)]
            del sequences
            gc.collect()
            for si, sub in enumerate(sub_chunks):
                print(f"  Sub-chunk {si+1}/{len(sub_chunks)}: {len(sub)} sequences")
                global_step, loss = train_on_sequences(
                    model, sub, tokenizer, nq_config, device,
                    epochs=EPOCHS_PER_CHUNK, global_step=global_step,
                    total_steps=total_steps_est,
                    chunk_name=f"Chunk{chunk_idx}-izumi-{si+1}"
                )
                del sub
                gc.collect()
            del sub_chunks
        else:
            global_step, loss = train_on_sequences(
                model, sequences, tokenizer, nq_config, device,
                epochs=EPOCHS_PER_CHUNK, global_step=global_step,
                total_steps=total_steps_est,
                chunk_name=f"Chunk{chunk_idx}-izumi"
            )
            del sequences

        gc.collect()
        datasets_used.append("izumi-lab/llm-japanese-dataset")
        training_log.append({"epoch": len(training_log) + 1, "loss": loss, "chunk": "izumi"})
        save_checkpoint(model, config, training_log, checkpoint, datasets_used)
        print(f"  Chunk {chunk_idx} done. Step: {global_step}")
    except Exception as e:
        print(f"  ERROR: {e}")

    # ========== Final: 最終チェックポイント保存 & 推論テスト ==========
    print(f"\n{'='*60}")
    print(f"=== Final: 最終保存 & 推論テスト ===")
    print(f"{'='*60}")
    save_checkpoint(model, config, training_log, checkpoint, datasets_used)
    print(f"Total optimization steps: {global_step}")
    print(f"Training log entries: {len(training_log)}")

    run_inference_test(model, tokenizer, max_seq_len, device)
    print("\n=== All chunks complete! ===")


if __name__ == "__main__":
    main()
