#!/usr/bin/env python3
"""Training script that generates data from HF Inference Endpoint."""
import os
import sys
import torch
import torch.nn.functional as F
import json
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

# HF Inference Endpoint
ENDPOINT_URL = "https://vvcci2ps4y3wfx7m.us-east4.gcp.endpoints.huggingface.cloud"

# Training params
NUM_CHUNKS = 8
EPOCHS_PER_CHUNK = 6
BATCH_SIZE = 1
GRAD_ACCUM = 4
LR = 5e-5
WARMUP_STEPS = 30

# Prompts to query the endpoint
QUERY_PROMPTS = [
    # 一般知識
    "日本の首都はどこですか？",
    "富士山の高さは何メートルですか？",
    "太陽系で一番大きい惑星は何ですか？",
    "水の化学式は何ですか？",
    "光の速さはどれくらいですか？",
    "日本で一番長い川は何ですか？",
    "地球の年齢はどれくらいですか？",
    "人間の体で一番大きい臓器は何ですか？",
    # 歴史
    "明治維新は何年に起きましたか？",
    "第二次世界大戦はいつ終わりましたか？",
    "源頼朝は何をした人ですか？",
    "江戸時代はいつからいつまでですか？",
    # 科学
    "DNAとは何ですか？",
    "重力とは何ですか？",
    "光合成とは何ですか？",
    "原子とは何ですか？",
    # 文化
    "俳句とは何ですか？",
    "歌舞伎とは何ですか？",
    "茶道について教えてください。",
    "日本の国花は何ですか？",
    # 地理
    "日本で一番大きい湖は何ですか？",
    "北海道の県庁所在地はどこですか？",
    "日本の人口はどれくらいですか？",
    "東京タワーの高さは何メートルですか？",
    # 数学・論理
    "円周率とは何ですか？",
    "三角形の内角の和は何度ですか？",
    "素数とは何ですか？",
    "ピタゴラスの定理とは何ですか？",
    # 技術
    "人工知能とは何ですか？",
    "量子コンピュータとは何ですか？",
    "インターネットとは何ですか？",
    "プログラミングとは何ですか？",
    # 日常
    "健康のために大切なことは何ですか？",
    "読書のメリットは何ですか？",
    "環境問題について教えてください。",
    "睡眠はなぜ大切ですか？",
    # 追加
    "月の直径はどれくらいですか？",
    "酸素の元素記号は何ですか？",
    "日本の国歌は何ですか？",
    "ノーベル賞とは何ですか？",
]

# High-quality reference answers for crafted QA
CRAFTED_QA = [
    "質問: 日本の首都はどこですか？\n回答: 日本の首都は東京です。",
    "質問: 富士山の高さは？\n回答: 富士山の高さは3776メートルです。",
    "質問: 太陽系で一番大きい惑星は？\n回答: 木星が太陽系で最も大きい惑星です。",
    "質問: 水の化学式は？\n回答: 水の化学式はH2Oです。",
    "質問: 光の速さは？\n回答: 光の速さは秒速約30万キロメートルです。",
    "質問: 日本で一番長い川は？\n回答: 信濃川が日本で最も長い川で、全長367kmです。",
    "質問: 地球の年齢はどれくらい？\n回答: 地球の年齢は約46億年です。",
    "質問: 人間の体で一番大きい臓器は？\n回答: 皮膚が人間の体で最も大きい臓器です。",
    "質問: 明治維新は何年に起きましたか？\n回答: 明治維新は1868年に起きました。",
    "質問: 円周率とは何ですか？\n回答: 円周率は円の周の長さと直径の比で、約3.14159です。",
    "質問: 三角形の内角の和は？\n回答: 三角形の内角の和は180度です。",
    "質問: 日本の国花は？\n回答: 日本の国花は桜と菊です。",
    "質問: 琵琶湖はどこにありますか？\n回答: 琵琶湖は滋賀県にある日本最大の湖です。",
    "質問: 人工知能とは？\n回答: 人工知能（AI）は、人間の知的能力をコンピュータで実現する技術です。",
    "質問: 量子コンピュータとは？\n回答: 量子コンピュータは量子力学の原理を利用して計算を行うコンピュータです。",
    "質問: DNAとは何ですか？\n回答: DNAはデオキシリボ核酸の略で、遺伝情報を保持する分子です。",
]


def query_endpoint(prompt, max_new_tokens=150, temperature=0.7, retries=3):
    """Query the HF inference endpoint."""
    payload = {
        "inputs": f"質問: {prompt}\n回答:",
        "parameters": {
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
            "do_sample": True,
            "top_p": 0.9,
        }
    }
    for attempt in range(retries):
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
                    return result[0].get("generated_text", "")
                elif isinstance(result, dict):
                    return result.get("generated_text", "")
            elif resp.status_code == 503:
                # Model loading, wait
                time.sleep(5)
                continue
            else:
                print(f"    HTTP {resp.status_code}: {resp.text[:100]}")
        except Exception as e:
            if attempt < retries - 1:
                time.sleep(2 * (attempt + 1))
            else:
                print(f"    Error: {e}")
    return ""


def generate_training_data():
    """Generate training data by querying the endpoint."""
    print(f"  Querying endpoint: {ENDPOINT_URL}")
    print(f"  Prompts: {len(QUERY_PROMPTS)}")

    generated_texts = []
    for i, prompt in enumerate(QUERY_PROMPTS):
        text = query_endpoint(prompt)
        if text and len(text) > 10:
            # Format as QA pair
            full_text = f"質問: {prompt}\n回答:{text}" if not text.startswith("質問:") else text
            generated_texts.append(full_text)
            if (i + 1) % 10 == 0:
                print(f"    Generated {i+1}/{len(QUERY_PROMPTS)} samples")
        time.sleep(0.2)  # Rate limiting

    print(f"  Endpoint generated: {len(generated_texts)} samples")
    return generated_texts


def tokenize_texts(texts, tokenizer, max_seq_len):
    sequences = []
    for t in texts:
        ids = tokenizer.encode(t)
        if len(ids) > 2:
            sequences.append(ids[:max_seq_len])
    return sequences


def train_chunk(model, config, tokenizer, chunk_texts, chunk_idx, num_chunks, device):
    """Train one chunk."""
    print(f"\n{'='*60}")
    print(f"  Chunk {chunk_idx+1}/{num_chunks}: {len(chunk_texts)} texts")
    print(f"{'='*60}")

    max_seq_len = config["max_seq_len"]
    sequences = tokenize_texts(chunk_texts, tokenizer, max_seq_len)
    print(f"  Sequences: {len(sequences)}")

    if not sequences:
        print("  No sequences, skipping.")
        return None

    steps_per_epoch = len(sequences) // BATCH_SIZE
    total_steps = (steps_per_epoch * EPOCHS_PER_CHUNK) // GRAD_ACCUM
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    min_lr_ratio = 0.1

    model.train()
    global_step = 0
    best_loss = float('inf')

    for epoch in range(EPOCHS_PER_CHUNK):
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
                    lr = LR * global_step / max(WARMUP_STEPS, 1)
                else:
                    progress = (global_step - WARMUP_STEPS) / max(total_steps - WARMUP_STEPS, 1)
                    cosine_decay = 0.5 * (1 + math.cos(math.pi * progress))
                    lr = LR * (min_lr_ratio + (1 - min_lr_ratio) * cosine_decay)
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
        print(f"  Chunk {chunk_idx+1} Epoch {epoch+1}/{EPOCHS_PER_CHUNK} | Loss: {avg_loss:.4f} | Steps: {global_step}")

        if avg_loss < best_loss:
            best_loss = avg_loss

    del optimizer
    gc.collect()
    return best_loss


def save_checkpoint(model, config, chunk_info):
    """Save model checkpoint."""
    ckpt = {
        "model_state": model.state_dict(),
        "config": config,
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "chunk_info": chunk_info,
        "data_source": "hf_endpoint",
    }
    torch.save(ckpt, CKPT_PATH)
    size_mb = os.path.getsize(CKPT_PATH) / (1024 * 1024)
    print(f"  Checkpoint saved: {CKPT_PATH} ({size_mb:.1f}MB)")


def main():
    print("=" * 60)
    print("  Training from HF Inference Endpoint")
    print(f"  Endpoint: {ENDPOINT_URL}")
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
    print(f"  Model: {params:,} params on {device}")

    tokenizer = NeuroQuantumTokenizer(model_file=TOKENIZER_PATH)
    del ckpt
    gc.collect()

    # Generate data from endpoint
    print("\n[2/4] Generating data from endpoint...")
    endpoint_texts = generate_training_data()

    # Combine with crafted QA (heavy repeat for high-quality signal)
    all_texts = endpoint_texts.copy()
    crafted_repeats = 60  # Repeat crafted QA heavily to dominate training
    for _ in range(crafted_repeats):
        all_texts.extend(CRAFTED_QA)

    print(f"\n  Total training texts: {len(all_texts)}")
    print(f"    - From endpoint: {len(endpoint_texts)}")
    print(f"    - Crafted QA: {crafted_repeats * len(CRAFTED_QA)}")

    # Also load HF datasets as before for extra data
    print("\n[3/4] Loading additional HF datasets (streaming)...")
    try:
        import signal
        from datasets import load_dataset

        class TimeoutError(Exception):
            pass

        def timeout_handler(signum, frame):
            raise TimeoutError("Timed out")

        ds_list = [
            {"id": "fujiki/japanese_alpaca_data", "format": "alpaca"},
            {"id": "kunishou/oasst1-chat-44k-ja", "format": "conversations"},
        ]

        for ds_info in ds_list:
            ds_id = ds_info["id"]
            fmt = ds_info["format"]
            try:
                print(f"  Loading {ds_id} (streaming, max=200)...")
                signal.signal(signal.SIGALRM, timeout_handler)
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
                            if len(text) > 10:
                                all_texts.append(text)
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
                                if len(text) > 10:
                                    all_texts.append(text)
                                    count += 1
                    if count >= 200:
                        break
                signal.alarm(0)
                print(f"    -> {count} samples")
            except TimeoutError:
                signal.alarm(0)
                print(f"    -> Timed out, skipping")
            except Exception as e:
                signal.alarm(0)
                print(f"    -> Failed: {e}")
            gc.collect()
    except ImportError:
        print("  datasets not available, skipping HF datasets")

    print(f"  Final total texts: {len(all_texts)}")

    # Split into chunks
    random.shuffle(all_texts)
    chunk_size = max(len(all_texts) // NUM_CHUNKS, 1)
    chunks = []
    for i in range(NUM_CHUNKS):
        start = i * chunk_size
        end = start + chunk_size if i < NUM_CHUNKS - 1 else len(all_texts)
        chunks.append(all_texts[start:end])

    del all_texts
    gc.collect()
    print(f"  Chunks: {[len(c) for c in chunks]}")

    # Train
    print("\n[4/4] Training...")
    results = []
    for i in range(NUM_CHUNKS):
        best_loss = train_chunk(model, cfg, tokenizer, chunks[i], i, NUM_CHUNKS, device)
        results.append({"chunk": i + 1, "best_loss": best_loss})
        save_checkpoint(model, cfg, {"chunks_completed": i + 1, "results": results})
        chunks[i] = None
        gc.collect()

    # Summary
    print("\n" + "=" * 60)
    print("  Training Complete!")
    print("=" * 60)
    for r in results:
        print(f"  Chunk {r['chunk']}: Loss = {r['best_loss']:.4f}" if r['best_loss'] else f"  Chunk {r['chunk']}: skipped")
    print(f"\n  Checkpoint: {CKPT_PATH}")

    # Quick inference test
    print("\n[Test] Quick inference...")
    model.eval()
    test_prompt = "質問: 日本の首都は"
    ids = tokenizer.encode(test_prompt)
    input_t = torch.tensor([ids], dtype=torch.long, device=device)
    with torch.no_grad():
        for _ in range(30):
            logits = model(input_t)
            next_logits = logits[0, -1, :] / 0.7
            probs = torch.softmax(next_logits, dim=-1)
            next_id = torch.multinomial(probs, 1).item()
            input_t = torch.cat([input_t, torch.tensor([[next_id]], device=device)], dim=1)
            if next_id == tokenizer.eos_id:
                break
    output = tokenizer.decode(input_t[0].tolist())
    print(f"  Prompt: {test_prompt}")
    print(f"  Output: {output}")


if __name__ == "__main__":
    main()
