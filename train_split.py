#!/usr/bin/env python3
"""
分割学習スクリプト: Hugging Faceデータセットを分割して学習する。
一度に全データを学習するとタイムアウトする問題を回避するため、
データセットをチャンクに分割し、各チャンクごとに学習・保存を行う。

使い方:
  # デフォルト設定で分割学習（4チャンクに分割）
  python train_split.py

  # チャンク数を指定
  python train_split.py --num_chunks 8

  # 特定のチャンクだけ学習（0始まり）
  python train_split.py --chunk_index 2

  # エポック数やサンプル数を指定
  python train_split.py --epochs 5 --max_samples 2000

  # QA形式で分割学習
  python train_split.py --mode qa --num_chunks 4

  # カスタムデータセットIDを指定して学習
  python train_split.py --dataset_ids "user/my-dataset" "user/another-dataset"

  # カスタムデータセット + 一般テキストモード
  python train_split.py --mode general --dataset_ids "range3/cc100-ja" --max_samples 5000
"""
import os
import sys
import argparse
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
SPLIT_STATE_PATH = os.path.join(os.path.dirname(__file__), "split_training_state.json")

# Default QA datasets
QA_DATASETS = [
    {"id": "fujiki/japanese_alpaca_data", "format": "alpaca"},
    {"id": "FreedomIntelligence/alpaca-gpt4-japanese", "format": "conversations"},
    {"id": "kunishou/oasst1-chat-44k-ja", "format": "conversations"},
    {"id": "izumi-lab/llm-japanese-dataset", "format": "izumi"},
]

# Default general datasets
GENERAL_DATASETS = [
    {"id": "izumi-lab/llm-japanese-dataset", "col": "output"},
    {"id": "kunishou/oasst1-chat-44k-ja", "col": "conversations"},
    {"id": "fujiki/japanese_alpaca_data", "col": "output"},
    {"id": "shi3z/Japanese_wikipedia_conversation_100K", "col": "conversations"},
    {"id": "FreedomIntelligence/alpaca-gpt4-japanese", "col": "conversations"},
]

CRAFTED_QA = [
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
]


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
        q_text = turn.get("value", turn.get("content", "")).strip() if isinstance(turn, dict) else (turn.strip() if isinstance(turn, str) else "")
        a_text = next_turn.get("value", next_turn.get("content", "")).strip() if isinstance(next_turn, dict) else (next_turn.strip() if isinstance(next_turn, str) else "")
        if q_text and a_text:
            pairs.append(f"質問: {q_text}\n回答: {a_text}")
        i += 2
    return "\n\n".join(pairs) if pairs else None


def format_qa_izumi(row):
    output = row.get("output", "").strip()
    instruction = row.get("input", row.get("instruction", "")).strip() if isinstance(row.get("input", ""), str) else ""
    if not output:
        return None
    return f"質問: {instruction}\n回答: {output}" if instruction else f"回答: {output}"


def extract_texts_general(ds, text_column, max_samples):
    texts = []
    n = min(max_samples, len(ds))
    for row in ds.select(range(n)):
        col_data = row.get(text_column)
        if isinstance(col_data, str) and len(col_data.strip()) > 4:
            texts.append(col_data.strip())
        elif isinstance(col_data, list):
            parts = []
            for turn in col_data:
                if isinstance(turn, dict) and "value" in turn:
                    parts.append(turn["value"])
                elif isinstance(turn, dict) and "content" in turn:
                    parts.append(turn["content"])
                elif isinstance(turn, str):
                    parts.append(turn)
            combined = "\n".join(parts)
            if len(combined.strip()) > 4:
                texts.append(combined.strip())
    return texts


def load_all_qa_texts(max_samples_per_dataset):
    """Load all QA texts from all datasets."""
    all_qa = []
    for ds_info in QA_DATASETS:
        ds_id = ds_info["id"]
        fmt = ds_info["format"]
        max_samples = max_samples_per_dataset
        if fmt == "izumi":
            max_samples = min(1000, max_samples)
        print(f"  Loading {ds_id}...")
        try:
            ds = load_dataset(ds_id, split="train", trust_remote_code=True)
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
    return all_qa


def load_all_general_texts(max_samples_per_dataset):
    """Load all general texts from all datasets."""
    all_texts = []
    for ds_info in GENERAL_DATASETS:
        ds_id = ds_info["id"]
        col = ds_info["col"]
        print(f"  Loading {ds_id}...")
        try:
            ds = load_dataset(ds_id, split="train", trust_remote_code=True)
            texts = extract_texts_general(ds, col, max_samples_per_dataset)
            print(f"    -> {len(texts)} texts")
            all_texts.extend(texts)
        except Exception as e:
            print(f"    -> ERROR: {e}")
    return all_texts


def load_custom_datasets(dataset_ids, max_samples, mode):
    """Load custom datasets by ID with auto-format detection."""
    all_texts = []
    for ds_id in dataset_ids:
        print(f"  Loading {ds_id}...")
        try:
            # まず通常ロードを試す、失敗したらstreaming
            try:
                ds = load_dataset(ds_id, split="train", trust_remote_code=True)
                is_streaming = False
            except Exception:
                ds = load_dataset(ds_id, split="train", streaming=True, trust_remote_code=True)
                is_streaming = True

            count = 0
            iterator = ds if is_streaming else ds.select(range(min(max_samples, len(ds))))
            for row in iterator:
                if count >= max_samples:
                    break
                text = None
                if mode == "qa":
                    # QA形式: 質問/回答ペアを自動検出
                    q = (row.get("question") or row.get("instruction") or row.get("input") or "")
                    if isinstance(q, str):
                        q = q.strip()
                    else:
                        q = ""
                    a = (row.get("answer") or row.get("output") or row.get("response") or "")
                    if isinstance(a, str):
                        a = a.strip()
                    else:
                        a = ""
                    if q and a and len(q) > 2 and len(a) > 2:
                        text = f"質問: {q}\n回答: {a}"
                    elif not q and a and len(a) > 10:
                        text = f"回答: {a}"
                    # conversations形式も試す
                    if not text:
                        text = format_qa_conversations(row)
                    # alpaca形式も試す
                    if not text and row.get("instruction"):
                        text = format_qa_alpaca(row)
                else:
                    # 一般テキスト: テキストフィールドを自動検出
                    for col in ["text", "content", "output", "sentence", "document"]:
                        val = row.get(col)
                        if isinstance(val, str) and len(val.strip()) > 10:
                            text = val.strip()
                            break
                    if not text:
                        convs = row.get("conversations", [])
                        if isinstance(convs, list) and convs:
                            parts = []
                            for turn in convs:
                                if isinstance(turn, dict):
                                    parts.append(turn.get("value", turn.get("content", "")))
                                elif isinstance(turn, str):
                                    parts.append(turn)
                            combined = "\n".join(parts)
                            if len(combined.strip()) > 10:
                                text = combined.strip()
                if text and len(text) > 10:
                    all_texts.append(text)
                    count += 1
            print(f"    -> {count} texts")
        except Exception as e:
            print(f"    -> ERROR: {e}")
    return all_texts


def split_into_chunks(data, num_chunks):
    """Split data into roughly equal chunks."""
    random.shuffle(data)
    chunk_size = math.ceil(len(data) / num_chunks)
    chunks = []
    for i in range(0, len(data), chunk_size):
        chunk = data[i:i + chunk_size]
        if chunk:
            chunks.append(chunk)
    return chunks


def tokenize_texts(texts, tokenizer, max_seq_len):
    sequences = []
    for t in texts:
        content_ids = tokenizer.encode(t, add_special=False)
        max_content = max_seq_len - 2
        if max_content <= 0:
            continue
        if len(content_ids) <= max_content:
            if len(content_ids) >= 2:
                seq = [tokenizer.bos_id] + content_ids + [tokenizer.eos_id]
                sequences.append(seq)
        else:
            stride = max(max_content // 2, 1)
            for start in range(0, len(content_ids) - max_content + 1, stride):
                chunk = content_ids[start:start + max_content]
                seq = [tokenizer.bos_id] + chunk + [tokenizer.eos_id]
                sequences.append(seq)
            remaining = content_ids[-max_content:]
            tail_seq = [tokenizer.bos_id] + remaining + [tokenizer.eos_id]
            if tail_seq != sequences[-1]:
                sequences.append(tail_seq)
    return sequences


def get_lr(step, total_steps, warmup_steps, max_lr, min_lr_ratio=0.1):
    if step < warmup_steps:
        return max_lr * step / max(warmup_steps, 1)
    progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
    cosine_decay = 0.5 * (1 + math.cos(math.pi * progress))
    return max_lr * (min_lr_ratio + (1 - min_lr_ratio) * cosine_decay)


def train_on_chunk(model, sequences, tokenizer, nq_config, device, args, chunk_idx, num_chunks):
    """Train model on a single chunk of data."""
    max_seq_len = nq_config.max_seq_len
    batch_size = args.batch_size
    grad_accum_steps = args.grad_accum_steps
    epochs = args.epochs_per_chunk
    lr = args.lr
    warmup_steps = args.warmup_steps
    grad_clip = args.grad_clip

    steps_per_epoch = len(sequences) // batch_size
    total_steps = (steps_per_epoch * epochs) // grad_accum_steps
    print(f"  Sequences: {len(sequences)}")
    print(f"  Steps/epoch: {steps_per_epoch}, Total opt steps: {total_steps}")
    print(f"  Effective batch size: {batch_size * grad_accum_steps}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)

    model.train()
    training_log = []
    global_step = 0
    best_loss = float('inf')

    for epoch in range(epochs):
        random.shuffle(sequences)
        total_loss = 0
        n_batches = 0
        optimizer.zero_grad()

        for i in range(0, len(sequences), batch_size):
            batch_seqs = sequences[i:i + batch_size]
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
            loss = loss / grad_accum_steps
            loss.backward()

            total_loss += loss.item() * grad_accum_steps
            n_batches += 1

            if n_batches % grad_accum_steps == 0:
                cur_lr = get_lr(global_step, total_steps, warmup_steps, lr)
                for pg in optimizer.param_groups:
                    pg['lr'] = cur_lr
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()
                optimizer.zero_grad()
                global_step += 1

            if n_batches % 200 == 0:
                avg = total_loss / n_batches
                cur_lr = get_lr(global_step, total_steps, warmup_steps, lr)
                print(f"    Chunk {chunk_idx+1}/{num_chunks} | Epoch {epoch+1}/{epochs} | "
                      f"Batch {n_batches} | Loss: {avg:.4f} | LR: {cur_lr:.2e}")

        if n_batches % grad_accum_steps != 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            optimizer.zero_grad()
            global_step += 1

        avg_loss = total_loss / max(n_batches, 1)
        print(f"  Chunk {chunk_idx+1}/{num_chunks} | Epoch {epoch+1}/{epochs} | Avg Loss: {avg_loss:.6f}")
        training_log.append({"epoch": epoch + 1, "loss": avg_loss})

        if avg_loss < best_loss:
            best_loss = avg_loss

    return training_log, best_loss


def save_checkpoint(model, config, training_log, original_ckpt, dataset_ids, mode):
    new_checkpoint = {
        "model_state": model.state_dict(),
        "config": config,
        "training_log": training_log,
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "datasets": list(set(original_ckpt.get("datasets", []) + dataset_ids)),
        "split_training": True,
    }
    if mode == "qa":
        new_checkpoint["qa_training"] = True
    torch.save(new_checkpoint, CKPT_PATH)
    size_mb = os.path.getsize(CKPT_PATH) / 1024 / 1024
    print(f"  Saved: {CKPT_PATH} ({size_mb:.1f} MB)")


def save_split_state(state):
    with open(SPLIT_STATE_PATH, "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=2)


def load_split_state():
    if os.path.exists(SPLIT_STATE_PATH):
        with open(SPLIT_STATE_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return None


def main():
    parser = argparse.ArgumentParser(description="分割学習: データセットをチャンクに分割して学習")
    parser.add_argument("--mode", choices=["qa", "general"], default="qa",
                        help="学習モード: qa=QA形式, general=一般テキスト (default: qa)")
    parser.add_argument("--num_chunks", type=int, default=4,
                        help="データセットを分割するチャンク数 (default: 4)")
    parser.add_argument("--chunk_index", type=int, default=None,
                        help="特定のチャンクのみ学習（0始まり）。指定なしで全チャンク順次学習")
    parser.add_argument("--epochs_per_chunk", type=int, default=5,
                        help="各チャンクあたりのエポック数 (default: 5)")
    parser.add_argument("--lr", type=float, default=3e-5,
                        help="学習率 (default: 3e-5)")
    parser.add_argument("--batch_size", type=int, default=4,
                        help="バッチサイズ (default: 4)")
    parser.add_argument("--grad_accum_steps", type=int, default=4,
                        help="勾配蓄積ステップ数 (default: 4)")
    parser.add_argument("--warmup_steps", type=int, default=20,
                        help="ウォームアップステップ数 (default: 20)")
    parser.add_argument("--grad_clip", type=float, default=1.0,
                        help="勾配クリッピング (default: 1.0)")
    parser.add_argument("--max_samples", type=int, default=2000,
                        help="各データセットからの最大サンプル数 (default: 2000)")
    parser.add_argument("--resume", action="store_true",
                        help="前回の分割学習を途中から再開")
    parser.add_argument("--crafted_repeat", type=int, default=20,
                        help="手作りQAの繰り返し回数（QAモードのみ、default: 20）")
    parser.add_argument("--dataset_ids", nargs="+", default=None,
                        help="カスタムHugging FaceデータセットID（スペース区切り）。指定時はデフォルトの代わりに使用")
    args = parser.parse_args()

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

    # Load data
    print(f"\n=== Loading datasets (mode: {args.mode}) ===")
    if args.dataset_ids:
        # カスタムデータセットIDが指定された場合
        print(f"  Custom datasets: {args.dataset_ids}")
        all_texts = load_custom_datasets(args.dataset_ids, args.max_samples, args.mode)
        if args.mode == "qa" and args.crafted_repeat > 0:
            for _ in range(args.crafted_repeat):
                all_texts.extend(CRAFTED_QA)
            print(f"  + {len(CRAFTED_QA) * args.crafted_repeat} crafted QA samples")
        dataset_ids = args.dataset_ids
    elif args.mode == "qa":
        all_texts = load_all_qa_texts(args.max_samples)
        # Add crafted QA
        for _ in range(args.crafted_repeat):
            all_texts.extend(CRAFTED_QA)
        print(f"  + {len(CRAFTED_QA) * args.crafted_repeat} crafted QA samples")
        dataset_ids = [d["id"] for d in QA_DATASETS]
    else:
        all_texts = load_all_general_texts(args.max_samples)
        dataset_ids = [d["id"] for d in GENERAL_DATASETS]

    print(f"\nTotal texts: {len(all_texts)}")

    if len(all_texts) == 0:
        print("ERROR: No texts loaded. Exiting.")
        return

    # Split into chunks
    chunks = split_into_chunks(all_texts, args.num_chunks)
    actual_num_chunks = len(chunks)
    print(f"\n=== Data split into {actual_num_chunks} chunks ===")
    for i, chunk in enumerate(chunks):
        print(f"  Chunk {i}: {len(chunk)} texts")

    # Determine which chunks to train
    if args.chunk_index is not None:
        if args.chunk_index >= actual_num_chunks:
            print(f"ERROR: chunk_index {args.chunk_index} >= num_chunks {actual_num_chunks}")
            return
        chunk_indices = [args.chunk_index]
    elif args.resume:
        state = load_split_state()
        if state and state.get("mode") == args.mode:
            last_completed = state.get("last_completed_chunk", -1)
            start_idx = last_completed + 1
            if start_idx >= actual_num_chunks:
                print("All chunks already completed! Use --chunk_index to retrain specific chunks.")
                return
            chunk_indices = list(range(start_idx, actual_num_chunks))
            print(f"Resuming from chunk {start_idx} (chunks {last_completed + 1} already done)")
        else:
            chunk_indices = list(range(actual_num_chunks))
    else:
        chunk_indices = list(range(actual_num_chunks))

    # Train on each chunk
    all_training_log = list(prev_log)

    for chunk_idx in chunk_indices:
        chunk_texts = chunks[chunk_idx]
        print(f"\n{'='*60}")
        print(f"=== Training Chunk {chunk_idx+1}/{actual_num_chunks} ({len(chunk_texts)} texts) ===")
        print(f"{'='*60}")

        # Tokenize this chunk
        sequences = tokenize_texts(chunk_texts, tokenizer, max_seq_len)
        print(f"  Tokenized: {len(sequences)} sequences")

        if len(sequences) == 0:
            print(f"  WARNING: No sequences in chunk {chunk_idx}. Skipping.")
            continue

        # Train
        chunk_log, best_loss = train_on_chunk(
            model, sequences, tokenizer, nq_config, device,
            args, chunk_idx, actual_num_chunks
        )

        # Append log with global epoch numbering
        for entry in chunk_log:
            all_training_log.append({
                "epoch": len(all_training_log) + 1,
                "loss": entry["loss"],
                "chunk": chunk_idx,
            })

        # Save checkpoint after each chunk
        print(f"\n  Saving checkpoint after chunk {chunk_idx+1}...")
        save_checkpoint(model, config, all_training_log, checkpoint, dataset_ids, args.mode)

        # Save split state for resume
        split_state = {
            "mode": args.mode,
            "num_chunks": actual_num_chunks,
            "last_completed_chunk": chunk_idx,
            "total_chunks_trained": chunk_idx + 1,
            "best_loss": best_loss,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "args": {
                "epochs_per_chunk": args.epochs_per_chunk,
                "lr": args.lr,
                "batch_size": args.batch_size,
                "max_samples": args.max_samples,
            }
        }
        save_split_state(split_state)
        print(f"  Split state saved (chunk {chunk_idx+1}/{actual_num_chunks} done)")

    # Final inference test
    print(f"\n=== Inference test ===")
    model.eval()
    if args.mode == "qa":
        test_prompts = [
            "質問: 日本の首都はどこですか？\n回答:",
            "質問: プログラミングとは何ですか？\n回答:",
            "質問: AIとは何ですか？\n回答:",
            "質問: 量子コンピュータとは何ですか？\n回答:",
        ]
    else:
        test_prompts = [
            "こんにちは",
            "量子コンピュータとは",
            "日本の首都は",
            "プログラミングを学ぶ",
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
                if args.mode == "qa":
                    partial = tokenizer.decode(generated[len(tokens):], skip_special=True)
                    if "質問:" in partial:
                        break

        generated_text = tokenizer.decode(generated[len(tokens):], skip_special=True)
        if args.mode == "qa" and "質問:" in generated_text:
            generated_text = generated_text[:generated_text.index("質問:")].strip()
        print(f'  {prompt.strip()} {generated_text}')

    print(f"\n=== Split training complete! ===")
    print(f"Trained on {len(chunk_indices)} chunk(s) out of {actual_num_chunks}")
    print("Done!")


if __name__ == "__main__":
    main()
