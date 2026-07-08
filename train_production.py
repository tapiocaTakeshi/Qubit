#!/usr/bin/env python3
"""
megabyte_100mb 本格学習スクリプト
15データセット + sentencepiece トークナイザー
"""
import sys
import os
import torch
import torch.nn.functional as F
from typing import List
import random

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from neuroquantum_layered import (
    NeuroQuantum,
    NeuroQuantumConfig,
    NeuroQuantumTokenizer,
    get_model_config_by_size,
)

# ========================================
# 15 Dataset IDs (simplified, more stable)
# ========================================
DATASETS = [
    ("wikitext", "wikitext-2-raw-v1", "WikiText-2"),
    ("wikitext", "wikitext-103-v1", "WikiText-103"),
    ("openwebtext", None, "OpenWebText"),
    ("big_patent", None, "Big Patent"),
    ("cc_news", None, "CC News"),
    ("c4", "en", "C4 English"),
    ("wikipedia", "20220301.en", "Wikipedia EN"),
    ("arxiv_dataset", None, "ArXiv"),
    ("billsum", None, "BillSum"),
    ("cnn_dailymail", "3.0.0", "CNN/DailyMail"),
    ("gigaword", None, "Gigaword"),
    ("multi_woz_v22", None, "MultiWOZ"),
    ("pubmed", None, "PubMed"),
    ("scientific_papers", "arxiv", "Scientific Papers"),
    ("imdb", None, "IMDB"),
]

def safe_load_dataset(dataset_id: str, split_name: str = None, max_samples: int = 500):
    """安全なデータセット読み込み"""
    try:
        from datasets import load_dataset

        if split_name:
            ds = load_dataset(dataset_id, split_name, streaming=True, split="train")
        else:
            ds = load_dataset(dataset_id, streaming=True, split="train")

        texts = []
        for sample in ds:
            if len(texts) >= max_samples:
                break

            # テキストを探す
            text = None
            for key in ['text', 'content', 'passage', 'summary', 'document', 'abstract']:
                if key in sample and isinstance(sample[key], str) and len(str(sample[key])) > 20:
                    text = str(sample[key]).strip()[:1000]
                    break

            if text and len(text) > 20:
                texts.append(text)

        return texts
    except Exception as e:
        return []

def tokenize_texts(texts: List[str], tokenizer, max_seq_len: int) -> List[List[int]]:
    """テキストをトークン化"""
    sequences = []

    for text in texts:
        try:
            token_ids = tokenizer.encode(text, add_special=False)

            if len(token_ids) < 2:
                continue

            # シーケンス構築
            max_content = max_seq_len - 4
            if len(token_ids) > max_content:
                token_ids = token_ids[:max_content]

            seq = [tokenizer.bof_id, tokenizer.bos_id] + token_ids + \
                  [tokenizer.eos_id, tokenizer.eof_id]

            # パディング
            pad_len = max_seq_len - len(seq)
            if pad_len > 0:
                seq = seq + [tokenizer.pad_id] * pad_len
            else:
                seq = seq[:max_seq_len]

            sequences.append(seq)
        except:
            continue

    return sequences

def main():
    print("=" * 80)
    print("🚀 NeuroQuantum megabyte_100mb - PRODUCTION TRAINING (15 Datasets)")
    print("=" * 80)

    # デバイス
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n✓ Device: {device}")

    # モデル設定
    config_dict = get_model_config_by_size("megabyte_100mb", vocab_size=32000)
    config = NeuroQuantumConfig(
        vocab_size=32000,
        embed_dim=config_dict["embed_dim"],
        hidden_dim=config_dict["hidden_dim"],
        num_heads=config_dict["num_heads"],
        num_layers=6,  # 増加
        max_seq_len=512,
        dropout=0.1,
        lambda_entangle=0.5,
    )

    # トークナイザー（sentencepiece対応）
    print("\n🔤 Building tokenizer with sentencepiece...")
    tokenizer = NeuroQuantumTokenizer(vocab_size=32000)

    # モデル
    print("\n🧠 Initializing NeuroQuantum model...")
    model = NeuroQuantum(config=config, tokenizer=tokenizer).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Parameters: {n_params:,}")

    # オプティマイザー
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)

    # データセット読み込み
    print("\n📚 Loading 15 datasets...")
    all_sequences = []
    loaded_count = 0

    for dataset_id, split, desc in DATASETS:
        print(f"  {desc:20s}... ", end="", flush=True)
        texts = safe_load_dataset(dataset_id, split, max_samples=500)

        if texts:
            sequences = tokenize_texts(texts, tokenizer, config.max_seq_len)
            all_sequences.extend(sequences)
            loaded_count += 1
            print(f"✓ {len(sequences)} seqs")
        else:
            print("✗")

    print(f"\n✓ Loaded {loaded_count}/15 datasets, Total sequences: {len(all_sequences)}")

    if not all_sequences:
        print("⚠️  No datasets loaded. Using synthetic data...")
        all_sequences = [
            [random.randint(0, config.vocab_size-1) for _ in range(512)]
            for _ in range(1000)
        ]

    # 学習ループ
    print("\n🚀 Starting training...")
    print(f"  Epochs: 3")
    print(f"  Batch size: 4")
    print(f"  Total batches per epoch: {len(all_sequences) // 4}")

    for epoch in range(3):
        print(f"\n📍 Epoch {epoch + 1}/3")

        model.train()
        random.shuffle(all_sequences)

        total_loss = 0.0
        batch_count = 0

        for batch_idx in range(0, len(all_sequences), 4):
            batch = all_sequences[batch_idx:batch_idx+4]

            if len(batch) < 2:
                continue

            input_ids = torch.tensor(batch, dtype=torch.long, device=device)
            labels = torch.tensor([s[1:] + [tokenizer.pad_id] for s in batch],
                                dtype=torch.long, device=device)

            optimizer.zero_grad()

            with torch.autocast(device, enabled=device=="cuda"):
                logits = model(input_ids)
                loss = F.cross_entropy(logits.view(-1, config.vocab_size), labels.view(-1))

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()
            batch_count += 1

            if (batch_idx // 4) % 50 == 0 and batch_idx > 0:
                avg_loss = total_loss / batch_count
                print(f"  Batch {batch_idx//4:4d}: Loss = {avg_loss:.4f}")

        scheduler.step()

        avg_loss = total_loss / max(batch_count, 1)
        print(f"  ✓ Epoch {epoch+1} Average Loss: {avg_loss:.4f}")

        # エポックごとにチェックポイント保存
        os.makedirs("./checkpoints", exist_ok=True)
        ckpt_path = f"./checkpoints/megabyte_100mb_epoch{epoch+1}.pt"
        torch.save(model.state_dict(), ckpt_path)
        print(f"  ✓ Checkpoint saved: {ckpt_path}")

    print("\n" + "=" * 80)
    print("✅ TRAINING COMPLETED!")
    print("=" * 80)
    print(f"\n📊 Final Results:")
    print(f"  Model: megabyte_100mb")
    print(f"  Parameters: {n_params:,}")
    print(f"  Datasets: {loaded_count}/15")
    print(f"  Total sequences: {len(all_sequences)}")
    print(f"  Epochs completed: 3")
    print(f"\n💾 Checkpoints saved in ./checkpoints/")

if __name__ == "__main__":
    main()
