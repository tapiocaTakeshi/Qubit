#!/usr/bin/env python3
"""
megabyte_100mb 安定型学習スクリプト
HuggingFace Datasets v3互換のデータセットを使用
"""
import sys
import os
import torch
import torch.nn.functional as F
from typing import List

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from neuroquantum_layered import (
    NeuroQuantum,
    NeuroQuantumConfig,
    NeuroQuantumTokenizer,
    get_model_config_by_size,
)

def load_dataset_stable(dataset_id: str, max_samples: int, desc: str) -> List[str]:
    """安定したデータセット読み込み"""
    try:
        from datasets import load_dataset
        import random

        print(f"  📂 {desc}... ", end="", flush=True)

        # streaming=True でデータセットをロード
        ds = load_dataset(dataset_id, streaming=True, split="train")

        texts = []
        count = 0

        for sample in ds:
            if count >= max_samples:
                break

            # テキストを探す
            text = None
            for key in ['text', 'content', 'passage']:
                if key in sample and isinstance(sample[key], str) and len(sample[key]) > 10:
                    text = sample[key].strip()
                    break

            if text:
                texts.append(text[:500])
                count += 1

        if texts:
            random.shuffle(texts)
            print(f"✓ {len(texts)}")
            return texts
        else:
            print("✗ No texts")
            return []

    except Exception as e:
        print(f"✗ {str(e)[:30]}")
        return []

def tokenize_batch(texts: List[str], tokenizer, max_seq_len: int) -> List[List[int]]:
    """バッチ処理"""
    sequences = []

    for text in texts:
        try:
            ids = tokenizer.encode(text, add_special=False)[:max_seq_len-2]
            if len(ids) < 2:
                continue

            seq = [tokenizer.bos_id] + ids + [tokenizer.eos_id]
            pad_len = max_seq_len - len(seq)
            seq = seq + [tokenizer.pad_id] * pad_len

            sequences.append(seq[:max_seq_len])
        except:
            continue

    return sequences

def main():
    print("=" * 70)
    print("🚀 NeuroQuantum megabyte_100mb - Stable Multi-Dataset Training")
    print("=" * 70)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n✓ Device: {device}")

    # モデル設定
    config_dict = get_model_config_by_size("megabyte_100mb", vocab_size=8000)
    config = NeuroQuantumConfig(
        vocab_size=8000,
        embed_dim=config_dict["embed_dim"],
        hidden_dim=config_dict["hidden_dim"],
        num_heads=config_dict["num_heads"],
        num_layers=4,
        max_seq_len=256,
        dropout=0.1,
        lambda_entangle=0.5,
    )

    print("\n🔤 Tokenizer...")
    tokenizer = NeuroQuantumTokenizer(vocab_size=8000)
    print(f"  Vocab size: {len(tokenizer)}")

    print("\n🧠 Model...")
    model = NeuroQuantum(config=config, tokenizer=tokenizer).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Parameters: {n_params:,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    # 複数のデータセットを試す
    print("\n📚 Loading datasets (try multiple sources)...")

    all_sequences = []

    # 試すデータセットのリスト（安定性重視）
    datasets_to_try = [
        ("wikitext", 2000, "WikiText"),
        ("openwebtext", 2000, "OpenWebText"),
        ("big_patent", 2000, "Big Patent"),
        ("trec-covid", 2000, "TREC-COVID"),
    ]

    for dataset_id, max_samples, desc in datasets_to_try:
        texts = load_dataset_stable(dataset_id, max_samples, desc)
        if texts:
            sequences = tokenize_batch(texts, tokenizer, config.max_seq_len)
            all_sequences.extend(sequences)

    print(f"\n✓ Total sequences: {len(all_sequences)}")

    if all_sequences:
        # 実データで学習
        print("\n🚀 Training with real data...")
        import random

        model.train()
        random.shuffle(all_sequences)

        batch_size = 2
        for batch_idx in range(0, min(100, len(all_sequences)), batch_size):
            batch = all_sequences[batch_idx:batch_idx+batch_size]

            if len(batch) < 2:
                continue

            input_ids = torch.tensor(batch, dtype=torch.long, device=device)
            labels = torch.tensor([s[1:] + [tokenizer.pad_id] for s in batch],
                                dtype=torch.long, device=device)

            optimizer.zero_grad()
            logits = model(input_ids)
            loss = F.cross_entropy(logits.view(-1, config.vocab_size), labels.view(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            if batch_idx % 20 == 0:
                print(f"  Batch {batch_idx:3d}: Loss = {loss.item():.4f}")
    else:
        # ダミーデータで学習
        print("\n⚠️  Using dummy data for demo...")
        model.train()

        for batch_idx in range(30):
            input_ids = torch.randint(0, config.vocab_size, (2, 256), device=device)
            labels = torch.randint(0, config.vocab_size, (2, 256), device=device)

            optimizer.zero_grad()
            logits = model(input_ids)
            loss = F.cross_entropy(logits.view(-1, config.vocab_size), labels.view(-1))
            loss.backward()
            optimizer.step()

            if batch_idx % 10 == 0:
                print(f"  Batch {batch_idx:3d}: Loss = {loss.item():.4f}")

    # チェックポイント保存
    os.makedirs("./checkpoints", exist_ok=True)
    ckpt_path = "./checkpoints/megabyte_100mb_stable.pt"
    torch.save(model.state_dict(), ckpt_path)

    print(f"\n✅ Checkpoint saved: {ckpt_path}")
    print("\n🎉 Training completed!")

if __name__ == "__main__":
    main()
