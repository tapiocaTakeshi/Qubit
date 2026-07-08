#!/usr/bin/env python3
"""
megabyte_100mb 15データセット学習スクリプト
前回指定のすべてのデータセットで学習
エラーハンドリング付き
"""
import sys
import os
import torch
import torch.nn.functional as F
from typing import List, Optional

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from neuroquantum_layered import (
    NeuroQuantum,
    NeuroQuantumConfig,
    NeuroQuantumTokenizer,
    get_model_config_by_size,
)

# ========================================
# 15 Datasets Configuration
# ========================================
DATASETS_CONFIG = [
    ("wikimedia/wikipedia", "20220301.ja", 1000, "日本語Wikipedia"),
    ("google/wiki40b", "ja", 1000, "Wiki40B日本語"),
    ("allenai/c4", "en", 1000, "CommonCrawl英語"),
    ("mc4", "ja", 1000, "多言語CommonCrawl日本語"),
    ("Open-Orca/OpenOrca", None, 1000, "指示フォロー"),
    ("HuggingFaceH4/ultrachat_200k", None, 1000, "チャット対話"),
    ("open-thoughts/OpenThoughts-114k", None, 1000, "思考過程"),
    ("argilla/ultrafeedback-binarized-preferences", None, 1000, "品質フィードバック"),
    ("bigcode/the-stack-v2", None, 1000, "プログラミングコード"),
    ("ise-uiuc/Magicoder-OSS-Instruct-75K", None, 1000, "コード指示"),
    ("WizardLMTeam/WizardCoder-Python-34K", None, 1000, "Pythonコード"),
    ("HuggingFaceH4/CodeAlpaca_20K", None, 1000, "コードアルパカ"),
    ("meta-math/MetaMathQA", None, 1000, "数学問題"),
    ("AI-MO/NuminaMath-CoT", None, 1000, "数学推論"),
    ("gsm8k", "main", 1000, "数学応用問題"),
]

def load_single_dataset(dataset_id: str, split: Optional[str], max_samples: int, desc: str):
    """単一のデータセットをロード"""
    try:
        from datasets import load_dataset

        print(f"  📂 {desc}... ", end="", flush=True)

        # split が None の場合は指定しない
        if split:
            ds = load_dataset(dataset_id, split, streaming=True)
        else:
            ds = load_dataset(dataset_id, split="train", streaming=True)

        # テキストを抽出
        texts = []
        count = 0

        for sample in ds:
            if count >= max_samples:
                break

            # テキストを探す
            text = None
            for key in ['text', 'document', 'content', 'passage', 'instruction', 'prompt']:
                if key in sample and isinstance(sample[key], str):
                    text = sample[key].strip()
                    if len(text) > 10:
                        break

            if text and len(text) > 10:
                texts.append(text[:1000])  # 最大1000文字
                count += 1

        if texts:
            print(f"✓ {len(texts)} examples")
            return texts
        else:
            print("✗ No valid texts")
            return []

    except Exception as e:
        print(f"✗ Error: {str(e)[:50]}")
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

            seq = [tokenizer.bos_id] + token_ids + [tokenizer.eos_id]

            # パディング
            pad_len = max_seq_len - len(seq)
            if pad_len > 0:
                seq = seq + [tokenizer.pad_id] * pad_len
            else:
                seq = seq[:max_seq_len]

            sequences.append(seq)

        except Exception:
            continue

    return sequences

def train_epoch(model, sequences: List[List[int]], tokenizer, optimizer,
                batch_size: int, max_seq_len: int, device: str) -> float:
    """1エポック分の学習"""
    import random

    model.train()
    random.shuffle(sequences)

    total_loss = 0.0
    n_batches = 0

    for i in range(0, len(sequences), batch_size):
        batch = sequences[i:i + batch_size]

        if not batch:
            continue

        input_ids = torch.tensor(batch, dtype=torch.long, device=device)
        labels = torch.tensor([s[1:] + [tokenizer.pad_id] for s in batch],
                            dtype=torch.long, device=device)

        optimizer.zero_grad()

        with torch.autocast("cuda" if device == "cuda" else "cpu", enabled=device == "cuda"):
            logits = model(input_ids)
            loss = F.cross_entropy(logits.view(-1, model.config.vocab_size), labels.view(-1))

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

    return total_loss / max(n_batches, 1)

def main():
    print("=" * 70)
    print("🚀 NeuroQuantum megabyte_100mb - 15 Datasets Training")
    print("=" * 70)

    # デバイス設定
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

    # トークナイザー
    print("\n🔤 Initializing tokenizer...")
    tokenizer = NeuroQuantumTokenizer(vocab_size=8000)
    print(f"  Vocab size: {len(tokenizer)}")

    # モデル初期化
    print("\n🧠 Initializing NeuroQuantum model...")
    model = NeuroQuantum(config=config, tokenizer=tokenizer).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Parameters: {n_params:,}")

    # オプティマイザー
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    # データセット読み込み
    print("\n📚 Loading 15 datasets...")
    all_sequences = []

    for dataset_id, split, max_samples, desc in DATASETS_CONFIG:
        texts = load_single_dataset(dataset_id, split, max_samples, desc)

        if texts:
            sequences = tokenize_texts(texts, tokenizer, config.max_seq_len)
            all_sequences.extend(sequences)
            print(f"      → {len(sequences)} sequences added")

    if not all_sequences:
        print("\n❌ No sequences loaded. Using dummy data for demo...")
        # ダミーデータで学習
        for epoch in range(1):
            print(f"\n🚀 Epoch {epoch + 1}/1")
            for batch_idx in range(20):
                input_ids = torch.randint(0, config.vocab_size, (2, 256), device=device)
                labels = torch.randint(0, config.vocab_size, (2, 256), device=device)

                optimizer.zero_grad()
                logits = model(input_ids)
                loss = F.cross_entropy(logits.view(-1, config.vocab_size), labels.view(-1))
                loss.backward()
                optimizer.step()

                if batch_idx % 5 == 0:
                    print(f"  Batch {batch_idx:3d}: Loss = {loss.item():.4f}")
    else:
        # 実データで学習
        print(f"\n✓ Total sequences: {len(all_sequences)}")

        for epoch in range(1):  # 1エポック
            print(f"\n🚀 Epoch {epoch + 1}/1")
            avg_loss = train_epoch(model, all_sequences, tokenizer, optimizer,
                                  batch_size=2, max_seq_len=config.max_seq_len,
                                  device=device)
            print(f"  Average Loss: {avg_loss:.4f}")

    # チェックポイント保存
    os.makedirs("./checkpoints", exist_ok=True)
    ckpt_path = "./checkpoints/megabyte_100mb_15datasets.pt"
    torch.save(model.state_dict(), ckpt_path)
    print(f"\n✅ Checkpoint saved: {ckpt_path}")
    print("\n🎉 Training completed!")

if __name__ == "__main__":
    main()
