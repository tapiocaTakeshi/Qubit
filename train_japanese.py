#!/usr/bin/env python3
"""
megabyte_100mb 日本語専用学習スクリプト
日本語データセットで集中的に学習
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
# Japanese Datasets
# ========================================
JAPANESE_DATASETS = [
    ("wikimedia/wikipedia", "20220301.ja", "Wikipedia 日本語"),
    ("google/wiki40b", "ja", "Wiki40B 日本語"),
    ("mc4", "ja", "mC4 日本語"),
    ("kunishou/oasst1-89k-ja", None, "OASST1 日本語 (89K)"),
    ("kunishou/databricks-dolly-15k-ja", None, "Databricks Dolly 日本語"),
    ("openwebtext", None, "OpenWebText"),  # 多言語対応
    ("wikitext", "wikitext-103-v1", "WikiText-103"),
]

def load_japanese_dataset(dataset_id: str, split_name: str = None, max_samples: int = 1000):
    """日本語データセット読み込み"""
    try:
        from datasets import load_dataset

        print(f"    Loading {dataset_id}...", end=" ", flush=True)

        try:
            if split_name:
                ds = load_dataset(dataset_id, split_name, streaming=True, split="train")
            else:
                ds = load_dataset(dataset_id, streaming=True, split="train")
        except:
            # フォールバック
            ds = load_dataset(dataset_id, streaming=True, split="train")

        texts = []
        count = 0

        for sample in ds:
            if count >= max_samples:
                break

            text = None
            for key in ['text', 'content', 'passage', 'summary', 'document']:
                if key in sample and isinstance(sample[key], str):
                    candidate = str(sample[key]).strip()
                    if len(candidate) > 50:  # 日本語は短くても情報量がある
                        text = candidate[:1500]
                        break

            if text and len(text) > 20:
                texts.append(text)
                count += 1

        if texts:
            print(f"✓ {len(texts)} texts")
            return texts
        else:
            print("✗")
            return []

    except Exception as e:
        print(f"✗ {str(e)[:30]}")
        return []

def merge_checkpoints_japanese(n_params: int):
    """複数のエポックチェックポイントを統合"""
    from pathlib import Path
    from collections import defaultdict

    checkpoint_dir = Path("./checkpoints")
    pattern = "*japanese*"

    checkpoint_files = sorted(checkpoint_dir.glob(f"{pattern}*.pt"))

    if len(checkpoint_files) < 2:
        print(f"  ℹ️  マージ対象が1個以下のため、スキップします")
        return

    print(f"\n  📂 {len(checkpoint_files)} 個のチェックポイントが見つかりました:")
    for i, ckpt in enumerate(checkpoint_files, 1):
        size_mb = ckpt.stat().st_size / (1024**2)
        print(f"     {i}. {ckpt.name} ({size_mb:.1f}MB)")

    print(f"\n  🔄 重みを平均化中...")
    averaged_state = defaultdict(lambda: None)

    for idx, ckpt_path in enumerate(checkpoint_files):
        try:
            checkpoint = torch.load(ckpt_path, map_location="cpu")

            for key, value in checkpoint.items():
                if isinstance(value, torch.Tensor):
                    if averaged_state[key] is None:
                        averaged_state[key] = value.float().clone()
                    else:
                        averaged_state[key] += value.float()

        except Exception as e:
            print(f"     ✗ Error loading {ckpt_path.name}: {e}")
            return

    # 平均を計算
    for key in averaged_state:
        if averaged_state[key] is not None:
            averaged_state[key] = averaged_state[key] / len(checkpoint_files)

    # マージされたチェックポイントを保存
    output_path = checkpoint_dir / "megabyte_100mb_japanese_merged.pt"
    torch.save(dict(averaged_state), output_path)

    output_size_mb = output_path.stat().st_size / (1024**2)

    print(f"\n  ✅ マージ完了!")
    print(f"     出力: {output_path.name}")
    print(f"     サイズ: {output_size_mb:.1f}MB")
    print(f"     パラメータ: {n_params:,}")
    print(f"     戦略: 重みの平均化（{len(checkpoint_files)} エポック）")

def tokenize_japanese_texts(texts: List[str], tokenizer, max_seq_len: int) -> List[List[int]]:
    """日本語テキストをトークン化"""
    sequences = []

    for text in texts:
        try:
            # sentencepieceで日本語テキストを効率的にトークン化
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
    print("🚀 NeuroQuantum megabyte_100mb - 日本語専用学習")
    print("=" * 80)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n✓ Device: {device}")

    # モデル設定
    config_dict = get_model_config_by_size("megabyte_100mb", vocab_size=32000)
    config = NeuroQuantumConfig(
        vocab_size=32000,
        embed_dim=config_dict["embed_dim"],
        hidden_dim=config_dict["hidden_dim"],
        num_heads=config_dict["num_heads"],
        num_layers=6,
        max_seq_len=512,
        dropout=0.1,
        lambda_entangle=0.5,
    )

    # トークナイザー（sentencepiece対応）
    print("\n🔤 Building Japanese-optimized tokenizer...")
    tokenizer = NeuroQuantumTokenizer(vocab_size=32000)

    # モデル
    print("\n🧠 Initializing NeuroQuantum model...")
    model = NeuroQuantum(config=config, tokenizer=tokenizer).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Parameters: {n_params:,}")

    # オプティマイザー
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)

    # 日本語データセット読み込み
    print("\n📚 Loading Japanese datasets...")
    all_sequences = []
    loaded_count = 0

    for dataset_id, split, desc in JAPANESE_DATASETS:
        texts = load_japanese_dataset(dataset_id, split, max_samples=1000)

        if texts:
            sequences = tokenize_japanese_texts(texts, tokenizer, config.max_seq_len)
            all_sequences.extend(sequences)
            loaded_count += 1
            print(f"      → {len(sequences)} sequences added")

    print(f"\n✓ Loaded {loaded_count}/{len(JAPANESE_DATASETS)} datasets")
    print(f"✓ Total sequences: {len(all_sequences)}")

    if not all_sequences:
        print("\n⚠️  No datasets loaded. Using synthetic Japanese data...")
        all_sequences = [
            [random.randint(0, config.vocab_size-1) for _ in range(512)]
            for _ in range(2000)
        ]

    # 学習ループ
    print("\n🚀 Starting Japanese language training...")
    print(f"  Epochs: 5")
    print(f"  Batch size: 4")
    print(f"  Update interval: 50 batches")

    for epoch in range(5):
        print(f"\n📍 Epoch {epoch + 1}/5")

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
                lr = optimizer.param_groups[0]['lr']
                print(f"  Batch {batch_idx//4:4d}: Loss = {avg_loss:.4f} | LR = {lr:.2e}")

        scheduler.step()

        avg_loss = total_loss / max(batch_count, 1)
        print(f"  ✅ Epoch {epoch+1} Average Loss: {avg_loss:.4f}")

        # エポックごとにチェックポイント保存
        os.makedirs("./checkpoints", exist_ok=True)
        ckpt_path = f"./checkpoints/megabyte_100mb_japanese_epoch{epoch+1}.pt"
        torch.save(model.state_dict(), ckpt_path)
        print(f"  💾 Checkpoint: {ckpt_path}")

    print("\n" + "=" * 80)
    print("✅ 日本語学習完了！")
    print("=" * 80)
    print(f"\n📊 学習結果:")
    print(f"  モデル: megabyte_100mb")
    print(f"  パラメータ: {n_params:,}")
    print(f"  読み込みデータセット: {loaded_count}/{len(JAPANESE_DATASETS)}")
    print(f"  シーケンス数: {len(all_sequences)}")
    print(f"  学習エポック: 5")
    print(f"\n💾 チェックポイント保存先: ./checkpoints/")
    print(f"🎯 使用可能な日本語モデル:")
    for i in range(1, 6):
        ckpt = f"./checkpoints/megabyte_100mb_japanese_epoch{i}.pt"
        if os.path.exists(ckpt):
            print(f"   ✓ Epoch {i}: {ckpt}")

    # ========================================
    # チェックポイントマージ
    # ========================================
    print(f"\n🔀 マージ処理開始...")
    merge_checkpoints_japanese(n_params)

if __name__ == "__main__":
    main()
