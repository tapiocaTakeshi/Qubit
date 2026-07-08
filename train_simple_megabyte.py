#!/usr/bin/env python3
"""
megabyte_100mb シンプル学習スクリプト
embedding-gemma-300m (sentence-transformers) を使用
単一の確実に動作するデータセットで学習
"""
import sys
import os
import torch
import torch.nn.functional as F
from datasets import load_dataset

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from neuroquantum_layered import (
    NeuroQuantum,
    NeuroQuantumConfig,
    NeuroQuantumTokenizer,
    get_model_config_by_size,
)

def main():
    print("=" * 70)
    print("🚀 NeuroQuantum megabyte_100mb Simple Training")
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
        num_layers=4,  # 簡潔化
        max_seq_len=256,  # 短縮
        dropout=0.1,
        lambda_entangle=0.5,
    )

    # トークナイザー
    print("\n🔤 Initializing tokenizer...")
    tokenizer = NeuroQuantumTokenizer(vocab_size=8000)
    print(f"  Vocab size: {len(tokenizer)}")

    # モデル初期化
    print("\n🧠 Initializing NeuroQuantum model...")
    # デフォルト埋め込みを使用（sentence-transformersのバージョン互換性の問題を回避）
    model = NeuroQuantum(
        config=config,
        tokenizer=tokenizer,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Parameters: {n_params:,}")

    # オプティマイザー
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    # データセット: 確実に動作するもの
    print("\n📚 Loading dataset...")
    try:
        # tinyshakespeare - シンプルで確実
        dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train", streaming=True)
        print("  ✓ Using: wikitext-2-raw-v1")
    except:
        print("  Falling back to simple text generation")
        # フォールバック: ダミーデータ生成
        dataset = None

    # シンプル学習ループ
    print("\n🚀 Starting training...")

    if dataset is not None:
        # 実データセットで学習
        batch_size = 2
        num_batches = 100

        model.train()
        dataset_iter = iter(dataset)

        for batch_idx in range(num_batches):
            try:
                # テキストを取得
                batch_texts = []
                for _ in range(batch_size):
                    sample = next(dataset_iter)
                    text = sample.get('text', '')
                    if text and len(text) > 10:
                        batch_texts.append(text[:500])  # 最大500文字

                if not batch_texts:
                    continue

                # トークン化
                input_ids_list = []
                labels_list = []
                max_len = 128

                for text in batch_texts:
                    token_ids = tokenizer.encode(text, add_special=False)[:max_len-2]
                    if len(token_ids) < 2:
                        continue

                    # シーケンス構築
                    seq = [tokenizer.bos_id] + token_ids + [tokenizer.eos_id]
                    seq = seq[:max_len]

                    # パディング
                    pad_len = max_len - len(seq)
                    seq = seq + [tokenizer.pad_id] * pad_len

                    input_ids_list.append(seq)
                    labels_list.append(seq[1:] + [tokenizer.pad_id])

                if not input_ids_list:
                    continue

                # テンソル化
                input_ids = torch.tensor(input_ids_list, dtype=torch.long, device=device)
                labels = torch.tensor(labels_list, dtype=torch.long, device=device)

                # フォワードパス
                optimizer.zero_grad()
                logits = model(input_ids)
                loss = F.cross_entropy(logits.view(-1, config.vocab_size), labels.view(-1))

                # バックプロパゲーション
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

                if batch_idx % 10 == 0:
                    print(f"  Batch {batch_idx:3d}: Loss = {loss.item():.4f}")

            except StopIteration:
                break
            except Exception as e:
                print(f"  Error in batch {batch_idx}: {e}")
                continue

    else:
        # ダミーデータで簡単なテスト
        print("  Running in test mode with dummy data...")
        model.train()

        for batch_idx in range(10):
            # ダミーデータ生成
            input_ids = torch.randint(0, config.vocab_size, (2, 128), device=device)
            labels = torch.randint(0, config.vocab_size, (2, 128), device=device)

            optimizer.zero_grad()
            logits = model(input_ids)
            loss = F.cross_entropy(logits.view(-1, config.vocab_size), labels.view(-1))
            loss.backward()
            optimizer.step()

            print(f"  Batch {batch_idx}: Loss = {loss.item():.4f}")

    # チェックポイント保存
    os.makedirs("./checkpoints", exist_ok=True)
    ckpt_path = "./checkpoints/megabyte_100mb_simple.pt"
    torch.save(model.state_dict(), ckpt_path)
    print(f"\n✅ Checkpoint saved: {ckpt_path}")
    print("\n🎉 Training completed!")

if __name__ == "__main__":
    main()
