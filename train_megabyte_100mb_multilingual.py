#!/usr/bin/env python3
"""
megabyte_100mb モデルの多言語・多分野学習スクリプト

embedding-gemma-300m（日本語対応テキストエンベディング）を使用して、
複数のHuggingFaceデータセットで学習します。

使用データセット：
1. wikimedia/wikipedia (英語・多言語)
2. google/wiki40b (40言語のWikipedia)
3. allenai/c4 (CommonCrawl)
4. mc4 (多言語CommonCrawl)
5. Open-Orca/OpenOrca (指示フォロー)
6. HuggingFaceH4/ultrachat_200k (チャット)
7. open-thoughts/OpenThoughts-114k (思考過程)
8. argilla/ultrafeedback-binarized-preferences (品質フィードバック)
9. bigcode/the-stack-v2 (コード)
10. ise-uiuc/Magicoder-OSS-Instruct-75K (コード指示)
11. WizardLMTeam/WizardCoder-Python-34K (Pythonコード)
12. HuggingFaceH4/CodeAlpaca_20K (コードアルパカ)
13. meta-math/MetaMathQA (数学)
14. AI-MO/NuminaMath-CoT (数学推論)
15. gsm8k (数学応用)

使用方法:
    python train_megabyte_100mb_multilingual.py \\
        --epochs 3 \\
        --batch-size 4 \\
        --save-dir ./checkpoints \\
        --max-samples-per-dataset 10000
"""

import argparse
import json
import os
import sys
from datetime import datetime
from typing import Optional, List

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dataset_utils import safe_load_dataset
from neuroquantum_layered import (
    NeuroQuantum,
    NeuroQuantumConfig,
    NeuroQuantumTokenizer,
    get_model_config_by_size,
)
from progress_logger import ProgressLogger

# ==========================================
# データセット定義
# ==========================================

DATASETS = [
    # テキスト・一般的なデータセット
    ("wikitext", "wikitext-103-v1"),  # WikiText
    ("openwebtext", None),  # OpenWebText

    # 指示フォロー・会話データ
    ("databricks/databricks-dolly-15k", None),  # Databricks Dolly
    ("allenai/real-toxicity-prompts", None),  # Prompts

    # 数学データ
    ("gsm8k", "main"),  # Grade School Math

    # コードデータ
    ("codeparrot/github-code", None),  # GitHub Code
]

progress = ProgressLogger("train_megabyte_100mb_multilingual")


def _first_str(row, keys):
    """行から最初に見つかった非空文字列を返す"""
    for k in keys:
        val = row.get(k)
        if isinstance(val, str) and val.strip():
            return val.strip()
    return None


def extract_generic_text(row):
    """行からテキストを抽出（形式自動検出）"""
    # 対話形式を試す
    instruction = _first_str(row, ("instruction", "prompt", "question", "query"))
    output = _first_str(row, ("output", "response", "answer", "completion", "text"))
    context = _first_str(row, ("context", "input", "document", "content"))

    if instruction and output:
        parts = [f"<USER> {instruction}"]
        if context:
            parts.append(f"<CONTEXT> {context}")
        parts.append(f"<ASSISTANT> {output}")
        return "\n".join(parts).strip()

    # 単純なテキスト列を試す
    text_candidates = ["text", "document", "content", "passage", "body"]
    for col in text_candidates:
        text = _first_str(row, (col,))
        if text:
            return text

    # 最後に非空の文字列値を探す
    for v in row.values():
        if isinstance(v, str) and v.strip() and len(v.strip()) > 4:
            return v.strip()

    return None


def load_dataset_texts(dataset_id: str, split: Optional[str], max_samples: int) -> List[str]:
    """データセットからテキストを抽出"""
    print(f"\n📂 Loading {dataset_id}...")

    try:
        ds = safe_load_dataset(dataset_id, split=split)
        if ds is None:
            print(f"⚠️  Failed to load {dataset_id}")
            return []

        texts = []
        n = len(ds) if max_samples is None or max_samples <= 0 else min(max_samples, len(ds))

        for i, row in enumerate(ds.select(range(n))):
            if i % 1000 == 0:
                print(f"  Processing {i}/{n}...")
            text = extract_generic_text(row)
            if text:
                texts.append(text)

        print(f"✓ Loaded {len(texts)} texts from {dataset_id}")
        return texts

    except Exception as e:
        print(f"✗ Error loading {dataset_id}: {e}")
        return []


def tokenize_texts(texts: List[str], tokenizer, max_seq_len: int) -> List[List[int]]:
    """テキストを学習用シーケンスへ変換"""
    sequences = []
    for t in texts:
        try:
            content_ids = tokenizer.encode(t, add_special=False)
            max_content = max_seq_len - 4  # BOS/EOS + BOF/EOF用スペース確保

            if max_content <= 0 or len(content_ids) < 2:
                continue

            if len(content_ids) <= max_content:
                seq = (
                    [tokenizer.bof_id, tokenizer.bos_id]
                    + content_ids
                    + [tokenizer.eos_id, tokenizer.eof_id]
                )
                sequences.append(seq)
            else:
                # 長い文を分割
                stride = max(max_content // 2, 1)
                starts = list(range(0, len(content_ids) - max_content + 1, stride))

                for idx, start in enumerate(starts):
                    chunk = content_ids[start : start + max_content]
                    prefix = (
                        [tokenizer.bof_id, tokenizer.bos_id]
                        if idx == 0
                        else [tokenizer.bos_id]
                    )
                    suffix = (
                        [tokenizer.eos_id, tokenizer.eof_id]
                        if idx == len(starts) - 1
                        else [tokenizer.eos_id]
                    )
                    sequences.append(prefix + chunk + suffix)
        except Exception as e:
            print(f"Warning: Failed to tokenize text: {e}")
            continue

    return sequences


def train_epoch(
    model,
    sequences: List[List[int]],
    tokenizer,
    optimizer,
    batch_size: int,
    max_seq_len: int,
    epoch: int,
    device: str,
    use_bf16: bool = False,
) -> float:
    """1エポック分の学習"""
    import random

    model.train()
    random.shuffle(sequences)
    total_loss = 0.0
    n_batches = 0

    for i in range(0, len(sequences), batch_size):
        batch = sequences[i : i + batch_size]
        if not batch:
            continue

        max_len = min(max(len(s) for s in batch), max_seq_len)
        input_ids, labels = [], []

        for s in batch:
            ids = s[:max_len]
            pad_len = max_len - len(ids)
            input_ids.append(ids + [tokenizer.pad_id] * pad_len)
            labels.append(ids[1:] + [tokenizer.pad_id] * (pad_len + 1))

        input_ids = torch.tensor(input_ids, dtype=torch.long, device=device)
        labels = torch.tensor(labels, dtype=torch.long, device=device)

        optimizer.zero_grad()

        with torch.autocast("cuda" if device == "cuda" else "cpu", enabled=use_bf16):
            logits = model(input_ids)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1))

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

        if (i // batch_size) % 10 == 0:
            avg_loss = total_loss / max(n_batches, 1)
            print(f"  Epoch {epoch}, Batch {i // batch_size}: Loss = {avg_loss:.4f}")

    return total_loss / max(n_batches, 1)


def main():
    parser = argparse.ArgumentParser(description="megabyte_100mb 多言語学習")
    parser.add_argument("--epochs", type=int, default=1, help="学習エポック数")
    parser.add_argument("--batch-size", type=int, default=2, help="バッチサイズ")
    parser.add_argument("--max-samples-per-dataset", type=int, default=5000, help="データセット当たりの最大サンプル数")
    parser.add_argument("--save-dir", type=str, default="./checkpoints", help="チェックポイント保存ディレクトリ")
    parser.add_argument("--use-gemma-300m", action="store_true", default=False, help="Google Generative AI APIで embedding-gemma-300m を使用")
    parser.add_argument("--use-sentence-transformers", action="store_true", default=True, help="sentence-transformersで google/embeddinggemma-300m を使用（デフォルト）")
    parser.add_argument("--no-external-embedding", dest="use_external", action="store_false", default=True, help="外部埋め込みを不使用")
    parser.add_argument("--vocab-size", type=int, default=32000, help="語彙サイズ")

    args = parser.parse_args()

    # デバイス設定
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # モデルサイズ設定を取得（megabyte_100mb）
    config_dict = get_model_config_by_size("megabyte_100mb", vocab_size=args.vocab_size)
    config = NeuroQuantumConfig(
        vocab_size=args.vocab_size,
        embed_dim=config_dict["embed_dim"],
        hidden_dim=config_dict["hidden_dim"],
        num_heads=config_dict["num_heads"],
        num_layers=config_dict["num_layers"],
        max_seq_len=config_dict["max_seq_len"],
        dropout=config_dict["dropout"],
        lambda_entangle=config_dict["entangle_strength"],
    )

    print(f"\n✨ NeuroQuantum Config:")
    print(f"  embed_dim: {config.embed_dim}")
    print(f"  hidden_dim: {config.hidden_dim}")
    print(f"  num_heads: {config.num_heads}")
    print(f"  num_layers: {config.num_layers}")
    print(f"  max_seq_len: {config.max_seq_len}")

    # トークナイザー初期化
    print("\n🔤 Initializing tokenizer...")
    tokenizer = NeuroQuantumTokenizer()
    print(f"  Tokenizer vocab size: {len(tokenizer)}")

    # モデル初期化（embedding-gemma-300m使用）
    print("\n🧠 Initializing NeuroQuantum model...")

    # 埋め込みモデルの選択
    use_sentence_transformers = args.use_external and args.use_sentence_transformers
    use_google_embedding = args.use_external and args.use_gemma_300m and not use_sentence_transformers

    if use_sentence_transformers:
        print("  Using: sentence-transformers (google/embeddinggemma-300m)")
        print("  ✓ Local execution (no API key needed)")
    elif use_google_embedding:
        google_api_key = os.getenv("GOOGLE_API_KEY")
        if not google_api_key:
            print("  ⚠️  GOOGLE_API_KEY not set. Falling back to sentence-transformers...")
            use_sentence_transformers = True
            use_google_embedding = False
        else:
            print("  Using: Google Generative AI API (embedding-gemma-300m)")
    else:
        print("  Using: Default token embeddings")

    model = NeuroQuantum(
        config=config,
        use_sentence_transformers=use_sentence_transformers,
        sentence_transformers_model="google/embeddinggemma-300m",
        use_google_embedding=use_google_embedding,
        google_api_key=os.getenv("GOOGLE_API_KEY") if use_google_embedding else None,
        google_model="models/embedding-gemma-300m",
        tokenizer=tokenizer,
    ).to(device)

    # パラメータ数
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total trainable parameters: {n_params:,}")

    # オプティマイザー
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)

    # チェックポイントディレクトリ作成
    os.makedirs(args.save_dir, exist_ok=True)

    # 各データセットで学習
    all_sequences = []
    print(f"\n📚 Loading all datasets...")

    for dataset_id, split in DATASETS:
        texts = load_dataset_texts(dataset_id, split, args.max_samples_per_dataset)
        if texts:
            sequences = tokenize_texts(texts, tokenizer, config.max_seq_len)
            all_sequences.extend(sequences)
            print(f"  Total sequences so far: {len(all_sequences)}")

    if not all_sequences:
        print("❌ No sequences loaded. Exiting.")
        return

    print(f"\n✅ Total sequences for training: {len(all_sequences)}")

    # 学習ループ
    print(f"\n🚀 Starting training for {args.epochs} epochs...")
    for epoch in range(args.epochs):
        print(f"\n━━━ Epoch {epoch + 1}/{args.epochs} ━━━")
        avg_loss = train_epoch(
            model,
            all_sequences,
            tokenizer,
            optimizer,
            args.batch_size,
            config.max_seq_len,
            epoch + 1,
            device,
            use_bf16=torch.cuda.is_available(),
        )
        print(f"  Average Loss: {avg_loss:.4f}")

        # チェックポイント保存
        ckpt_path = os.path.join(
            args.save_dir,
            f"megabyte_100mb_multilingual_epoch{epoch+1}.pt"
        )
        torch.save(model.state_dict(), ckpt_path)
        print(f"  ✓ Checkpoint saved to {ckpt_path}")

    print("\n✨ Training completed!")
    print(f"Final checkpoint: {ckpt_path}")


if __name__ == "__main__":
    main()
