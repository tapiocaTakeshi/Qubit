#!/usr/bin/env python3
"""
wikimedia/wikipedia 全データ学習スクリプト (NeuroQuantum アーキテクチャ)

`wikimedia/wikipedia` データセットの **全記事** を 3 エポック学習する。

全記事（日本語版で約140万件、英語版で約640万件）はメモリに載らないため、
HuggingFace datasets の streaming モードでデータを逐次読み込み、
一定サイズのバッファ単位で学習する「ストリーミング・シャード学習」方式を採る。
これによりメモリ使用量を一定に保ったまま、データ全体を 3 回（3エポック）通過できる。

設定は環境変数で上書き可能:
    WIKI_CONFIG       学習対象の config 名 (デフォルト: 20231101.ja)
                      複数指定する場合はカンマ区切り (例: "20231101.ja,20231101.en")
    WIKI_DATASET      データセット ID (デフォルト: wikimedia/wikipedia)
    WIKI_EPOCHS       エポック数 (デフォルト: 3)
    WIKI_MAX_SAMPLES  各 config から使う最大記事数。デバッグ用。
                      未指定または 0 のとき「全データ」(デフォルト: 全データ)
    WIKI_TOKENIZER_SAMPLES  トークナイザ語彙構築に使う記事数 (デフォルト: 200000)
    WIKI_BUFFER_SIZE  1 シャードあたりの学習シーケンス数 (デフォルト: 50000)

使い方:
    python train_wikipedia.py                       # 日本語版・全データ・3エポック
    WIKI_CONFIG=20231101.en python train_wikipedia.py
    WIKI_MAX_SAMPLES=5000 python train_wikipedia.py # 動作確認用に少量で
"""
import os
import sys
import json
import math
import random
from datetime import datetime, timezone

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(__file__))
from dataset_utils import safe_load_dataset, sync_checkpoint_to_network_volume
from neuroquantum_layered import (
    NeuroQuantum,
    NeuroQuantumConfig,
    NeuroQuantumTokenizer,
    get_gpu_adaptive_config,
)
from progress_logger import ProgressLogger

progress = ProgressLogger("train_wikipedia")

# ============================================================
# 設定
# ============================================================
DATASET_ID = os.environ.get("WIKI_DATASET", "wikimedia/wikipedia")
WIKI_CONFIGS = [
    c.strip()
    for c in os.environ.get("WIKI_CONFIG", "20231101.ja").split(",")
    if c.strip()
]
EPOCHS = int(os.environ.get("WIKI_EPOCHS", "3"))
# 0 / 未設定 = 全データ
_max_samples_env = int(os.environ.get("WIKI_MAX_SAMPLES", "0"))
MAX_SAMPLES = _max_samples_env if _max_samples_env > 0 else None
TOKENIZER_SAMPLES = int(os.environ.get("WIKI_TOKENIZER_SAMPLES", "200000"))
BUFFER_SIZE = int(os.environ.get("WIKI_BUFFER_SIZE", "50000"))

LR = 5e-4
GRAD_CLIP = 1.0

# GPU 性能に応じてニューロン数 / バッチサイズ等を自動決定
CONFIG = get_gpu_adaptive_config(vocab_size=32000)
BATCH_SIZE = CONFIG["batch_size"]
MAX_SEQ_LEN = CONFIG["max_seq_len"]

CKPT_PATH = os.path.join(os.path.dirname(__file__), "neuroq_wikipedia_checkpoint.pt")
TOKENIZER_PREFIX = os.path.join(os.path.dirname(__file__), "neuroq_wikipedia_tokenizer")


# ============================================================
# データセット読み込み (streaming)
# ============================================================
def iter_wiki_texts(configs, max_samples=None):
    """wikimedia/wikipedia の本文テキストを streaming で逐次 yield する。

    Args:
        configs: 読み込む config 名のリスト (例: ["20231101.ja"])
        max_samples: 各 config から取り出す最大記事数。None で全件。

    Yields:
        記事本文 (str)
    """
    for cfg in configs:
        progress.info(f"Streaming {DATASET_ID} ({cfg})...")
        try:
            ds = safe_load_dataset(
                DATASET_ID, split="train", streaming=True, name=cfg
            )
        except Exception as e:
            progress.log_dataset_error(f"{DATASET_ID}:{cfg}", str(e))
            continue

        count = 0
        for row in ds:
            text = (row.get("text") or "").strip()
            if len(text) > 4:
                yield text
                count += 1
                if max_samples is not None and count >= max_samples:
                    break
        progress.log_dataset_loaded(f"{DATASET_ID}:{cfg}", count)


def tokenize_text(text, tokenizer, max_seq_len):
    """1 記事を学習シーケンス（複数可）にトークナイズする。

    各チャンクは必ず BOS で始まり EOS で終わるため、モデルは文の境界を学習でき、
    文の途中から生成を始めてしまう問題を避けられる。
    """
    sequences = []
    content_ids = tokenizer.encode(text, add_special=False)
    max_content = max_seq_len - 2  # BOS / EOS の分を確保
    if max_content <= 0:
        return sequences
    if len(content_ids) <= max_content:
        if len(content_ids) >= 2:
            seq = [tokenizer.bof_id, tokenizer.bos_id] + content_ids + [tokenizer.eos_id, tokenizer.eof_id]
            sequences.append(seq)
        return sequences

    stride = max(max_content // 2, 1)
    chunks = list(range(0, len(content_ids) - max_content + 1, stride))
    for idx, start in enumerate(chunks):
        chunk = content_ids[start:start + max_content]
        prefix = [tokenizer.bof_id, tokenizer.bos_id] if idx == 0 else [tokenizer.bos_id]
        suffix = [tokenizer.eos_id, tokenizer.eof_id] if idx == len(chunks) - 1 else [tokenizer.eos_id]
        sequences.append(prefix + chunk + suffix)
    remaining = content_ids[-max_content:]
    tail_seq = [tokenizer.bos_id] + remaining + [tokenizer.eos_id, tokenizer.eof_id]
    if not sequences or tail_seq != sequences[-1]:
        sequences.append(tail_seq)
    return sequences


# ============================================================
# 学習
# ============================================================
def train_on_buffer(model, sequences, tokenizer, optimizer, device, epoch, shard):
    """バッファ（シャード）内のシーケンスで 1 パス学習する。"""
    model.train()
    random.shuffle(sequences)
    total_loss = 0.0
    n_batches = 0

    for i in range(0, len(sequences), BATCH_SIZE):
        batch_seqs = sequences[i:i + BATCH_SIZE]
        if not batch_seqs:
            continue

        max_len = min(max(len(s) for s in batch_seqs), MAX_SEQ_LEN)
        input_ids, labels = [], []
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
            shift_logits.view(-1, model.config.vocab_size),
            shift_labels.view(-1),
            ignore_index=-100,
        )

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

        if n_batches % 50 == 0:
            avg = total_loss / n_batches
            progress.log_batch(epoch=epoch + 1, batch=n_batches, loss=avg,
                               extra={"shard": shard})

    return total_loss, n_batches


def build_tokenizer(device):
    """先頭 TOKENIZER_SAMPLES 件の記事から SentencePiece 語彙を構築する。

    既存の語彙ファイルがあれば再利用する。全データのトークナイズ前に語彙が必要なため、
    語彙構築には先頭の記事サンプルを用いる（学習自体は全データで行う）。
    """
    model_file = TOKENIZER_PREFIX + ".model"
    if os.path.isfile(model_file):
        progress.info(f"Reusing existing tokenizer: {model_file}")
        tokenizer = NeuroQuantumTokenizer(vocab_size=CONFIG["vocab_size"], model_file=model_file)
        actual = tokenizer.actual_vocab_size or tokenizer.vocab_size
        return tokenizer, actual

    progress.info(f"=== Building SentencePiece tokenizer (sample={TOKENIZER_SAMPLES}) ===")
    sample_texts = []
    for text in iter_wiki_texts(WIKI_CONFIGS, max_samples=TOKENIZER_SAMPLES):
        sample_texts.append(text)
        if len(sample_texts) >= TOKENIZER_SAMPLES:
            break
    progress.info(f"Tokenizer corpus: {len(sample_texts)} articles")

    tokenizer = NeuroQuantumTokenizer(vocab_size=CONFIG["vocab_size"])
    tokenizer.build_vocab(
        sample_texts,
        model_prefix=TOKENIZER_PREFIX,
        character_coverage=0.9995,
    )
    actual = tokenizer.actual_vocab_size or tokenizer.vocab_size
    progress.info(f"Actual vocab size: {actual}")
    return tokenizer, actual


def save_checkpoint(model, actual_vocab, training_log, total_articles, total_sequences):
    checkpoint = {
        "model_state": model.state_dict(),
        "config": {
            "vocab_size": actual_vocab,
            "embed_dim": CONFIG["embed_dim"],
            "hidden_dim": CONFIG["hidden_dim"],
            "num_heads": CONFIG["num_heads"],
            "num_layers": CONFIG["num_layers"],
            "max_seq_len": CONFIG["max_seq_len"],
            "entangle_strength": CONFIG["entangle_strength"],
            "dropout": CONFIG["dropout"],
            "architecture": "neuroquantum",
        },
        "training_log": training_log,
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "datasets": [f"{DATASET_ID}:{c}" for c in WIKI_CONFIGS],
        "total_articles": total_articles,
        "total_sequences": total_sequences,
    }
    torch.save(checkpoint, CKPT_PATH)
    size_mb = os.path.getsize(CKPT_PATH) / 1024 / 1024
    progress.info(f"Saved checkpoint: {CKPT_PATH} ({size_mb:.1f} MB)")
    sync_checkpoint_to_network_volume(CKPT_PATH, TOKENIZER_PREFIX + ".model")


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    progress.info(f"Device: {device}")
    progress.info(
        f"Dataset: {DATASET_ID} configs={WIKI_CONFIGS} epochs={EPOCHS} "
        f"max_samples={'ALL' if MAX_SAMPLES is None else MAX_SAMPLES} "
        f"batch_size={BATCH_SIZE} buffer={BUFFER_SIZE}"
    )

    # Step 1: トークナイザ構築（先頭サンプルから）
    tokenizer, actual_vocab = build_tokenizer(device)
    CONFIG["vocab_size"] = actual_vocab

    # Step 2: モデル構築
    progress.info("=== Building NeuroQuantum model ===")
    nq_config = NeuroQuantumConfig(
        vocab_size=actual_vocab,
        embed_dim=CONFIG["embed_dim"],
        hidden_dim=CONFIG["hidden_dim"],
        num_heads=CONFIG["num_heads"],
        num_layers=CONFIG["num_layers"],
        max_seq_len=CONFIG["max_seq_len"],
        dropout=CONFIG["dropout"],
        lambda_entangle=CONFIG["entangle_strength"],
    )
    model = NeuroQuantum(config=nq_config).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    progress.info(f"Parameters: {n_params:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)

    # Step 3: ストリーミング・シャード学習 (3 エポック = データ全体を 3 周)
    progress.info("=== Training (streaming shards over full Wikipedia) ===")
    progress.start_training(epochs=EPOCHS, total_sequences=-1, batch_size=BATCH_SIZE, lr=LR)
    training_log = []
    total_articles = 0
    total_sequences = 0

    for epoch in range(EPOCHS):
        progress.start_epoch(epoch + 1, EPOCHS)
        buffer = []
        epoch_loss = 0.0
        epoch_batches = 0
        shard = 0
        epoch_articles = 0

        for text in iter_wiki_texts(WIKI_CONFIGS, max_samples=MAX_SAMPLES):
            epoch_articles += 1
            for seq in tokenize_text(text, tokenizer, MAX_SEQ_LEN):
                buffer.append(seq)

            if len(buffer) >= BUFFER_SIZE:
                shard += 1
                loss_sum, n_b = train_on_buffer(
                    model, buffer, tokenizer, optimizer, device, epoch, shard
                )
                epoch_loss += loss_sum
                epoch_batches += n_b
                total_sequences += len(buffer)
                progress.info(
                    f"  Epoch {epoch+1} shard {shard} | articles~{epoch_articles} | "
                    f"shard_loss={loss_sum / max(n_b, 1):.4f}"
                )
                buffer = []

        # 残りバッファを学習
        if buffer:
            shard += 1
            loss_sum, n_b = train_on_buffer(
                model, buffer, tokenizer, optimizer, device, epoch, shard
            )
            epoch_loss += loss_sum
            epoch_batches += n_b
            total_sequences += len(buffer)
            buffer = []

        avg_loss = epoch_loss / max(epoch_batches, 1)
        progress.log_epoch(epoch=epoch + 1, total_epochs=EPOCHS, loss=avg_loss,
                           extra={"articles": epoch_articles, "shards": shard})
        training_log.append({"epoch": epoch + 1, "loss": avg_loss, "articles": epoch_articles})
        total_articles = epoch_articles  # 各エポックで全件通過するので同数

        # エポックごとにチェックポイント保存（中断しても続きから再開可）
        save_checkpoint(model, actual_vocab, training_log, total_articles, total_sequences)

    progress.end_training(
        final_loss=training_log[-1]["loss"] if training_log else 0.0,
        checkpoint_path=CKPT_PATH,
    )

    # Step 4: 学習履歴を保存
    history_path = os.path.join(os.path.dirname(__file__), "training_history_wikipedia.json")
    history = {
        "architecture": "neuroquantum",
        "config": CONFIG,
        "datasets": [f"{DATASET_ID}:{c}" for c in WIKI_CONFIGS],
        "epochs": EPOCHS,
        "total_articles_per_epoch": total_articles,
        "total_sequences": total_sequences,
        "parameters": n_params,
        "training_log": training_log,
        "trained_at": datetime.now(timezone.utc).isoformat(),
    }
    with open(history_path, "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)
    progress.info(f"Training history saved: {history_path}")

    # Step 5: 簡易推論テスト
    progress.info("=== Inference test ===")
    model.eval()
    test_prompts = ["日本の歴史", "量子力学とは", "東京は", "人工知能の研究", "地球の構造"]
    for prompt in test_prompts:
        tokens = tokenizer.encode(prompt, add_special=True)
        input_tensor = torch.tensor([tokens], dtype=torch.long, device=device)
        generated = list(tokens)
        with torch.no_grad():
            for _ in range(60):
                seq = input_tensor[:, -MAX_SEQ_LEN:]
                logits = model(seq)[:, -1, :] / 0.7
                topk_vals = torch.topk(logits, 40)[0]
                logits[logits < topk_vals[:, -1:]] = float("-inf")
                for prev in set(generated[-50:]):
                    if prev < logits.size(-1):
                        logits[0, prev] /= 1.3
                probs = F.softmax(logits, dim=-1)
                nxt = torch.multinomial(probs, 1)
                nxt_id = nxt.item()
                if nxt_id in (tokenizer.eos_id, tokenizer.eof_id):
                    break
                if nxt_id in (tokenizer.pad_id, tokenizer.bof_id):
                    input_tensor = torch.cat([input_tensor, nxt], dim=1)
                    continue
                generated.append(nxt_id)
                input_tensor = torch.cat([input_tensor, nxt], dim=1)
        generated_text = tokenizer.decode(generated[len(tokens):], skip_special=True)
        print(f'  "{prompt}" -> "{generated_text}"')

    progress.info("Done!")


if __name__ == "__main__":
    main()
