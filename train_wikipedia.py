#!/usr/bin/env python3
"""
wikimedia/wikipedia 全データ学習スクリプト (NeuroQuantum アーキテクチャ)

`wikimedia/wikipedia` データセットの **全記事** を 3 エポック学習する。

## データ読み込み方式（バッチ処理）

デフォルトは map-style（非streaming）でのバッチ処理。`wikimedia/wikipedia` は
Parquet/Arrow 形式なので、`load_dataset` は Arrow ファイルをメモリマップする。
全記事（日本語版で約140万件、英語版で約640万件）をRAMに載せず、ディスクから
逐次読みながら、**バッチ単位でトークナイズして即学習・破棄**することで
メモリ使用量を一定に保つ。

streaming に対するバッチ処理の利点:
  - 3 エポック回しても初回 DL 1 回でキャッシュを再利用（streaming は毎エポック再取得）
  - 全記事を完全シャッフルできる（streaming はバッファ内シャッフルのみ）
  - データ件数が既知なので進捗・スケジューリングが正確

ディスク容量が厳しい場合は WIKI_STREAMING=1 で streaming にフォールバックできる。

設定は環境変数で上書き可能:
    WIKI_CONFIG       学習対象の config 名 (デフォルト: 20231101.ja)
                      複数指定する場合はカンマ区切り (例: "20231101.ja,20231101.en")
    WIKI_DATASET      データセット ID (デフォルト: wikimedia/wikipedia)
    WIKI_EPOCHS       エポック数 (デフォルト: 3)
    WIKI_MAX_SAMPLES  各 config から使う最大記事数。デバッグ用。
                      未指定または 0 のとき「全データ」(デフォルト: 全データ)
    WIKI_TOKENIZER_SAMPLES  トークナイザ語彙構築に使う記事数 (デフォルト: 200000)
    WIKI_STREAMING    1 で streaming フォールバック (デフォルト: 0 = バッチ処理)

使い方:
    python train_wikipedia.py                       # 日本語版・全データ・3エポック
    WIKI_CONFIG=20231101.en python train_wikipedia.py
    WIKI_MAX_SAMPLES=5000 python train_wikipedia.py # 動作確認用に少量で
    WIKI_STREAMING=1 python train_wikipedia.py      # ディスク節約 streaming
"""
import os
import sys
import json
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
USE_STREAMING = os.environ.get("WIKI_STREAMING", "0") == "1"
# 途中保存の間隔（バッチ数）。コンテナ再起動を跨いで再開できるよう、
# エポック終了を待たず定期的にチェックポイントを保存する。
CHECKPOINT_EVERY = int(os.environ.get("WIKI_CHECKPOINT_EVERY", "2000"))

LR = 5e-4
GRAD_CLIP = 1.0

# GPU 性能に応じてニューロン数 / バッチサイズ等を自動決定
CONFIG = get_gpu_adaptive_config(vocab_size=32000)
BATCH_SIZE = CONFIG["batch_size"]
MAX_SEQ_LEN = CONFIG["max_seq_len"]

CKPT_PATH = os.path.join(os.path.dirname(__file__), "neuroq_wikipedia_checkpoint.pt")
TOKENIZER_PREFIX = os.path.join(os.path.dirname(__file__), "neuroq_wikipedia_tokenizer")


# ============================================================
# データセット読み込み
# ============================================================
def load_map_datasets():
    """各 config を map-style（非streaming）で読み込み、リストで返す。

    Arrow メモリマップにより、巨大データでも RAM 使用量を抑えつつランダムアクセス
    （完全シャッフル）が可能。初回のみディスクへ DL し、以降はキャッシュを再利用する。
    """
    datasets = []
    for cfg in WIKI_CONFIGS:
        progress.info(f"Loading {DATASET_ID} ({cfg}) [map-style/batched]...")
        try:
            ds = safe_load_dataset(DATASET_ID, split="train", name=cfg)
            progress.log_dataset_loaded(f"{DATASET_ID}:{cfg}", len(ds))
            datasets.append((cfg, ds))
        except Exception as e:
            progress.log_dataset_error(f"{DATASET_ID}:{cfg}", str(e))
    return datasets


def iter_texts_mapstyle(datasets, epoch, max_samples=None):
    """map-style データセットを完全シャッフルしてテキストを yield する。"""
    for cfg, ds in datasets:
        n = len(ds)
        if max_samples is not None:
            n = min(n, max_samples)
        # 索引マッピングのみをシャッフル（データ本体はコピーしない）
        shuffled = ds.shuffle(seed=epoch).select(range(n)) if n < len(ds) else ds.shuffle(seed=epoch)
        for row in shuffled:
            text = (row.get("text") or "").strip()
            if len(text) > 4:
                yield text


def iter_texts_streaming(epoch, max_samples=None):
    """streaming で各 config のテキストを逐次 yield する（フォールバック）。"""
    for cfg in WIKI_CONFIGS:
        progress.info(f"Streaming {DATASET_ID} ({cfg})...")
        try:
            ds = safe_load_dataset(DATASET_ID, split="train", streaming=True, name=cfg)
            ds = ds.shuffle(seed=epoch, buffer_size=10000)
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
def train_batch(model, batch_seqs, tokenizer, optimizer, device):
    """BATCH_SIZE 件のシーケンスで 1 ステップ学習し、loss を返す。"""
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
    return loss.item()


def build_tokenizer(text_iter_factory):
    """先頭 TOKENIZER_SAMPLES 件の記事から SentencePiece 語彙を構築する。

    既存の語彙ファイルがあれば再利用する。全データのトークナイズ前に語彙が必要なため、
    語彙構築には記事サンプルを用いる（学習自体は全データで行う）。

    Args:
        text_iter_factory: テキストイテレータを返すファクトリ（引数なし）
    """
    model_file = TOKENIZER_PREFIX + ".model"
    if os.path.isfile(model_file):
        progress.info(f"Reusing existing tokenizer: {model_file}")
        tokenizer = NeuroQuantumTokenizer(vocab_size=CONFIG["vocab_size"], model_file=model_file)
        return tokenizer, (tokenizer.actual_vocab_size or tokenizer.vocab_size)

    progress.info(f"=== Building SentencePiece tokenizer (sample={TOKENIZER_SAMPLES}) ===")
    sample_texts = []
    for text in text_iter_factory():
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


def save_checkpoint(model, optimizer, actual_vocab, training_log, total_articles,
                    total_sequences, epoch, global_batch, quiet=False):
    """チェックポイントを保存する。

    再開に必要な optimizer 状態・現在エポック・累計バッチ数も含める。
    途中保存（quiet=True）ではログを簡潔にする。一時ファイルに書いてから
    rename することで、保存中の再起動による破損を防ぐ。
    """
    checkpoint = {
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
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
        "epoch": epoch,                 # 0-indexed: 現在処理中（未完了）のエポック
        "global_batch": global_batch,   # 累計学習バッチ数（進捗の目安）
    }
    tmp_path = CKPT_PATH + ".tmp"
    torch.save(checkpoint, tmp_path)
    os.replace(tmp_path, CKPT_PATH)
    if not quiet:
        size_mb = os.path.getsize(CKPT_PATH) / 1024 / 1024
        progress.info(f"Saved checkpoint: {CKPT_PATH} ({size_mb:.1f} MB)")
        sync_checkpoint_to_network_volume(CKPT_PATH, TOKENIZER_PREFIX + ".model")


def try_resume(model, optimizer, device):
    """既存チェックポイントがあれば model/optimizer に読み込み、再開情報を返す。

    Returns:
        (start_epoch, training_log, total_sequences, global_batch)
        チェックポイントが無ければ (0, [], 0, 0)
    """
    if not os.path.isfile(CKPT_PATH):
        return 0, [], 0, 0
    try:
        ckpt = torch.load(CKPT_PATH, map_location=device)
        model.load_state_dict(ckpt["model_state"])
        if "optimizer_state" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer_state"])
        start_epoch = ckpt.get("epoch", 0)
        training_log = ckpt.get("training_log", [])
        total_sequences = ckpt.get("total_sequences", 0)
        global_batch = ckpt.get("global_batch", 0)
        progress.info(
            f"=== Resuming from checkpoint: epoch={start_epoch} "
            f"global_batch={global_batch} total_sequences={total_sequences} ==="
        )
        return start_epoch, training_log, total_sequences, global_batch
    except Exception as e:
        progress.info(f"Checkpoint found but failed to load ({e}). Starting fresh.")
        return 0, [], 0, 0


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mode = "streaming" if USE_STREAMING else "map-style/batched"
    progress.info(f"Device: {device}")
    progress.info(
        f"Dataset: {DATASET_ID} configs={WIKI_CONFIGS} epochs={EPOCHS} "
        f"max_samples={'ALL' if MAX_SAMPLES is None else MAX_SAMPLES} "
        f"batch_size={BATCH_SIZE} mode={mode}"
    )

    # データソースの準備とテキストイテレータの定義
    if USE_STREAMING:
        map_datasets = None
        def text_iter(epoch=0):
            return iter_texts_streaming(epoch, max_samples=MAX_SAMPLES)
    else:
        map_datasets = load_map_datasets()
        if not map_datasets:
            progress.info("ERROR: no datasets loaded. Aborting.")
            return
        def text_iter(epoch=0):
            return iter_texts_mapstyle(map_datasets, epoch, max_samples=MAX_SAMPLES)

    # Step 1: トークナイザ構築（先頭サンプルから）
    tokenizer, actual_vocab = build_tokenizer(lambda: text_iter(epoch=0))
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

    # 既存チェックポイントがあれば再開（コンテナ再起動を跨いで継続）
    start_epoch, training_log, total_sequences, global_batch = try_resume(model, optimizer, device)

    # Step 3: バッチ処理学習 (3 エポック = データ全体を 3 周)
    progress.info("=== Training (batched over full Wikipedia) ===")
    progress.start_training(epochs=EPOCHS, total_sequences=-1, batch_size=BATCH_SIZE, lr=LR)
    total_articles = 0

    for epoch in range(start_epoch, EPOCHS):
        progress.start_epoch(epoch + 1, EPOCHS)
        model.train()
        batch_buf = []          # 学習用に貯める BATCH_SIZE 分のシーケンス
        epoch_loss = 0.0
        epoch_batches = 0
        epoch_articles = 0

        for text in text_iter(epoch=epoch):
            epoch_articles += 1
            for seq in tokenize_text(text, tokenizer, MAX_SEQ_LEN):
                batch_buf.append(seq)
                total_sequences += 1
                # BATCH_SIZE 溜まったら即学習して破棄（メモリは 1 バッチ分のみ）
                if len(batch_buf) >= BATCH_SIZE:
                    loss = train_batch(model, batch_buf, tokenizer, optimizer, device)
                    epoch_loss += loss
                    epoch_batches += 1
                    global_batch += 1
                    batch_buf = []
                    if epoch_batches % 50 == 0:
                        progress.log_batch(
                            epoch=epoch + 1, batch=epoch_batches,
                            loss=epoch_loss / epoch_batches,
                            extra={"articles": epoch_articles},
                        )
                    # 定期チェックポイント（再起動耐性）
                    if global_batch % CHECKPOINT_EVERY == 0:
                        save_checkpoint(
                            model, optimizer, actual_vocab, training_log,
                            epoch_articles, total_sequences, epoch, global_batch,
                            quiet=True,
                        )

        # 端数バッチを学習
        if batch_buf:
            loss = train_batch(model, batch_buf, tokenizer, optimizer, device)
            epoch_loss += loss
            epoch_batches += 1
            global_batch += 1
            batch_buf = []

        avg_loss = epoch_loss / max(epoch_batches, 1)
        progress.log_epoch(epoch=epoch + 1, total_epochs=EPOCHS, loss=avg_loss,
                           extra={"articles": epoch_articles, "batches": epoch_batches})
        training_log.append({"epoch": epoch + 1, "loss": avg_loss, "articles": epoch_articles})
        total_articles = epoch_articles  # 各エポックで全件通過するので同数

        # エポック完了時に保存。epoch+1 を保存することで次回はこのエポックを飛ばして再開
        save_checkpoint(model, optimizer, actual_vocab, training_log,
                        total_articles, total_sequences, epoch + 1, global_batch)

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
        "mode": mode,
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
