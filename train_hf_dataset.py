#!/usr/bin/env python3
"""
Hugging Face データセットIDを指定して学習するユニバーサルスクリプト。

任意のHFデータセットに対応し、様々な列名形式を自動検出する。
デフォルトではすべてのデータを使用して学習を実行する。

複数のデータセットで学習しても、常に1つのptファイル (neuroq_small_checkpoint.pt) に上書き保存されます。

使い方:
    # 1. 基本的な使用方法 (全データを学習):
    python train_hf_dataset.py --dataset-id "llm-jp/databricks-dolly-15k-ja" --epochs 3

    # 2. 異なるデータセットで学習（チェックポイントに上書き）:
    python train_hf_dataset.py --dataset-id "openwebtext" --split "train" --epochs 5

    # 3. マックスサンプルを制限してテスト:
    python train_hf_dataset.py --dataset-id "wikitext" --split "train" \
        --max-samples 1000 --epochs 1 --vocab-size 4000

    # 4. HF Hub にアップロード:
    HF_TOKEN=hf_xxx python train_hf_dataset.py \
        --dataset-id "llm-jp/databricks-dolly-15k-ja" \
        --epochs 3 --upload --repo-id tapiocatakeshi/Qubit

    # 5. 既存チェックポイントから追加学習（同じ1つのファイルで継続）:
    python train_hf_dataset.py --dataset-id "llm-jp/databricks-dolly-15k-ja" \
        --resume --reset-epochs --epochs 3
"""
import argparse
import json
import os
import sys
from datetime import datetime, timezone
from typing import Optional

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dataset_utils import safe_load_dataset, sync_checkpoint_to_network_volume
from neuroquantum_layered import (
    NeuroQuantum,
    NeuroQuantumConfig,
    NeuroQuantumTokenizer,
    get_model_config_by_size,
)
from progress_logger import ProgressLogger

DEFAULT_REPO_ID = "tapiocatakeshi/Qubit"

progress = ProgressLogger("train_hf_dataset")


def _first_str(row, keys):
    """row から keys の最初に見つかった非空文字列を返す。無ければ None。"""
    for k in keys:
        val = row.get(k)
        if isinstance(val, str) and val.strip():
            return val.strip()
    return None


def detect_text_column(row):
    """行から適切なテキスト列を自動検出する。"""
    # テキスト列の候補を優先順位付きで試す
    text_candidates = [
        ("text",),  # wikitext, openwebtext などはこれ
        ("document", "content"),
        ("passage", "article"),
        ("sentence",),
        ("body",),
        ("description", "desc"),
        ("summary",),
    ]
    return _first_str(row, [col for cols in text_candidates for col in cols])


def extract_dialogue_format(row):
    """行から対話形式のテキストを抽出する。"""
    # instruction+output 形式
    instruction = _first_str(row, ("instruction", "prompt", "question", "query"))
    context = _first_str(row, ("context", "input"))
    output = _first_str(row, ("output", "response", "answer", "completion"))

    if instruction and output:
        parts = [f"<USER> {instruction}"]
        if context:
            parts.append(context)
        parts.append(f"<ASSISTANT> {output}")
        return "\n".join(parts).strip()

    return None


def extract_generic_text(row):
    """行からテキストを抽出する（形式自動検出）。"""
    # まず対話形式を試す
    dialogue = extract_dialogue_format(row)
    if dialogue:
        return dialogue

    # 次に単純なテキスト列を試す
    text = detect_text_column(row)
    if text:
        return text

    # 最後に非空の文字列値を探す
    for v in row.values():
        if isinstance(v, str) and v.strip() and len(v.strip()) > 4:
            return v.strip()

    return None


def extract_texts(ds, max_samples: Optional[int] = None):
    """データセットからテキストを抽出する。

    自動的に列形式を検出し、対話形式またはテキスト形式で抽出。
    max_samples が None または <= 0 の場合、全件を使用する。
    """
    texts = []
    n = len(ds) if max_samples is None or max_samples <= 0 else min(max_samples, len(ds))

    for row in ds.select(range(n)):
        text = extract_generic_text(row)
        if text:
            texts.append(text)

    return texts


def tokenize_texts(texts, tokenizer, max_seq_len):
    """テキストを BOS/EOS + BOF/EOF で囲まれた学習用シーケンスへ変換する。"""
    sequences = []
    for t in texts:
        content_ids = tokenizer.encode(t, add_special=False)
        max_content = max_seq_len - 2
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
    return sequences


def train_epoch(
    model,
    sequences,
    tokenizer,
    optimizer,
    batch_size,
    max_seq_len,
    epoch,
    device,
    save_every=0,
    on_save=None,
):
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
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1
        if n_batches % 25 == 0:
            progress.log_batch(epoch=epoch + 1, batch=n_batches, loss=total_loss / n_batches)
        if save_every and on_save and n_batches % save_every == 0:
            avg = total_loss / max(n_batches, 1)
            progress.info(
                f"[checkpoint] epoch {epoch + 1} batch {n_batches} avg_loss={avg:.4f} を保存中..."
            )
            on_save(epoch=epoch, batch=n_batches, loss=avg)

    return total_loss / max(n_batches, 1)


def upload_checkpoint_to_hf(ckpt_path: str, repo_id: str, hf_token: str, tokenizer_path: str | None = None):
    """チェックポイントを HuggingFace Hub にアップロードする。"""
    from huggingface_hub import HfApi

    api = HfApi(token=hf_token)
    api.create_repo(repo_id=repo_id, exist_ok=True)

    ckpt_basename = os.path.basename(ckpt_path)
    progress.info(f"Uploading {ckpt_basename} → {repo_id} ...")
    api.upload_file(
        path_or_fileobj=ckpt_path,
        path_in_repo=ckpt_basename,
        repo_id=repo_id,
        commit_message=f"Add {ckpt_basename} (NeuroQuantum small)",
    )
    progress.info(f"  ✅ {ckpt_basename} uploaded")

    if tokenizer_path and os.path.isfile(tokenizer_path):
        tok_basename = os.path.basename(tokenizer_path)
        progress.info(f"Uploading {tok_basename} → {repo_id} ...")
        api.upload_file(
            path_or_fileobj=tokenizer_path,
            path_in_repo=tok_basename,
            repo_id=repo_id,
            commit_message=f"Add tokenizer for {ckpt_basename}",
        )
        progress.info(f"  ✅ {tok_basename} uploaded")

    return f"https://huggingface.co/{repo_id}"


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument(
        "--dataset-id",
        required=True,
        help="Hugging Face データセット ID (例: llm-jp/databricks-dolly-15k-ja, openwebtext)",
    )
    p.add_argument(
        "--split",
        default=None,
        help="読み込む split (指定なしで自動検出。複数ある場合は最初のものを使用)",
    )
    p.add_argument(
        "--max-samples",
        type=int,
        default=0,
        help="使用する最大サンプル数。0 または負値で全件を使用 (デフォルト)",
    )
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--vocab-size", type=int, default=32000)
    p.add_argument("--batch-size", type=int, default=None, help="未指定なら small 設定 (4) を使用")
    p.add_argument("--max-seq-len", type=int, default=None, help="未指定なら small 設定 (4096) を使用")
    p.add_argument(
        "--ckpt-name",
        default=None,
        help="チェックポイント名 (デフォルト: neuroq_small_checkpoint.pt)",
    )
    p.add_argument(
        "--save-every",
        type=int,
        default=0,
        help="N バッチごとに中間チェックポイントを保存 (0 で無効)",
    )
    p.add_argument(
        "--resume",
        action="store_true",
        help="既存チェックポイントから学習を再開/追加学習する",
    )
    p.add_argument(
        "--reset-epochs",
        action="store_true",
        help="--resume 時に training_log のエポック数を無視し、重みのみ流用して 0 エポック目から学習",
    )
    p.add_argument("--upload", action="store_true", help="HF Hub にアップロードする")
    p.add_argument("--repo-id", default=DEFAULT_REPO_ID)
    p.add_argument("--hf-token", default=os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN"))
    p.add_argument(
        "--tokenizer-prefix",
        default=None,
        help="トークナイザーのプリフィックス (デフォルト: neuroq_small_tokenizer)",
    )
    return p.parse_args()


def get_default_names(dataset_id: str):
    """デフォルトのチェックポイント名とトークナイザープリフィックスを返す。"""
    ckpt_name = "neuroq_small_checkpoint.pt"
    tokenizer_prefix = "neuroq_small_tokenizer"
    return ckpt_name, tokenizer_prefix


def auto_detect_split(dataset_id: str):
    """データセットの利用可能な split を自動検出（最初のものを返す）。"""
    try:
        from datasets import get_dataset_config_names, get_dataset_split_names

        splits = get_dataset_split_names(dataset_id)
        if splits:
            progress.info(f"Available splits: {splits}")
            return splits[0]
    except Exception as e:
        progress.warning(f"split auto-detection failed: {e}")
    return "train"


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    progress.info(f"Device: {device}")

    # ---- チェックポイント名とトークナイザープリフィックスの確定 ----
    ckpt_name = args.ckpt_name or get_default_names(args.dataset_id)[0]
    tokenizer_prefix_arg = args.tokenizer_prefix or get_default_names(args.dataset_id)[1]

    # ---- small モデル設定 ----
    CONFIG = get_model_config_by_size("small", vocab_size=args.vocab_size)
    batch_size = args.batch_size or CONFIG["batch_size"]
    max_seq_len = args.max_seq_len or CONFIG["max_seq_len"]

    # ---- split の自動検出 ----
    split = args.split or auto_detect_split(args.dataset_id)

    # ---- データセット読み込み ----
    progress.info(f"=== Loading {args.dataset_id} (split={split}) ===")
    ds = safe_load_dataset(args.dataset_id, split=split)
    use_all = args.max_samples is None or args.max_samples <= 0
    effective_max_samples = len(ds) if use_all else min(args.max_samples, len(ds))
    progress.info(
        f"Dataset size: {len(ds)} / using "
        f"{'ALL ' + str(effective_max_samples) if use_all else effective_max_samples} samples"
    )

    # ---- テキスト抽出 ----
    texts = extract_texts(ds, args.max_samples)
    progress.log_dataset_loaded(args.dataset_id, len(texts))
    if not texts:
        raise RuntimeError(f"{args.dataset_id} からテキストを取得できませんでした")

    # ---- チェックポイント読み込み（再開/追加学習時）----
    ckpt_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ckpt_name)
    resume_ckpt = None
    if args.resume:
        if os.path.isfile(ckpt_path):
            progress.info(f"=== Resuming from checkpoint: {ckpt_path} ===")
            resume_ckpt = torch.load(ckpt_path, map_location="cpu")
        else:
            progress.warning(f"--resume 指定ですが {ckpt_path} が見つかりません。新規学習として開始します。")

    # ---- トークナイザー構築 / ロード ----
    tokenizer_prefix = os.path.join(os.path.dirname(os.path.abspath(__file__)), tokenizer_prefix_arg)
    tokenizer_model_path = tokenizer_prefix + ".model"
    if resume_ckpt is not None and os.path.isfile(tokenizer_model_path):
        progress.info(f"=== Loading existing tokenizer: {tokenizer_model_path} ===")
        ckpt_vocab = resume_ckpt.get("config", {}).get("vocab_size", args.vocab_size)
        tokenizer = NeuroQuantumTokenizer(vocab_size=ckpt_vocab, model_file=tokenizer_model_path)
    else:
        progress.info("=== Building SentencePiece tokenizer ===")
        tokenizer = NeuroQuantumTokenizer(vocab_size=args.vocab_size)
        tokenizer.build_vocab(texts, model_prefix=tokenizer_prefix, character_coverage=0.9995)
    actual_vocab = tokenizer.actual_vocab_size or tokenizer.vocab_size
    CONFIG["vocab_size"] = actual_vocab
    progress.info(f"Actual vocab size: {actual_vocab}")

    # ---- モデル構築 ----
    progress.info("=== Building NeuroQuantum (small) ===")
    nq_config = NeuroQuantumConfig(
        vocab_size=actual_vocab,
        embed_dim=CONFIG["embed_dim"],
        hidden_dim=CONFIG["hidden_dim"],
        num_heads=CONFIG["num_heads"],
        num_layers=CONFIG["num_layers"],
        max_seq_len=max_seq_len,
        dropout=CONFIG["dropout"],
        lambda_entangle=CONFIG["entangle_strength"],
    )
    model = NeuroQuantum(config=nq_config).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    progress.info(f"Parameters: {n_params:,}")

    if resume_ckpt is not None and "model_state" in resume_ckpt:
        model.load_state_dict(resume_ckpt["model_state"], strict=False)
        progress.info("Resumed model weights from checkpoint.")

    # ---- トークナイズ ----
    progress.info("=== Tokenizing ===")
    sequences = tokenize_texts(texts, tokenizer, max_seq_len)
    progress.info(f"Training sequences: {len(sequences)}")
    if not sequences:
        raise RuntimeError("学習シーケンスが 0 件です。--max-samples を増やしてください。")

    # ---- 学習 ----
    progress.info("=== Training ===")
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    training_log = []
    start_epoch = 0
    if resume_ckpt is not None:
        training_log = list(resume_ckpt.get("training_log", []))
        if args.reset_epochs:
            start_epoch = 0
            progress.info("--reset-epochs: 重みのみ流用し、学習を 0 エポック目から開始します。")
        else:
            start_epoch = len(training_log)
        if resume_ckpt.get("optimizer_state") is not None and not args.reset_epochs:
            try:
                optimizer.load_state_dict(resume_ckpt["optimizer_state"])
                progress.info("Resumed optimizer state.")
            except Exception as e:
                progress.warning(f"optimizer state の復元に失敗 (新規optimizer で継続): {e}")
        for _ in range(start_epoch):
            scheduler.step()
        progress.info(
            f"Resuming at epoch {start_epoch + 1}/{args.epochs} "
            f"(completed epochs: {start_epoch})"
        )
        if start_epoch >= args.epochs:
            progress.info("既に全エポック完了済みのチェックポイントです。学習をスキップします.")

    progress.start_training(epochs=args.epochs, total_sequences=len(sequences), batch_size=batch_size, lr=args.lr)

    def save_checkpoint(epoch, batch=None, loss=None, final=False):
        """チェックポイントを保存する。"""
        checkpoint = {
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "config": nq_config.__dict__,
            "training_log": training_log,
            "batch": batch,
            "epoch": epoch,
            "loss": loss,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        torch.save(checkpoint, ckpt_path)
        progress.info(f"Checkpoint saved: {ckpt_path}")
        if final:
            sync_checkpoint_to_network_volume(ckpt_path)

    for epoch in range(start_epoch, args.epochs):
        avg_loss = train_epoch(
            model,
            sequences,
            tokenizer,
            optimizer,
            batch_size,
            max_seq_len,
            epoch,
            device,
            save_every=args.save_every,
            on_save=save_checkpoint,
        )
        scheduler.step()
        training_log.append({"epoch": epoch, "avg_loss": avg_loss})
        save_checkpoint(epoch=epoch, loss=avg_loss)
        progress.log_epoch(epoch=epoch + 1, loss=avg_loss)

    progress.end_training()
    save_checkpoint(epoch=args.epochs - 1, loss=avg_loss, final=True)

    # ---- HF Hub へのアップロード ----
    if args.upload and args.hf_token:
        upload_checkpoint_to_hf(ckpt_path, args.repo_id, args.hf_token, tokenizer_model_path)

    progress.info("=== Training complete ===")
    progress.info(f"Checkpoint: {ckpt_path}")
    progress.info(f"Tokenizer: {tokenizer_model_path}")


if __name__ == "__main__":
    main()
