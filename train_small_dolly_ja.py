#!/usr/bin/env python3
"""
NeuroQuantum "small" モデルを日本語指示データセット databricks-dolly-15k-ja
(llm-jp/databricks-dolly-15k-ja) で学習し、チェックポイントを Hugging Face
Hub にアップロードする。

databricks-dolly-15k-ja は LLM-jp が公開した Databricks dolly-15k の日本語
翻訳版で、約 15,000 件の指示追従データを含む。各サンプルは

    instruction : 指示文
    input       : 補助的な文脈 (空の場合あり)
    output      : 模範解答
    category    : タスク種別

という列を持つ。本スクリプトは各サンプルを

    <USER> {instruction}
    {input}            # input がある場合のみ
    <ASSISTANT> {output}

という会話形式へ整形して言語モデルの学習に用いる。OASST 等で事前学習済みの
チェックポイントから `--resume` で追加学習(指示チューニング)する用途にも
適している。

使い方:
    # 1. 全サンプル学習 (デフォルト, 約 15k 件):
    python train_small_dolly_ja.py --epochs 3

    # 2. 既存チェックポイントから dolly-15k-ja で追加学習:
    python train_small_dolly_ja.py --resume \
        --ckpt-name neuroq_small_oasst_ja_checkpoint.pt \
        --tokenizer-prefix neuroq_small_oasst_ja_tokenizer \
        --reset-epochs --epochs 3

    # 3. 学習 + HF アップロード:
    HF_TOKEN=hf_xxx python train_small_dolly_ja.py \
        --epochs 3 --upload --repo-id tapiocatakeshi/Qubit

    # 4. 一部のみで動作確認 (CPU でも数分):
    python train_small_dolly_ja.py --max-samples 200 --epochs 1 --vocab-size 4000

備考:
    - llm-jp/databricks-dolly-15k-ja は train split のみを公開している。
    - 列名はデータセットにより `input`/`context`、`output`/`response` の
      ゆらぎがあるため、両方を許容している。
"""
import argparse
import json
import os
import sys
from datetime import datetime, timezone

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

DATASET_ID = "llm-jp/databricks-dolly-15k-ja"
DEFAULT_SPLIT = "train"  # dolly-15k-ja は train split のみを公開している
DEFAULT_REPO_ID = "tapiocatakeshi/Qubit"
DEFAULT_CHECKPOINT_NAME = "neuroq_small_dolly_ja_checkpoint.pt"

progress = ProgressLogger("train_small_dolly_ja")


def _first_str(row, keys):
    """row から keys の最初に見つかった非空文字列を返す。無ければ None。"""
    for k in keys:
        val = row.get(k)
        if isinstance(val, str) and val.strip():
            return val.strip()
    return None


def extract_dolly_texts(ds, max_samples: int):
    """databricks-dolly-15k-ja の各サンプルを会話テキストへ整形する。

    `instruction` を <USER>、任意の `input`/`context` を続け、`output`/`response`
    を <ASSISTANT> として 1 ターン会話を構築する。
    `max_samples <= 0` の場合はデータセット全件を使用する。
    """
    texts = []
    n = len(ds) if max_samples is None or max_samples <= 0 else min(max_samples, len(ds))
    for row in ds.select(range(n)):
        instruction = _first_str(row, ("instruction", "input"))
        if not instruction:
            continue
        # `input`/`context` は補助文脈。instruction と被らない列のみ採用する。
        context = _first_str(row, ("context",))
        if context is None and "input" in row and instruction != _first_str(row, ("input",)):
            context = _first_str(row, ("input",))
        answer = _first_str(row, ("output", "response"))

        parts = [f"<USER> {instruction}"]
        if context:
            parts.append(context)
        if answer:
            parts.append(f"<ASSISTANT> {answer}")
        combined = "\n".join(parts).strip()
        if len(combined) > 4:
            texts.append(combined)
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
        # 長時間の CPU 学習でも進捗を失わないよう、一定間隔で中間チェックポイントを保存する。
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
        commit_message=f"Add {ckpt_basename} (NeuroQuantum small / dolly-15k-ja)",
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
    p.add_argument("--dataset-id", default=DATASET_ID, help="HF データセット ID (デフォルト: databricks-dolly-15k-ja)")
    p.add_argument("--split", default=DEFAULT_SPLIT, help="読み込む split (dolly-15k-ja は train)")
    p.add_argument(
        "--max-samples",
        type=int,
        default=0,
        help="使用する最大サンプル数。0 または負値で全件 (約 15k) を使用",
    )
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--vocab-size", type=int, default=32000)
    p.add_argument("--batch-size", type=int, default=None, help="未指定なら small 設定 (4) を使用")
    p.add_argument("--max-seq-len", type=int, default=None, help="未指定なら small 設定 (4096) を使用")
    p.add_argument("--ckpt-name", default=DEFAULT_CHECKPOINT_NAME)
    p.add_argument(
        "--save-every",
        type=int,
        default=500,
        help="N バッチごとに中間チェックポイントを保存 (0 で無効)。長時間 CPU 学習の進捗保護用",
    )
    p.add_argument(
        "--resume",
        action="store_true",
        help="既存チェックポイント (--ckpt-name) と保存済みトークナイザーから学習を再開/追加学習する。"
        " 事前学習済みモデルへ dolly-15k-ja で指示チューニングする場合に使用",
    )
    p.add_argument(
        "--reset-epochs",
        action="store_true",
        help="--resume 時に training_log のエポック数を無視し、重みのみ流用して 0 エポック目から学習する。"
        " 別データセットで事前学習したチェックポイントを dolly で追加学習する際に推奨",
    )
    p.add_argument("--upload", action="store_true", help="HF Hub にアップロードする")
    p.add_argument("--repo-id", default=DEFAULT_REPO_ID)
    p.add_argument("--hf-token", default=os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN"))
    p.add_argument("--tokenizer-prefix", default="neuroq_small_dolly_ja_tokenizer")
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    progress.info(f"Device: {device}")

    # ---- small モデル設定 ----
    CONFIG = get_model_config_by_size("small", vocab_size=args.vocab_size)
    batch_size = args.batch_size or CONFIG["batch_size"]
    max_seq_len = args.max_seq_len or CONFIG["max_seq_len"]

    # ---- データセット読み込み ----
    progress.info(f"=== Loading {args.dataset_id} (split={args.split}) ===")
    ds = safe_load_dataset(args.dataset_id, split=args.split)
    use_all = args.max_samples is None or args.max_samples <= 0
    effective_max_samples = len(ds) if use_all else min(args.max_samples, len(ds))
    progress.info(
        f"Dataset size: {len(ds)} / using "
        f"{'ALL ' + str(effective_max_samples) if use_all else effective_max_samples} samples"
    )
    texts = extract_dolly_texts(ds, args.max_samples)
    progress.log_dataset_loaded(args.dataset_id, len(texts))
    if not texts:
        raise RuntimeError("dolly-15k-ja からテキストを取得できませんでした")

    # ---- チェックポイントのロード (再開/追加学習時) ----
    ckpt_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), args.ckpt_name)
    resume_ckpt = None
    if args.resume:
        if os.path.isfile(ckpt_path):
            progress.info(f"=== Resuming from checkpoint: {ckpt_path} ===")
            resume_ckpt = torch.load(ckpt_path, map_location="cpu")
        else:
            progress.warning(
                f"--resume 指定ですが {ckpt_path} が見つかりません。新規学習として開始します。"
            )

    # ---- トークナイザー構築 / ロード ----
    tokenizer_prefix = os.path.join(os.path.dirname(os.path.abspath(__file__)), args.tokenizer_prefix)
    tokenizer_model_path = tokenizer_prefix + ".model"
    if resume_ckpt is not None and os.path.isfile(tokenizer_model_path):
        # 再開時は既存トークナイザーを再利用し、語彙のズレと再構築コストを避ける。
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
            # 別データセットで事前学習したチェックポイントを dolly で追加学習する場合、
            # エポックカウントはリセットしつつ過去ログは履歴として保持する。
            start_epoch = 0
            progress.info("--reset-epochs: 重みのみ流用し、dolly 学習を 0 エポック目から開始します。")
        else:
            # training_log には完了済みエポックのみ追加されるため、その数が次の開始エポック。
            start_epoch = len(training_log)
        if resume_ckpt.get("optimizer_state") is not None and not args.reset_epochs:
            try:
                optimizer.load_state_dict(resume_ckpt["optimizer_state"])
                progress.info("Resumed optimizer state.")
            except Exception as e:  # noqa: BLE001
                progress.warning(f"optimizer state の復元に失敗 (新規optimizerで継続): {e}")
        for _ in range(start_epoch):
            scheduler.step()
        progress.info(
            f"Resuming at epoch {start_epoch + 1}/{args.epochs} "
            f"(completed epochs: {start_epoch})"
        )
        if start_epoch >= args.epochs:
            progress.info("既に全エポック完了済みのチェックポイントです。学習をスキップします。")

    progress.start_training(epochs=args.epochs, total_sequences=len(sequences), batch_size=batch_size, lr=args.lr)

    def save_checkpoint(epoch, batch=None, loss=None, final=False):
        """中間／最終チェックポイントを保存する共通処理。"""
        checkpoint = {
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "config": {
                "vocab_size": actual_vocab,
                "embed_dim": CONFIG["embed_dim"],
                "hidden_dim": CONFIG["hidden_dim"],
                "num_heads": CONFIG["num_heads"],
                "num_layers": CONFIG["num_layers"],
                "max_seq_len": max_seq_len,
                "entangle_strength": CONFIG["entangle_strength"],
                "dropout": CONFIG["dropout"],
                "architecture": "neuroquantum",
                "model_size": "small",
            },
            "training_log": training_log,
            "trained_at": datetime.now(timezone.utc).isoformat(),
            "datasets": [args.dataset_id],
            "max_samples": effective_max_samples,
            "epochs": args.epochs,
            "lr": args.lr,
            "progress": {
                "epoch": epoch + 1,
                "batch": batch,
                "loss": loss,
                "final": final,
            },
        }
        torch.save(checkpoint, ckpt_path)
        size_mb = os.path.getsize(ckpt_path) / 1024 / 1024
        progress.info(f"Saved: {ckpt_path} ({size_mb:.1f} MB)")
        sync_checkpoint_to_network_volume(ckpt_path, tokenizer_path=tokenizer_model_path)

    for epoch in range(start_epoch, args.epochs):
        progress.start_epoch(epoch + 1, args.epochs)
        avg = train_epoch(
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
        progress.log_epoch(epoch=epoch + 1, total_epochs=args.epochs, loss=avg)
        training_log.append({"epoch": epoch + 1, "loss": avg, "dataset": args.dataset_id})
        # 各エポック終了時にもチェックポイントを保存する。
        save_checkpoint(epoch=epoch, loss=avg)

    # ---- チェックポイント保存 (最終) ----
    progress.info("=== Saving checkpoint ===")
    final_loss = training_log[-1]["loss"] if training_log else None
    save_checkpoint(epoch=args.epochs - 1, loss=final_loss, final=True)
    progress.end_training(final_loss=final_loss, checkpoint_path=ckpt_path)

    # ---- 学習履歴を保存 ----
    history_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "training_history_small_dolly_ja.json",
    )
    history = {
        "architecture": "neuroquantum",
        "model_size": "small",
        "config": CONFIG,
        "datasets": [args.dataset_id],
        "total_texts": len(texts),
        "total_sequences": len(sequences),
        "parameters": n_params,
        "training_log": training_log,
        "max_samples": effective_max_samples,
        "epochs": args.epochs,
        "lr": args.lr,
        "trained_at": datetime.now(timezone.utc).isoformat(),
    }
    with open(history_path, "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)
    progress.info(f"Training history saved: {history_path}")

    # ---- HF アップロード ----
    if args.upload:
        if not args.hf_token:
            progress.warning(
                "HF_TOKEN が設定されていないため、HF アップロードをスキップします。"
                " `HF_TOKEN=hf_xxx python ... --upload` で再実行してください。"
            )
        else:
            url = upload_checkpoint_to_hf(
                ckpt_path=ckpt_path,
                repo_id=args.repo_id,
                hf_token=args.hf_token,
                tokenizer_path=tokenizer_model_path,
            )
            progress.info(f"Model URL: {url}")
    else:
        progress.info(
            "HF アップロードを行うには --upload と HF_TOKEN を指定してください。"
            f" 例: HF_TOKEN=hf_xxx python {os.path.basename(__file__)} --upload --repo-id {args.repo_id}"
        )

    progress.info("Done!")


if __name__ == "__main__":
    main()
