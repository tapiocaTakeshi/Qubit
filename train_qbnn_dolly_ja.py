#!/usr/bin/env python3
"""
Entangled QBNN (E-QBNN) を日本語指示データセット databricks-dolly-15k-ja
(llm-jp/databricks-dolly-15k-ja) で学習し、チェックポイントを保存する。

本スクリプトは `qbnn_layered.py` の `EQBNNGenerativeAI` / `EQBNNGenerativeModel`
(層間エンタングルメントを持つ量子ビットニューラルネットワーク = QBNN) を
dolly-15k-ja で学習させるための専用エントリポイントである。
`train_small_dolly_ja.py` が NeuroQuantum (Transformer 系) を学習するのに対し、
こちらは QBNN アーキテクチャを学習する。

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

という会話形式へ整形して QBNN の学習に用いる。

学習済み重みは既定で既存の pt ファイル neuroq_small_oasst_ja_checkpoint.pt へ
上書き保存する (ユーザー指定)。別ファイルへ保存したい場合は --ckpt-name を指定する。

使い方:
    # 1. 全サンプル (約 15k 件) を 3 エポックで学習し既存 pt へ上書き保存 (デフォルト):
    python train_qbnn_dolly_ja.py

    # 2. エポック数・ニューロン数を変えて学習:
    python train_qbnn_dolly_ja.py --epochs 3 --num-neurons 2048

    # 3. 学習 + HF アップロード:
    HF_TOKEN=hf_xxx python train_qbnn_dolly_ja.py \
        --epochs 3 --upload --repo-id tapiocatakeshi/Qubit

    # 4. 別の pt ファイルへ保存 (上書きしたくない場合):
    python train_qbnn_dolly_ja.py --ckpt-name qbnn_dolly_ja_checkpoint.pt

    # 5. 一部のみで動作確認:
    python train_qbnn_dolly_ja.py --max-samples 200 --epochs 1

備考:
    - llm-jp/databricks-dolly-15k-ja は train split のみを公開している。
    - 列名はデータセットにより `input`/`context`、`output`/`response` の
      ゆらぎがあるため、両方を許容している。
    - QBNN の SimpleTokenizer は日本語を自動検出し文字単位でトークナイズする。
    - 既定の保存先は NeuroQuantum 由来の名前だが、保存後の中身は QBNN になる。
"""
import argparse
import json
import os
import pickle
import sys
from datetime import datetime, timezone

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dataset_utils import safe_load_dataset, sync_checkpoint_to_network_volume
from qbnn_layered import EQBNNGenerativeAI
from progress_logger import ProgressLogger

DATASET_ID = "llm-jp/databricks-dolly-15k-ja"
DEFAULT_SPLIT = "train"  # dolly-15k-ja は train split のみを公開している
DEFAULT_REPO_ID = "tapiocatakeshi/Qubit"
# 学習結果は既存の pt ファイルへ上書き保存する (ユーザー指定)。
# 注意: このファイルは元々 NeuroQuantum (oasst) の重みだが、本スクリプト実行後は
# QBNN (E-QBNN / dolly-15k-ja) の重み・トークナイザーに置き換わる。
DEFAULT_CHECKPOINT_NAME = "neuroq_small_oasst_ja_checkpoint.pt"

progress = ProgressLogger("train_qbnn_dolly_ja")


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


def save_checkpoint(ai, ckpt_path: str, args, n_texts: int, final_loss):
    """QBNN モデルとトークナイザーをチェックポイントへ保存する。

    EQBNNGenerativeModel は state_dict で、SimpleTokenizer は語彙辞書を
    pickle 互換の形でまとめて torch.save する。
    """
    if os.path.isfile(ckpt_path):
        progress.warning(
            f"既存ファイルを上書きします: {ckpt_path} "
            "(元の内容は失われます)"
        )
    tok = ai.tokenizer
    checkpoint = {
        "model_state": ai.model.state_dict(),
        "tokenizer": {
            "word2idx": tok.word2idx,
            "idx2word": tok.idx2word,
            "vocab_size": tok.vocab_size,
            "max_vocab_size": tok.max_vocab_size,
            "use_char": tok.use_char,
            "is_japanese": tok.is_japanese,
        },
        "config": {
            "architecture": "eqbnn",
            "embed_dim": ai.embed_dim,
            "hidden_dims": ai.hidden_dims,
            "entangle_strength": ai.entangle_strength,
            "vocab_size": tok.vocab_size,
            "seq_length": args.seq_length,
        },
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "datasets": [args.dataset_id],
        "total_texts": n_texts,
        "epochs": args.epochs,
        "lr": args.lr,
        "batch_size": args.batch_size,
        "final_loss": final_loss,
    }
    torch.save(checkpoint, ckpt_path)
    size_mb = os.path.getsize(ckpt_path) / 1024 / 1024
    progress.info(f"Saved: {ckpt_path} ({size_mb:.1f} MB)")

    # トークナイザーを単体でも保存しておく (推論時の利便性のため)。
    tok_path = os.path.splitext(ckpt_path)[0] + "_tokenizer.pkl"
    with open(tok_path, "wb") as f:
        pickle.dump(checkpoint["tokenizer"], f)
    progress.info(f"Saved tokenizer: {tok_path}")

    sync_checkpoint_to_network_volume(ckpt_path, tokenizer_path=tok_path)
    return tok_path


def upload_checkpoint_to_hf(ckpt_path, repo_id, hf_token, tokenizer_path=None):
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
        commit_message=f"Add {ckpt_basename} (QBNN / dolly-15k-ja)",
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
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--dataset-id", default=DATASET_ID, help="HF データセット ID")
    p.add_argument("--split", default=DEFAULT_SPLIT, help="読み込む split (dolly-15k-ja は train)")
    p.add_argument(
        "--max-samples",
        type=int,
        default=0,
        help="使用する最大サンプル数。0 または負値で全件 (約 15k) を使用",
    )
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--seq-length", type=int, default=64)
    p.add_argument("--embed-dim", type=int, default=128)
    p.add_argument("--num-neurons", type=int, default=1024, help="各 E-QBNN 層のニューロン数")
    p.add_argument("--entangle-strength", type=float, default=0.5)
    p.add_argument("--max-vocab-size", type=int, default=8000)
    p.add_argument("--ckpt-name", default=DEFAULT_CHECKPOINT_NAME)
    p.add_argument("--upload", action="store_true", help="HF Hub にアップロードする")
    p.add_argument("--repo-id", default=DEFAULT_REPO_ID)
    p.add_argument(
        "--hf-token",
        default=os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN"),
    )
    return p.parse_args()


def main():
    args = parse_args()

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
    progress.info(f"Formatted conversation texts: {len(texts)}")

    # ---- QBNN 構築 ----
    progress.info("=== Building Entangled QBNN (E-QBNN) ===")
    ai = EQBNNGenerativeAI(
        embed_dim=args.embed_dim,
        num_neurons=args.num_neurons,
        entangle_strength=args.entangle_strength,
        max_vocab_size=args.max_vocab_size,
    )

    # ---- 学習 ----
    progress.info(
        f"=== Training QBNN: epochs={args.epochs}, batch_size={args.batch_size}, "
        f"lr={args.lr}, seq_length={args.seq_length} ==="
    )
    progress.start_training(
        epochs=args.epochs,
        total_sequences=len(texts),
        batch_size=args.batch_size,
        lr=args.lr,
    )
    ai.train(
        texts,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        seq_length=args.seq_length,
    )
    n_params = sum(p.numel() for p in ai.model.parameters())
    progress.info(f"Parameters: {n_params:,}")
    progress.info(ai.get_entanglement_report())

    # ---- チェックポイント保存 ----
    progress.info("=== Saving checkpoint ===")
    ckpt_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), args.ckpt_name)
    final_loss = None  # EQBNNGenerativeAI.train は loss を返さないため None
    tok_path = save_checkpoint(ai, ckpt_path, args, len(texts), final_loss)
    progress.end_training(final_loss=final_loss, checkpoint_path=ckpt_path)

    # ---- 学習履歴を保存 ----
    history_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "training_history_qbnn_dolly_ja.json",
    )
    history = {
        "architecture": "eqbnn",
        "config": {
            "embed_dim": args.embed_dim,
            "num_neurons": args.num_neurons,
            "hidden_dims": ai.hidden_dims,
            "entangle_strength": args.entangle_strength,
            "vocab_size": ai.tokenizer.vocab_size,
            "seq_length": args.seq_length,
        },
        "datasets": [args.dataset_id],
        "total_texts": len(texts),
        "parameters": n_params,
        "max_samples": effective_max_samples,
        "epochs": args.epochs,
        "lr": args.lr,
        "batch_size": args.batch_size,
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
                tokenizer_path=tok_path,
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
