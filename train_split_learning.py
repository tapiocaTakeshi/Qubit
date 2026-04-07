#!/usr/bin/env python3
"""
分割学習 (Split Learning) トレーニングスクリプト

モデルをカットレイヤーで分割し、クライアント・サーバー間で
分散学習を行う。ローカルシミュレーションとネットワーク分散の
両方をサポート。

使い方:
  # ローカルシミュレーション（同一マシン上でクライアント・サーバーを模擬）
  python train_split_learning.py --mode local --cut_layer 3

  # ネットワーク分散: サーバー起動
  python train_split_learning.py --mode server --host 0.0.0.0 --port 9000 --cut_layer 3

  # ネットワーク分散: クライアント起動
  python train_split_learning.py --mode client --server_host 192.168.1.100 --server_port 9000 --cut_layer 3

  # カットレイヤーを自動選択（中間点で分割）
  python train_split_learning.py --mode local

  # QAモードで学習
  python train_split_learning.py --mode local --data_mode qa --cut_layer 2

  # カスタムデータセットを使用
  python train_split_learning.py --mode local --dataset_ids "user/my-dataset"
"""

import os
import sys
import argparse
import random
import math
import time
import json
import logging
import threading
from datetime import datetime, timezone

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(__file__))
from neuroquantum_layered import (
    NeuroQuantum, NeuroQuantumConfig, NeuroQuantumTokenizer,
    migrate_legacy_state_dict,
)
from split_learning import (
    SplitLearningTrainer,
    SplitLearningServer,
    SplitLearningClient,
    merge_split_models,
    split_model,
)
from dataset_utils import safe_load_dataset, sync_checkpoint_to_network_volume
from progress_logger import ProgressLogger

progress = ProgressLogger("split_learning")
logger = logging.getLogger(__name__)

CKPT_PATH = os.path.join(os.path.dirname(__file__), "neuroq_checkpoint.pt")

# デフォルトQAデータセット
QA_DATASETS = [
    {"id": "fujiki/japanese_alpaca_data", "format": "alpaca"},
    {"id": "FreedomIntelligence/alpaca-gpt4-japanese", "format": "conversations"},
    {"id": "kunishou/oasst1-chat-44k-ja", "format": "conversations"},
    {"id": "izumi-lab/llm-japanese-dataset", "format": "izumi"},
]

GENERAL_DATASETS = [
    {"id": "izumi-lab/llm-japanese-dataset", "col": "output"},
    {"id": "kunishou/oasst1-chat-44k-ja", "col": "conversations"},
    {"id": "fujiki/japanese_alpaca_data", "col": "output"},
    {"id": "shi3z/Japanese_wikipedia_conversation_100K", "col": "conversations"},
    {"id": "FreedomIntelligence/alpaca-gpt4-japanese", "col": "conversations"},
]

CRAFTED_QA = [
    "質問: 日本の首都はどこですか？\n回答: 日本の首都は東京です。",
    "質問: 富士山の高さはどのくらいですか？\n回答: 富士山の高さは3,776メートルです。日本で最も高い山です。",
    "質問: プログラミングとは何ですか？\n回答: プログラミングとは、コンピュータに実行させる命令を書くことです。",
    "質問: 人工知能とは何ですか？\n回答: 人工知能（AI）は、人間の知能を模倣するコンピュータシステムです。",
    "質問: 量子コンピュータとは何ですか？\n回答: 量子コンピュータは、量子力学の原理を利用して計算を行うコンピュータです。",
]


# ========================================
# データ読み込み（train_split.pyと共通）
# ========================================

def format_qa_alpaca(row):
    instruction = row.get("instruction", "").strip()
    inp = row.get("input", "").strip()
    output = row.get("output", "").strip()
    if not instruction or not output:
        return None
    q = f"{instruction}\n{inp}" if inp else instruction
    return f"質問: {q}\n回答: {output}"


def format_qa_conversations(row):
    convs = row.get("conversations", [])
    if not convs:
        return None
    pairs = []
    i = 0
    while i < len(convs) - 1:
        turn, next_turn = convs[i], convs[i + 1]
        q_text = turn.get("value", turn.get("content", "")).strip() if isinstance(turn, dict) else (turn.strip() if isinstance(turn, str) else "")
        a_text = next_turn.get("value", next_turn.get("content", "")).strip() if isinstance(next_turn, dict) else (next_turn.strip() if isinstance(next_turn, str) else "")
        if q_text and a_text:
            pairs.append(f"質問: {q_text}\n回答: {a_text}")
        i += 2
    return "\n\n".join(pairs) if pairs else None


def format_qa_izumi(row):
    output = row.get("output", "").strip()
    instruction = row.get("input", row.get("instruction", "")).strip() if isinstance(row.get("input", ""), str) else ""
    if not output:
        return None
    return f"質問: {instruction}\n回答: {output}" if instruction else f"回答: {output}"


def load_qa_texts(max_samples):
    all_qa = []
    for ds_info in QA_DATASETS:
        ds_id = ds_info["id"]
        fmt = ds_info["format"]
        ms = min(1000, max_samples) if fmt == "izumi" else max_samples
        logger.info(f"  Loading {ds_id}...")
        try:
            ds = safe_load_dataset(ds_id, split="train")
            n = min(ms, len(ds))
            count = 0
            for row in ds.select(range(n)):
                if fmt == "alpaca":
                    text = format_qa_alpaca(row)
                elif fmt == "conversations":
                    text = format_qa_conversations(row)
                elif fmt == "izumi":
                    text = format_qa_izumi(row)
                else:
                    continue
                if text and len(text) > 10:
                    all_qa.append(text)
                    count += 1
            logger.info(f"    -> {count} QA samples")
        except Exception as e:
            logger.warning(f"    -> ERROR: {e}")
    return all_qa


def load_general_texts(max_samples):
    all_texts = []
    for ds_info in GENERAL_DATASETS:
        ds_id = ds_info["id"]
        col = ds_info["col"]
        logger.info(f"  Loading {ds_id}...")
        try:
            ds = safe_load_dataset(ds_id, split="train")
            n = min(max_samples, len(ds))
            for row in ds.select(range(n)):
                col_data = row.get(col)
                if isinstance(col_data, str) and len(col_data.strip()) > 4:
                    all_texts.append(col_data.strip())
                elif isinstance(col_data, list):
                    parts = []
                    for turn in col_data:
                        if isinstance(turn, dict) and "value" in turn:
                            parts.append(turn["value"])
                        elif isinstance(turn, dict) and "content" in turn:
                            parts.append(turn["content"])
                        elif isinstance(turn, str):
                            parts.append(turn)
                    combined = "\n".join(parts)
                    if len(combined.strip()) > 4:
                        all_texts.append(combined.strip())
            logger.info(f"    -> {len(all_texts)} texts")
        except Exception as e:
            logger.warning(f"    -> ERROR: {e}")
    return all_texts


def tokenize_texts(texts, tokenizer, max_seq_len):
    sequences = []
    for t in texts:
        content_ids = tokenizer.encode(t, add_special=False)
        max_content = max_seq_len - 2
        if max_content <= 0:
            continue
        if len(content_ids) <= max_content:
            if len(content_ids) >= 2:
                seq = [tokenizer.bof_id, tokenizer.bos_id] + content_ids + [tokenizer.eos_id, tokenizer.eof_id]
                sequences.append(seq)
        else:
            stride = max(max_content // 2, 1)
            chunks = list(range(0, len(content_ids) - max_content + 1, stride))
            for idx, start in enumerate(chunks):
                chunk = content_ids[start:start + max_content]
                prefix = [tokenizer.bof_id, tokenizer.bos_id] if idx == 0 else [tokenizer.bos_id]
                suffix = [tokenizer.eos_id, tokenizer.eof_id] if idx == len(chunks) - 1 else [tokenizer.eos_id]
                seq = prefix + chunk + suffix
                sequences.append(seq)
            remaining = content_ids[-max_content:]
            tail_seq = [tokenizer.bos_id] + remaining + [tokenizer.eos_id, tokenizer.eof_id]
            if tail_seq != sequences[-1]:
                sequences.append(tail_seq)
    return sequences


def get_lr(step, total_steps, warmup_steps, max_lr, min_lr_ratio=0.1):
    if step < warmup_steps:
        return max_lr * step / max(warmup_steps, 1)
    progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
    cosine_decay = 0.5 * (1 + math.cos(math.pi * progress))
    return max_lr * (min_lr_ratio + (1 - min_lr_ratio) * cosine_decay)


# ========================================
# ローカル分割学習
# ========================================

def train_local_split(model, tokenizer, sequences, config, nq_config, device, args):
    """
    ローカルで分割学習をシミュレートする。
    同一マシン上でクライアント・サーバーの分割学習を再現。
    """
    cut_layer = args.cut_layer
    if cut_layer is None:
        cut_layer = max(1, nq_config.num_layers // 2)
    logger.info(f"Cut layer: {cut_layer} / {nq_config.num_layers} layers")
    logger.info(f"  Client: embedding + {cut_layer} blocks")
    logger.info(f"  Server: {nq_config.num_layers - cut_layer} blocks + output head")

    trainer = SplitLearningTrainer(
        model=model,
        cut_layer=cut_layer,
        tokenizer=tokenizer,
        device=device,
        lr=args.lr,
        grad_clip=args.grad_clip,
    )

    client_params = sum(p.numel() for p in trainer.client.parameters())
    server_params = sum(p.numel() for p in trainer.server.parameters())
    logger.info(f"  Client params: {client_params:,}")
    logger.info(f"  Server params: {server_params:,}")

    max_seq_len = nq_config.max_seq_len
    batch_size = args.batch_size
    grad_accum_steps = args.grad_accum_steps
    epochs = args.epochs
    warmup_steps = args.warmup_steps

    steps_per_epoch = len(sequences) // batch_size
    total_steps = (steps_per_epoch * epochs) // grad_accum_steps
    logger.info(f"  Sequences: {len(sequences)}, Steps/epoch: {steps_per_epoch}")
    logger.info(f"  Effective batch size: {batch_size * grad_accum_steps}")

    progress.start_training(
        epochs=epochs, total_sequences=len(sequences),
        batch_size=batch_size, lr=args.lr,
    )

    training_log = []
    global_step = 0
    best_loss = float("inf")
    start_time = time.time()

    for epoch in range(epochs):
        progress.start_epoch(epoch + 1, epochs)
        random.shuffle(sequences)
        total_loss = 0
        n_batches = 0
        accum_loss = 0

        for i in range(0, len(sequences), batch_size):
            # タイムアウトチェック
            if args.max_minutes and (time.time() - start_time) >= args.max_minutes * 60:
                elapsed = (time.time() - start_time) / 60
                logger.info(f"  TIMEOUT: {elapsed:.1f}分経過。安全に中断します。")
                avg_loss = total_loss / max(n_batches, 1)
                training_log.append({
                    "epoch": epoch + 1, "loss": avg_loss, "timed_out": True
                })
                return trainer, training_log, best_loss

            batch_seqs = sequences[i:i + batch_size]
            if not batch_seqs:
                continue

            max_len = min(max(len(s) for s in batch_seqs), max_seq_len)
            input_ids = []
            labels = []
            for s in batch_seqs:
                ids = s[:max_len]
                pad_len = max_len - len(ids)
                input_ids.append(ids + [tokenizer.pad_id] * pad_len)
                labels.append(ids + [-100] * pad_len)

            input_t = torch.tensor(input_ids, dtype=torch.long, device=device)
            labels_t = torch.tensor(labels, dtype=torch.long, device=device)

            loss = trainer.train_step(input_t, labels_t)
            total_loss += loss
            accum_loss += loss
            n_batches += 1

            if n_batches % grad_accum_steps == 0:
                # 学習率スケジューリング
                cur_lr = get_lr(global_step, total_steps, warmup_steps, args.lr)
                for opt in [trainer.client_optimizer, trainer.server_optimizer]:
                    for pg in opt.param_groups:
                        pg["lr"] = cur_lr
                global_step += 1

            if n_batches % 100 == 0:
                avg = total_loss / n_batches
                progress.log_batch(epoch=epoch + 1, batch=n_batches, loss=avg)

        avg_loss = total_loss / max(n_batches, 1)
        progress.log_epoch(epoch=epoch + 1, total_epochs=epochs, loss=avg_loss)
        training_log.append({"epoch": epoch + 1, "loss": avg_loss})

        if avg_loss < best_loss:
            best_loss = avg_loss

    return trainer, training_log, best_loss


# ========================================
# ネットワーク分散: サーバーモード
# ========================================

def run_server_mode(model, device, args):
    """サーバーモードで分割学習を実行する。"""
    cut_layer = args.cut_layer
    if cut_layer is None:
        cut_layer = max(1, model.config.num_layers // 2)

    logger.info(f"Starting Split Learning Server")
    logger.info(f"  Host: {args.host}:{args.port}")
    logger.info(f"  Cut layer: {cut_layer} / {model.config.num_layers}")

    server = SplitLearningServer(
        model=model,
        cut_layer=cut_layer,
        device=device,
        host=args.host,
        port=args.port,
        lr=args.lr,
        grad_clip=args.grad_clip,
    )

    server_params = sum(p.numel() for p in server.server_model.parameters())
    logger.info(f"  Server params: {server_params:,}")
    logger.info("  Waiting for client connections...")

    try:
        server.start()
    except KeyboardInterrupt:
        logger.info("Server shutting down...")
        server.running = False

    return server


# ========================================
# ネットワーク分散: クライアントモード
# ========================================

def run_client_mode(model, tokenizer, sequences, nq_config, device, args):
    """クライアントモードで分割学習を実行する。"""
    cut_layer = args.cut_layer
    if cut_layer is None:
        cut_layer = max(1, nq_config.num_layers // 2)

    logger.info(f"Starting Split Learning Client")
    logger.info(f"  Server: {args.server_host}:{args.server_port}")
    logger.info(f"  Cut layer: {cut_layer} / {nq_config.num_layers}")

    client = SplitLearningClient(
        model=model,
        cut_layer=cut_layer,
        device=device,
        server_host=args.server_host,
        server_port=args.server_port,
        lr=args.lr,
        grad_clip=args.grad_clip,
    )

    client_params = sum(p.numel() for p in client.client_model.parameters())
    logger.info(f"  Client params: {client_params:,}")

    # サーバーに接続
    logger.info("  Connecting to server...")
    client.connect()
    logger.info("  Connected!")

    max_seq_len = nq_config.max_seq_len
    batch_size = args.batch_size
    epochs = args.epochs

    training_log = []
    best_loss = float("inf")
    start_time = time.time()

    progress.start_training(epochs=epochs, total_sequences=len(sequences),
                            batch_size=batch_size, lr=args.lr)

    try:
        for epoch in range(epochs):
            progress.start_epoch(epoch + 1, epochs)
            random.shuffle(sequences)
            total_loss = 0
            n_batches = 0

            for i in range(0, len(sequences), batch_size):
                if args.max_minutes and (time.time() - start_time) >= args.max_minutes * 60:
                    logger.info("TIMEOUT reached")
                    break

                batch_seqs = sequences[i:i + batch_size]
                if not batch_seqs:
                    continue

                max_len = min(max(len(s) for s in batch_seqs), max_seq_len)
                input_ids = []
                labels = []
                for s in batch_seqs:
                    ids = s[:max_len]
                    pad_len = max_len - len(ids)
                    input_ids.append(ids + [tokenizer.pad_id] * pad_len)
                    labels.append(ids + [-100] * pad_len)

                input_t = torch.tensor(input_ids, dtype=torch.long, device=device)
                labels_t = torch.tensor(labels, dtype=torch.long, device=device)

                loss = client.train_step(input_t, labels_t)
                total_loss += loss
                n_batches += 1

                if n_batches % 100 == 0:
                    avg = total_loss / n_batches
                    progress.log_batch(epoch=epoch + 1, batch=n_batches, loss=avg)

            avg_loss = total_loss / max(n_batches, 1)
            progress.log_epoch(epoch=epoch + 1, total_epochs=epochs, loss=avg_loss)
            training_log.append({"epoch": epoch + 1, "loss": avg_loss})

            if avg_loss < best_loss:
                best_loss = avg_loss

    finally:
        client.disconnect()

    return client, training_log, best_loss


# ========================================
# チェックポイント保存
# ========================================

def save_checkpoint(model, config, training_log, original_ckpt, args):
    """学習結果をチェックポイントとして保存する。"""
    new_checkpoint = {
        "model_state": model.state_dict(),
        "config": config,
        "training_log": training_log,
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "datasets": original_ckpt.get("datasets", []),
        "split_learning": True,
        "split_learning_config": {
            "cut_layer": args.cut_layer,
            "mode": args.mode,
        },
    }
    torch.save(new_checkpoint, CKPT_PATH)
    size_mb = os.path.getsize(CKPT_PATH) / 1024 / 1024
    logger.info(f"  Saved: {CKPT_PATH} ({size_mb:.1f} MB)")
    sync_checkpoint_to_network_volume(CKPT_PATH)


# ========================================
# 推論テスト
# ========================================

def run_inference_test(model, tokenizer, device, max_seq_len, data_mode="qa"):
    """学習後の推論テスト。"""
    model.eval()
    if data_mode == "qa":
        test_prompts = [
            "質問: 日本の首都はどこですか？\n回答:",
            "質問: プログラミングとは何ですか？\n回答:",
            "質問: AIとは何ですか？\n回答:",
        ]
    else:
        test_prompts = [
            "こんにちは",
            "量子コンピュータとは",
            "日本の首都は",
        ]

    for prompt in test_prompts:
        tokens = tokenizer.encode(prompt, add_special=True)
        input_tensor = torch.tensor([tokens], dtype=torch.long, device=device)
        generated = list(tokens)

        with torch.no_grad():
            for _ in range(80):
                seq = input_tensor[:, -max_seq_len:]
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
                if data_mode == "qa":
                    partial = tokenizer.decode(generated[len(tokens):], skip_special=True)
                    if "質問:" in partial:
                        break

        generated_text = tokenizer.decode(generated[len(tokens):], skip_special=True)
        if data_mode == "qa" and "質問:" in generated_text:
            generated_text = generated_text[:generated_text.index("質問:")].strip()
        logger.info(f"  {prompt.strip()} {generated_text}")


# ========================================
# メイン
# ========================================

def main():
    parser = argparse.ArgumentParser(description="分割学習 (Split Learning) トレーニング")

    # モード選択
    parser.add_argument("--mode", choices=["local", "server", "client"], default="local",
                        help="実行モード: local=ローカルシミュレーション, server=サーバー起動, "
                             "client=クライアント起動 (default: local)")

    # モデル分割
    parser.add_argument("--cut_layer", type=int, default=None,
                        help="カットレイヤーのインデックス (1 ~ num_layers-1)。"
                             "未指定時は中間点で自動分割")

    # データ設定
    parser.add_argument("--data_mode", choices=["qa", "general"], default="qa",
                        help="データモード: qa=QA形式, general=一般テキスト (default: qa)")
    parser.add_argument("--max_samples", type=int, default=2000,
                        help="各データセットからの最大サンプル数 (default: 2000)")
    parser.add_argument("--dataset_ids", nargs="+", default=None,
                        help="カスタムHugging FaceデータセットID")
    parser.add_argument("--crafted_repeat", type=int, default=10,
                        help="手作りQAの繰り返し回数 (default: 10)")

    # 学習ハイパーパラメータ
    parser.add_argument("--epochs", type=int, default=5,
                        help="エポック数 (default: 5)")
    parser.add_argument("--lr", type=float, default=3e-5,
                        help="学習率 (default: 3e-5)")
    parser.add_argument("--batch_size", type=int, default=4,
                        help="バッチサイズ (default: 4)")
    parser.add_argument("--grad_accum_steps", type=int, default=4,
                        help="勾配蓄積ステップ数 (default: 4)")
    parser.add_argument("--warmup_steps", type=int, default=20,
                        help="ウォームアップステップ数 (default: 20)")
    parser.add_argument("--grad_clip", type=float, default=1.0,
                        help="勾配クリッピング (default: 1.0)")
    parser.add_argument("--max_minutes", type=float, default=None,
                        help="最大学習時間（分）")

    # ネットワーク設定（サーバーモード）
    parser.add_argument("--host", type=str, default="0.0.0.0",
                        help="サーバーバインドアドレス (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=9000,
                        help="サーバーポート (default: 9000)")

    # ネットワーク設定（クライアントモード）
    parser.add_argument("--server_host", type=str, default="localhost",
                        help="サーバーホスト (default: localhost)")
    parser.add_argument("--server_port", type=int, default=9000,
                        help="サーバーポート (default: 9000)")

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    # チェックポイント読み込み
    logger.info("=== Loading checkpoint ===")
    if not os.path.exists(CKPT_PATH):
        logger.error(f"Checkpoint not found: {CKPT_PATH}")
        logger.error("Run train_local.py first to create initial checkpoint.")
        sys.exit(1)

    checkpoint = torch.load(CKPT_PATH, map_location="cpu")
    config = checkpoint["config"]
    prev_log = checkpoint.get("training_log", [])
    logger.info(f"Config: embed_dim={config['embed_dim']}, layers={config['num_layers']}, "
                f"vocab={config['vocab_size']}")

    # トークナイザー読み込み
    tokenizer_path = os.path.join(os.path.dirname(__file__), "neuroq_tokenizer.model")
    tokenizer = NeuroQuantumTokenizer(vocab_size=config["vocab_size"], model_file=tokenizer_path)

    # モデル構築
    nq_config = NeuroQuantumConfig(
        vocab_size=config["vocab_size"],
        embed_dim=config["embed_dim"],
        hidden_dim=config.get("hidden_dim", config["embed_dim"] * 2),
        num_heads=config["num_heads"],
        num_layers=config["num_layers"],
        max_seq_len=config["max_seq_len"],
        dropout=config.get("dropout", 0.1),
        lambda_entangle=config.get("entangle_strength", 0.5),
    )
    model = NeuroQuantum(config=nq_config).to(device)
    migrated = migrate_legacy_state_dict(checkpoint["model_state"], model)
    model.load_state_dict(migrated)
    n_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model loaded: {n_params:,} parameters")

    # カットレイヤーの自動設定
    if args.cut_layer is None:
        args.cut_layer = max(1, nq_config.num_layers // 2)
        logger.info(f"Auto cut_layer: {args.cut_layer}")

    # ========================================
    # サーバーモード
    # ========================================
    if args.mode == "server":
        run_server_mode(model, device, args)
        return

    # ========================================
    # データ読み込み（local/clientモード共通）
    # ========================================
    logger.info(f"\n=== Loading datasets (mode: {args.data_mode}) ===")
    if args.data_mode == "qa":
        all_texts = load_qa_texts(args.max_samples)
        for _ in range(args.crafted_repeat):
            all_texts.extend(CRAFTED_QA)
        logger.info(f"  + {len(CRAFTED_QA) * args.crafted_repeat} crafted QA samples")
    else:
        all_texts = load_general_texts(args.max_samples)

    logger.info(f"Total texts: {len(all_texts)}")
    if len(all_texts) == 0:
        logger.error("No texts loaded. Exiting.")
        sys.exit(1)

    # トークナイズ
    sequences = tokenize_texts(all_texts, tokenizer, nq_config.max_seq_len)
    logger.info(f"Tokenized: {len(sequences)} sequences")

    # ========================================
    # ローカルモード
    # ========================================
    if args.mode == "local":
        logger.info("\n=== Split Learning (Local Simulation) ===")
        trainer, training_log, best_loss = train_local_split(
            model, tokenizer, sequences, config, nq_config, device, args
        )

        # モデル統合
        logger.info("\n=== Merging split models ===")
        merged_model = trainer.get_merged_model().to(device)

        # チェックポイント保存
        all_log = list(prev_log) + [
            {**entry, "split_learning": True} for entry in training_log
        ]
        save_checkpoint(merged_model, config, all_log, checkpoint, args)

        # 推論テスト
        logger.info("\n=== Inference test ===")
        run_inference_test(merged_model, tokenizer, device, nq_config.max_seq_len, args.data_mode)

    # ========================================
    # クライアントモード
    # ========================================
    elif args.mode == "client":
        logger.info("\n=== Split Learning (Client Mode) ===")
        client, training_log, best_loss = run_client_mode(
            model, tokenizer, sequences, nq_config, device, args
        )

        # クライアント側のstate_dictのみ保存（完全モデルの再構築にはサーバー側も必要）
        client_ckpt_path = os.path.join(os.path.dirname(__file__), "neuroq_client_checkpoint.pt")
        torch.save({
            "client_state": client.client_model.state_dict(),
            "config": config,
            "cut_layer": args.cut_layer,
            "training_log": training_log,
            "trained_at": datetime.now(timezone.utc).isoformat(),
        }, client_ckpt_path)
        logger.info(f"Client checkpoint saved: {client_ckpt_path}")

    logger.info("\n=== Split Learning complete! ===")
    logger.info(f"Mode: {args.mode}, Cut layer: {args.cut_layer}/{nq_config.num_layers}")
    logger.info("Done!")


if __name__ == "__main__":
    main()
