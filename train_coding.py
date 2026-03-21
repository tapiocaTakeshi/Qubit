#!/usr/bin/env python3
"""HuggingFace Endpointからコーディング関連データを取得して学習するスクリプト。"""
import os
import sys
import json
import time
import random
import math
import gc
import requests
import torch
import torch.nn.functional as F
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(__file__))
from neuroquantum_layered import NeuroQuantum, NeuroQuantumConfig, NeuroQuantumTokenizer

CKPT_PATH = os.path.join(os.path.dirname(__file__), "neuroq_checkpoint.pt")
TOKENIZER_PATH = os.path.join(os.path.dirname(__file__), "neuroq_tokenizer.model")
ENDPOINT_URL = "https://vvcci2ps4y3wfx7m.us-east4.gcp.endpoints.huggingface.cloud"

BATCH_SIZE = 4
GRAD_ACCUM = 4
LR = 3e-5
WARMUP_STEPS = 20
MIN_LR_RATIO = 0.1
EPOCHS = 3

# コーディング関連プロンプト
PROMPTS = [
    # Python
    "Pythonでリストの要素をソートするコードは",
    "Pythonのクラスの定義方法は",
    "Pythonでファイルを読み書きするには",
    "Pythonのデコレータとは",
    "Pythonのリスト内包表記は",
    "Pythonの例外処理の書き方は",
    "Pythonでデータベースに接続するには",
    "Pythonのジェネレータとは",
    "Pythonで非同期処理を行うには",
    "Pythonのユニットテストの書き方は",
    # JavaScript
    "JavaScriptのasync/awaitの使い方は",
    "JavaScriptのPromiseとは",
    "JavaScriptでDOM操作を行うには",
    "JavaScriptのクロージャとは",
    "JavaScriptのアロー関数の書き方は",
    "ReactでコンポーネントをStateで管理するには",
    "Node.jsでHTTPサーバーを作るには",
    "TypeScriptの型定義の方法は",
    # アルゴリズム
    "バブルソートのアルゴリズムは",
    "二分探索の実装方法は",
    "クイックソートのアルゴリズムは",
    "ハッシュテーブルの仕組みは",
    "再帰関数の基本的な書き方は",
    "動的計画法とは",
    "グラフ探索アルゴリズムのBFSは",
    "スタックとキューのデータ構造は",
    # Web開発
    "HTMLの基本構造は",
    "CSSでフレックスボックスを使うには",
    "REST APIの設計原則は",
    "SQLでデータを検索するクエリは",
    "GitHubでプルリクエストを作る方法は",
    "Dockerコンテナの基本的な使い方は",
    "CI/CDパイプラインとは",
    "マイクロサービスアーキテクチャとは",
    # その他言語
    "Rustの所有権システムとは",
    "Goのゴルーチンとは",
    "Javaのインターフェースとは",
    "C言語のポインタとは",
    "SQLのJOIN操作の種類は",
    "正規表現の基本的な書き方は",
    # 設計パターン・概念
    "オブジェクト指向プログラミングの原則は",
    "デザインパターンのシングルトンとは",
    "MVCアーキテクチャとは",
    "テスト駆動開発(TDD)とは",
    "アジャイル開発の手法は",
    "関数型プログラミングの特徴は",
    "バージョン管理システムGitの基本は",
    "コードレビューのベストプラクティスは",
    "リファクタリングの基本的な手法は",
    "セキュアコーディングの原則は",
]


def fetch_texts(prompts, max_new_tokens=250):
    texts = []
    for i, prompt in enumerate(prompts):
        try:
            resp = requests.post(
                ENDPOINT_URL,
                json={"inputs": prompt, "parameters": {"max_new_tokens": max_new_tokens, "temperature": 0.7, "top_p": 0.9}},
                headers={"Content-Type": "application/json"},
                timeout=30,
            )
            if resp.status_code == 200:
                data = resp.json()
                if isinstance(data, list) and data:
                    text = data[0].get("generated_text", "")
                    full_text = prompt + text
                    if len(full_text) > 20:
                        texts.append(full_text)
                        if (i + 1) % 10 == 0:
                            print(f"  取得済み: {len(texts)}/{i+1} prompts")
            else:
                print(f"  HTTP {resp.status_code} for prompt {i+1}")
            time.sleep(0.3)
        except Exception as e:
            print(f"  Error on prompt {i+1}: {e}")
            time.sleep(1)
    return texts


def tokenize_texts(texts, tok, max_seq_len):
    sequences = []
    for t in texts:
        ids = tok.encode(t, add_special=True)
        if len(ids) <= max_seq_len:
            if len(ids) >= 4:
                sequences.append(ids)
        else:
            stride = max(max_seq_len // 2, 1)
            for start in range(0, len(ids) - max_seq_len + 1, stride):
                sequences.append(ids[start:start + max_seq_len])
    return sequences


def train(model, tokenizer, cfg, device, sequences):
    if not sequences:
        print("No sequences to train on!")
        return None

    max_seq_len = cfg["max_seq_len"]
    steps_per_epoch = len(sequences) // BATCH_SIZE
    total_steps = (steps_per_epoch * EPOCHS) // GRAD_ACCUM
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)

    model.train()
    global_step = 0
    best_loss = float('inf')

    for epoch in range(EPOCHS):
        random.shuffle(sequences)
        total_loss = 0
        n_batches = 0
        optimizer.zero_grad()

        for i in range(0, len(sequences), BATCH_SIZE):
            batch_seqs = sequences[i:i + BATCH_SIZE]
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

            input_ids_t = torch.tensor(input_ids, dtype=torch.long, device=device)
            labels_t = torch.tensor(labels, dtype=torch.long, device=device)

            logits = model(input_ids_t)
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels_t[..., 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, cfg["vocab_size"]),
                shift_labels.view(-1),
                ignore_index=-100
            )
            loss = loss / GRAD_ACCUM
            loss.backward()

            total_loss += loss.item() * GRAD_ACCUM
            n_batches += 1

            if n_batches % GRAD_ACCUM == 0:
                if global_step < WARMUP_STEPS:
                    cur_lr = LR * global_step / max(WARMUP_STEPS, 1)
                else:
                    progress = (global_step - WARMUP_STEPS) / max(total_steps - WARMUP_STEPS, 1)
                    cosine_decay = 0.5 * (1 + math.cos(math.pi * progress))
                    cur_lr = LR * (MIN_LR_RATIO + (1 - MIN_LR_RATIO) * cosine_decay)
                for pg in optimizer.param_groups:
                    pg['lr'] = cur_lr
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()
                global_step += 1

        if n_batches % GRAD_ACCUM != 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()
            global_step += 1

        avg_loss = total_loss / max(n_batches, 1)
        print(f"  Epoch {epoch+1}/{EPOCHS} | Loss: {avg_loss:.4f} | Steps: {global_step}")
        if avg_loss < best_loss:
            best_loss = avg_loss

    del optimizer
    gc.collect()
    return best_loss


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    checkpoint = torch.load(CKPT_PATH, map_location="cpu")
    cfg = checkpoint["config"]
    tokenizer = NeuroQuantumTokenizer(vocab_size=cfg["vocab_size"], model_file=TOKENIZER_PATH)

    nq_config = NeuroQuantumConfig(
        vocab_size=cfg["vocab_size"],
        embed_dim=cfg["embed_dim"],
        hidden_dim=cfg.get("hidden_dim", cfg["embed_dim"] * 2),
        num_heads=cfg["num_heads"],
        num_layers=cfg["num_layers"],
        max_seq_len=cfg["max_seq_len"],
        dropout=cfg.get("dropout", 0.1),
        lambda_entangle=cfg.get("entangle_strength", 0.5),
    )
    model = NeuroQuantum(config=nq_config).to(device)
    model.load_state_dict(checkpoint["model_state"])
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model loaded: {n_params:,} params")

    # Phase 1: Fetch coding data
    print(f"\n=== Phase 1: コーディングデータ取得 ===")
    print(f"Endpoint: {ENDPOINT_URL}")
    print(f"Prompts: {len(PROMPTS)}")

    texts = fetch_texts(PROMPTS, max_new_tokens=300)
    print(f"\n取得テキスト数: {len(texts)}")

    if not texts:
        print("No data fetched, aborting.")
        return

    # Phase 2: Tokenize
    print(f"\n=== Phase 2: トークナイズ ===")
    sequences = tokenize_texts(texts, tokenizer, cfg["max_seq_len"])
    print(f"シーケンス数: {len(sequences)}")
    del texts
    gc.collect()

    # Phase 3: Train
    print(f"\n=== Phase 3: コーディング学習 ({EPOCHS} epochs) ===")
    best_loss = train(model, tokenizer, cfg, device, sequences)
    print(f"\nBest Loss: {best_loss:.4f}")

    del sequences
    gc.collect()

    # Save
    model.eval()
    prev_log = checkpoint.get("training_log", [])
    new_checkpoint = {
        "model_state": model.state_dict(),
        "config": cfg,
        "training_log": prev_log + [{"type": "coding", "info": f"loss={best_loss:.4f}", "endpoint": ENDPOINT_URL}],
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "datasets": list(set(checkpoint.get("datasets", []) + ["coding-endpoint"])),
        "qa_training": checkpoint.get("qa_training", False),
    }
    torch.save(new_checkpoint, CKPT_PATH)
    print(f"Checkpoint saved!")

    # Test
    print(f"\n=== テスト推論 ===")
    import api
    api.model = model
    api.tokenizer = tokenizer
    api.config = cfg
    api.device = device
    from api import generate_text

    test_prompts = [
        "質問: Pythonでリストをソートする方法は？\n回答:",
        "質問: JavaScriptのasync/awaitとは？\n回答:",
        "質問: REST APIとは何ですか？\n回答:",
        "質問: Gitの基本的な使い方は？\n回答:",
    ]
    for prompt in test_prompts:
        result = generate_text(prompt, max_new_tokens=150, temperature=0.7, top_k=40, top_p=0.9, repetition_penalty=1.3)
        q = prompt.split("\n")[0]
        print(f"\n{q}")
        print(f"回答: {result}")

    print(f"\n=== コーディング学習完了! ===")


if __name__ == "__main__":
    main()
