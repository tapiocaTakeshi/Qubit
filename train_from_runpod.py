#!/usr/bin/env python3
"""
RunPod Serverless API から推論結果を取得して学習するスクリプト

RunPod の /runsync (同期) または /run + /status (非同期) エンドポイントを使用して
NeuroQuantum モデルの推論結果を収集し、ローカルモデルの学習データとして活用する。

使い方:
  # 環境変数でRunPod設定を指定
  export RUNPOD_API_KEY="your-api-key"
  export RUNPOD_ENDPOINT_ID="your-endpoint-id"

  # デフォルト実行
  python train_from_runpod.py

  # カスタムパラメータ
  python train_from_runpod.py --epochs 8 --lr 3e-5 --chunks 4

  # 推論のみ（学習スキップ）
  python train_from_runpod.py --infer-only

  # 学習のみ（RunPod推論スキップ、crafted QAのみ）
  python train_from_runpod.py --skip-runpod
"""
import os
import sys
import torch
import torch.nn.functional as F
import json
import math
import random
import gc
import requests
import time
import argparse
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(__file__))
from neuroquantum_layered import NeuroQuantum, NeuroQuantumConfig, NeuroQuantumTokenizer

CKPT_PATH = os.path.join(os.path.dirname(__file__), "neuroq_checkpoint.pt")
TOKENIZER_PATH = os.path.join(os.path.dirname(__file__), "neuroq_tokenizer.model")

# RunPod API
RUNPOD_API_KEY = os.environ.get("RUNPOD_API_KEY", "")
RUNPOD_ENDPOINT_ID = os.environ.get("RUNPOD_ENDPOINT_ID", "")
RUNPOD_BASE_URL = f"https://api.runpod.ai/v2/{RUNPOD_ENDPOINT_ID}"

# Training params
NUM_CHUNKS = 8
EPOCHS_PER_CHUNK = 6
BATCH_SIZE = 1
GRAD_ACCUM = 4
LR = 5e-5
WARMUP_STEPS = 30

# Prompts to query the RunPod endpoint
QUERY_PROMPTS = [
    # 一般知識
    "日本の首都はどこですか？",
    "富士山の高さは何メートルですか？",
    "太陽系で一番大きい惑星は何ですか？",
    "水の化学式は何ですか？",
    "光の速さはどれくらいですか？",
    "日本で一番長い川は何ですか？",
    "地球の年齢はどれくらいですか？",
    "人間の体で一番大きい臓器は何ですか？",
    # 歴史
    "明治維新は何年に起きましたか？",
    "第二次世界大戦はいつ終わりましたか？",
    "源頼朝は何をした人ですか？",
    "江戸時代はいつからいつまでですか？",
    # 科学
    "DNAとは何ですか？",
    "重力とは何ですか？",
    "光合成とは何ですか？",
    "原子とは何ですか？",
    # 文化
    "俳句とは何ですか？",
    "歌舞伎とは何ですか？",
    "茶道について教えてください。",
    "日本の国花は何ですか？",
    # 地理
    "日本で一番大きい湖は何ですか？",
    "北海道の県庁所在地はどこですか？",
    "日本の人口はどれくらいですか？",
    "東京タワーの高さは何メートルですか？",
    # 数学・論理
    "円周率とは何ですか？",
    "三角形の内角の和は何度ですか？",
    "素数とは何ですか？",
    "ピタゴラスの定理とは何ですか？",
    # 技術
    "人工知能とは何ですか？",
    "量子コンピュータとは何ですか？",
    "インターネットとは何ですか？",
    "プログラミングとは何ですか？",
    # 日常
    "健康のために大切なことは何ですか？",
    "読書のメリットは何ですか？",
    "環境問題について教えてください。",
    "睡眠はなぜ大切ですか？",
    # 追加
    "月の直径はどれくらいですか？",
    "酸素の元素記号は何ですか？",
    "日本の国歌は何ですか？",
    "ノーベル賞とは何ですか？",
]

# High-quality reference answers for crafted QA
CRAFTED_QA = [
    "質問: 日本の首都はどこですか？\n回答: 日本の首都は東京です。",
    "質問: 富士山の高さは？\n回答: 富士山の高さは3776メートルです。",
    "質問: 太陽系で一番大きい惑星は？\n回答: 木星が太陽系で最も大きい惑星です。",
    "質問: 水の化学式は？\n回答: 水の化学式はH2Oです。",
    "質問: 光の速さは？\n回答: 光の速さは秒速約30万キロメートルです。",
    "質問: 日本で一番長い川は？\n回答: 信濃川が日本で最も長い川で、全長367kmです。",
    "質問: 地球の年齢はどれくらい？\n回答: 地球の年齢は約46億年です。",
    "質問: 人間の体で一番大きい臓器は？\n回答: 皮膚が人間の体で最も大きい臓器です。",
    "質問: 明治維新は何年に起きましたか？\n回答: 明治維新は1868年に起きました。",
    "質問: 円周率とは何ですか？\n回答: 円周率は円の周の長さと直径の比で、約3.14159です。",
    "質問: 三角形の内角の和は？\n回答: 三角形の内角の和は180度です。",
    "質問: 日本の国花は？\n回答: 日本の国花は桜と菊です。",
    "質問: 琵琶湖はどこにありますか？\n回答: 琵琶湖は滋賀県にある日本最大の湖です。",
    "質問: 人工知能とは？\n回答: 人工知能（AI）は、人間の知的能力をコンピュータで実現する技術です。",
    "質問: 量子コンピュータとは？\n回答: 量子コンピュータは量子力学の原理を利用して計算を行うコンピュータです。",
    "質問: DNAとは何ですか？\n回答: DNAはデオキシリボ核酸の略で、遺伝情報を保持する分子です。",
]


# ── RunPod API helpers ────────────────────────────────────────

def _runpod_headers():
    """RunPod API認証ヘッダーを返す。"""
    return {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {RUNPOD_API_KEY}",
    }


def runpod_runsync(payload, timeout=60, retries=3):
    """RunPod /runsync エンドポイントに同期リクエストを送信。"""
    url = f"{RUNPOD_BASE_URL}/runsync"
    for attempt in range(retries):
        try:
            resp = requests.post(
                url,
                headers=_runpod_headers(),
                json={"input": payload},
                timeout=timeout,
            )
            if resp.status_code == 200:
                result = resp.json()
                status = result.get("status")
                if status == "COMPLETED":
                    return result.get("output", {})
                elif status == "FAILED":
                    print(f"    RunPod job failed: {result.get('error', 'unknown')}")
                    return {}
                elif status == "IN_QUEUE" or status == "IN_PROGRESS":
                    # Fell through to async, poll for result
                    job_id = result.get("id")
                    if job_id:
                        return _poll_runpod_status(job_id, timeout=timeout)
                return result.get("output", result)
            elif resp.status_code == 401:
                print("    RunPod API認証エラー: RUNPOD_API_KEYを確認してください")
                return {}
            elif resp.status_code == 404:
                print("    RunPodエンドポイントが見つかりません: RUNPOD_ENDPOINT_IDを確認してください")
                return {}
            else:
                print(f"    HTTP {resp.status_code}: {resp.text[:200]}")
        except requests.exceptions.Timeout:
            print(f"    タイムアウト (試行 {attempt+1}/{retries})")
        except Exception as e:
            print(f"    エラー: {e}")

        if attempt < retries - 1:
            wait = 2 ** (attempt + 1)
            print(f"    {wait}秒後にリトライ...")
            time.sleep(wait)

    return {}


def runpod_run_async(payload, retries=3):
    """RunPod /run エンドポイントに非同期リクエストを送信。ジョブIDを返す。"""
    url = f"{RUNPOD_BASE_URL}/run"
    for attempt in range(retries):
        try:
            resp = requests.post(
                url,
                headers=_runpod_headers(),
                json={"input": payload},
                timeout=30,
            )
            if resp.status_code == 200:
                return resp.json().get("id", "")
            else:
                print(f"    HTTP {resp.status_code}: {resp.text[:200]}")
        except Exception as e:
            print(f"    エラー: {e}")

        if attempt < retries - 1:
            wait = 2 ** (attempt + 1)
            time.sleep(wait)

    return ""


def _poll_runpod_status(job_id, timeout=120, poll_interval=2):
    """RunPod /status/{job_id} をポーリングして結果を取得。"""
    url = f"{RUNPOD_BASE_URL}/status/{job_id}"
    start_time = time.time()

    while time.time() - start_time < timeout:
        try:
            resp = requests.get(url, headers=_runpod_headers(), timeout=15)
            if resp.status_code == 200:
                result = resp.json()
                status = result.get("status")
                if status == "COMPLETED":
                    return result.get("output", {})
                elif status == "FAILED":
                    print(f"    ジョブ失敗: {result.get('error', 'unknown')}")
                    return {}
                elif status in ("IN_QUEUE", "IN_PROGRESS"):
                    time.sleep(poll_interval)
                    continue
        except Exception as e:
            print(f"    ステータス確認エラー: {e}")

        time.sleep(poll_interval)

    print(f"    ジョブ {job_id} タイムアウト")
    return {}


# ── Inference via RunPod ──────────────────────────────────────

def query_runpod(prompt, max_new_tokens=150, temperature=0.7):
    """RunPod エンドポイントで推論を実行。"""
    payload = {
        "prompt": f"質問: {prompt}\n回答:",
        "max_new_tokens": max_new_tokens,
        "temperature": temperature,
        "top_p": 0.9,
    }
    result = runpod_runsync(payload, timeout=60)
    if isinstance(result, dict):
        return result.get("generated_text", "")
    return ""


def query_runpod_batch(prompts, max_new_tokens=150, temperature=0.7):
    """複数プロンプトを非同期で一括推論。"""
    job_ids = []
    for prompt in prompts:
        payload = {
            "prompt": f"質問: {prompt}\n回答:",
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
            "top_p": 0.9,
        }
        job_id = runpod_run_async(payload)
        if job_id:
            job_ids.append((prompt, job_id))
        time.sleep(0.1)  # Rate limiting

    results = []
    for prompt, job_id in job_ids:
        output = _poll_runpod_status(job_id, timeout=120)
        text = output.get("generated_text", "") if isinstance(output, dict) else ""
        if text:
            results.append((prompt, text))

    return results


def train_qa_via_runpod(qa_pairs, epochs=4, lr=3e-5, repeat=3):
    """RunPod エンドポイント上でQA学習を実行。"""
    print(f"\n  RunPod上でQA学習を実行中...")
    print(f"  QAペア: {len(qa_pairs)}, エポック: {epochs}, リピート: {repeat}")

    chunk_size = max(len(qa_pairs) // 2, 1)
    chunks = [qa_pairs[i:i+chunk_size] for i in range(0, len(qa_pairs), chunk_size)]

    results = []
    for i, chunk in enumerate(chunks):
        print(f"\n  チャンク {i+1}/{len(chunks)} ({len(chunk)} QAペア)...")
        result = runpod_runsync({
            "action": "train_qa",
            "qa_pairs": chunk,
            "epochs": epochs,
            "lr": lr,
            "batch_size": 4,
            "grad_accum_steps": 4,
            "repeat": repeat,
        }, timeout=600)
        status = result.get("status", result.get("error", "unknown")) if isinstance(result, dict) else "unknown"
        final_loss = result.get("final_loss", "N/A") if isinstance(result, dict) else "N/A"
        print(f"    状態: {status}, 最終Loss: {final_loss}")
        results.append(result)

    return results


# ── Training data generation ──────────────────────────────────

def generate_training_data():
    """RunPodエンドポイントから推論結果を収集して学習データを生成。"""
    print(f"  RunPodエンドポイントに問い合わせ中...")
    print(f"  プロンプト数: {len(QUERY_PROMPTS)}")

    generated_texts = []

    # バッチ処理で効率化
    batch_size = 5
    for batch_start in range(0, len(QUERY_PROMPTS), batch_size):
        batch = QUERY_PROMPTS[batch_start:batch_start + batch_size]
        results = query_runpod_batch(batch)

        for prompt, text in results:
            if text and len(text) > 10:
                full_text = f"質問: {prompt}\n回答:{text}" if not text.startswith("質問:") else text
                generated_texts.append(full_text)

        done = min(batch_start + batch_size, len(QUERY_PROMPTS))
        print(f"    生成済み: {done}/{len(QUERY_PROMPTS)} ({len(generated_texts)} サンプル取得)")

    print(f"  RunPod生成結果: {len(generated_texts)} サンプル")
    return generated_texts


# ── Local training ────────────────────────────────────────────

def tokenize_texts(texts, tokenizer, max_seq_len):
    sequences = []
    for t in texts:
        ids = tokenizer.encode(t)
        if len(ids) > 2:
            sequences.append(ids[:max_seq_len])
    return sequences


def train_chunk(model, config, tokenizer, chunk_texts, chunk_idx, num_chunks, device):
    """Train one chunk locally."""
    print(f"\n{'='*60}")
    print(f"  Chunk {chunk_idx+1}/{num_chunks}: {len(chunk_texts)} texts")
    print(f"{'='*60}")

    max_seq_len = config["max_seq_len"]
    sequences = tokenize_texts(chunk_texts, tokenizer, max_seq_len)
    print(f"  Sequences: {len(sequences)}")

    if not sequences:
        print("  No sequences, skipping.")
        return None

    steps_per_epoch = len(sequences) // BATCH_SIZE
    total_steps = (steps_per_epoch * EPOCHS_PER_CHUNK) // GRAD_ACCUM
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    min_lr_ratio = 0.1

    model.train()
    global_step = 0
    best_loss = float('inf')

    for epoch in range(EPOCHS_PER_CHUNK):
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
                shift_logits.view(-1, config["vocab_size"]),
                shift_labels.view(-1),
                ignore_index=-100
            )
            loss = loss / GRAD_ACCUM
            loss.backward()

            total_loss += loss.item() * GRAD_ACCUM
            n_batches += 1

            if n_batches % GRAD_ACCUM == 0:
                if global_step < WARMUP_STEPS:
                    lr = LR * global_step / max(WARMUP_STEPS, 1)
                else:
                    progress = (global_step - WARMUP_STEPS) / max(total_steps - WARMUP_STEPS, 1)
                    cosine_decay = 0.5 * (1 + math.cos(math.pi * progress))
                    lr = LR * (min_lr_ratio + (1 - min_lr_ratio) * cosine_decay)
                for pg in optimizer.param_groups:
                    pg['lr'] = lr
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
        print(f"  Chunk {chunk_idx+1} Epoch {epoch+1}/{EPOCHS_PER_CHUNK} | Loss: {avg_loss:.4f} | Steps: {global_step}")

        if avg_loss < best_loss:
            best_loss = avg_loss

    del optimizer
    gc.collect()
    return best_loss


def save_checkpoint(model, config, chunk_info):
    """Save model checkpoint."""
    ckpt = {
        "model_state": model.state_dict(),
        "config": config,
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "chunk_info": chunk_info,
        "data_source": "runpod",
    }
    torch.save(ckpt, CKPT_PATH)
    size_mb = os.path.getsize(CKPT_PATH) / (1024 * 1024)
    print(f"  Checkpoint saved: {CKPT_PATH} ({size_mb:.1f}MB)")


# ── Main ──────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="RunPod APIから推論結果を取得してローカルモデルを学習",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--endpoint-id", default=RUNPOD_ENDPOINT_ID, help="RunPod Endpoint ID")
    parser.add_argument("--api-key", default=RUNPOD_API_KEY, help="RunPod API Key")
    parser.add_argument("--chunks", type=int, default=NUM_CHUNKS, help="学習チャンク数")
    parser.add_argument("--epochs", type=int, default=EPOCHS_PER_CHUNK, help="チャンクあたりのエポック数")
    parser.add_argument("--lr", type=float, default=LR, help="学習率")
    parser.add_argument("--skip-runpod", action="store_true", help="RunPod推論をスキップ（crafted QAのみ）")
    parser.add_argument("--infer-only", action="store_true", help="推論テストのみ実行")
    parser.add_argument("--remote-train", action="store_true", help="RunPod上でQA学習を実行")
    parser.add_argument("--prompt", action="append", help="推論テスト用プロンプト")
    args = parser.parse_args()

    # Update globals from args
    global RUNPOD_API_KEY, RUNPOD_ENDPOINT_ID, RUNPOD_BASE_URL
    global NUM_CHUNKS, EPOCHS_PER_CHUNK, LR
    if args.api_key:
        RUNPOD_API_KEY = args.api_key
    if args.endpoint_id:
        RUNPOD_ENDPOINT_ID = args.endpoint_id
        RUNPOD_BASE_URL = f"https://api.runpod.ai/v2/{RUNPOD_ENDPOINT_ID}"
    NUM_CHUNKS = args.chunks
    EPOCHS_PER_CHUNK = args.epochs
    LR = args.lr

    # Validate RunPod config
    if not args.skip_runpod:
        if not RUNPOD_API_KEY:
            print("エラー: RUNPOD_API_KEY が設定されていません")
            print("  export RUNPOD_API_KEY='your-api-key'")
            sys.exit(1)
        if not RUNPOD_ENDPOINT_ID:
            print("エラー: RUNPOD_ENDPOINT_ID が設定されていません")
            print("  export RUNPOD_ENDPOINT_ID='your-endpoint-id'")
            sys.exit(1)

    print("=" * 60)
    print("  RunPod APIからの学習")
    print(f"  Endpoint: {RUNPOD_BASE_URL}")
    print(f"  モード: {'推論のみ' if args.infer_only else 'リモート学習' if args.remote_train else 'ローカル学習'}")
    print("=" * 60)

    # ── Infer-only mode ───────────────────────────────────────
    if args.infer_only:
        prompts = args.prompt or ["日本の首都は", "量子コンピュータとは", "人工知能の未来は"]
        print(f"\n[推論テスト] {len(prompts)} プロンプト")
        for i, prompt in enumerate(prompts):
            print(f"\n  [{i+1}/{len(prompts)}] {prompt}")
            text = query_runpod(prompt, max_new_tokens=100, temperature=0.7)
            print(f"  → {text if text else '(応答なし)'}")
        return

    # ── Remote train mode (train on RunPod endpoint) ──────────
    if args.remote_train:
        qa_pairs = [
            {"question": q.split("\n")[0].replace("質問: ", ""),
             "answer": q.split("\n")[1].replace("回答: ", "")}
            for q in CRAFTED_QA
        ]
        results = train_qa_via_runpod(qa_pairs, epochs=args.epochs, lr=args.lr)
        success = sum(1 for r in results if isinstance(r, dict) and r.get("status") == "training_complete")
        print(f"\n  リモート学習完了: {success}/{len(results)} チャンク成功")
        return

    # ── Local training mode ───────────────────────────────────

    # Load model
    print("\n[1/4] モデルロード中...")
    device = torch.device("cpu")
    ckpt = torch.load(CKPT_PATH, map_location=device, weights_only=False)
    cfg = ckpt["config"]
    model_config = NeuroQuantumConfig(
        vocab_size=cfg["vocab_size"],
        embed_dim=cfg["embed_dim"],
        hidden_dim=cfg["hidden_dim"],
        num_heads=cfg["num_heads"],
        num_layers=cfg["num_layers"],
        max_seq_len=cfg["max_seq_len"],
        lambda_entangle=cfg.get("entangle_strength", cfg.get("lambda_entangle", 0.5)),
        dropout=cfg.get("dropout", 0.1),
    )
    model = NeuroQuantum(model_config)
    model.load_state_dict(ckpt.get("model_state_dict") or ckpt["model_state"])
    model.to(device)
    params = sum(p.numel() for p in model.parameters())
    print(f"  Model: {params:,} params on {device}")

    tokenizer = NeuroQuantumTokenizer(model_file=TOKENIZER_PATH)
    del ckpt
    gc.collect()

    # Generate data from RunPod endpoint
    if not args.skip_runpod:
        print("\n[2/4] RunPodから学習データを生成中...")
        endpoint_texts = generate_training_data()
    else:
        print("\n[2/4] RunPod推論スキップ (crafted QAのみ使用)")
        endpoint_texts = []

    # Combine with crafted QA
    all_texts = endpoint_texts.copy()
    crafted_repeats = 60
    for _ in range(crafted_repeats):
        all_texts.extend(CRAFTED_QA)

    print(f"\n  学習テキスト合計: {len(all_texts)}")
    print(f"    - RunPod生成: {len(endpoint_texts)}")
    print(f"    - Crafted QA: {crafted_repeats * len(CRAFTED_QA)}")

    # Load additional HF datasets
    print("\n[3/4] HuggingFaceデータセットを追加ロード中...")
    try:
        import signal
        from datasets import load_dataset

        class DatasetTimeoutError(Exception):
            pass

        def timeout_handler(signum, frame):
            raise DatasetTimeoutError("Timed out")

        ds_list = [
            {"id": "fujiki/japanese_alpaca_data", "format": "alpaca"},
            {"id": "kunishou/oasst1-chat-44k-ja", "format": "conversations"},
        ]

        for ds_info in ds_list:
            ds_id = ds_info["id"]
            fmt = ds_info["format"]
            try:
                print(f"  Loading {ds_id} (streaming, max=200)...")
                signal.signal(signal.SIGALRM, timeout_handler)
                signal.alarm(60)
                ds = load_dataset(ds_id, split="train", streaming=True)
                count = 0
                for row in ds:
                    if fmt == "alpaca":
                        inst = row.get("instruction", "")
                        out = row.get("output", "")
                        inp = row.get("input", "")
                        if inst and out:
                            q = f"{inst}\n{inp}".strip() if inp else inst
                            text = f"質問: {q}\n回答: {out}"
                            if len(text) > 10:
                                all_texts.append(text)
                                count += 1
                    elif fmt == "conversations":
                        convs = row.get("conversations", [])
                        if convs and len(convs) >= 2:
                            parts = []
                            for c in convs:
                                role = c.get("from", "")
                                val = c.get("value", "")
                                if role == "human":
                                    parts.append(f"質問: {val}")
                                elif role == "gpt":
                                    parts.append(f"回答: {val}")
                            if parts:
                                text = "\n".join(parts)
                                if len(text) > 10:
                                    all_texts.append(text)
                                    count += 1
                    if count >= 200:
                        break
                signal.alarm(0)
                print(f"    -> {count} samples")
            except DatasetTimeoutError:
                signal.alarm(0)
                print(f"    -> Timed out, skipping")
            except Exception as e:
                signal.alarm(0)
                print(f"    -> Failed: {e}")
            gc.collect()
    except ImportError:
        print("  datasetsが利用不可、HFデータセットをスキップ")

    print(f"  最終テキスト数: {len(all_texts)}")

    # Split into chunks and train
    random.shuffle(all_texts)
    chunk_size = max(len(all_texts) // NUM_CHUNKS, 1)
    chunks = []
    for i in range(NUM_CHUNKS):
        start = i * chunk_size
        end = start + chunk_size if i < NUM_CHUNKS - 1 else len(all_texts)
        chunks.append(all_texts[start:end])

    del all_texts
    gc.collect()
    print(f"  チャンク: {[len(c) for c in chunks]}")

    # Train
    print("\n[4/4] 学習中...")
    results = []
    for i in range(NUM_CHUNKS):
        best_loss = train_chunk(model, cfg, tokenizer, chunks[i], i, NUM_CHUNKS, device)
        results.append({"chunk": i + 1, "best_loss": best_loss})
        save_checkpoint(model, cfg, {"chunks_completed": i + 1, "results": results})
        chunks[i] = None
        gc.collect()

    # Summary
    print("\n" + "=" * 60)
    print("  学習完了!")
    print("=" * 60)
    for r in results:
        if r['best_loss']:
            print(f"  Chunk {r['chunk']}: Loss = {r['best_loss']:.4f}")
        else:
            print(f"  Chunk {r['chunk']}: skipped")
    print(f"\n  Checkpoint: {CKPT_PATH}")

    # Quick inference test
    print("\n[テスト] 推論テスト...")
    model.eval()
    test_prompt = "質問: 日本の首都は"
    ids = tokenizer.encode(test_prompt)
    input_t = torch.tensor([ids], dtype=torch.long, device=device)
    with torch.no_grad():
        for _ in range(30):
            logits = model(input_t)
            next_logits = logits[0, -1, :] / 0.7
            probs = torch.softmax(next_logits, dim=-1)
            next_id = torch.multinomial(probs, 1).item()
            input_t = torch.cat([input_t, torch.tensor([[next_id]], device=device)], dim=1)
            if next_id == tokenizer.eos_id:
                break
    output = tokenizer.decode(input_t[0].tolist())
    print(f"  プロンプト: {test_prompt}")
    print(f"  出力: {output}")


if __name__ == "__main__":
    main()
