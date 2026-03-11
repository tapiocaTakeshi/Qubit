#!/usr/bin/env python3
"""
Model Training & Inference API for NeuroQuantum.
Provides REST endpoints for training and text generation.
"""
import os
import sys
import torch
import torch.nn.functional as F
import json
import math
import random
import threading
from datetime import datetime, timezone
from typing import Optional, List
from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel

sys.path.insert(0, os.path.dirname(__file__))
from neuroquantum_layered import NeuroQuantum, NeuroQuantumConfig, NeuroQuantumTokenizer

app = FastAPI(title="NeuroQuantum API", version="1.0.0")

# Global state
model = None
tokenizer = None
config = None
device = None
training_status = {"running": False, "log": [], "message": "idle"}
CKPT_PATH = os.path.join(os.path.dirname(__file__), "neuroq_checkpoint.pt")
TOKENIZER_PATH = os.path.join(os.path.dirname(__file__), "neuroq_tokenizer.model")


# ========================================
# Request/Response models
# ========================================

class InferenceRequest(BaseModel):
    prompt: str
    max_new_tokens: int = 100
    temperature: float = 0.7
    top_k: int = 40
    top_p: float = 0.9
    repetition_penalty: float = 1.3


class InferenceResponse(BaseModel):
    prompt: str
    generated_text: str
    tokens_generated: int


class TrainRequest(BaseModel):
    dataset_ids: Optional[List[str]] = None
    epochs: int = 10
    lr: float = 1e-4
    batch_size: int = 4
    grad_accum_steps: int = 8
    warmup_steps: int = 100
    max_samples_per_dataset: int = 5000


class TrainQARequest(BaseModel):
    dataset_id: Optional[str] = None
    epochs: int = 20
    lr: float = 3e-5
    batch_size: int = 4
    grad_accum_steps: int = 4
    warmup_steps: int = 30
    max_samples_per_dataset: int = 1500


class TrainResponse(BaseModel):
    status: str
    message: str


class TrainStatusResponse(BaseModel):
    running: bool
    log: list
    message: str


# ========================================
# Model loading
# ========================================

def load_model():
    """Load model and tokenizer from checkpoint."""
    global model, tokenizer, config, device

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not os.path.exists(CKPT_PATH):
        raise RuntimeError(f"Checkpoint not found: {CKPT_PATH}")

    checkpoint = torch.load(CKPT_PATH, map_location="cpu")
    config = checkpoint["config"]

    tokenizer = NeuroQuantumTokenizer(
        vocab_size=config["vocab_size"], model_file=TOKENIZER_PATH
    )

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
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model loaded: {n_params:,} params on {device}")
    return checkpoint


# ========================================
# Inference
# ========================================

def generate_text(prompt: str, max_new_tokens: int = 100, temperature: float = 0.7,
                  top_k: int = 40, top_p: float = 0.9, repetition_penalty: float = 1.3) -> str:
    """Generate text from prompt."""
    global model, tokenizer, config, device

    tokens = tokenizer.encode(prompt, add_special=True)
    input_tensor = torch.tensor([tokens], dtype=torch.long, device=device)
    generated = list(tokens)
    max_seq_len = config["max_seq_len"]

    model.eval()
    with torch.no_grad():
        for _ in range(max_new_tokens):
            seq = input_tensor[:, -max_seq_len:]
            logits = model(seq)[:, -1, :] / max(temperature, 1e-5)

            # Top-K filtering
            if top_k > 0:
                k = min(top_k, logits.size(-1))
                topk_vals = torch.topk(logits, k)[0]
                logits[logits < topk_vals[:, -1:]] = float('-inf')

            # Top-P filtering
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                to_remove = cumulative_probs > top_p
                to_remove[:, 1:] = to_remove[:, :-1].clone()
                to_remove[:, 0] = False
                indices_to_remove = sorted_indices[to_remove]
                logits[0, indices_to_remove] = float('-inf')

            # Repetition penalty
            if repetition_penalty > 1.0 and len(generated) > 1:
                for prev in set(generated[-50:]):
                    if prev < logits.size(-1):
                        logits[0, prev] /= repetition_penalty

            probs = F.softmax(logits, dim=-1)
            nxt = torch.multinomial(probs, 1)
            nxt_id = nxt.item()

            if nxt_id == tokenizer.eos_id:
                break
            if nxt_id == tokenizer.pad_id:
                continue

            generated.append(nxt_id)
            input_tensor = torch.cat([input_tensor, nxt], dim=1)

    generated_text = tokenizer.decode(generated[len(tokens):], skip_special=True)
    return generated_text


# ========================================
# Training
# ========================================

def extract_texts(ds, text_column, max_samples):
    """Extract text from dataset."""
    texts = []
    n = min(max_samples, len(ds))
    for row in ds.select(range(n)):
        col_data = row.get(text_column)
        if isinstance(col_data, str) and len(col_data.strip()) > 4:
            texts.append(col_data.strip())
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
                texts.append(combined.strip())
    return texts


def tokenize_texts(texts, tok, max_seq_len):
    """Tokenize texts into training sequences."""
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


def get_lr(step, total_steps, warmup_steps, max_lr):
    """Learning rate with linear warmup and cosine decay."""
    if step < warmup_steps:
        return max_lr * step / max(warmup_steps, 1)
    progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
    return max_lr * 0.5 * (1 + math.cos(math.pi * progress))


# Default datasets
DEFAULT_DATASETS = [
    {"id": "izumi-lab/llm-japanese-dataset", "col": "output"},
    {"id": "kunishou/oasst1-chat-44k-ja", "col": "conversations"},
    {"id": "fujiki/japanese_alpaca_data", "col": "output"},
    {"id": "shi3z/Japanese_wikipedia_conversation_100K", "col": "conversations"},
    {"id": "FreedomIntelligence/alpaca-gpt4-japanese", "col": "conversations"},
]


def run_training(req: TrainRequest):
    """Run training in background thread."""
    global model, tokenizer, config, device, training_status
    from datasets import load_dataset

    training_status = {"running": True, "log": [], "message": "Loading datasets..."}

    try:
        # Load datasets
        all_texts = []
        datasets_to_use = DEFAULT_DATASETS
        if req.dataset_ids:
            datasets_to_use = [{"id": did, "col": "text"} for did in req.dataset_ids]

        for ds_info in datasets_to_use:
            try:
                training_status["message"] = f"Loading {ds_info['id']}..."
                ds = load_dataset(ds_info["id"], split="train")
                texts = extract_texts(ds, ds_info["col"], req.max_samples_per_dataset)
                all_texts.extend(texts)
                training_status["log"].append(f"Loaded {ds_info['id']}: {len(texts)} texts")
            except Exception as e:
                training_status["log"].append(f"Error loading {ds_info['id']}: {e}")

        # Also load cc100-ja
        try:
            training_status["message"] = "Loading cc100-ja..."
            ds_cc = load_dataset("range3/cc100-ja", split="train", streaming=True)
            cc_texts = []
            for i, row in enumerate(ds_cc):
                if i >= req.max_samples_per_dataset:
                    break
                text = row.get("text", "").strip()
                if len(text) > 10:
                    cc_texts.append(text)
            all_texts.extend(cc_texts)
            training_status["log"].append(f"Loaded cc100-ja: {len(cc_texts)} texts")
        except Exception as e:
            training_status["log"].append(f"Error loading cc100-ja: {e}")

        training_status["message"] = "Tokenizing..."
        max_seq_len = config["max_seq_len"]
        sequences = tokenize_texts(all_texts, tokenizer, max_seq_len)
        training_status["log"].append(f"Total: {len(all_texts)} texts -> {len(sequences)} sequences")

        # Training
        steps_per_epoch = len(sequences) // req.batch_size
        total_steps = (steps_per_epoch * req.epochs) // req.grad_accum_steps
        optimizer = torch.optim.AdamW(model.parameters(), lr=req.lr, weight_decay=0.01)

        model.train()
        global_step = 0

        for epoch in range(req.epochs):
            random.shuffle(sequences)
            total_loss = 0
            n_batches = 0
            optimizer.zero_grad()

            training_status["message"] = f"Training epoch {epoch+1}/{req.epochs}..."

            for i in range(0, len(sequences), req.batch_size):
                batch_seqs = sequences[i:i + req.batch_size]
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
                loss = loss / req.grad_accum_steps
                loss.backward()

                total_loss += loss.item() * req.grad_accum_steps
                n_batches += 1

                if n_batches % req.grad_accum_steps == 0:
                    lr = get_lr(global_step, total_steps, req.warmup_steps, req.lr)
                    for pg in optimizer.param_groups:
                        pg['lr'] = lr
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    optimizer.zero_grad()
                    global_step += 1

            if n_batches % req.grad_accum_steps != 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()
                global_step += 1

            avg_loss = total_loss / max(n_batches, 1)
            msg = f"Epoch {epoch+1}/{req.epochs} | Loss: {avg_loss:.4f}"
            training_status["log"].append(msg)
            training_status["message"] = msg

        # Save checkpoint
        model.eval()
        checkpoint = torch.load(CKPT_PATH, map_location="cpu")
        prev_log = checkpoint.get("training_log", [])
        new_log_entries = [{"epoch": len(prev_log) + i + 1, "loss": float(l.split("Loss: ")[1])}
                          for i, l in enumerate(training_status["log"]) if "Loss:" in l]

        new_checkpoint = {
            "model_state": model.state_dict(),
            "config": config,
            "training_log": prev_log + new_log_entries,
            "trained_at": datetime.now(timezone.utc).isoformat(),
            "datasets": list(set(
                checkpoint.get("datasets", []) +
                [d["id"] for d in datasets_to_use] +
                ["range3/cc100-ja"]
            )),
        }
        torch.save(new_checkpoint, CKPT_PATH)
        training_status["log"].append(f"Checkpoint saved: {CKPT_PATH}")
        training_status["message"] = "Training complete!"
        training_status["running"] = False

    except Exception as e:
        import traceback
        training_status["running"] = False
        training_status["message"] = f"Error: {e}"
        training_status["log"].append(traceback.format_exc())
        if model is not None:
            model.eval()


# ========================================
# QA Training
# ========================================

QA_DATASETS_INFO = [
    {"id": "fujiki/japanese_alpaca_data", "format": "alpaca"},
    {"id": "FreedomIntelligence/alpaca-gpt4-japanese", "format": "conversations"},
    {"id": "kunishou/oasst1-chat-44k-ja", "format": "conversations"},
    {"id": "izumi-lab/llm-japanese-dataset", "format": "izumi"},
]

CRAFTED_QA = [
    "質問: 日本の首都はどこですか？\n回答: 日本の首都は東京です。",
    "質問: 富士山の高さはどのくらいですか？\n回答: 富士山の高さは3,776メートルです。日本で最も高い山です。",
    "質問: プログラミングとは何ですか？\n回答: プログラミングとは、コンピュータに実行させる命令を書くことです。Python、Java、C++などのプログラミング言語を使って、ソフトウェアやアプリケーションを作成します。",
    "質問: 人工知能とは何ですか？\n回答: 人工知能（AI）は、人間の知能を模倣するコンピュータシステムです。機械学習、深層学習、自然言語処理などの技術を含みます。",
    "質問: 量子コンピュータとは何ですか？\n回答: 量子コンピュータは、量子力学の原理を利用して計算を行うコンピュータです。量子ビット（キュービット）を使い、従来のコンピュータでは困難な問題を解くことができます。",
    "質問: 機械学習とは何ですか？\n回答: 機械学習は、データからパターンを学習し、予測や判断を行う人工知能の一分野です。教師あり学習、教師なし学習、強化学習などの手法があります。",
    "質問: Pythonとは何ですか？\n回答: Pythonは、読みやすく書きやすい汎用プログラミング言語です。データ分析、AI開発、Web開発など幅広い分野で使われています。",
    "質問: インターネットとは何ですか？\n回答: インターネットは、世界中のコンピュータネットワークを相互に接続した通信網です。Webサイトの閲覧、メール、動画配信などのサービスを支えています。",
    "質問: 太陽系にはいくつの惑星がありますか？\n回答: 太陽系には8つの惑星があります。水星、金星、地球、火星、木星、土星、天王星、海王星です。",
    "質問: 光の速さはどのくらいですか？\n回答: 光の速さは秒速約30万キロメートル（299,792,458 m/s）です。これは宇宙で最も速い速度です。",
    "質問: 日本語の文字体系について教えてください。\n回答: 日本語には、ひらがな、カタカナ、漢字の3種類の文字体系があります。ひらがなは46文字、カタカナも46文字あり、漢字は数千字が日常的に使われています。",
    "質問: 地球の年齢はどのくらいですか？\n回答: 地球の年齢は約46億年です。太陽系の形成とほぼ同時期に誕生しました。",
    "質問: DNAとは何ですか？\n回答: DNAはデオキシリボ核酸の略で、生物の遺伝情報を保持する分子です。二重らせん構造を持ち、アデニン、チミン、グアニン、シトシンの4つの塩基から構成されています。",
    "質問: 民主主義とは何ですか？\n回答: 民主主義は、国民が主権を持ち、選挙などを通じて政治に参加する政治体制です。言論の自由、法の支配、人権の尊重が基本原則です。",
    "質問: 相対性理論とは何ですか？\n回答: 相対性理論はアインシュタインが提唱した物理学の理論です。特殊相対性理論と一般相対性理論があり、時間と空間が相対的であること、重力が時空の歪みであることを示しました。",
    "質問: 北海道の特徴を教えてください。\n回答: 北海道は日本最北の島で、面積は日本最大です。冬は寒く雪が多く、夏は涼しいです。農業、酪農、漁業が盛んで、ラベンダー畑や温泉など観光地も多いです。",
    "質問: 地震はなぜ起きるのですか？\n回答: 地震は、地球の表面を覆うプレートが動くことで起きます。プレート同士がぶつかったり、すれ違ったりする際にエネルギーが蓄積され、それが解放されるときに地震が発生します。",
    "質問: 水は何度で沸騰しますか？\n回答: 水は標準気圧（1気圧）のもとで100度（摂氏）で沸騰します。気圧が低い高山では、沸点はこれより低くなります。",
    "質問: 日本の四季について教えてください。\n回答: 日本には春、夏、秋、冬の四季があります。春は桜が咲き、夏は暑く湿度が高いです。秋は紅葉が美しく、冬は寒く雪が降る地域もあります。四季の変化は日本文化に大きな影響を与えています。",
    "質問: ニューラルネットワークとは何ですか？\n回答: ニューラルネットワークは、人間の脳の神経回路を模倣した機械学習モデルです。入力層、隠れ層、出力層からなり、重みを学習することでパターン認識や予測を行います。深層学習の基盤技術です。",
    "質問: 酸素はどのような元素ですか？\n回答: 酸素は原子番号8の元素で、元素記号はOです。空気中に約21%含まれ、生物の呼吸に不可欠です。また、物質の燃焼にも必要で、水（H2O）の構成元素でもあります。",
    "質問: 東京タワーについて教えてください。\n回答: 東京タワーは1958年に完成した電波塔で、高さは333メートルです。東京都港区芝公園にあり、赤と白の鮮やかな外観が特徴的です。展望台からは東京の景色を一望できます。",
    "質問: 円周率とは何ですか？\n回答: 円周率（π）は、円の直径に対する円周の比率です。約3.14159で、無限に続く無理数です。数学、物理学、工学など多くの分野で重要な定数です。",
    "質問: 織田信長はどんな人物ですか？\n回答: 織田信長は戦国時代の武将で、天下統一を目指しました。鉄砲を活用した長篠の戦いや、楽市楽座などの経済政策で知られます。1582年の本能寺の変で明智光秀に討たれました。",
    "質問: クラウドコンピューティングとは何ですか？\n回答: クラウドコンピューティングは、インターネット経由でサーバー、ストレージ、データベースなどのコンピューティングリソースを利用するサービスです。AWS、Azure、GCPなどが代表的なプロバイダーです。",
]


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


def run_qa_training(req: TrainQARequest):
    """Run QA-format training in background thread."""
    global model, tokenizer, config, device, training_status
    from datasets import load_dataset

    training_status = {"running": True, "log": [], "message": "Loading QA datasets..."}
    min_lr_ratio = 0.1

    try:
        all_qa = []

        # If custom dataset_id is specified, use it exclusively
        if req.dataset_id:
            ds_id = req.dataset_id
            try:
                training_status["message"] = f"Loading {ds_id}..."
                ds = load_dataset(ds_id, split="train", streaming=True)
                count = 0
                for row in ds:
                    if count >= req.max_samples_per_dataset:
                        break
                    # Auto-detect QA format
                    q = row.get("question", row.get("instruction", "")).strip()
                    a = row.get("answer", row.get("output", "")).strip()
                    if q and a and len(q) > 2 and len(a) > 2:
                        all_qa.append(f"質問: {q}\n回答: {a}")
                        count += 1
                training_status["log"].append(f"Loaded {ds_id}: {count} QA samples")
            except Exception as e:
                training_status["log"].append(f"Error loading {ds_id}: {e}")
        else:
            # Default: use built-in QA datasets
            for ds_info in QA_DATASETS_INFO:
                ds_id = ds_info["id"]
                fmt = ds_info["format"]
                max_samples = req.max_samples_per_dataset
                if fmt == "izumi":
                    max_samples = min(1000, max_samples)
                try:
                    training_status["message"] = f"Loading {ds_id}..."
                    ds = load_dataset(ds_id, split="train")
                    n = min(max_samples, len(ds))
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
                    training_status["log"].append(f"Loaded {ds_id}: {count} QA samples")
                except Exception as e:
                    training_status["log"].append(f"Error loading {ds_id}: {e}")

        # Add crafted QA x40
        for _ in range(40):
            all_qa.extend(CRAFTED_QA)
        training_status["log"].append(f"Added {len(CRAFTED_QA) * 40} crafted QA samples")
        training_status["log"].append(f"Total QA texts: {len(all_qa)}")

        # Tokenize
        training_status["message"] = "Tokenizing..."
        max_seq_len = config["max_seq_len"]
        sequences = tokenize_texts(all_qa, tokenizer, max_seq_len)
        training_status["log"].append(f"Training sequences: {len(sequences)}")

        # Training setup
        steps_per_epoch = len(sequences) // req.batch_size
        total_steps = (steps_per_epoch * req.epochs) // req.grad_accum_steps
        optimizer = torch.optim.AdamW(model.parameters(), lr=req.lr, weight_decay=0.01)

        model.train()
        global_step = 0
        best_loss = float('inf')

        for epoch in range(req.epochs):
            random.shuffle(sequences)
            total_loss = 0
            n_batches = 0
            optimizer.zero_grad()
            training_status["message"] = f"QA Training epoch {epoch+1}/{req.epochs}..."

            for i in range(0, len(sequences), req.batch_size):
                batch_seqs = sequences[i:i + req.batch_size]
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
                loss = loss / req.grad_accum_steps
                loss.backward()

                total_loss += loss.item() * req.grad_accum_steps
                n_batches += 1

                if n_batches % req.grad_accum_steps == 0:
                    progress = (global_step - req.warmup_steps) / max(total_steps - req.warmup_steps, 1)
                    if global_step < req.warmup_steps:
                        lr = req.lr * global_step / max(req.warmup_steps, 1)
                    else:
                        cosine_decay = 0.5 * (1 + math.cos(math.pi * progress))
                        lr = req.lr * (min_lr_ratio + (1 - min_lr_ratio) * cosine_decay)
                    for pg in optimizer.param_groups:
                        pg['lr'] = lr
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    optimizer.zero_grad()
                    global_step += 1

            if n_batches % req.grad_accum_steps != 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()
                global_step += 1

            avg_loss = total_loss / max(n_batches, 1)
            msg = f"Epoch {epoch+1}/{req.epochs} | Loss: {avg_loss:.4f}"
            training_status["log"].append(msg)
            training_status["message"] = msg

            extra_ds = [req.dataset_id] if req.dataset_id else []
            if avg_loss < best_loss:
                best_loss = avg_loss
                save_qa_checkpoint(model, config, training_status, epoch + 1, extra_ds)

            # Periodic save every 5 epochs
            if (epoch + 1) % 5 == 0:
                save_qa_checkpoint(model, config, training_status, epoch + 1, extra_ds)

        # Final save
        extra_ds = [req.dataset_id] if req.dataset_id else []
        save_qa_checkpoint(model, config, training_status, req.epochs, extra_ds)
        model.eval()
        training_status["message"] = f"QA Training complete! Best loss: {best_loss:.4f}"
        training_status["running"] = False

    except Exception as e:
        import traceback
        training_status["running"] = False
        training_status["message"] = f"Error: {e}"
        training_status["log"].append(traceback.format_exc())
        if model is not None:
            model.eval()


def save_qa_checkpoint(mdl, cfg, status, epoch_num, extra_datasets=None):
    """Save QA training checkpoint."""
    checkpoint = torch.load(CKPT_PATH, map_location="cpu")
    prev_log = checkpoint.get("training_log", [])
    new_log_entries = [
        {"epoch": len(prev_log) + i + 1, "loss": float(l.split("Loss: ")[1])}
        for i, l in enumerate(status["log"]) if "Loss:" in l
    ]
    ds_list = checkpoint.get("datasets", []) + [d["id"] for d in QA_DATASETS_INFO]
    if extra_datasets:
        ds_list += extra_datasets
    new_checkpoint = {
        "model_state": mdl.state_dict(),
        "config": cfg,
        "training_log": prev_log + new_log_entries,
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "datasets": list(set(ds_list)),
        "qa_training": True,
        "qa_high_epoch": True,
    }
    torch.save(new_checkpoint, CKPT_PATH)
    status["log"].append(f"Checkpoint saved at epoch {epoch_num}")


# ========================================
# API Endpoints
# ========================================

@app.on_event("startup")
async def startup():
    load_model()


@app.get("/")
async def root():
    return {
        "name": "NeuroQuantum API",
        "version": "1.0.0",
        "model": "neuroquantum",
        "config": config,
        "parameters": sum(p.numel() for p in model.parameters()) if model else 0,
    }


@app.post("/inference", response_model=InferenceResponse)
async def inference(req: InferenceRequest):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    if training_status["running"]:
        raise HTTPException(status_code=503, detail="Model is currently training")

    text = generate_text(
        req.prompt,
        max_new_tokens=req.max_new_tokens,
        temperature=req.temperature,
        top_k=req.top_k,
        top_p=req.top_p,
        repetition_penalty=req.repetition_penalty,
    )
    tokens_count = len(tokenizer.encode(text, add_special=False)) if text else 0
    return InferenceResponse(
        prompt=req.prompt,
        generated_text=text,
        tokens_generated=tokens_count,
    )


@app.post("/train", response_model=TrainResponse)
async def train(req: TrainRequest, background_tasks: BackgroundTasks):
    if training_status["running"]:
        raise HTTPException(status_code=409, detail="Training already in progress")

    background_tasks.add_task(run_training, req)
    return TrainResponse(
        status="started",
        message=f"Training started: {req.epochs} epochs, lr={req.lr}, "
                f"effective_batch={req.batch_size * req.grad_accum_steps}",
    )


@app.post("/train/qa", response_model=TrainResponse)
async def train_qa(req: TrainQARequest, background_tasks: BackgroundTasks):
    """QA形式の日本語データで高エポック学習。"""
    if training_status["running"]:
        raise HTTPException(status_code=409, detail="Training already in progress")

    background_tasks.add_task(run_qa_training, req)
    return TrainResponse(
        status="started",
        message=f"QA Training started: {req.epochs} epochs, lr={req.lr}, "
                f"effective_batch={req.batch_size * req.grad_accum_steps}",
    )


@app.get("/train/status", response_model=TrainStatusResponse)
async def train_status():
    return TrainStatusResponse(**training_status)


@app.post("/reload")
async def reload_model():
    """Reload model from latest checkpoint."""
    try:
        load_model()
        return {"status": "reloaded", "message": "Model reloaded from checkpoint"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
