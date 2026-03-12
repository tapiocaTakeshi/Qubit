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


class TrainMarkdownRequest(BaseModel):
    epochs: int = 25
    lr: float = 3e-5
    batch_size: int = 4
    grad_accum_steps: int = 4
    warmup_steps: int = 20


class TrainSplitRequest(BaseModel):
    mode: str = "qa"  # "qa" or "general"
    dataset_ids: Optional[List[str]] = None  # カスタムデータセットIDリスト（指定時はデフォルトの代わりに使用）
    num_chunks: int = 4
    chunk_index: Optional[int] = None  # None = all chunks, int = specific chunk
    epochs_per_chunk: int = 5
    lr: float = 3e-5
    batch_size: int = 4
    grad_accum_steps: int = 4
    warmup_steps: int = 20
    max_samples_per_dataset: int = 2000
    crafted_repeat: int = 20
    resume: bool = False


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
    """Tokenize texts into training sequences.

    Each chunk always starts with BOS and ends with EOS so the model
    learns proper sentence boundaries and avoids mid-sentence generation.
    """
    sequences = []
    for t in texts:
        content_ids = tok.encode(t, add_special=False)
        max_content = max_seq_len - 2  # Reserve slots for BOS and EOS
        if max_content <= 0:
            continue
        if len(content_ids) <= max_content:
            if len(content_ids) >= 2:
                seq = [tok.bos_id] + content_ids + [tok.eos_id]
                sequences.append(seq)
        else:
            stride = max(max_content // 2, 1)
            for start in range(0, len(content_ids) - max_content + 1, stride):
                chunk = content_ids[start:start + max_content]
                seq = [tok.bos_id] + chunk + [tok.eos_id]
                sequences.append(seq)
            remaining = content_ids[-max_content:]
            tail_seq = [tok.bos_id] + remaining + [tok.eos_id]
            if tail_seq != sequences[-1]:
                sequences.append(tail_seq)
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
                    a = row.get("answer", row.get("output", row.get("response", ""))).strip()
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
# Markdown Training
# ========================================

MARKDOWN_QA = [
    "質問: 日本の首都はどこですか？\n回答: ## 日本の首都\n\n日本の首都は**東京**です。\n\n### 基本情報\n- **人口**: 約1,400万人\n- **面積**: 約2,194 km²\n- **特徴**: 世界最大級の都市圏",
    "質問: プログラミング言語の種類を教えてください。\n回答: ## プログラミング言語の種類\n\n主要なプログラミング言語を以下にまとめます。\n\n### 人気言語ランキング\n| 順位 | 言語 | 用途 |\n|------|------|------|\n| 1 | Python | AI・データ分析 |\n| 2 | JavaScript | Web開発 |\n| 3 | Java | エンタープライズ |\n| 4 | C++ | システム開発 |\n\n### 特徴\n- **Python**: 読みやすく初心者向け\n- **JavaScript**: ブラウザで動作\n- **Java**: 大規模システムに適する",
    "質問: 機械学習の種類を教えてください。\n回答: ## 機械学習の種類\n\n機械学習は大きく**3つ**に分類されます。\n\n### 1. 教師あり学習\n- 正解ラベル付きデータで学習\n- 例: 画像分類、スパム検出\n\n### 2. 教師なし学習\n- ラベルなしデータからパターンを発見\n- 例: クラスタリング、次元削減\n\n### 3. 強化学習\n- 試行錯誤で最適な行動を学習\n- 例: ゲームAI、ロボット制御\n\n> **ポイント**: 目的に応じて適切な手法を選択することが重要です。",
    "質問: Pythonの基本的な文法を教えてください。\n回答: ## Pythonの基本文法\n\n### 変数と型\n```python\nname = \"太郎\"  # 文字列\nage = 25        # 整数\npi = 3.14       # 浮動小数点\n```\n\n### 条件分岐\n```python\nif age >= 18:\n    print(\"成人です\")\nelse:\n    print(\"未成年です\")\n```\n\n### リスト操作\n```python\nfruits = [\"りんご\", \"みかん\", \"バナナ\"]\nfor fruit in fruits:\n    print(fruit)\n```\n\n> **Note**: Pythonはインデントでブロックを表現します。",
    "質問: 太陽系の惑星について教えてください。\n回答: ## 太陽系の惑星\n\n太陽系には**8つの惑星**があります。\n\n### 地球型惑星（岩石惑星）\n1. **水星** - 最も太陽に近い\n2. **金星** - 最も高温\n3. **地球** - 生命が存在\n4. **火星** - 赤い惑星\n\n### 木星型惑星（巨大ガス惑星）\n5. **木星** - 最大の惑星\n6. **土星** - 美しい環を持つ\n7. **天王星** - 横倒しで自転\n8. **海王星** - 最も遠い惑星\n\n| 惑星 | 直径(km) | 太陽からの距離(億km) |\n|------|----------|--------------------|\n| 水星 | 4,879 | 0.58 |\n| 地球 | 12,742 | 1.50 |\n| 木星 | 139,820 | 7.78 |",
    "質問: ニューラルネットワークとは何ですか？\n回答: ## ニューラルネットワーク\n\n人間の脳の**神経回路**を模倣した機械学習モデルです。\n\n### 基本構造\n1. **入力層**: データを受け取る\n2. **隠れ層**: 特徴を抽出・変換\n3. **出力層**: 結果を出力\n\n### 主要なアーキテクチャ\n- **CNN**: 画像認識に特化\n- **RNN/LSTM**: 時系列データ処理\n- **Transformer**: 自然言語処理の主流\n- **GAN**: 画像生成\n\n### 活性化関数\n```\nReLU: f(x) = max(0, x)\nSigmoid: f(x) = 1 / (1 + e^(-x))\nTanh: f(x) = (e^x - e^(-x)) / (e^x + e^(-x))\n```\n\n> 深層学習（ディープラーニング）は、多層のニューラルネットワークを用いた学習手法です。",
    "質問: 日本の歴史の時代区分を教えてください。\n回答: ## 日本の歴史 - 時代区分\n\n### 古代\n- **縄文時代**: 約1万年前〜\n- **弥生時代**: 紀元前3世紀〜\n- **古墳時代**: 3世紀〜\n- **飛鳥時代**: 592年〜\n- **奈良時代**: 710年〜\n- **平安時代**: 794年〜\n\n### 中世\n- **鎌倉時代**: 1185年〜\n- **室町時代**: 1336年〜\n- **戦国時代**: 1467年〜\n\n### 近世・近代\n- **安土桃山時代**: 1573年〜\n- **江戸時代**: 1603年〜\n- **明治時代**: 1868年〜\n- **大正時代**: 1912年〜\n- **昭和時代**: 1926年〜\n\n### 現代\n- **平成**: 1989年〜\n- **令和**: 2019年〜",
    "質問: HTTPステータスコードについて教えてください。\n回答: ## HTTPステータスコード\n\nHTTPレスポンスの状態を示す3桁の数字コードです。\n\n### 2xx 成功\n| コード | 意味 |\n|--------|------|\n| `200` | OK - 成功 |\n| `201` | Created - 作成成功 |\n| `204` | No Content - 内容なし |\n\n### 3xx リダイレクト\n| コード | 意味 |\n|--------|------|\n| `301` | 恒久的な移動 |\n| `302` | 一時的な移動 |\n\n### 4xx クライアントエラー\n| コード | 意味 |\n|--------|------|\n| `400` | Bad Request |\n| `401` | Unauthorized |\n| `403` | Forbidden |\n| `404` | Not Found |\n\n### 5xx サーバーエラー\n| コード | 意味 |\n|--------|------|\n| `500` | Internal Server Error |\n| `502` | Bad Gateway |\n| `503` | Service Unavailable |",
    "質問: Gitの基本コマンドを教えてください。\n回答: ## Gitの基本コマンド\n\n### 初期設定\n```bash\ngit init              # リポジトリ初期化\ngit clone <url>       # リポジトリをクローン\n```\n\n### 基本操作\n```bash\ngit add <file>        # ステージング\ngit commit -m \"msg\"   # コミット\ngit push origin main  # プッシュ\ngit pull origin main  # プル\n```\n\n### ブランチ操作\n```bash\ngit branch <name>     # ブランチ作成\ngit checkout <name>   # ブランチ切替\ngit merge <name>      # マージ\n```\n\n### 確認コマンド\n- `git status` - 状態確認\n- `git log` - 履歴表示\n- `git diff` - 差分確認\n\n> **ベストプラクティス**: こまめにコミットし、わかりやすいコミットメッセージを書きましょう。",
    "質問: データベースの種類を教えてください。\n回答: ## データベースの種類\n\n### リレーショナルDB（RDB）\nテーブル形式でデータを管理します。\n\n| DB | 特徴 |\n|-----|------|\n| **MySQL** | オープンソース、Web開発で人気 |\n| **PostgreSQL** | 高機能、拡張性が高い |\n| **SQLite** | 軽量、組み込み向け |\n\n### NoSQL\n- **MongoDB**: ドキュメント型、JSON形式\n- **Redis**: キーバリュー型、高速キャッシュ\n- **Cassandra**: カラム型、大規模分散処理\n- **Neo4j**: グラフ型、関係性の表現\n\n### 選択基準\n1. データの構造（構造化 vs 非構造化）\n2. スケーラビリティの要件\n3. 一貫性 vs 可用性のトレードオフ",
    "質問: APIとは何ですか？\n回答: ## API（Application Programming Interface）\n\nソフトウェア同士が通信するための**インターフェース**です。\n\n### REST API\n最も一般的なAPI設計スタイルです。\n\n#### HTTPメソッド\n| メソッド | 操作 | 例 |\n|----------|------|-----|\n| `GET` | 取得 | ユーザー一覧取得 |\n| `POST` | 作成 | 新規ユーザー登録 |\n| `PUT` | 更新 | ユーザー情報更新 |\n| `DELETE` | 削除 | ユーザー削除 |\n\n### レスポンス例\n```json\n{\n  \"status\": 200,\n  \"data\": {\n    \"id\": 1,\n    \"name\": \"太郎\"\n  }\n}\n```\n\n> **ポイント**: RESTful APIは*ステートレス*で、各リクエストが独立しています。",
    "質問: 量子コンピュータの仕組みを教えてください。\n回答: ## 量子コンピュータ\n\n量子力学の原理を利用した**次世代のコンピュータ**です。\n\n### 古典コンピュータとの違い\n| 項目 | 古典 | 量子 |\n|------|------|------|\n| 基本単位 | ビット（0/1） | 量子ビット（重ね合わせ） |\n| 処理方式 | 逐次処理 | 並列処理 |\n| 得意分野 | 汎用計算 | 最適化・暗号 |\n\n### 主要な原理\n1. **重ね合わせ**: 0と1を同時に表現\n2. **エンタングルメント**: 量子もつれによる相関\n3. **干渉**: 正解の確率を増幅\n\n### 応用分野\n- 創薬シミュレーション\n- 金融リスク分析\n- 暗号解読\n- 材料科学",
    "質問: CSSの基本を教えてください。\n回答: ## CSSの基本\n\nWebページの**見た目を装飾**するスタイルシート言語です。\n\n### セレクタと宣言\n```css\n/* 要素セレクタ */\nh1 {\n  color: #333;\n  font-size: 24px;\n}\n\n/* クラスセレクタ */\n.container {\n  max-width: 1200px;\n  margin: 0 auto;\n  padding: 20px;\n}\n```\n\n### Flexbox レイアウト\n```css\n.flex-container {\n  display: flex;\n  justify-content: center;\n  align-items: center;\n  gap: 16px;\n}\n```\n\n### よく使うプロパティ\n- `margin` / `padding`: 余白\n- `color` / `background`: 色\n- `font-size` / `font-weight`: 文字装飾\n- `border` / `border-radius`: 枠線\n\n> **Tips**: レスポンシブデザインには`@media`クエリを使います。",
    "質問: 富士山について教えてください。\n回答: ## 富士山\n\n日本の**最高峰**であり、*世界文化遺産*に登録されています。\n\n### 基本データ\n- **標高**: 3,776m\n- **所在地**: 静岡県・山梨県\n- **種類**: 成層火山（活火山）\n- **最終噴火**: 1707年（宝永噴火）\n\n### 登山シーズン\n| 月 | 状況 |\n|----|------|\n| 7月 | 開山（山梨県側） |\n| 8月 | ベストシーズン |\n| 9月 | 閉山 |\n\n### 登山ルート\n1. **吉田ルート**: 最も人気、初心者向け\n2. **富士宮ルート**: 最短距離\n3. **須走ルート**: 砂走りが特徴\n4. **御殿場ルート**: 最長、上級者向け\n\n> 富士山は日本人の心の象徴であり、古来より信仰の対象とされてきました。",
    "質問: セキュリティ対策について教えてください。\n回答: ## Webセキュリティ対策\n\n### 主要な脅威と対策\n\n#### 1. XSS（クロスサイトスクリプティング）\n- **対策**: 出力時のエスケープ処理\n```html\n<!-- 危険 -->\n<p>{{user_input}}</p>\n<!-- 安全 -->\n<p>{{user_input | escape}}</p>\n```\n\n#### 2. SQLインジェクション\n- **対策**: プリペアドステートメントの使用\n```python\n# 危険\ncursor.execute(f\"SELECT * FROM users WHERE id={id}\")\n# 安全\ncursor.execute(\"SELECT * FROM users WHERE id=?\", (id,))\n```\n\n#### 3. CSRF\n- **対策**: CSRFトークンの使用\n\n### セキュリティチェックリスト\n- [ ] HTTPS通信の強制\n- [ ] 入力値バリデーション\n- [ ] パスワードのハッシュ化\n- [ ] 適切な認証・認可\n- [ ] セキュリティヘッダーの設定",
    "質問: Docker とは何ですか？\n回答: ## Docker\n\nアプリケーションを**コンテナ**として仮想化するプラットフォームです。\n\n### 基本コマンド\n```bash\n# イメージ操作\ndocker pull nginx           # イメージ取得\ndocker build -t myapp .     # イメージ作成\n\n# コンテナ操作\ndocker run -d -p 80:80 nginx  # 起動\ndocker ps                     # 一覧\ndocker stop <id>              # 停止\n```\n\n### Dockerfile例\n```dockerfile\nFROM python:3.11-slim\nWORKDIR /app\nCOPY requirements.txt .\nRUN pip install -r requirements.txt\nCOPY . .\nCMD [\"python\", \"app.py\"]\n```\n\n### VMとの比較\n| 項目 | Docker | VM |\n|------|--------|----|\n| 起動速度 | 秒単位 | 分単位 |\n| リソース | 軽量 | 重い |\n| 分離レベル | プロセス | 完全 |",
    "質問: 天気について説明してください。\n回答: ## 天気のメカニズム\n\n### 天気を決める要素\n1. **気温**: 大気の温度\n2. **気圧**: 空気の重さによる圧力\n3. **湿度**: 空気中の水蒸気量\n4. **風**: 気圧差による空気の移動\n\n### 主な天気現象\n- **晴れ**: 高気圧に覆われた状態\n- **雨**: 水蒸気が凝結して降下\n- **雪**: 水蒸気が氷晶として降下\n- **台風**: 熱帯低気圧が発達\n\n### 天気図の記号\n| 記号 | 意味 |\n|------|------|\n| ○ | 快晴 |\n| ◎ | 曇り |\n| ● | 雨 |\n\n> 天気予報は**数値予報モデル**によるスーパーコンピュータのシミュレーションで行われます。",
    "質問: 数学の基本公式を教えてください。\n回答: ## 数学の基本公式\n\n### 代数\n- 二次方程式の解: `x = (-b ± √(b²-4ac)) / 2a`\n- 因数分解: `a²-b² = (a+b)(a-b)`\n\n### 幾何学\n| 図形 | 面積公式 |\n|------|----------|\n| 円 | `S = πr²` |\n| 三角形 | `S = ½ × 底辺 × 高さ` |\n| 長方形 | `S = 縦 × 横` |\n\n### 三角関数\n```\nsin²θ + cos²θ = 1\ntan θ = sin θ / cos θ\n```\n\n### 微積分\n1. **微分**: `f'(x) = lim(h→0) [f(x+h) - f(x)] / h`\n2. **積分**: `∫f(x)dx = F(x) + C`\n\n> **重要**: これらの公式は*物理学*や*工学*の基礎となります。",
    "質問: 健康的な食事について教えてください。\n回答: ## 健康的な食事ガイド\n\n### 五大栄養素\n1. **炭水化物**: エネルギー源\n2. **タンパク質**: 体を作る\n3. **脂質**: エネルギー貯蔵\n4. **ビタミン**: 体の調子を整える\n5. **ミネラル**: 骨・血液の材料\n\n### 1日の推奨摂取量\n| 栄養素 | 成人男性 | 成人女性 |\n|--------|----------|----------|\n| カロリー | 2,200kcal | 1,800kcal |\n| タンパク質 | 65g | 50g |\n| 食物繊維 | 21g | 18g |\n\n### バランスの良い食事のポイント\n- 主食・主菜・副菜を揃える\n- **野菜**は1日350g以上\n- *塩分*は控えめに（1日6g未満）\n- 水分を十分に摂取\n\n> 「医食同源」- 食事は最良の薬です。",
    "質問: 人工知能の歴史を教えてください。\n回答: ## 人工知能（AI）の歴史\n\n### 年表\n| 年代 | 出来事 |\n|------|--------|\n| 1950 | チューリングテストの提唱 |\n| 1956 | 「人工知能」という用語の誕生 |\n| 1960s | 第1次AIブーム（探索・推論） |\n| 1980s | 第2次AIブーム（エキスパートシステム） |\n| 2012 | 深層学習の躍進（AlexNet） |\n| 2022 | 生成AI（ChatGPT）の登場 |\n\n### AIの3つの波\n1. **第1次ブーム**: ルールベース\n   - 限定的な問題解決\n2. **第2次ブーム**: 知識ベース\n   - エキスパートシステムの活用\n3. **第3次ブーム**: 機械学習・深層学習\n   - ビッグデータと計算力の向上\n\n> 現在は**第3次AIブーム**の最中であり、*生成AI*が社会に大きな影響を与えています。",
    "質問: 環境問題について教えてください。\n回答: ## 地球の環境問題\n\n### 主要な環境問題\n\n#### 1. 地球温暖化\n- **原因**: CO2などの温室効果ガスの増加\n- **影響**: 海面上昇、異常気象\n- **対策**: 再生可能エネルギーの推進\n\n#### 2. 生物多様性の喪失\n- 毎年約4万種が絶滅の危機\n- 森林破壊と生息地の減少\n\n#### 3. 海洋プラスチック汚染\n- 年間約800万トンが海に流入\n\n### SDGs関連目標\n| 目標 | 内容 |\n|------|------|\n| 7 | エネルギーをみんなに |\n| 13 | 気候変動に具体的な対策を |\n| 14 | 海の豊かさを守ろう |\n| 15 | 陸の豊かさも守ろう |\n\n### 私たちにできること\n- [ ] マイバッグ・マイボトルを使う\n- [ ] 省エネを心がける\n- [ ] フードロスを減らす\n- [ ] 公共交通機関を利用する",
]


def run_markdown_training(req: TrainMarkdownRequest):
    """Run markdown format training in background thread."""
    global model, tokenizer, config, device, training_status
    training_status = {"running": True, "log": [], "message": "Preparing markdown training data..."}
    min_lr_ratio = 0.1

    try:
        # Build markdown training corpus: repeat to reinforce the pattern
        all_texts = []
        for _ in range(80):
            all_texts.extend(MARKDOWN_QA)
        training_status["log"].append(f"Markdown QA samples: {len(MARKDOWN_QA)} x 80 = {len(all_texts)}")

        # Tokenize
        training_status["message"] = "Tokenizing..."
        max_seq_len = config["max_seq_len"]
        sequences = tokenize_texts(all_texts, tokenizer, max_seq_len)
        training_status["log"].append(f"Training sequences: {len(sequences)}")

        # Training
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
            training_status["message"] = f"Markdown Training epoch {epoch+1}/{req.epochs}..."

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
                    if global_step < req.warmup_steps:
                        lr = req.lr * global_step / max(req.warmup_steps, 1)
                    else:
                        progress = (global_step - req.warmup_steps) / max(total_steps - req.warmup_steps, 1)
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

            if avg_loss < best_loss:
                best_loss = avg_loss
                save_qa_checkpoint(model, config, training_status, epoch + 1, ["markdown_format"])

            if (epoch + 1) % 5 == 0:
                save_qa_checkpoint(model, config, training_status, epoch + 1, ["markdown_format"])

        save_qa_checkpoint(model, config, training_status, req.epochs, ["markdown_format"])
        model.eval()
        training_status["message"] = f"Markdown Training complete! Best loss: {best_loss:.4f}"
        training_status["running"] = False

    except Exception as e:
        import traceback
        training_status["running"] = False
        training_status["message"] = f"Error: {e}"
        training_status["log"].append(traceback.format_exc())
        if model is not None:
            model.eval()


# ========================================
# Split Training
# ========================================

SPLIT_STATE_PATH = os.path.join(os.path.dirname(__file__), "split_training_state.json")

GENERAL_DATASETS_INFO = [
    {"id": "izumi-lab/llm-japanese-dataset", "col": "output"},
    {"id": "kunishou/oasst1-chat-44k-ja", "col": "conversations"},
    {"id": "fujiki/japanese_alpaca_data", "col": "output"},
    {"id": "shi3z/Japanese_wikipedia_conversation_100K", "col": "conversations"},
    {"id": "FreedomIntelligence/alpaca-gpt4-japanese", "col": "conversations"},
]


def _load_all_qa_texts(max_samples):
    """Load all QA texts from all datasets."""
    from datasets import load_dataset as _load_ds
    all_qa = []
    for ds_info in QA_DATASETS_INFO:
        ds_id = ds_info["id"]
        fmt = ds_info["format"]
        ms = min(1000, max_samples) if fmt == "izumi" else max_samples
        try:
            training_status["message"] = f"Loading {ds_id}..."
            ds = _load_ds(ds_id, split="train", trust_remote_code=True)
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
            training_status["log"].append(f"Loaded {ds_id}: {count} QA samples")
        except Exception as e:
            training_status["log"].append(f"Error loading {ds_id}: {e}")
    return all_qa


def _load_all_general_texts(max_samples):
    """Load all general texts from all datasets."""
    from datasets import load_dataset as _load_ds
    all_texts = []
    for ds_info in GENERAL_DATASETS_INFO:
        ds_id = ds_info["id"]
        col = ds_info["col"]
        try:
            training_status["message"] = f"Loading {ds_id}..."
            ds = _load_ds(ds_id, split="train", trust_remote_code=True)
            texts = extract_texts(ds, col, max_samples)
            all_texts.extend(texts)
            training_status["log"].append(f"Loaded {ds_id}: {len(texts)} texts")
        except Exception as e:
            training_status["log"].append(f"Error loading {ds_id}: {e}")
    return all_texts


def _load_custom_datasets(dataset_ids, max_samples, mode):
    """Load custom datasets by ID. Auto-detects format."""
    from datasets import load_dataset as _load_ds
    all_texts = []
    for ds_id in dataset_ids:
        try:
            training_status["message"] = f"Loading {ds_id}..."
            # まずstreaming=Trueで試す（大規模データセット対応）
            try:
                ds = _load_ds(ds_id, split="train", trust_remote_code=True)
                is_streaming = False
            except Exception:
                ds = _load_ds(ds_id, split="train", streaming=True, trust_remote_code=True)
                is_streaming = True

            count = 0
            iterator = ds if is_streaming else ds.select(range(min(max_samples, len(ds))))
            for row in iterator:
                if count >= max_samples:
                    break
                text = None
                if mode == "qa":
                    # QA形式: 質問/回答ペアを自動検出
                    q = (row.get("question") or row.get("instruction") or row.get("input") or "")
                    if isinstance(q, str):
                        q = q.strip()
                    else:
                        q = ""
                    a = (row.get("answer") or row.get("output") or row.get("response") or "")
                    if isinstance(a, str):
                        a = a.strip()
                    else:
                        a = ""
                    if q and a and len(q) > 2 and len(a) > 2:
                        text = f"質問: {q}\n回答: {a}"
                    elif not q and a and len(a) > 10:
                        text = f"回答: {a}"
                    # conversations形式も試す
                    if not text:
                        convs = row.get("conversations", [])
                        if isinstance(convs, list) and len(convs) >= 2:
                            text = format_qa_conversations(row)
                    # alpaca形式も試す
                    if not text and row.get("instruction"):
                        text = format_qa_alpaca(row)
                else:
                    # 一般テキスト: 利用可能なテキストフィールドを自動検出
                    for col in ["text", "content", "output", "sentence", "document"]:
                        val = row.get(col)
                        if isinstance(val, str) and len(val.strip()) > 10:
                            text = val.strip()
                            break
                    if not text:
                        convs = row.get("conversations", [])
                        if isinstance(convs, list) and convs:
                            parts = []
                            for turn in convs:
                                if isinstance(turn, dict):
                                    parts.append(turn.get("value", turn.get("content", "")))
                                elif isinstance(turn, str):
                                    parts.append(turn)
                            combined = "\n".join(parts)
                            if len(combined.strip()) > 10:
                                text = combined.strip()
                if text and len(text) > 10:
                    all_texts.append(text)
                    count += 1
            training_status["log"].append(f"Loaded {ds_id}: {count} texts")
        except Exception as e:
            training_status["log"].append(f"Error loading {ds_id}: {e}")
    return all_texts



def _split_into_chunks(data, num_chunks):
    """Split data into roughly equal chunks."""
    random.shuffle(data)
    chunk_size = math.ceil(len(data) / num_chunks)
    chunks = []
    for i in range(0, len(data), chunk_size):
        chunk = data[i:i + chunk_size]
        if chunk:
            chunks.append(chunk)
    return chunks


def _save_split_state(state):
    with open(SPLIT_STATE_PATH, "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=2)


def _load_split_state():
    if os.path.exists(SPLIT_STATE_PATH):
        with open(SPLIT_STATE_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return None


def run_split_training(req: TrainSplitRequest):
    """Run split dataset training in background thread."""
    global model, tokenizer, config, device, training_status
    from datasets import load_dataset as _load_ds

    training_status = {"running": True, "log": [], "message": "Loading datasets for split training..."}
    min_lr_ratio = 0.1

    try:
        # Load data
        if req.dataset_ids:
            # カスタムデータセットIDが指定された場合
            all_texts = _load_custom_datasets(req.dataset_ids, req.max_samples_per_dataset, req.mode)
            if req.mode == "qa" and req.crafted_repeat > 0:
                for _ in range(req.crafted_repeat):
                    all_texts.extend(CRAFTED_QA)
                training_status["log"].append(f"Added {len(CRAFTED_QA) * req.crafted_repeat} crafted QA samples")
            dataset_ids = req.dataset_ids
        elif req.mode == "qa":
            all_texts = _load_all_qa_texts(req.max_samples_per_dataset)
            for _ in range(req.crafted_repeat):
                all_texts.extend(CRAFTED_QA)
            training_status["log"].append(f"Added {len(CRAFTED_QA) * req.crafted_repeat} crafted QA samples")
            dataset_ids = [d["id"] for d in QA_DATASETS_INFO]
        else:
            all_texts = _load_all_general_texts(req.max_samples_per_dataset)
            dataset_ids = [d["id"] for d in GENERAL_DATASETS_INFO]

        training_status["log"].append(f"Total texts: {len(all_texts)}")

        if len(all_texts) == 0:
            training_status["message"] = "Error: No texts loaded"
            training_status["running"] = False
            return

        # Split into chunks
        chunks = _split_into_chunks(all_texts, req.num_chunks)
        actual_num_chunks = len(chunks)
        for i, chunk in enumerate(chunks):
            training_status["log"].append(f"Chunk {i}: {len(chunk)} texts")

        # Determine which chunks to train
        if req.chunk_index is not None:
            if req.chunk_index >= actual_num_chunks:
                training_status["message"] = f"Error: chunk_index {req.chunk_index} >= {actual_num_chunks}"
                training_status["running"] = False
                return
            chunk_indices = [req.chunk_index]
        elif req.resume:
            state = _load_split_state()
            if state and state.get("mode") == req.mode:
                start_idx = state.get("last_completed_chunk", -1) + 1
                if start_idx >= actual_num_chunks:
                    training_status["message"] = "All chunks already completed!"
                    training_status["running"] = False
                    return
                chunk_indices = list(range(start_idx, actual_num_chunks))
            else:
                chunk_indices = list(range(actual_num_chunks))
        else:
            chunk_indices = list(range(actual_num_chunks))

        max_seq_len = config["max_seq_len"]

        # Train on each chunk
        for chunk_idx in chunk_indices:
            chunk_texts = chunks[chunk_idx]
            training_status["message"] = f"Chunk {chunk_idx+1}/{actual_num_chunks}: Tokenizing..."

            sequences = tokenize_texts(chunk_texts, tokenizer, max_seq_len)
            if len(sequences) == 0:
                training_status["log"].append(f"Chunk {chunk_idx}: No sequences, skipping")
                continue

            training_status["log"].append(f"Chunk {chunk_idx}: {len(sequences)} sequences")

            # Training setup
            steps_per_epoch = len(sequences) // req.batch_size
            total_steps = (steps_per_epoch * req.epochs_per_chunk) // req.grad_accum_steps
            optimizer = torch.optim.AdamW(model.parameters(), lr=req.lr, weight_decay=0.01)

            model.train()
            global_step = 0
            best_loss = float('inf')

            for epoch in range(req.epochs_per_chunk):
                random.shuffle(sequences)
                total_loss = 0
                n_batches = 0
                optimizer.zero_grad()
                training_status["message"] = (
                    f"Chunk {chunk_idx+1}/{actual_num_chunks} | "
                    f"Epoch {epoch+1}/{req.epochs_per_chunk}..."
                )

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
                        if global_step < req.warmup_steps:
                            lr = req.lr * global_step / max(req.warmup_steps, 1)
                        else:
                            progress = (global_step - req.warmup_steps) / max(total_steps - req.warmup_steps, 1)
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
                msg = (f"Chunk {chunk_idx+1}/{actual_num_chunks} | "
                       f"Epoch {epoch+1}/{req.epochs_per_chunk} | Loss: {avg_loss:.4f}")
                training_status["log"].append(msg)
                training_status["message"] = msg

                if avg_loss < best_loss:
                    best_loss = avg_loss

            # Save checkpoint after each chunk
            extra_ds = dataset_ids
            save_qa_checkpoint(model, config, training_status, chunk_idx + 1, extra_ds)
            training_status["log"].append(f"Chunk {chunk_idx+1}/{actual_num_chunks} complete, best loss: {best_loss:.4f}")

            # Save split state
            _save_split_state({
                "mode": req.mode,
                "num_chunks": actual_num_chunks,
                "last_completed_chunk": chunk_idx,
                "best_loss": best_loss,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            })

        model.eval()
        training_status["message"] = f"Split training complete! {len(chunk_indices)} chunks trained"
        training_status["running"] = False

    except Exception as e:
        import traceback
        training_status["running"] = False
        training_status["message"] = f"Error: {e}"
        training_status["log"].append(traceback.format_exc())
        if model is not None:
            model.eval()


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


@app.post("/train/markdown", response_model=TrainResponse)
async def train_markdown(req: TrainMarkdownRequest, background_tasks: BackgroundTasks):
    """マークダウン形式出力の学習。"""
    if training_status["running"]:
        raise HTTPException(status_code=409, detail="Training already in progress")

    background_tasks.add_task(run_markdown_training, req)
    return TrainResponse(
        status="started",
        message=f"Markdown Training started: {req.epochs} epochs, lr={req.lr}, "
                f"effective_batch={req.batch_size * req.grad_accum_steps}",
    )


@app.post("/train/split", response_model=TrainResponse)
async def train_split(req: TrainSplitRequest, background_tasks: BackgroundTasks):
    """データセットを分割して学習（タイムアウト回避）。"""
    if training_status["running"]:
        raise HTTPException(status_code=409, detail="Training already in progress")

    chunk_desc = f"chunk {req.chunk_index}" if req.chunk_index is not None else f"all {req.num_chunks} chunks"
    background_tasks.add_task(run_split_training, req)
    return TrainResponse(
        status="started",
        message=f"Split Training started: mode={req.mode}, {chunk_desc}, "
                f"{req.epochs_per_chunk} epochs/chunk, lr={req.lr}",
    )


@app.get("/train/split/status")
async def train_split_status():
    """分割学習の進捗状態を取得。"""
    state = _load_split_state()
    return {
        "training_running": training_status["running"],
        "split_state": state,
        "current_status": training_status["message"],
        "log": training_status["log"],
    }


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
