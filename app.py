"""
neuroQ - QBNN-Transformer on Hugging Face Spaces
データセットID入力 → 自動ロード → 学習 → 推論
CPU最適化版
"""

import os
import json
import math
import time
import gradio as gr
import torch
import torch.nn as nn
import torch.nn.functional as F
from huggingface_hub import HfApi, hf_hub_download
from huggingface_hub.utils import EntryNotFoundError

datasets_lib = None  # 遅延インポート

# ============================================================
# 設定
# ============================================================

REPO_ID = "tapiocaTakeshi/neuroQ"
CKPT_FILE = "qbnn_checkpoint.pt"
HISTORY_FILE = "training_history.json"

MODEL_CONFIG = {
    "vocab_size": 200,
    "embed_dim": 64,
    "num_heads": 2,
    "num_layers": 2,
    "max_seq_len": 32,
    "entangle_strength": 0.12,
}

# ============================================================
# QBNN レイヤー
# ============================================================

class QBNNLayer(nn.Module):
    def __init__(self, dim, lam=0.12):
        super().__init__()
        self.linear = nn.Linear(dim, dim)
        self.J = nn.Parameter(torch.randn(dim, dim) * 0.01)
        self.lam = lam
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        h = self.linear(x)
        delta = torch.einsum('bsd,od->bso', torch.tanh(x), self.J) * torch.tanh(h)
        return self.norm(F.gelu(h + self.lam * delta))


class QBNNTransformer(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        d = cfg["embed_dim"]
        self.embed     = nn.Embedding(cfg["vocab_size"], d)
        self.pos_embed = nn.Embedding(cfg["max_seq_len"], d)
        self.layers = nn.ModuleList([
            nn.ModuleDict({
                "attn":  nn.MultiheadAttention(d, cfg["num_heads"], batch_first=True),
                "qbnn":  QBNNLayer(d, cfg["entangle_strength"]),
                "norm1": nn.LayerNorm(d),
                "norm2": nn.LayerNorm(d),
            }) for _ in range(cfg["num_layers"])
        ])
        self.head = nn.Linear(d, cfg["vocab_size"])
        self.cfg  = cfg

    def forward(self, x):
        B, S = x.shape
        h = self.embed(x) + self.pos_embed(torch.arange(S, device=x.device).unsqueeze(0))
        for layer in self.layers:
            a, _ = layer["attn"](h, h, h)
            h = layer["norm1"](h + a)
            h = layer["norm2"](h + layer["qbnn"](h))
        return self.head(h)

    def generate(self, tokens, max_new=30, temperature=0.8):
        self.eval()
        with torch.no_grad():
            for _ in range(max_new):
                seq    = tokens[:, -self.cfg["max_seq_len"]:]
                logits = self(seq)[:, -1, :] / max(temperature, 1e-5)
                nxt    = torch.multinomial(F.softmax(logits, dim=-1), 1)
                tokens = torch.cat([tokens, nxt], dim=1)
                if nxt.item() == 1:
                    break
        return tokens

# ============================================================
# 文字レベルトークナイザー
# ============================================================

class CharTokenizer:
    def __init__(self):
        chars = list(
            "あいうえおかきくけこさしすせそたちつてとなにぬねのはひふへほまみむめもやゆよらりるれろわをん"
            "アイウエオカキクケコサシスセソタチツテトナニヌネノハヒフヘホマミムメモヤユヨラリルレロワヲン"
            "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
            "、。？！「」（）・ー\n "
        )
        self.stoi = {"<PAD>": 0, "<EOS>": 1, "<UNK>": 2}
        for i, c in enumerate(chars):
            self.stoi[c] = i + 3
        self.itos = {v: k for k, v in self.stoi.items()}
        self.vocab_size = len(self.stoi)

    def encode(self, text, max_len=32):
        return [self.stoi.get(c, 2) for c in str(text)[:max_len]]

    def decode(self, ids):
        return "".join(self.itos.get(i, "?") for i in ids if i not in (0, 1, 2))

# ============================================================
# グローバル状態
# ============================================================

tokenizer = CharTokenizer()
MODEL_CONFIG["vocab_size"] = tokenizer.vocab_size
model      = QBNNTransformer(MODEL_CONFIG)
is_trained = False
stop_flag  = False
training_history = []

# ============================================================
# チェックポイント保存・読込
# ============================================================

def save_checkpoint():
    token = os.environ.get("HF_TOKEN")
    if not token:
        return "⚠️ HF_TOKEN が Secrets に未設定です"
    try:
        torch.save({"model_state": model.state_dict(), "config": MODEL_CONFIG}, CKPT_FILE)
        HfApi().upload_file(
            path_or_fileobj=CKPT_FILE, path_in_repo=CKPT_FILE,
            repo_id=REPO_ID, repo_type="space", token=token,
        )
        os.remove(CKPT_FILE)
        save_training_history(token)
        return "✅ チェックポイントを保存しました"
    except Exception as e:
        return f"❌ 保存エラー: {e}"

def save_training_history(token=None):
    if token is None:
        token = os.environ.get("HF_TOKEN")
    if not token:
        return
    try:
        with open(HISTORY_FILE, "w", encoding="utf-8") as f:
            json.dump(training_history, f, ensure_ascii=False, indent=2)
        HfApi().upload_file(
            path_or_fileobj=HISTORY_FILE, path_in_repo=HISTORY_FILE,
            repo_id=REPO_ID, repo_type="space", token=token,
        )
        os.remove(HISTORY_FILE)
    except Exception:
        pass

def load_training_history():
    global training_history
    try:
        path = hf_hub_download(repo_id=REPO_ID, filename=HISTORY_FILE, repo_type="space")
        with open(path, "r", encoding="utf-8") as f:
            training_history = json.load(f)
    except (EntryNotFoundError, Exception):
        training_history = []

def add_history_entry(dataset_id, epochs, samples, final_loss):
    import datetime
    training_history.append({
        "dataset_id": dataset_id.strip(),
        "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M"),
        "epochs": int(epochs),
        "samples": int(samples),
        "final_loss": round(float(final_loss), 4),
    })

def get_history_display():
    if not training_history:
        return "（学習履歴はまだありません）"
    lines = []
    for i, entry in enumerate(training_history):
        lines.append(
            f"[{i+1}]  {entry['dataset_id']}\n"
            f"     📅 {entry.get('timestamp', '?')}  |  "
            f"🔁 {entry.get('epochs', '?')} epochs  |  "
            f"📊 {entry.get('samples', '?')} samples  |  "
            f"📉 Loss: {entry.get('final_loss', '?')}"
        )
    return "\n\n".join(lines)

def get_history_choices():
    if not training_history:
        return []
    return [
        f"[{i+1}] {entry['dataset_id']} ({entry.get('timestamp', '?')})"
        for i, entry in enumerate(training_history)
    ]

def delete_history_entry(selected):
    if not selected:
        return "⚠️ 削除する項目を選択してください", get_history_display(), gr.update(choices=get_history_choices(), value=None)
    try:
        idx = int(selected.split("]")[0].replace("[", "")) - 1
        if 0 <= idx < len(training_history):
            removed = training_history.pop(idx)
            token = os.environ.get("HF_TOKEN")
            if token:
                save_training_history(token)
            return (
                f"✅ 削除しました: {removed['dataset_id']}",
                get_history_display(),
                gr.update(choices=get_history_choices(), value=None),
            )
        return "❌ 無効なインデックスです", get_history_display(), gr.update(choices=get_history_choices(), value=None)
    except Exception as e:
        return f"❌ エラー: {e}", get_history_display(), gr.update(choices=get_history_choices(), value=None)

def refresh_history():
    load_training_history()
    return get_history_display(), gr.update(choices=get_history_choices(), value=None)

def load_checkpoint():
    global is_trained
    try:
        path = hf_hub_download(repo_id=REPO_ID, filename=CKPT_FILE, repo_type="space")
        model.load_state_dict(torch.load(path, map_location="cpu")["model_state"])
        model.eval()
        is_trained = True
        load_training_history()
        return "✅ チェックポイントを読み込みました"
    except EntryNotFoundError:
        return "⚠️ チェックポイントが見つかりません。先に学習してください。"
    except Exception as e:
        return f"❌ 読込エラー: {e}"

# ============================================================
# データセット読み込み
# ============================================================

TEXT_CANDIDATES = ["text", "content", "sentence", "question", "answer",
                   "context", "abstract", "body", "description", "input", "output"]

def load_dataset_texts(dataset_id, text_column, split, max_samples):
    global datasets_lib
    if datasets_lib is None:
        import datasets as _ds
        datasets_lib = _ds
    try:
        ds = datasets_lib.load_dataset(dataset_id.strip(), split=split, trust_remote_code=False)
    except Exception as e:
        return None, f"❌ データセット読み込みエラー: {e}"

    cols = ds.column_names
    col  = text_column.strip() if text_column.strip() in cols else next(
        (c for c in TEXT_CANDIDATES if c in cols), None)
    if col is None:
        return None, f"❌ テキストカラムが見つかりません。カラム一覧: {cols}"

    texts = [row[col].strip() for row in ds.select(range(min(int(max_samples), len(ds))))
             if isinstance(row.get(col), str) and len(row[col].strip()) > 4]

    if not texts:
        return None, "❌ 有効なテキストが見つかりませんでした。"
    return texts, f"✅ '{col}' カラムから {len(texts)} 件取得"

def preview_dataset(dataset_id, text_column, split, max_samples, progress=gr.Progress()):
    if not dataset_id.strip():
        return "⚠️ データセットIDを入力してください", ""
    progress(0.3, desc="読み込み中…")
    texts, msg = load_dataset_texts(dataset_id, text_column, split, max_samples)
    progress(1.0)
    if texts is None:
        return msg, ""
    return msg, "【先頭5件プレビュー】\n\n" + "\n—\n".join(texts[:5])

# ============================================================
# 学習ループ
# ============================================================

def train_on_dataset(dataset_id, text_column, split, max_samples,
                     epochs, lr, progress=gr.Progress()):
    global model, is_trained, stop_flag
    stop_flag = False
    is_trained = False

    if not dataset_id.strip():
        yield "⚠️ データセットIDを入力してください", ""
        return

    yield "📥 データセットを読み込んでいます...", ""
    texts, msg = load_dataset_texts(dataset_id, text_column, split, max_samples)
    if texts is None:
        yield msg, ""
        return

    yield f"{msg}\n⚙️ データを準備中...", ""

    data = []
    for t in texts:
        ids = tokenizer.encode(t, max_len=MODEL_CONFIG["max_seq_len"] + 1)
        if len(ids) >= 2:
            data.append(ids)

    if not data:
        yield "❌ 学習可能なデータがありません", ""
        return

    model = QBNNTransformer(MODEL_CONFIG)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(lr))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, int(epochs))

    log = [
        f"📊 データ件数 : {len(data)} 件",
        f"🔁 エポック数 : {epochs}  |  学習率 : {lr}",
        "=" * 44,
    ]

    for epoch in range(int(epochs)):
        if stop_flag:
            log.append("⏹ 停止しました")
            break

        model.train()
        total, count = 0.0, 0
        for ids in data:
            L = min(len(ids) - 1, MODEL_CONFIG["max_seq_len"])
            x = torch.tensor([ids[:L]], dtype=torch.long)
            y = torch.tensor([ids[1:L+1]], dtype=torch.long)
            loss = F.cross_entropy(model(x).reshape(-1, tokenizer.vocab_size), y.reshape(-1))
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total += loss.item()
            count += 1
        scheduler.step()

        avg  = total / max(count, 1)
        done = int(20 * (epoch + 1) / int(epochs))
        bar  = "\u2588" * done + "\u2591" * (20 - done)
        line = f"Epoch {epoch+1:>3}/{int(epochs)}  [{bar}]  Loss: {avg:.4f}"
        log.append(line)
        progress((epoch + 1) / int(epochs), desc=line)
        yield "\n".join(log), ""

    is_trained = True
    model.eval()
    final_loss = avg if 'avg' in dir() else 0.0
    add_history_entry(dataset_id, epochs, len(data), final_loss)
    save_msg = save_checkpoint()
    log += ["", save_msg, "🎉 学習完了！「💬 チャット」タブで試してみてください。"]
    yield "\n".join(log), "✅ 完了"

def stop_training():
    global stop_flag
    stop_flag = True
    return "⏹ 停止リクエストを送信…"

# ============================================================
# 推論
# ============================================================

def chat(message, history, temperature):
    global is_trained
    if not is_trained:
        msg = load_checkpoint()
        if not is_trained:
            return history + [(message, f"⚠️ {msg}")]
    ids = tokenizer.encode(message)
    if not ids:
        return history + [(message, "（入力が空です）")]
    out = model.generate(torch.tensor([ids], dtype=torch.long),
                         max_new=30, temperature=float(temperature))
    resp = tokenizer.decode(out[0, len(ids):].tolist()) or "（生成結果が空です。エポック数を増やしてみてください）"
    return history + [(message, resp)]

# ============================================================
# Gradio UI - デザインシステム
# ============================================================

CSS = """
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=Noto+Sans+JP:wght@300;400;500;700&family=Inter:wght@300;400;500;600&display=swap');

/* ── ベース ─────────────────────────────────── */
*  { box-sizing: border-box; }
body, .gradio-container {
    font-family: 'Noto Sans JP', 'Inter', sans-serif !important;
    background: linear-gradient(168deg, #050510 0%, #0a0a1a 35%, #0d0822 70%, #08061a 100%) !important;
    min-height: 100vh;
}
.gradio-container { max-width: 960px !important; margin: 0 auto !important; }

/* ── ヘッダー ───────────────────────────────── */
.hdr {
    text-align: center; padding: 40px 0 16px; position: relative;
    overflow: hidden;
}
.hdr::before {
    content: '';  position: absolute; top: -40px; left: 50%; transform: translateX(-50%);
    width: 400px; height: 400px;
    background: radial-gradient(circle, rgba(139,92,246,.10) 0%, rgba(99,102,241,.05) 30%, transparent 65%);
    pointer-events: none; z-index: 0;
}
.hdr > * { position: relative; z-index: 1; }
.hdr .tagline {
    font-family: 'IBM Plex Mono', monospace; font-size: .68em;
    color: #4a4a6a; letter-spacing: .4em; text-transform: uppercase;
    margin-bottom: 8px;
}
.hdr h1 {
    font-size: 3.2em; margin: 0; letter-spacing: -0.03em;
    font-weight: 700; line-height: 1;
    background: linear-gradient(135deg, #e0d4ff 0%, #a78bfa 30%, #7c3aed 70%, #6d28d9 100%);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    filter: drop-shadow(0 0 40px rgba(139,92,246,.2));
    animation: titleGlow 4s ease-in-out infinite alternate;
}
@keyframes titleGlow {
    0%   { filter: drop-shadow(0 0 30px rgba(139,92,246,.15)); }
    100% { filter: drop-shadow(0 0 50px rgba(139,92,246,.3)); }
}
.hdr .sub {
    font-family: 'IBM Plex Mono', monospace; font-size: .72em;
    color: #3e3e5a; margin-top: 8px;
}
.glow-line {
    width: 140px; height: 2px; margin: 18px auto 0;
    background: linear-gradient(90deg, transparent, #8b5cf6 30%, #a78bfa 50%, #8b5cf6 70%, transparent);
    border-radius: 2px; opacity: .7;
    animation: linePulse 3s ease-in-out infinite;
}
@keyframes linePulse {
    0%, 100% { opacity: .5; width: 120px; }
    50%      { opacity: .9; width: 180px; }
}

/* ── パネル ──────────────────────────────────── */
.panel {
    background: linear-gradient(135deg, rgba(15,15,30,.6) 0%, rgba(10,10,25,.5) 100%);
    backdrop-filter: blur(16px); -webkit-backdrop-filter: blur(16px);
    border: 1px solid rgba(139,92,246,.15);
    border-radius: 16px; padding: 18px 20px; margin-bottom: 14px;
    box-shadow: 0 8px 32px rgba(0,0,0,.2), inset 0 1px 0 rgba(255,255,255,.03);
    transition: border-color .3s ease;
}
.panel:hover { border-color: rgba(139,92,246,.3); }

.panel-glow {
    background: linear-gradient(135deg, rgba(139,92,246,.07) 0%, rgba(99,102,241,.04) 50%, rgba(15,15,30,.5) 100%);
    backdrop-filter: blur(16px);
    border: 1px solid rgba(139,92,246,.25);
    border-radius: 16px; padding: 18px 20px; margin-bottom: 14px;
    box-shadow: 0 0 50px rgba(139,92,246,.05), 0 8px 32px rgba(0,0,0,.15),
                inset 0 1px 0 rgba(255,255,255,.04);
}

/* ── バッジ ──────────────────────────────────── */
.badge {
    display: inline-block;
    background: rgba(139,92,246,.08);
    border: 1px solid rgba(139,92,246,.25);
    border-radius: 8px; padding: 4px 12px;
    font-family: 'IBM Plex Mono', monospace; font-size: .72em;
    color: #a78bfa;
    transition: all .25s cubic-bezier(.4,0,.2,1);
    cursor: default;
}
.badge:hover {
    background: rgba(139,92,246,.18);
    border-color: rgba(139,92,246,.5);
    color: #c4b5fd;
    transform: translateY(-2px);
    box-shadow: 0 4px 16px rgba(139,92,246,.15);
}
.badge-step {
    background: linear-gradient(135deg, rgba(139,92,246,.3), rgba(99,102,241,.2));
    border: 1px solid rgba(139,92,246,.5);
    border-radius: 8px; padding: 4px 14px;
    font-family: 'IBM Plex Mono', monospace; font-size: .72em; font-weight: 600;
    color: #e0d4ff; letter-spacing: .1em;
    text-shadow: 0 0 10px rgba(139,92,246,.3);
}

/* ── フォーム ──────────────────────────────── */
.mono textarea, .mono input {
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: .82em !important;
    background: rgba(5,5,15,.85) !important;
    color: #86efac !important;
    border: 1px solid rgba(139,92,246,.12) !important;
    border-radius: 10px !important;
    transition: all .25s ease !important;
}
.mono textarea:focus, .mono input:focus {
    border-color: rgba(139,92,246,.4) !important;
    box-shadow: 0 0 0 3px rgba(139,92,246,.08) !important;
}

/* ── タブ ────────────────────────────────────── */
.tabs > .tab-nav {
    border-bottom: 1px solid rgba(139,92,246,.1) !important;
    background: transparent !important;
}
.tabs > .tab-nav > button {
    font-family: 'Noto Sans JP', sans-serif !important;
    font-weight: 500 !important; font-size: .88em !important;
    border-radius: 12px 12px 0 0 !important;
    padding: 10px 20px !important;
    transition: all .3s cubic-bezier(.4,0,.2,1) !important;
    border: none !important;
    color: #6b6b8a !important;
}
.tabs > .tab-nav > button:hover {
    color: #a78bfa !important;
    background: rgba(139,92,246,.05) !important;
}
.tabs > .tab-nav > button.selected {
    background: rgba(139,92,246,.1) !important;
    border-bottom: 2px solid #8b5cf6 !important;
    color: #c4b5fd !important;
}

/* ── ボタン ──────────────────────────────────── */
button.primary {
    background: linear-gradient(135deg, #7c3aed 0%, #6d28d9 50%, #5b21b6 100%) !important;
    border: none !important; border-radius: 12px !important;
    font-weight: 500 !important; font-size: .92em !important;
    box-shadow: 0 4px 20px rgba(109,40,217,.3), inset 0 1px 0 rgba(255,255,255,.1) !important;
    transition: all .3s cubic-bezier(.4,0,.2,1) !important;
    text-shadow: 0 1px 2px rgba(0,0,0,.2) !important;
}
button.primary:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 30px rgba(109,40,217,.45), inset 0 1px 0 rgba(255,255,255,.15) !important;
    filter: brightness(1.1) !important;
}
button.primary:active { transform: translateY(0) !important; }

button.secondary {
    border: 1px solid rgba(139,92,246,.3) !important;
    border-radius: 12px !important;
    background: rgba(139,92,246,.06) !important;
    transition: all .25s ease !important;
}
button.secondary:hover {
    background: rgba(139,92,246,.12) !important;
    border-color: rgba(139,92,246,.5) !important;
}

button.stop {
    background: linear-gradient(135deg, #dc2626 0%, #b91c1c 100%) !important;
    border: none !important; border-radius: 12px !important;
    box-shadow: 0 4px 16px rgba(185,28,28,.25) !important;
    transition: all .25s ease !important;
}
button.stop:hover { box-shadow: 0 6px 24px rgba(185,28,28,.4) !important; }

/* ── チャット ────────────────────────────────── */
.chatbot {
    border-radius: 16px !important;
    border: 1px solid rgba(139,92,246,.12) !important;
    background: rgba(8,8,18,.6) !important;
    box-shadow: inset 0 2px 20px rgba(0,0,0,.15) !important;
}
.chatbot .message { border-radius: 14px !important; }

/* ── 言語セクション ─────────────────────────── */
.lang-section {
    padding: 10px 0 8px;
    border-bottom: 1px solid rgba(139,92,246,.06);
    display: flex; flex-wrap: wrap; align-items: center; gap: 4px;
}
.lang-section:last-child { border-bottom: none; }
.lang-label {
    color: #94a3b8; font-size: .84em; font-weight: 500;
    min-width: 90px;
}

/* ── アニメーション ─────────────────────────── */
@keyframes fadeInUp {
    from { opacity: 0; transform: translateY(16px); }
    to   { opacity: 1; transform: translateY(0); }
}
@keyframes pulseGlow {
    0%, 100% { box-shadow: 0 0 20px rgba(139,92,246,.06); }
    50%      { box-shadow: 0 0 40px rgba(139,92,246,.12); }
}
.animate-in { animation: fadeInUp .6s cubic-bezier(.4,0,.2,1) both; }
.animate-in-delay { animation: fadeInUp .6s cubic-bezier(.4,0,.2,1) .15s both; }
.pulse-glow { animation: pulseGlow 4s ease-in-out infinite; }

/* ── スクロールバー & フッター ───────────── */
footer { display: none !important; }
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: rgba(139,92,246,.25); border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: rgba(139,92,246,.45); }
"""

# ============================================================
# おすすめデータセット（言語別）
# ============================================================

RECOMMENDED = {
    "\U0001f1ef\U0001f1f5 日本語": [
        "kunishou/databricks-dolly-15k-ja",
        "llm-book/aio-passages",
        "izumi-lab/open-text-books",
        "shunk031/JGLUE",
        "range3/wiki40b-ja",
        "kunishou/oasst1-chat-44k-ja",
    ],
    "\U0001f1fa\U0001f1f8 英語": [
        "roneneldan/TinyStories",
        "ag_news",
        "wikitext",
        "tatsu-lab/alpaca",
        "databricks/databricks-dolly-15k",
        "openai/gsm8k",
    ],
    "\U0001f1e8\U0001f1f3 中国語": [
        "silk-road/alpaca-data-gpt4-chinese",
        "shibing624/sharegpt_gpt4",
        "FreedomIntelligence/huatuo_encyclopedia_qa",
    ],
    "\U0001f1f0\U0001f1f7 韓国語": [
        "heegyu/namuwiki-extracted",
        "nlpai-lab/kullm-v2",
    ],
    "\U0001f310 多言語": [
        "facebook/flores",
        "mc4",
        "wikipedia",
        "csebuetnlp/xlsum",
    ],
}

# ============================================================
# Gradio アプリ構築
# ============================================================

with gr.Blocks(title="neuroQ \u2013 QBNN Dataset Trainer", css=CSS,
               theme=gr.themes.Base(primary_hue="violet", neutral_hue="slate")) as demo:

    # ── ヘッダー ──────────────────────────────────────
    gr.HTML("""
    <div class='hdr'>
        <div class='tagline'>quantum \u00b7 neural \u00b7 network</div>
        <h1>neuroQ</h1>
        <div class='sub'>QBNN-Transformer \u00b7 CPU Edition \u00b7 Dataset Trainer</div>
        <div class='glow-line'></div>
    </div>
    """)

    with gr.Tabs():

        # ══════════════════════════════════════════════
        # タブ① データセット学習
        # ══════════════════════════════════════════════
        with gr.Tab("\u269b データセット学習"):

            gr.HTML("<div class='panel-glow animate-in'>"
                    "<span class='badge-step'>STEP 1</span>"
                    "<span style='color:#cbd5e1;font-size:.88em;margin-left:10px;'>"
                    "HuggingFace データセットIDを入力してプレビュー</span></div>")

            with gr.Row():
                dataset_id_box = gr.Textbox(
                    label="\U0001f4e6 データセットID（author/name 形式）",
                    placeholder="例: kunishou/databricks-dolly-15k-ja",
                    scale=3,
                )
                text_col_box = gr.Textbox(
                    label="\U0001f4dd テキストカラム名（空=自動）",
                    value="text", scale=1,
                )

            with gr.Row():
                split_box = gr.Dropdown(
                    choices=["train", "validation", "test"],
                    value="train", label="\U0001f4c2 スプリット", scale=1,
                )
                max_samples_box = gr.Slider(
                    10, 2000, value=300, step=10,
                    label="\U0001f4ca 最大サンプル数", scale=3,
                )

            preview_btn = gr.Button("\U0001f50d プレビュー確認", size="sm", variant="secondary")
            preview_status  = gr.Textbox(label="", interactive=False, lines=1)
            preview_content = gr.Textbox(label="プレビュー", interactive=False,
                                         lines=5, elem_classes="mono")

            # おすすめデータセット（言語別）
            rec_sections = []
            for lang, datasets in RECOMMENDED.items():
                badges = "".join(
                    f"<span class='badge' style='margin:3px;'>{d}</span>"
                    for d in datasets
                )
                rec_sections.append(
                    f"<div class='lang-section'>"
                    f"<span class='lang-label'>{lang}</span>"
                    f"<div style='display:flex;flex-wrap:wrap;gap:4px;'>{badges}</div></div>"
                )
            rec_html = "\n".join(rec_sections)
            gr.HTML(
                f"<div class='panel animate-in-delay' style='margin-top:12px;'>"
                f"<div style='color:#6d6d8a;font-family:\"IBM Plex Mono\",monospace;"
                f"font-size:.68em;margin-bottom:12px;letter-spacing:.25em;text-transform:uppercase;'>"
                f"\U0001f4da Recommended Datasets</div>"
                f"{rec_html}</div>"
            )

            gr.HTML("<div class='panel-glow animate-in' style='margin-top:8px;'>"
                    "<span class='badge-step'>STEP 2</span>"
                    "<span style='color:#cbd5e1;font-size:.88em;margin-left:10px;'>"
                    "学習パラメータを設定して学習開始</span></div>")

            with gr.Row():
                epochs_box = gr.Slider(5, 300, value=30, step=5, label="\U0001f501 エポック数")
                lr_box     = gr.Slider(0.0001, 0.01, value=0.001, step=0.0001, label="\U0001f4c8 学習率")

            with gr.Row():
                train_btn = gr.Button("\u25b6  学習開始", variant="primary", scale=3)
                stop_btn  = gr.Button("\u23f9  停止",    variant="stop",    scale=1)

            stop_status = gr.Textbox(label="", interactive=False, lines=1)
            train_log   = gr.Textbox(label="\U0001f4cb 学習ログ", interactive=False,
                                     lines=16, elem_classes="mono",
                                     placeholder="学習ログがここに表示されます...")

            preview_btn.click(
                fn=preview_dataset,
                inputs=[dataset_id_box, text_col_box, split_box, max_samples_box],
                outputs=[preview_status, preview_content],
                show_progress=True,
            )
            train_btn.click(
                fn=train_on_dataset,
                inputs=[dataset_id_box, text_col_box, split_box, max_samples_box,
                        epochs_box, lr_box],
                outputs=[train_log, stop_status],
                show_progress=True,
            )
            stop_btn.click(fn=stop_training, outputs=stop_status)

        # ══════════════════════════════════════════════
        # タブ② チャット
        # ══════════════════════════════════════════════
        with gr.Tab("\U0001f4ac チャット"):
            gr.HTML("<div class='panel' style='text-align:center;padding:12px;'>"
                    "<span style='color:#6d6d8a;font-family:\"IBM Plex Mono\",monospace;"
                    "font-size:.72em;letter-spacing:.2em;text-transform:uppercase;'>"
                    "\u269b QBNN Inference Engine</span></div>")
            chatbot = gr.Chatbot(height=420, label="neuroQ",
                                 bubble_full_width=False)
            with gr.Row():
                msg_in   = gr.Textbox(placeholder="メッセージを入力して Enter...",
                                      label="", scale=4, show_label=False,
                                      container=False)
                send_btn = gr.Button("送信 \u2192", variant="primary", scale=1)
            temp_sl = gr.Slider(0.1, 2.0, value=0.8, step=0.05, label="\U0001f321 Temperature")
            with gr.Row():
                load_btn  = gr.Button("\U0001f4e5 チェックポイント読込", size="sm", variant="secondary")
                clear_btn = gr.Button("\U0001f5d1 クリア", size="sm", variant="secondary")
            load_out = gr.Textbox(label="", interactive=False, lines=1)

            send_btn.click(fn=chat, inputs=[msg_in, chatbot, temp_sl], outputs=chatbot)
            msg_in.submit(fn=chat, inputs=[msg_in, chatbot, temp_sl], outputs=chatbot)
            load_btn.click(fn=load_checkpoint, outputs=load_out)
            clear_btn.click(fn=lambda: [], outputs=chatbot)

        # ══════════════════════════════════════════════
        # タブ③ 学習履歴
        # ══════════════════════════════════════════════
        with gr.Tab("\U0001f4cb 学習履歴"):
            gr.HTML("<div class='panel-glow'>"
                    "<span class='badge-step'>HISTORY</span>"
                    "<span style='color:#cbd5e1;font-size:.88em;margin-left:10px;'>"
                    "学習済みデータセットの一覧管理</span></div>")

            history_display = gr.Textbox(
                label="学習済みデータセット一覧",
                value=get_history_display(),
                interactive=False,
                lines=10,
                elem_classes="mono",
            )

            with gr.Row():
                history_dropdown = gr.Dropdown(
                    choices=get_history_choices(),
                    label="\U0001f5d1 削除する項目を選択",
                    scale=3,
                )
                delete_btn = gr.Button("\U0001f5d1 削除", variant="stop", scale=1)

            with gr.Row():
                refresh_btn = gr.Button("\U0001f504 履歴を更新", size="sm", variant="secondary")
            delete_status = gr.Textbox(label="", interactive=False, lines=1)

            delete_btn.click(
                fn=delete_history_entry,
                inputs=[history_dropdown],
                outputs=[delete_status, history_display, history_dropdown],
            )
            refresh_btn.click(
                fn=refresh_history,
                outputs=[history_display, history_dropdown],
            )

        # ══════════════════════════════════════════════
        # タブ④ モデル情報
        # ══════════════════════════════════════════════
        with gr.Tab("\u2139 モデル情報"):
            p = sum(v.numel() for v in model.parameters())
            config_rows = [
                ("Embedding Dim",   MODEL_CONFIG['embed_dim']),
                ("Attention Heads",  MODEL_CONFIG['num_heads']),
                ("Layers",           MODEL_CONFIG['num_layers']),
                ("Max Seq Length",   MODEL_CONFIG['max_seq_len']),
                ("Vocab Size",       MODEL_CONFIG['vocab_size']),
                ("Total Params",     f"{p:,}"),
                ("Entangle \u03bb",  MODEL_CONFIG['entangle_strength']),
            ]
            table_html = "".join(
                f"<tr>"
                f"<td style='padding:10px 14px 10px 0;border-bottom:1px solid rgba(139,92,246,.06);"
                f"color:#94a3b8;font-size:.88em;'>{k}</td>"
                f"<td style='padding:10px 0;border-bottom:1px solid rgba(139,92,246,.06);"
                f"font-family:IBM Plex Mono,monospace;color:#c4b5fd;font-weight:600;"
                f"font-size:.92em;'>{v}</td></tr>"
                for k, v in config_rows
            )
            gr.HTML(f"""
            <div class='panel pulse-glow'>
                <div style='font-family:"IBM Plex Mono",monospace;color:#8b5cf6;font-size:.72em;
                            margin-bottom:16px;letter-spacing:.2em;text-transform:uppercase;'>
                    \u269b Model Configuration
                </div>
                <table style='color:#cbd5e1;font-size:.88em;border-collapse:collapse;width:100%;'>
                    {table_html}
                </table>
            </div>
            <div class='panel' style='margin-top:10px;'>
                <div style='color:#5a5a7a;font-size:.76em;font-family:"IBM Plex Mono",monospace;line-height:2.2;'>
                    \u26a0 CPU only \u2014 大規模学習には Colab / GCP を推奨<br>
                    \u26a0 HF_TOKEN を Secrets に設定するとチェックポイントが自動保存されます<br>
                    \u269b Architecture: QBNN-Transformer (量子もつれ層 + Multi-Head Attention)
                </div>
            </div>
            """)

    demo.load(fn=load_checkpoint, outputs=gr.Textbox(visible=False))

if __name__ == "__main__":
    demo.launch()