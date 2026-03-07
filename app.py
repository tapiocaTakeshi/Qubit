"""
neuroQ - QBNN-Transformer on Hugging Face Spaces
データセットID入力 → 自動ロード → 学習 → 推論
CPU最適化版
"""

import os
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
        return "✅ チェックポイントを保存しました"
    except Exception as e:
        return f"❌ 保存エラー: {e}"

def load_checkpoint():
    global is_trained
    try:
        path = hf_hub_download(repo_id=REPO_ID, filename=CKPT_FILE, repo_type="space")
        model.load_state_dict(torch.load(path, map_location="cpu")["model_state"])
        model.eval()
        is_trained = True
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
        bar  = "█" * done + "░" * (20 - done)
        line = f"Epoch {epoch+1:>3}/{int(epochs)}  [{bar}]  Loss: {avg:.4f}"
        log.append(line)
        progress((epoch + 1) / int(epochs), desc=line)
        yield "\n".join(log), ""

    is_trained = True
    model.eval()
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
# Gradio UI
# ============================================================

CSS = """
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=Noto+Sans+JP:wght@300;500&display=swap');
body, .gradio-container { font-family: 'Noto Sans JP', sans-serif !important; background: #0b0b14 !important; }
.hdr { text-align:center; padding:24px 0 4px; }
.hdr h1 { font-size:2.2em; color:#e2e8f0; margin:0; letter-spacing:-0.02em; }
.hdr .sub { font-family:'IBM Plex Mono',monospace; font-size:0.75em; color:#4a4a6a; margin-top:4px; }
.badge { display:inline-block; background:rgba(139,92,246,.18); border:1px solid rgba(139,92,246,.45); border-radius:4px; padding:2px 9px; font-family:'IBM Plex Mono',monospace; font-size:.75em; color:#a78bfa; }
.panel { background:rgba(255,255,255,.025); border:1px solid rgba(139,92,246,.25); border-radius:12px; padding:14px; margin-bottom:12px; }
.mono textarea, .mono input { font-family:'IBM Plex Mono',monospace !important; font-size:.82em !important; background:#05050d !important; color:#86efac !important; }
"""

RECOMMENDED = [
    "kunishou/databricks-dolly-15k-ja",
    "llm-book/aio-passages",
    "izumi-lab/open-text-books",
    "roneneldan/TinyStories",
    "ag_news",
]

with gr.Blocks(title="neuroQ – Dataset Trainer", css=CSS,
               theme=gr.themes.Base(primary_hue="violet", neutral_hue="slate")) as demo:

    gr.HTML("""
    <div class='hdr'>
        <div style='font-family:"IBM Plex Mono",monospace;font-size:.8em;color:#5a5a7a;letter-spacing:.3em;margin-bottom:4px;'>
            QUANTUM · NEURAL · NETWORK
        </div>
        <h1>neuro<span style='color:#8b5cf6;'>Q</span></h1>
        <div class='sub'>QBNN-Transformer · CPU Edition · Dataset Trainer</div>
    </div>
    """)

    with gr.Tabs():

        # ── タブ① データセット学習 ─────────────────────────
        with gr.Tab("⚛ データセット学習"):

            gr.HTML("<div class='panel'><span class='badge'>STEP 1</span>"
                    " <span style='color:#94a3b8;font-size:.88em;'>"
                    "HuggingFace データセットIDを入力してプレビュー</span></div>")

            with gr.Row():
                dataset_id_box = gr.Textbox(
                    label="📦 データセットID（author/name 形式）",
                    placeholder="例: kunishou/databricks-dolly-15k-ja",
                    scale=3,
                )
                text_col_box = gr.Textbox(
                    label="📝 テキストカラム名（空=自動）",
                    value="text", scale=1,
                )

            with gr.Row():
                split_box = gr.Dropdown(
                    choices=["train", "validation", "test"],
                    value="train", label="📂 スプリット", scale=1,
                )
                max_samples_box = gr.Slider(
                    10, 2000, value=300, step=10,
                    label="📊 最大サンプル数", scale=3,
                )

            preview_btn = gr.Button("🔍 プレビュー確認", size="sm", variant="secondary")
            preview_status  = gr.Textbox(label="", interactive=False, lines=1)
            preview_content = gr.Textbox(label="プレビュー", interactive=False,
                                         lines=5, elem_classes="mono")

            # おすすめ
            rec_html = "".join(f"<span class='badge' style='margin:3px;'>{d}</span>" for d in RECOMMENDED)
            gr.HTML(f"<div style='margin:6px 0 14px;'><div style='color:#4a4a6a;font-family:\"IBM Plex Mono\",monospace;font-size:.73em;margin-bottom:6px;'>RECOMMENDED</div>{rec_html}</div>")

            gr.HTML("<hr style='border-color:rgba(139,92,246,.2);margin:4px 0 14px;'>")
            gr.HTML("<div class='panel'><span class='badge'>STEP 2</span>"
                    " <span style='color:#94a3b8;font-size:.88em;'>学習パラメータを設定して学習開始</span></div>")

            with gr.Row():
                epochs_box = gr.Slider(5, 300, value=30, step=5, label="🔁 エポック数")
                lr_box     = gr.Slider(0.0001, 0.01, value=0.001, step=0.0001, label="📈 学習率")

            with gr.Row():
                train_btn = gr.Button("▶  学習開始", variant="primary", scale=3)
                stop_btn  = gr.Button("⏹  停止",    variant="stop",    scale=1)

            stop_status = gr.Textbox(label="", interactive=False, lines=1)
            train_log   = gr.Textbox(label="学習ログ", interactive=False,
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

        # ── タブ② チャット ────────────────────────────────
        with gr.Tab("💬 チャット"):
            chatbot = gr.Chatbot(height=420, label="neuroQ")
            with gr.Row():
                msg_in   = gr.Textbox(placeholder="入力してEnter...", label="", scale=4, show_label=False)
                send_btn = gr.Button("送信", variant="primary", scale=1)
            temp_sl = gr.Slider(0.1, 2.0, value=0.8, step=0.05, label="🌡 Temperature")
            with gr.Row():
                load_btn  = gr.Button("📥 チェックポイント読込", size="sm")
                clear_btn = gr.Button("🗑 クリア",              size="sm")
            load_out = gr.Textbox(label="", interactive=False, lines=1)

            send_btn.click(fn=chat, inputs=[msg_in, chatbot, temp_sl], outputs=chatbot)
            msg_in.submit(fn=chat, inputs=[msg_in, chatbot, temp_sl], outputs=chatbot)
            load_btn.click(fn=load_checkpoint, outputs=load_out)
            clear_btn.click(fn=lambda: [], outputs=chatbot)

        # ── タブ③ モデル情報 ──────────────────────────────
        with gr.Tab("ℹ モデル情報"):
            p = sum(v.numel() for v in model.parameters())
            gr.HTML(f"""
            <div class='panel'>
                <div style='font-family:"IBM Plex Mono",monospace;color:#8b5cf6;font-size:.82em;margin-bottom:10px;'>MODEL CONFIG</div>
                <table style='color:#cbd5e1;font-size:.88em;border-collapse:collapse;width:100%;'>
                    {''.join(f"<tr><td style='padding:5px 0;border-bottom:1px solid rgba(139,92,246,.1);'>{k}</td><td style='font-family:IBM Plex Mono,monospace;color:#a78bfa;'>{v}</td></tr>"
                    for k,v in [('embed_dim',MODEL_CONFIG['embed_dim']),('num_heads',MODEL_CONFIG['num_heads']),
                                 ('num_layers',MODEL_CONFIG['num_layers']),('max_seq_len',MODEL_CONFIG['max_seq_len']),
                                 ('vocab_size',MODEL_CONFIG['vocab_size']),('total params',f'{p:,}'),
                                 ('entangle_strength λ',MODEL_CONFIG['entangle_strength'])])}
                </table>
            </div>
            <div style='color:#4a4a6a;font-size:.78em;font-family:"IBM Plex Mono",monospace;margin-top:10px;line-height:1.9;'>
                ⚠ CPUのみ。大規模学習には Colab / GCP を推奨<br>
                ⚠ HF_TOKEN を Secrets に設定するとチェックポイントが自動保存されます
            </div>
            """)

    demo.load(fn=load_checkpoint, outputs=gr.Textbox(visible=False))

if __name__ == "__main__":
    demo.launch()