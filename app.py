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
    try:
        if not dataset_id.strip():
            return "⚠️ データセットIDを入力してください", ""

        # カンマまたは改行区切りで複数データセットIDを分割
        dataset_ids = [d.strip() for d in dataset_id.replace("\n", ",").split(",") if d.strip()]
        if not dataset_ids:
            return "⚠️ データセットIDを入力してください", ""

        all_texts = []
        messages = []
        for i, did in enumerate(dataset_ids):
            progress((i + 0.5) / len(dataset_ids), desc=f"読み込み中… ({did})")
            texts, msg = load_dataset_texts(did, text_column, split, max_samples)
            messages.append(f"📦 {did}: {msg}")
            if texts:
                all_texts.extend(texts)
        progress(1.0)

        status = "\n".join(messages)
        if not all_texts:
            return status + "\n❌ 有効なテキストが見つかりませんでした。", ""

        status += f"\n\n✅ 合計 {len(all_texts)} 件のテキストを取得"
        preview = "【先頭5件プレビュー】\n\n" + "\n—\n".join(all_texts[:5])
        return status, preview
    except Exception as e:
        import traceback
        return f"🚨 エラーが発生しました:\n{traceback.format_exc()}", ""

# ============================================================
# 学習ループ
# ============================================================

def train_on_dataset(dataset_id, text_column, split, max_samples,
                     epochs, lr, progress=gr.Progress()):
    try:
        global model, is_trained, stop_flag
        stop_flag = False
        is_trained = False

        if not dataset_id.strip():
            yield "⚠️ データセットIDを入力してください", ""
            return

        # カンマまたは改行区切りで複数データセットIDを分割
        dataset_ids = [d.strip() for d in dataset_id.replace("\n", ",").split(",") if d.strip()]
        if not dataset_ids:
            yield "⚠️ データセットIDを入力してください", ""
            return

        # ── 複数データセットからテキストを一括読み込み ──
        all_texts = []
        load_log = []
        load_log.append(f"📥 {len(dataset_ids)} 個のデータセットを読み込みます...")
        yield "\n".join(load_log), ""

        for i, did in enumerate(dataset_ids):
            load_log.append(f"\n📦 [{i+1}/{len(dataset_ids)}] {did} を読み込み中...")
            yield "\n".join(load_log), ""

            texts, msg = load_dataset_texts(did, text_column, split, max_samples)
            load_log.append(f"   {msg}")
            if texts:
                load_log.append(f"   → {len(texts)} 件取得")
                all_texts.extend(texts)
            else:
                load_log.append(f"   → スキップ（テキスト取得不可）")
            yield "\n".join(load_log), ""

        if not all_texts:
            load_log.append("\n❌ すべてのデータセットから有効なテキストが見つかりませんでした")
            yield "\n".join(load_log), ""
            return

        load_log.append(f"\n✅ 合計: {len(all_texts)} 件のテキストを取得")
        load_log.append("⚙️ データを準備中...")
        yield "\n".join(load_log), ""

        data = []
        for t in all_texts:
            ids = tokenizer.encode(t, max_len=MODEL_CONFIG["max_seq_len"] + 1)
            if len(ids) >= 2:
                data.append(ids)

        if not data:
            yield "❌ 学習可能なデータがありません", ""
            return

        model = QBNNTransformer(MODEL_CONFIG)
        optimizer = torch.optim.AdamW(model.parameters(), lr=float(lr))
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, int(epochs))

        log = load_log + [
            "",
            "=" * 44,
            f"📊 学習データ件数     : {len(data)} 件 （{len(dataset_ids)} データセット）",
            f"🔁 エポック数         : {epochs}  |  学習率 : {lr}",
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
                
                # 10ステップごとに一時的な経過をUIに表示
                if count % 10 == 0 or count == len(data):
                    step_loss = total / count
                    temp_line = f"▶ Epoch {epoch+1}/{int(epochs)}  [Step {count}/{len(data)}]  Loss: {step_loss:.4f} ..."
                    progress((epoch + (count/len(data))) / int(epochs), desc=f"Epoch {epoch+1} - Step {count}/{len(data)}")
                    yield "\n".join(log + [temp_line]), ""

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
        final_loss = avg if 'avg' in dir() else 0.0
        # 学習履歴に全データセットIDを記録
        combined_id = ", ".join(dataset_ids)
        add_history_entry(combined_id, epochs, len(data), final_loss)
        save_msg = save_checkpoint()
        log += ["", save_msg, "🎉 学習完了！「💬 チャット」タブで試してみてください。"]
        yield "\n".join(log), "✅ 完了"
    except Exception as e:
        import traceback
        err_msg = f"\n\n🚨 予期せぬエラーが発生しました:\n{e}\n\n{traceback.format_exc()}"
        if 'log' in locals() and isinstance(log, list):
            yield "\n".join(log + [err_msg]), "❌ エラー"
        elif 'load_log' in locals() and isinstance(load_log, list):
             yield "\n".join(load_log + [err_msg]), "❌ エラー"
        else:
            yield err_msg, "❌ エラー"

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
/* シンプルなベーススタイル */
.mono textarea, .mono input {
    font-family: monospace !important;
}
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

with gr.Blocks(title="neuroQ \u2013 QBNN Dataset Trainer", css=CSS) as demo:

    # ── ヘッダー ──────────────────────────────────────
    gr.Markdown("# neuroQ\n**QBNN-Transformer · CPU Edition · Dataset Trainer**")

    with gr.Tabs():

        # ══════════════════════════════════════════════
        # タブ① データセット学習
        # ══════════════════════════════════════════════
        with gr.Tab("\u269b データセット学習"):
            gr.Markdown("### STEP 1: HuggingFace データセットIDを入力（カンマ区切りで複数OK）")

            with gr.Row():
                dataset_id_box = gr.Textbox(
                    lines=3,
                    label="📦 データセットID（改行 または カンマ区切りで複数指定）",
                    placeholder="【例】\nkunishou/databricks-dolly-15k-ja\nag_news\nroneneldan/TinyStories",
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
            with gr.Accordion("\U0001f4da おすすめデータセット", open=False):
                for lang, datasets in RECOMMENDED.items():
                    gr.Markdown(f"**{lang}**: {', '.join(datasets)}")

            gr.Markdown("### STEP 2: 学習パラメータを設定して学習開始")

            with gr.Row():
                epochs_box = gr.Slider(5, 300, value=30, step=5, label="\U0001f501 エポック数")
                lr_box     = gr.Slider(0.0001, 0.01, value=0.001, step=0.0001, label="\U0001f4c8 学習率")

            with gr.Row():
                train_btn = gr.Button("\u25b6  学習開始", variant="primary", scale=3)
                stop_btn  = gr.Button("\u23f9  停止",    variant="stop",    scale=1)

            stop_status = gr.Textbox(label="", interactive=False, lines=1)
            train_log   = gr.Textbox(label="\U0001f4cb 学習ログ", interactive=False,
                                     lines=16, elem_classes="mono",
                                     placeholder="学習ログがここに表示されます...",
                                     autoscroll=True)

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
            gr.Markdown("### \u269b QBNN Inference Engine")
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
            gr.Markdown("### 学習履歴の管理\n履歴の一覧確認と削除が行えます。")

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
            md = f"""### \u269b Model Configuration
| Parameter | Value |
| --- | --- |
| **Embedding Dim** | {MODEL_CONFIG['embed_dim']} |
| **Attention Heads** | {MODEL_CONFIG['num_heads']} |
| **Layers** | {MODEL_CONFIG['num_layers']} |
| **Max Seq Length** | {MODEL_CONFIG['max_seq_len']} |
| **Vocab Size** | {MODEL_CONFIG['vocab_size']} |
| **Total Params** | {p:,} |
| **Entangle λ** | {MODEL_CONFIG['entangle_strength']} |

### \u26a0 注意事項
- **CPU only** — 大規模学習には Colab / GCP を推奨
- **HF_TOKEN** を Secrets に設定するとチェックポイントが自動保存されます
- **Architecture**: QBNN-Transformer (量子もつれ層 + Multi-Head Attention)
"""
            gr.Markdown(md)

    demo.load(fn=load_checkpoint, outputs=gr.Textbox(visible=False))

if __name__ == "__main__":
    demo.launch()