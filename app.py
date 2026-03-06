#!/usr/bin/env python3
"""
🧠⚛️ NeuroQ - 脳型量子ビットネットワーク チャットボット
Hugging Face Spaces 用 Gradio インターフェース

QBNN (Quantum Bit Neural Network) Transformer による日本語対話AI
"""

import os
import sys
import torch
import torch.nn.functional as F
import warnings
import time

warnings.filterwarnings('ignore')

# パス設定
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

# NeuroQ モジュールのインポート
from neuroquantum_layered import NeuroQuantum, NeuroQuantumConfig, NeuroQuantumTokenizer

import gradio as gr

# ========================================
# モデルのロード（起動時に1回だけ実行）
# ========================================

print("🧠⚛️ NeuroQ モデルをロード中...")

# デバイス検出
if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
    DEVICE_NAME = "🎮 NVIDIA GPU (CUDA)"
elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
    DEVICE_NAME = "🍎 Apple Silicon GPU (MPS)"
else:
    DEVICE = torch.device("cpu")
    DEVICE_NAME = "💻 CPU"

print(f"🖥️ デバイス: {DEVICE_NAME}")

# チェックポイント検索
CHECKPOINT_CANDIDATES = [
    os.path.join(SCRIPT_DIR, 'neuroq_checkpoint.pt'),
    os.path.join(SCRIPT_DIR, 'checkpoints', 'neuroq_checkpoint.pt'),
]

checkpoint_path = None
for cp in CHECKPOINT_CANDIDATES:
    if os.path.exists(cp):
        checkpoint_path = cp
        break

# トークナイザー検索
TOKENIZER_CANDIDATES = [
    os.path.join(SCRIPT_DIR, 'neuroq_tokenizer_8k.model'),
    os.path.join(SCRIPT_DIR, 'neuroq_tokenizer.model'),
    os.path.join(SCRIPT_DIR, 'neuroq_tokenizer_ja.model'),
]

tokenizer_path = None
for tp in TOKENIZER_CANDIDATES:
    if os.path.exists(tp):
        tokenizer_path = tp
        break

# モデル初期化
MODEL = None
TOKENIZER = None
MODEL_INFO = {}

if checkpoint_path and tokenizer_path:
    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

        # 設定の取得
        if isinstance(checkpoint, dict) and 'config' in checkpoint:
            cfg = checkpoint['config']
            vocab_size = cfg.get('vocab_size', 8000)
            embed_dim = cfg.get('embed_dim', 256)
            num_heads = cfg.get('num_heads', 8)
            num_layers = cfg.get('num_layers', 5)
            max_seq_len = cfg.get('max_seq_len', 512)
        else:
            vocab_size = 8000
            embed_dim = 256
            num_heads = 8
            num_layers = 5
            max_seq_len = 512

        config = NeuroQuantumConfig(
            vocab_size=vocab_size,
            embed_dim=embed_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            max_seq_len=max_seq_len,
        )

        MODEL = NeuroQuantum(config)
        state_dict = checkpoint.get('model_state_dict', checkpoint)
        result = MODEL.load_state_dict(state_dict, strict=False)
        MODEL = MODEL.to(DEVICE)
        MODEL.eval()

        TOKENIZER = NeuroQuantumTokenizer(
            vocab_size=vocab_size,
            model_file=tokenizer_path,
        )

        total_params = sum(p.numel() for p in MODEL.parameters())
        phase = checkpoint.get('phase', '不明') if isinstance(checkpoint, dict) else '不明'

        MODEL_INFO = {
            'params': f"{total_params:,}",
            'embed_dim': embed_dim,
            'num_heads': num_heads,
            'num_layers': num_layers,
            'max_seq_len': max_seq_len,
            'phase': phase,
            'device': DEVICE_NAME,
        }

        print(f"✅ モデルロード完了: {total_params:,} パラメータ")
        print(f"   embed_dim={embed_dim}, heads={num_heads}, layers={num_layers}")
        print(f"   学習フェーズ: {phase}")

    except Exception as e:
        print(f"❌ モデルロードエラー: {e}")
        import traceback
        traceback.print_exc()
else:
    print(f"⚠️ チェックポイントまたはトークナイザーが見つかりません")
    print(f"   checkpoint: {checkpoint_path}")
    print(f"   tokenizer: {tokenizer_path}")


# ========================================
# テキスト生成
# ========================================

@torch.no_grad()
def generate_text(prompt: str, max_new_tokens: int = 150,
                  temperature: float = 0.8, top_k: int = 40,
                  top_p: float = 0.92, repetition_penalty: float = 1.2):
    """NeuroQモデルでテキスト生成"""
    if MODEL is None or TOKENIZER is None:
        return "⚠️ モデルが読み込まれていません。チェックポイントを確認してください。"

    try:
        tokens = TOKENIZER.encode(prompt, add_special=True)
        max_seq = MODEL_INFO.get('max_seq_len', 512)

        if len(tokens) > max_seq - 10:
            tokens = tokens[-(max_seq - 10):]

        input_ids = torch.tensor([tokens], dtype=torch.long).to(DEVICE)
        generated_tokens = []
        generated_text_so_far = set()

        for _ in range(max_new_tokens):
            if input_ids.size(1) > max_seq:
                input_ids = input_ids[:, -max_seq:]

            logits = MODEL(input_ids)
            next_logits = logits[:, -1, :] / max(temperature, 0.01)

            # Repetition Penalty
            if repetition_penalty > 1.0 and generated_tokens:
                for token_id in set(generated_tokens[-50:]):
                    next_logits[0, token_id] /= repetition_penalty

            # Top-K
            if top_k > 0:
                top_k_val = min(top_k, next_logits.size(-1))
                indices_to_remove = next_logits < torch.topk(next_logits, top_k_val)[0][..., -1, None]
                next_logits[indices_to_remove] = -float('inf')

            # Top-P (Nucleus Sampling)
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(next_logits, descending=True)
                probs = F.softmax(sorted_logits, dim=-1)
                cumulative_probs = torch.cumsum(probs, dim=-1)
                remove_mask = cumulative_probs - probs > top_p
                sorted_logits[remove_mask] = -float('inf')
                next_logits = sorted_logits.scatter(1, sorted_indices, sorted_logits)

            probs = F.softmax(next_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            token_id = next_token.item()

            # EOS判定
            if hasattr(TOKENIZER, 'eos_id') and token_id == TOKENIZER.eos_id:
                break

            generated_tokens.append(token_id)
            input_ids = torch.cat([input_ids, next_token], dim=1)

            # 繰り返し検出
            decoded = TOKENIZER.decode(generated_tokens)
            if len(decoded) > 20:
                last_chunk = decoded[-20:]
                if last_chunk in generated_text_so_far:
                    break
                generated_text_so_far.add(last_chunk)

        result = TOKENIZER.decode(generated_tokens)
        return result.strip()

    except Exception as e:
        return f"⚠️ 生成エラー: {str(e)}"


# ========================================
# Gradio インターフェース
# ========================================

def respond(message, history, system_message, max_tokens, temperature, top_p,
            top_k, repetition_penalty):
    """Gradio チャット応答関数"""

    # プロンプト構築
    prompt_parts = []

    if system_message:
        prompt_parts.append(f"システム: {system_message}")

    # 会話履歴を含める（最新3ターンまで）
    if history:
        recent = history[-3:]
        for turn in recent:
            if turn.get("role") == "user":
                prompt_parts.append(f"ユーザー: {turn['content']}")
            elif turn.get("role") == "assistant":
                prompt_parts.append(f"アシスタント: {turn['content']}")

    prompt_parts.append(f"ユーザー: {message}")
    prompt_parts.append("アシスタント:")

    full_prompt = "\n".join(prompt_parts)

    # テキスト生成
    response = generate_text(
        prompt=full_prompt,
        max_new_tokens=int(max_tokens),
        temperature=temperature,
        top_k=int(top_k),
        top_p=top_p,
        repetition_penalty=repetition_penalty,
    )

    # ストリーミング風に出力
    partial = ""
    for char in response:
        partial += char
        yield partial


# モデル情報文字列
def get_model_info_text():
    if not MODEL_INFO:
        return "モデルが読み込まれていません"
    return (
        f"🧠 パラメータ: {MODEL_INFO.get('params', '?')}\n"
        f"📐 次元: embed={MODEL_INFO.get('embed_dim', '?')}, "
        f"heads={MODEL_INFO.get('num_heads', '?')}, "
        f"layers={MODEL_INFO.get('num_layers', '?')}\n"
        f"📏 最大シーケンス長: {MODEL_INFO.get('max_seq_len', '?')}\n"
        f"🎓 学習フェーズ: {MODEL_INFO.get('phase', '?')}\n"
        f"🖥️ デバイス: {MODEL_INFO.get('device', '?')}"
    )


# ========================================
# Gradio UI
# ========================================

THEME = gr.themes.Soft(
    primary_hue="violet",
    secondary_hue="blue",
    neutral_hue="slate",
    font=gr.themes.GoogleFont("Noto Sans JP"),
)

CSS = """
.gradio-container {
    max-width: 900px !important;
}
h1 {
    text-align: center;
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-size: 2.5em !important;
}
.model-info {
    background: linear-gradient(135deg, #1a1a2e, #16213e);
    border: 1px solid #0f3460;
    border-radius: 12px;
    padding: 16px;
    color: #e0e0e0;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.85em;
    white-space: pre-wrap;
}
footer { display: none !important; }
"""

with gr.Blocks(theme=THEME, css=CSS, title="🧠⚛️ NeuroQ チャット") as demo:

    gr.Markdown(
        """
        # 🧠⚛️ NeuroQ
        ### 脳型量子ビットネットワーク (QBNN) による日本語生成AI
        """
    )

    with gr.Row():
        with gr.Column(scale=4):
            chatbot = gr.ChatInterface(
                respond,
                type="messages",
                additional_inputs=[
                    gr.Textbox(
                        value="あなたは親切で知識豊富なAIアシスタントです。日本語で丁寧に回答してください。",
                        label="🎭 システムプロンプト",
                        lines=2,
                    ),
                    gr.Slider(
                        minimum=10, maximum=500, value=150, step=10,
                        label="📏 最大トークン数",
                    ),
                    gr.Slider(
                        minimum=0.1, maximum=2.0, value=0.8, step=0.1,
                        label="🌡️ Temperature",
                    ),
                    gr.Slider(
                        minimum=0.1, maximum=1.0, value=0.92, step=0.05,
                        label="🎯 Top-P",
                    ),
                    gr.Slider(
                        minimum=1, maximum=100, value=40, step=5,
                        label="🔝 Top-K",
                    ),
                    gr.Slider(
                        minimum=1.0, maximum=2.0, value=1.2, step=0.1,
                        label="🔄 繰り返しペナルティ",
                    ),
                ],
                examples=[
                    ["こんにちは！あなたは何ができますか？"],
                    ["日本語で俳句を作ってください"],
                    ["量子コンピューティングについて教えてください"],
                    ["東京の観光スポットを3つ教えてください"],
                ],
                fill_height=True,
            )

    with gr.Accordion("📊 モデル情報", open=False):
        gr.Markdown(
            f'<div class="model-info">{get_model_info_text()}</div>'
        )
        gr.Markdown(
            """
            ---
            **NeuroQ** は QBNN (Quantum Bit Neural Network) Transformer アーキテクチャを採用した
            実験的な日本語生成AIモデルです。独自の量子もつれ層により、通常のTransformerとは異なる
            表現学習を実現します。

            - 🏗️ アーキテクチャ: QBNN-Transformer
            - 📖 日本語3段階学習: 事前学習 → SFT → DPO
            - ⚛️ 量子もつれ層: Bloch球マッピング + 動的エンタングルメント
            - 🔤 トークナイザー: SentencePiece (8k語彙)

            [📂 GitHubリポジトリ](https://github.com/tapiocaTakeshi/NeuroQ)
            """
        )


if __name__ == "__main__":
    demo.launch()
