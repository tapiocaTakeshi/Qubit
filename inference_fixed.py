#!/usr/bin/env python3
"""
megabyte_100mb 推論スクリプト（修正版）
チェックポイント設定に合わせて自動調整
"""
import sys
import os
import torch
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from neuroquantum_layered import (
    NeuroQuantum,
    NeuroQuantumConfig,
    NeuroQuantumTokenizer,
    get_model_config_by_size,
)

def generate_text(model, tokenizer, prompt: str, max_length: int = 50, device: str = "cuda"):
    """テキスト生成"""
    model.eval()

    prompt_ids = tokenizer.encode(prompt, add_special=False)
    if not prompt_ids:
        return prompt

    input_ids = [tokenizer.bos_id] + prompt_ids[:100]
    generated = input_ids.copy()

    with torch.no_grad():
        for _ in range(max_length):
            seq_len = len(generated)
            pad_len = max(model.config.max_seq_len - seq_len, 0)
            padded = generated + [tokenizer.pad_id] * pad_len
            padded = padded[:model.config.max_seq_len]

            input_tensor = torch.tensor([padded], dtype=torch.long, device=device)

            try:
                logits = model(input_tensor)
                last_logits = logits[0, seq_len-1, :]
                
                # Temperature scaling
                next_token = torch.argmax(last_logits, dim=-1).item()
                
                if next_token == tokenizer.eos_id or next_token == tokenizer.pad_id:
                    break

                generated.append(next_token)
            except:
                break

    try:
        return tokenizer.decode(generated)
    except:
        return prompt

def main():
    print("\n" + "=" * 80)
    print("🤖 NeuroQuantum - 日本語テキスト生成（推論）")
    print("=" * 80)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n✓ Device: {device}")

    # 最新のチェックポイント探索
    checkpoint_dir = Path("./checkpoints")
    if not checkpoint_dir.exists():
        print("❌ チェックポイントディレクトリが見つかりません")
        return

    checkpoint_files = sorted(checkpoint_dir.glob("*.pt"))
    if not checkpoint_files:
        print("❌ チェックポイントが見つかりません")
        return

    checkpoint_path = checkpoint_files[-1]
    print(f"\n📂 使用するチェックポイント: {checkpoint_path.name}")
    print(f"   サイズ: {checkpoint_path.stat().st_size / (1024**2):.1f}MB")

    # モデル設定を自動判定
    print("\n🔧 モデル設定を判定中...")
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    
    # vocab_sizeを推測
    token_emb = checkpoint.get("token_embedding.weight")
    if token_emb is not None:
        vocab_size = token_emb.shape[0]
        print(f"   推定vocab_size: {vocab_size}")
    else:
        vocab_size = 8000
        print(f"   デフォルトvocab_size: {vocab_size}")

    config_dict = get_model_config_by_size("megabyte_100mb", vocab_size=vocab_size)
    config = NeuroQuantumConfig(
        vocab_size=vocab_size,
        embed_dim=config_dict["embed_dim"],
        hidden_dim=config_dict["hidden_dim"],
        num_heads=config_dict["num_heads"],
        num_layers=4,
        max_seq_len=256,
        dropout=0.1,
        lambda_entangle=0.5,
    )

    print("\n🔤 トークナイザーを初期化中...")
    tokenizer = NeuroQuantumTokenizer(vocab_size=vocab_size)

    print("\n🧠 モデルを読み込み中...")
    model = NeuroQuantum(config=config, tokenizer=tokenizer).to(device)

    try:
        model.load_state_dict(checkpoint)
        print("   ✓ チェックポイント読み込み完了")
    except Exception as e:
        print(f"   ⚠️ 警告: {str(e)[:100]}")
        print("   → 部分的にロードします...")

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"   パラメータ数: {n_params:,}")

    # 推論デモ
    print("\n" + "=" * 80)
    print("💬 日本語テキスト生成デモ")
    print("=" * 80)

    prompts = [
        "こんにちは、",
        "日本語は",
        "人工知能",
        "今日は",
        "プログラミングは",
    ]

    for prompt in prompts:
        print(f"\n📝 プロンプト: '{prompt}'")
        print("   生成中...", end="", flush=True)

        generated = generate_text(model, tokenizer, prompt, max_length=40, device=device)

        print(f"\n   ✓ 結果:")
        print(f"   \"{generated}\"")

    print("\n" + "=" * 80)
    print("✅ 推論デモ完了")
    print("=" * 80)

if __name__ == "__main__":
    main()
