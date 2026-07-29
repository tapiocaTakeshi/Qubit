#!/usr/bin/env python3
"""
megabyte_100mb 推論スクリプト
チェックポイントから日本語テキストを生成
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

def load_model(checkpoint_path: str, tokenizer_path: str):
    """train_hf_dataset.py が保存したチェックポイントからモデル/トークナイザーを復元する。

    Returns:
        (model, tokenizer, config_dict, device)
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    config_dict = checkpoint["config"]

    tokenizer = NeuroQuantumTokenizer(vocab_size=config_dict["vocab_size"], model_file=tokenizer_path)

    config = NeuroQuantumConfig(
        vocab_size=config_dict["vocab_size"],
        embed_dim=config_dict["embed_dim"],
        hidden_dim=config_dict["hidden_dim"],
        num_heads=config_dict["num_heads"],
        num_layers=config_dict["num_layers"],
        max_seq_len=config_dict["max_seq_len"],
        dropout=config_dict["dropout"],
        lambda_entangle=config_dict["lambda_entangle"],
    )

    model = NeuroQuantum(config=config).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    return model, tokenizer, config_dict, device


def find_latest_checkpoint():
    """最低ロスのチェックポイントを探す"""
    checkpoint_dir = Path("./checkpoints")

    if not checkpoint_dir.exists():
        return None

    # SFT best を優先（最低ロス）
    sft_best_files = sorted(checkpoint_dir.glob("*sft_best.pt"))
    if sft_best_files:
        return sft_best_files[-1]

    # その他の best を探す
    best_files = sorted(checkpoint_dir.glob("*_best.pt"))
    if best_files:
        return best_files[-1]

    # マージされたチェックポイント
    merged_files = sorted(checkpoint_dir.glob("*merged.pt"))
    if merged_files:
        return merged_files[-1]

    # 通常のチェックポイント
    checkpoint_files = sorted(checkpoint_dir.glob("*.pt"))
    if checkpoint_files:
        return checkpoint_files[-1]

    return None

def generate_text(model, tokenizer, prompt: str, max_length: int = 100, device: str = "cuda"):
    """テキスト生成"""
    model.eval()

    prompt_ids = tokenizer.encode(prompt, add_special=False)

    if not prompt_ids:
        return ""

    input_ids = [tokenizer.bof_id, tokenizer.bos_id] + prompt_ids
    generated = input_ids.copy()

    with torch.no_grad():
        for _ in range(max_length):
            seq_len = len(generated)
            pad_len = max(512 - seq_len, 0)
            padded = generated + [tokenizer.pad_id] * pad_len
            padded = padded[:512]

            input_tensor = torch.tensor([padded], dtype=torch.long, device=device)

            try:
                logits = model(input_tensor)
                last_logits = logits[0, seq_len-1, :]
                next_token = torch.argmax(last_logits, dim=-1).item()

                if next_token == tokenizer.eos_id:
                    break

                generated.append(next_token)
            except Exception as e:
                print(f"      (エラー: {e})")
                break

    try:
        generated_text = tokenizer.decode(generated)
        return generated_text
    except:
        return ""

def main():
    print("\n" + "=" * 80)
    print("🤖 NeuroQuantum megabyte_100mb - テキスト生成（推論）")
    print("=" * 80)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n✓ Device: {device}")

    print("\n📂 チェックポイント検索中...")
    checkpoint_path = find_latest_checkpoint()

    if not checkpoint_path:
        print("❌ チェックポイントが見つかりません")
        return

    print(f"   ✓ 見つかったチェックポイント: {checkpoint_path.name}")
    print(f"   サイズ: {checkpoint_path.stat().st_size / (1024**2):.1f}MB")

    config_dict = get_model_config_by_size("megabyte_100mb", vocab_size=32000)
    config = NeuroQuantumConfig(
        vocab_size=32000,
        embed_dim=1024,
        hidden_dim=2048,
        num_heads=16,
        num_layers=10,
        max_seq_len=512,
        dropout=0.1,
        lambda_entangle=0.5,
    )

    print("\n🔤 トークナイザーを初期化中...")
    tokenizer = NeuroQuantumTokenizer(vocab_size=32000)

    print("\n🧠 モデルを読み込み中...")
    model = NeuroQuantum(config=config, tokenizer=tokenizer).to(device)

    try:
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint)
        print("   ✓ チェックポイント読み込み完了")
    except Exception as e:
        print(f"   ❌ エラー: {e}")
        return

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"   パラメータ数: {n_params:,}")

    print("\n" + "=" * 80)
    print("💬 テキスト生成デモ")
    print("=" * 80)

    prompts = [
        "こんにちは",
        "日本語",
        "人工知能",
        "自然言語処理は",
        "今日の天気は",
    ]

    for prompt in prompts:
        print(f"\n📝 プロンプト: '{prompt}'")
        print("   生成中...", end="", flush=True)

        generated = generate_text(model, tokenizer, prompt, max_length=50, device=device)

        if generated:
            print(f"\n   ✓ 生成結果:")
            print(f"   \"{generated[:200]}\"")
        else:
            print("   ❌ テキスト生成に失敗しました")

    print("\n" + "=" * 80)
    print("✅ 推論デモ完了")
    print("=" * 80)

if __name__ == "__main__":
    main()
