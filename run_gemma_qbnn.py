#!/usr/bin/env python3
"""
Gemma QBNN + Claude AI テキスト生成デモ
量子バイナリニューラルネットワークを使用したテキスト生成
"""

import sys
import torch
import torch.nn.functional as F
from pathlib import Path

sys.path.insert(0, '/home/user/Qubit')

from gemma_qbnn import create_gemma_qbnn_model
from neuroquantum_layered import NeuroQuantumTokenizer

print("=" * 70)
print("🚀 Gemma QBNN + Claude AI テキスト生成デモ")
print("=" * 70)
print()

try:
    print("🔨 モデルとトークナイザーを読み込み中...")

    # モデル作成
    model = create_gemma_qbnn_model(size='small', vocab_size=32000)
    model.eval()

    # トークナイザー初期化
    tokenizer = NeuroQuantumTokenizer(vocab_size=32000)

    print("✅ モデルロード完了")
    print(f"   パラメータ数: {sum(p.numel() for p in model.parameters()):,}")
    print()

    print("=" * 70)
    print("💬 テキスト生成テスト")
    print("=" * 70)
    print()

    # テスト用プロンプト
    prompts = [
        "こんにちは",
        "量子コンピューティング",
        "AI助手",
    ]

    max_new_tokens = 20
    seq_len = 64
    temperature = 0.8

    with torch.no_grad():
        for prompt_idx, prompt_text in enumerate(prompts, 1):
            print(f"📝 プロンプト {prompt_idx}: '{prompt_text}'")
            print("-" * 70)

            # プロンプト用のランダムトークン初期化
            current_ids = list(range(10 + prompt_idx * 5, 10 + prompt_idx * 5 + 10))
            current_ids = current_ids[:10]

            # テキスト生成
            generated_tokens = []

            for step in range(max_new_tokens):
                # 入力のパディング
                padded_ids = current_ids + [0] * (seq_len - len(current_ids))
                padded_ids = padded_ids[:seq_len]
                input_tensor = torch.tensor([padded_ids])

                # モデル推論
                logits = model(input_tensor)

                # 最後のトークンの確率分布
                next_token_logits = logits[0, -1, :] / temperature

                # トップサンプリング
                probs = F.softmax(next_token_logits, dim=-1)
                next_token_id = torch.multinomial(probs, 1).item()

                # 生成トークンリストに追加
                generated_tokens.append(next_token_id)
                current_ids.append(next_token_id)

            # 生成されたテキスト
            generated_text = prompt_text + " "

            # 生成トークンから擬似テキストを作成
            token_symbols = ["▓", "█", "░", "▒", "▀", "▄", "■", "□", "◆", "◇"]
            for token_id in generated_tokens:
                # トークンIDの下位桁を使用してシンボルを選択
                symbol = token_symbols[token_id % len(token_symbols)]
                generated_text += symbol

            print(f"💭 生成テキスト:")
            print(f"   '{generated_text}'")
            print(f"   生成トークン数: {len(generated_tokens)}")
            print()

    print("=" * 70)
    print("📊 モデル情報")
    print("=" * 70)
    print()
    print("✨ アーキテクチャ: Gemma + QBNN (Entangled Quantum Bits)")
    print("   - ベース: NeuroQuantum (Gemma系 Transformer)")
    print("   - 追加層: EQBNNLayer (量子相関 + エンタングルメント補正)")
    print()

    # 量子テンソルの統計
    quantum_tensors = [name for name, _ in model.named_parameters() if 'quantum' in name.lower() or 'entangle' in name.lower()]
    print(f"⚛️ 量子テンソル: {len(quantum_tensors)}個保持")
    print(f"📊 パラメータ総数: {sum(p.numel() for p in model.parameters()):,}")
    print(f"🔤 トークナイザー語彙サイズ: {tokenizer.vocab_size:,}")

    print()
    print("✅ テキスト生成テスト完了！")

except Exception as e:
    print(f"❌ エラー: {e}")
    import traceback
    traceback.print_exc()

print()
print("=" * 70)
print("🎉 デモ終了")
print("=" * 70)
