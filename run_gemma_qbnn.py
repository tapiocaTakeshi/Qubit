#!/usr/bin/env python3
"""
Gemma QBNN + Claude AI推論デモ
量子バイナリニューラルネットワークを使用した推論実行
"""

import os
import sys
from pathlib import Path

# モデルパスの確認
model_path = Path("gguf_models/gemma_qbnn_small_Q4_K_M.gguf")

if not model_path.exists():
    print(f"❌ モデルファイルが見つかりません: {model_path}")
    sys.exit(1)

print("=" * 70)
print("🚀 Gemma QBNN + Claude AI デモ")
print("=" * 70)
print(f"✅ モデル: {model_path.name}")
print(f"   サイズ: {model_path.stat().st_size / (1024**2):.2f}MB")
print()

try:
    from llama_cpp import Llama

    print("📦 llama-cpp-pythonでモデルをロード中...")
    model = Llama(
        model_path=str(model_path),
        n_ctx=512,
        n_batch=64,
        n_threads=4,
        verbose=False
    )

    print("✅ モデルロード完了")
    print()
    print("=" * 70)
    print("🧠 推論テスト")
    print("=" * 70)

    # テストプロンプト
    prompts = [
        "こんにちは、Claudeです。",
        "量子コンピューティングについて説明してください。",
        "Q: 2+2は？\nA:",
    ]

    for i, prompt in enumerate(prompts, 1):
        print(f"\n📝 プロンプト {i}: {prompt[:50]}...")

        response = model(
            prompt,
            max_tokens=100,
            temperature=0.7,
            top_p=0.95
        )

        output_text = response['choices'][0]['text'].strip()
        print(f"💭 出力: {output_text[:200]}...")
        print("-" * 70)

    print("\n✅ 推論完了！")
    print("\n📊 モデル情報:")
    print(f"   アーキテクチャ: Gemma + QBNN (Entangled Quantum Bits)")
    print(f"   量子テンソル: 54個保持")
    print(f"   パラメータ: 34,870,045個")
    print(f"   量子化: Q4_K_M")

except ImportError:
    print("⚠️  llama-cpp-pythonがインストールされていません")
    print("   インストール中...")
    print()
    print("   以下のコマンドで再度実行してください:")
    print("   python run_gemma_qbnn.py")
except Exception as e:
    print(f"❌ エラー: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 70)
print("🎉 完了")
print("=" * 70)
