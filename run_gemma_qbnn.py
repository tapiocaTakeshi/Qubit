#!/usr/bin/env python3
"""
Gemma QBNN + Claude AI推論デモ
量子バイナリニューラルネットワークを使用した推論実行
"""

import sys
import torch
import torch.nn.functional as F

sys.path.insert(0, '/home/user/Qubit')

from gemma_qbnn import create_gemma_qbnn_model

print("=" * 70)
print("🚀 Gemma QBNN + Claude AI デモ")
print("=" * 70)
print()

try:
    print("🔨 モデル作成中...")
    model = create_gemma_qbnn_model(size='small', vocab_size=32000)
    model.eval()

    print("✅ モデルロード完了")
    print(f"   パラメータ数: {sum(p.numel() for p in model.parameters()):,}")
    print()
    print("=" * 70)
    print("🧠 推論テスト")
    print("=" * 70)
    print()

    # テスト入力
    batch_size = 2
    seq_len = 32
    vocab_size = 32000

    with torch.no_grad():
        # 複数のテストプロンプト
        test_cases = [
            ("こんにちは", [12345, 23456]),
            ("量子コンピューティング", [12346, 23457, 34567]),
            ("AI助手のClaudeです", [12347, 23458, 34568, 45678]),
        ]

        for i, (prompt_text, token_ids) in enumerate(test_cases, 1):
            print(f"📝 テスト {i}: '{prompt_text}'")

            # パディング
            input_tokens = token_ids + [0] * (seq_len - len(token_ids))
            input_tensor = torch.tensor([input_tokens[:seq_len]] * batch_size)

            print(f"   入力トークン数: {len(token_ids)}")

            # 推論実行
            logits = model(input_tensor)

            # トークンの確率を計算
            probs = F.softmax(logits[:, -1, :], dim=-1)
            top_k = torch.topk(probs[0], k=5)

            print(f"   次のトークン候補（確率）:")
            for rank, (prob, token_id) in enumerate(zip(top_k.values, top_k.indices), 1):
                print(f"     {rank}. Token ID: {token_id.item():5d} (確率: {prob.item():.4f})")

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

    print()
    print("✅ 推論テスト完了！")

except Exception as e:
    print(f"❌ エラー: {e}")
    import traceback
    traceback.print_exc()

print()
print("=" * 70)
print("🎉 デモ終了")
print("=" * 70)
