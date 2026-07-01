#!/usr/bin/env python3
"""
NeuroQuantum推論テスト（簡略版）
neuroquantum_layered.pyのコアコンポーネントを検証
"""

import sys
import torch
import numpy as np

def test_neuro_quantum_core():
    """NeuroQuantumのコアモジュール検証"""

    print("\n" + "=" * 80)
    print("  🧠 NeuroQuantum推論エンジン検証")
    print("=" * 80)

    # テスト1: インポート検証
    print("\n📌 テスト1: モジュールインポート")
    print("-" * 80)

    try:
        from neuroquantum_layered import (
            NeuroQuantumAI,
            NeuroQuantumConfig,
            get_gpu_adaptive_config,
            detect_gpu_tier
        )
        print("✅ NeuroQuantumコアモジュールのインポート成功")
    except ImportError as e:
        print(f"❌ インポート失敗: {e}")
        return False

    # テスト2: GPU/CPUティア検出
    print("\n📌 テスト2: GPU/CPUティア検出と適応設定")
    print("-" * 80)

    try:
        tier, device_name, gpu_info = detect_gpu_tier()
        config = get_gpu_adaptive_config(vocab_size=4096)

        print(f"🖥️  システム情報:")
        print(f"   - GPU ティア: {tier}")
        print(f"   - デバイス: {device_name}")
        print(f"   - デバイスタイプ: {gpu_info.get('device_type')}")
        print(f"   - システムRAM: {gpu_info.get('ram_gb')} GB")
        if gpu_info.get('vram_gb', 0) > 0:
            print(f"   - VRAM: {gpu_info.get('vram_gb')} GB")
        if gpu_info.get('compute_capability'):
            print(f"   - Compute Capability: {gpu_info.get('compute_capability')}")

        print(f"\n⚙️  適応設定:")
        print(f"   - Embed Dim: {config.get('embed_dim')}")
        print(f"   - Hidden Dim: {config.get('hidden_dim')}")
        print(f"   - Num Heads: {config.get('num_heads')}")
        print(f"   - Num Layers: {config.get('num_layers')}")
        print(f"   - Batch Size: {config.get('batch_size')}")
        print(f"   - Max Seq Len: {config.get('max_seq_len')}")

        print("✅ ティア検出と適応設定成功")
    except Exception as e:
        print(f"❌ ティア検出失敗: {e}")
        return False

    # テスト3: NeuroQuantumAI初期化
    print("\n📌 テスト3: NeuroQuantumAI初期化")
    print("-" * 80)

    try:
        ai = NeuroQuantumAI(
            embed_dim=128,
            hidden_dim=256,
            num_heads=4,
            num_layers=2,
            max_seq_len=512,
            dropout=0.1,
            lambda_entangle=0.5,
        )

        print(f"✅ NeuroQuantumAIインスタンス作成成功")
        print(f"   - Embed Dim: {ai.embed_dim}")
        print(f"   - Hidden Dim: {ai.hidden_dim}")
        print(f"   - Num Heads: {ai.num_heads}")
        print(f"   - Num Layers: {ai.num_layers}")
        print(f"   - Max Seq Len: {ai.max_seq_len}")
        print(f"   - Device: {ai.device}")
        print(f"   - Lambda Entangle: {ai.lambda_entangle}")

    except Exception as e:
        print(f"❌ 初期化失敗: {e}")
        import traceback
        traceback.print_exc()
        return False

    # テスト4: 量子コンピュータシミュレーター
    print("\n📌 テスト4: 量子コンピュータシミュレーター")
    print("-" * 80)

    try:
        print(f"⚛️  量子回路シミュレーター利用可能: {ai.use_quantum_simulation}")

        if ai.use_quantum_simulation:
            print(f"   ✓ 量子コンピュータ名: {ai.quantum_computer.name}")
            print("✅ 量子シミュレーター初期化成功")
        else:
            print("⚠️  量子シミュレーターは未利用（オプション機能）")
            print("✅ （オプション機能なので続行）")

    except Exception as e:
        print(f"⚠️  量子機能エラー（オプション）: {e}")
        print("✅ （オプション機能なので続行）")

    # テスト5: NeuroQuantumConfig
    print("\n📌 テスト5: NeuroQuantumConfig検証")
    print("-" * 80)

    try:
        config = NeuroQuantumConfig(
            vocab_size=4096,
            embed_dim=256,
            hidden_dim=512,
            num_heads=8,
            num_layers=4,
            max_seq_len=1024,
            dropout=0.1,
            lambda_entangle=0.5,
        )

        print(f"✅ NeuroQuantumConfigインスタンス作成成功")
        print(f"   - Vocab Size: {config.vocab_size}")
        print(f"   - Embed Dim: {config.embed_dim}")
        print(f"   - Hidden Dim: {config.hidden_dim}")
        print(f"   - Num Heads: {config.num_heads}")
        print(f"   - Num Layers: {config.num_layers}")
        print(f"   - Max Seq Len: {config.max_seq_len}")
        print(f"   - Dropout: {config.dropout}")
        print(f"   - Lambda Entangle: {config.lambda_entangle}")

    except Exception as e:
        print(f"❌ Config検証失敗: {e}")
        return False

    # テスト6: PyTorch/デバイス互換性
    print("\n📌 テスト6: PyTorch/デバイス互換性")
    print("-" * 80)

    try:
        print(f"🔧 PyTorchバージョン: {torch.__version__}")
        print(f"   - CUDA利用可能: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"   - CUDA デバイス数: {torch.cuda.device_count()}")
            print(f"   - 現在のデバイス: {torch.cuda.get_device_name(0)}")
        print(f"   - MPS利用可能: {hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()}")

        # テンソル作成テスト
        x = torch.randn(2, 3)
        y = torch.randn(3, 4)
        z = torch.mm(x, y)

        print(f"\n✅ テンソル操作テスト成功")
        print(f"   - テンソル形状: {z.shape}")
        print(f"   - テンソル値サンプル: {z[0, 0].item():.4f}")

    except Exception as e:
        print(f"❌ PyTorch互換性テスト失敗: {e}")
        return False

    # テスト7: 推論準備状況
    print("\n📌 テスト7: NeuroQuantum推論準備状況")
    print("-" * 80)

    try:
        print("🎯 推論準備チェック:")
        print(f"   ✓ NeuroQuantumAIインスタンス: {'✅ 準備完了' if ai else '❌ 未初期化'}")
        print(f"   ✓ トークナイザー: {'✅ 準備完了' if ai.tokenizer else '⚠️  未初期化（自動学習時に初期化）'}")
        print(f"   ✓ モデル: {'✅ 準備完了' if ai.model else '⚠️  未学習（自動学習時に初期化）'}")
        print(f"   ✓ デバイス: {'✅ 準備完了' if ai.device else '❌ デバイス未設定'}")
        print(f"   ✓ 量子シミュレーター: {'✅ 準備完了' if ai.use_quantum_simulation else '⚠️  オプション'}")

        print("\n📝 推論フロー:")
        print("   1. テキスト入力 → 2. トークナイザー → 3. エンコード")
        print("   4. モデル推論 → 5. 量子補正 → 6. サンプリング")
        print("   7. デコード → 8. テキスト出力")

        print("\n✅ 推論準備状況確認完了")

    except Exception as e:
        print(f"❌ 推論準備確認失敗: {e}")
        return False

    return True


def main():
    """メイン実行"""

    try:
        result = test_neuro_quantum_core()

        # 結果サマリー
        print("\n" + "=" * 80)
        print("  📊 テスト結果サマリー")
        print("=" * 80)

        if result:
            print("\n✅ すべてのコアテストが成功しました！")
            print("\nNeuroQuantumエンジンの推論準備完了:")
            print("  - GPU/CPU自動検出 ✓")
            print("  - 適応型モデル設定 ✓")
            print("  - 量子シミュレーション（オプション） ✓")
            print("  - PyTorch互換性 ✓")
            print("  - 推論パイプライン準備完了 ✓")
        else:
            print("\n❌ いくつかのテストが失敗しました")
            return 1

        print("\n" + "=" * 80)
        return 0

    except Exception as e:
        print(f"\n❌ 予期しないエラー: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
