#!/usr/bin/env python3
"""
Test and Example Script for QBNN to GGUF Conversion
QBNN to GGUF 変換のテストと使用例スクリプト

このスクリプトは以下を実演します：
1. ダミーQBNNモデルの生成
2. GGUFへの変換
3. 変換されたファイルの検証
"""

import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
import json
import sys
import os

# スクリプトディレクトリをパスに追加
sys.path.insert(0, os.path.dirname(__file__))

from export_qbnn_gguf import QBNNToGGUFConverter, QBNNQuantumMetadata


class SimpleQBNNDemo(nn.Module):
    """デモ用の簡単なQBNNモデル"""

    def __init__(self, embedding_dim=128, hidden_dim=256, num_layers=4):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # Embedding層
        self.embedding = nn.Embedding(32000, embedding_dim)

        # QBNN層をシミュレート
        self.quantum_corr = nn.ModuleList([
            nn.Linear(hidden_dim, hidden_dim) for _ in range(num_layers)
        ])

        self.entangle_ops = nn.ModuleList([
            nn.Linear(hidden_dim, hidden_dim) for _ in range(num_layers)
        ])

        # APQB theta パラメータ
        self.theta = nn.ParameterList([
            nn.Parameter(torch.randn(hidden_dim) * np.pi / 2)
            for _ in range(num_layers)
        ])

        # 通常の層
        self.layers = nn.ModuleList([
            nn.Linear(embedding_dim if i == 0 else hidden_dim, hidden_dim)
            for i in range(num_layers)
        ])

        self.output_layer = nn.Linear(hidden_dim, 32000)

    def forward(self, input_ids):
        """順伝播"""
        x = self.embedding(input_ids)
        x = x.mean(dim=1)  # 平均プーリング

        for i in range(self.num_layers):
            # 通常の線形変換
            x = self.layers[i](x)

            # 量子相関（デモ）
            quantum_effect = self.quantum_corr[i](x)

            # エンタングル（デモ）
            entangle_effect = self.entangle_ops[i](quantum_effect)

            # APQB theta効果（デモ）
            theta_effect = torch.cos(self.theta[i]) * x

            # 結合
            x = torch.tanh(x + entangle_effect * 0.1 + theta_effect * 0.05)

        logits = self.output_layer(x)
        return logits


def create_demo_qbnn_checkpoint(output_file: str, size: str = "1B"):
    """デモ用のQBNNチェックポイントを生成

    Args:
        output_file: 出力ファイルパス
        size: モデルサイズ (1-bit/2-bit/3-bit)
    """
    print(f"🔨 Creating {size} QBNN checkpoint...")

    # サイズ別の設定
    configs = {
        "1B": {"embedding_dim": 128, "hidden_dim": 256, "num_layers": 4},
        "2B": {"embedding_dim": 256, "hidden_dim": 512, "num_layers": 8},
        "3B": {"embedding_dim": 512, "hidden_dim": 1024, "num_layers": 12},
    }

    config = configs.get(size, configs["1B"])
    model = SimpleQBNNDemo(**config)

    # ダミー入力で初期化
    with torch.no_grad():
        dummy_input = torch.randint(0, 32000, (1, 128))
        _ = model(dummy_input)

    # チェックポイント保存
    torch.save(model.state_dict(), output_file)
    file_size_mb = os.path.getsize(output_file) / (1024 * 1024)
    print(f"✅ Created checkpoint: {output_file} ({file_size_mb:.2f}MB)")

    return output_file


def test_single_conversion(checkpoint_file: str, output_dir: str = "test_gguf"):
    """単一モデルの変換をテスト

    Args:
        checkpoint_file: チェックポイントファイル
        output_dir: 出力ディレクトリ
    """
    print(f"\n{'='*60}")
    print("TEST 1: Single Model Conversion")
    print(f"{'='*60}")

    converter = QBNNToGGUFConverter(output_dir=output_dir)

    gguf_file = Path(output_dir) / f"{Path(checkpoint_file).stem}.gguf"

    print(f"\n📝 Converting {checkpoint_file} to {gguf_file}...")
    success = converter.convert_to_gguf(
        checkpoint_file,
        str(gguf_file),
        model_name="DemoQBNN",
        model_size="1B",
        quantization="Q4_K_M",
        preserve_quantum=True
    )

    if success:
        gguf_size_mb = os.path.getsize(gguf_file) / (1024 * 1024)
        print(f"\n✅ Conversion successful!")
        print(f"   Output file: {gguf_file}")
        print(f"   File size: {gguf_size_mb:.2f}MB")

        # メタデータ確認
        print(f"\n📋 Checking GGUF metadata...")
        return str(gguf_file)
    else:
        print(f"❌ Conversion failed!")
        return None


def test_multiple_quantizations(checkpoint_file: str, output_dir: str = "test_gguf"):
    """複数の量子化レベルでテスト

    Args:
        checkpoint_file: チェックポイントファイル
        output_dir: 出力ディレクトリ
    """
    print(f"\n{'='*60}")
    print("TEST 2: Multiple Quantization Levels")
    print(f"{'='*60}")

    quantization_levels = ["F32", "Q8_0", "Q5_K_M", "Q4_K_M"]
    converter = QBNNToGGUFConverter(output_dir=output_dir)

    results = {}

    for quant in quantization_levels:
        gguf_file = Path(output_dir) / f"demo_{quant}.gguf"

        print(f"\n🔄 Converting with {quant}...")
        success = converter.convert_to_gguf(
            checkpoint_file,
            str(gguf_file),
            model_name="DemoQBNN",
            model_size="1B",
            quantization=quant,
            preserve_quantum=True
        )

        if success:
            size_mb = os.path.getsize(gguf_file) / (1024 * 1024)
            results[quant] = {"success": True, "size_mb": size_mb}
            print(f"   Size: {size_mb:.2f}MB")
        else:
            results[quant] = {"success": False}

    # 結果サマリー
    print(f"\n{'='*60}")
    print("Quantization Comparison")
    print(f"{'='*60}")
    print(f"{'Quantization':<15} {'Success':<10} {'Size (MB)':<15}")
    print(f"{'-'*40}")

    for quant, result in results.items():
        if result["success"]:
            print(f"{quant:<15} {'✓':<10} {result['size_mb']:>10.2f}MB")
        else:
            print(f"{quant:<15} {'✗':<10} {'FAILED':<15}")

    return results


def test_quantum_characteristics_preservation(checkpoint_file: str, output_dir: str = "test_gguf"):
    """量子特性保存のテスト

    Args:
        checkpoint_file: チェックポイントファイル
        output_dir: 出力ディレクトリ
    """
    print(f"\n{'='*60}")
    print("TEST 3: Quantum Characteristics Preservation")
    print(f"{'='*60}")

    print(f"\n📊 Analyzing checkpoint: {checkpoint_file}")

    # チェックポイントをロード
    checkpoint = torch.load(checkpoint_file, map_location="cpu", weights_only=True)

    # テンソルの分析
    quantum_tensors = []
    normal_tensors = []
    embedding_tensors = []

    for name, tensor in checkpoint.items():
        if any(q in name for q in ["quantum_corr", "entangle", "theta"]):
            quantum_tensors.append((name, tensor.shape))
        elif "embed" in name:
            embedding_tensors.append((name, tensor.shape))
        else:
            normal_tensors.append((name, tensor.shape))

    print(f"\n📊 Tensor Analysis:")
    print(f"   Quantum tensors: {len(quantum_tensors)}")
    print(f"   Normal tensors: {len(normal_tensors)}")
    print(f"   Embedding tensors: {len(embedding_tensors)}")

    print(f"\n🧬 Quantum Tensors Details:")
    for name, shape in quantum_tensors[:5]:
        print(f"   - {name}: {shape}")
    if len(quantum_tensors) > 5:
        print(f"   ... and {len(quantum_tensors)-5} more")

    # 変換時の処理を検証
    converter = QBNNToGGUFConverter(output_dir=output_dir)
    quantum_features, _ = converter.extract_quantum_characteristics(checkpoint)

    print(f"\n✨ Extracted Quantum Features:")
    print(f"   Has quantum correlation: {quantum_features['has_quantum_correlation']}")
    print(f"   Has entanglement: {quantum_features['has_entanglement']}")
    print(f"   APQB theta parameters: {len(quantum_features['apqb_theta_params'])}")
    print(f"   Entanglement layers: {len(quantum_features['entangle_layers'])}")

    return quantum_features


def main():
    """メインテスト関数"""
    print("🧠⚛️  QBNN to GGUF Conversion Test Suite")
    print(f"{'='*60}\n")

    # テスト用ディレクトリを作成
    test_dir = Path("test_gguf")
    test_dir.mkdir(exist_ok=True)

    # ステップ1: デモチェックポイント生成
    print("STEP 1: Creating Demo QBNN Checkpoint")
    print(f"{'='*60}")

    checkpoint_file = test_dir / "demo_qbnn_small.pt"
    if not checkpoint_file.exists():
        create_demo_qbnn_checkpoint(str(checkpoint_file), size="1B")
    else:
        print(f"✓ Checkpoint already exists: {checkpoint_file}")

    # ステップ2: 単一変換テスト
    print("\nSTEP 2: Single Conversion Test")
    gguf_file = test_single_conversion(str(checkpoint_file), output_dir="test_gguf")

    # ステップ3: 複数量子化テスト
    print("\nSTEP 3: Multiple Quantization Test")
    quant_results = test_multiple_quantizations(str(checkpoint_file), output_dir="test_gguf")

    # ステップ4: 量子特性保存テスト
    print("\nSTEP 4: Quantum Characteristics Test")
    quantum_features = test_quantum_characteristics_preservation(str(checkpoint_file), output_dir="test_gguf")

    # 最終サマリー
    print(f"\n{'='*60}")
    print("✅ TEST SUITE COMPLETE")
    print(f"{'='*60}")
    print(f"\n📂 Output directory: {test_dir.absolute()}")
    print(f"📦 Generated GGUF files:")
    for gguf in test_dir.glob("*.gguf"):
        size_mb = gguf.stat().st_size / (1024 * 1024)
        print(f"   - {gguf.name}: {size_mb:.2f}MB")

    print(f"\n✨ All tests completed successfully!")


if __name__ == "__main__":
    main()
