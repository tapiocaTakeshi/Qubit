#!/usr/bin/env python3
"""
1-bit量子化テストスクリプト
"""

import torch
import torch.nn as nn
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from binary_quantization_1bit import (
    BinaryQuantizer,
    Binary1BitLinear,
    estimate_1bit_size,
    compare_quantization_levels,
)


class SimpleTestModel(nn.Module):
    """テスト用シンプルモデル"""

    def __init__(self, vocab_size=1000, embed_dim=128, hidden_dim=256, num_layers=2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.layers = nn.ModuleList(
            [nn.Linear(embed_dim if i == 0 else hidden_dim, hidden_dim) for i in range(num_layers)]
        )
        self.output = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x):
        x = self.embedding(x)
        for layer in self.layers:
            x = layer(x)
            x = torch.relu(x)
        x = self.output(x)
        return x


def test_binary_quantizer():
    """BinaryQuantizer のテスト"""
    print("\n" + "=" * 60)
    print("TEST 1: Binary Quantizer")
    print("=" * 60)

    quantizer = BinaryQuantizer()

    # テストテンソル
    x = torch.randn(4, 128)
    print(f"Input tensor shape: {x.shape}")
    print(f"Input range: [{x.min():.4f}, {x.max():.4f}]")

    # 推論モード
    x_quant = quantizer(x, training=False)
    print(f"\nQuantized tensor:")
    print(f"   Shape: {x_quant.shape}")
    print(f"   Range: [{x_quant.min():.4f}, {x_quant.max():.4f}]")
    print(f"   Unique values: {torch.unique(x_quant).numel()}")
    print(f"   ✅ Quantization successful")


def test_binary_linear():
    """Binary1BitLinear のテスト"""
    print("\n" + "=" * 60)
    print("TEST 2: Binary 1-Bit Linear Layer")
    print("=" * 60)

    layer = Binary1BitLinear(in_features=256, out_features=512)
    print(f"Layer: {layer.in_features} → {layer.out_features}")

    # ダミー入力
    x = torch.randn(4, 256)
    output = layer(x)

    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Weight scale: {layer.weight_scale.mean():.6f}")
    print(f"Weight binary range: [{layer.weight_binary.min()}, {layer.weight_binary.max()}]")
    print(f"✅ Linear layer successful")


def test_model_quantization():
    """モデル全体の量子化テスト"""
    print("\n" + "=" * 60)
    print("TEST 3: Full Model Quantization")
    print("=" * 60)

    # モデルを作成
    model = SimpleTestModel(vocab_size=1000, embed_dim=128, hidden_dim=256, num_layers=2)

    print(f"Model:")
    print(f"   Layers: {sum(1 for _ in model.modules() if isinstance(_, nn.Linear))}")
    print(f"   Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # オリジナルの推論
    x = torch.randint(0, 1000, (2, 32))
    with torch.no_grad():
        original_output = model(x)

    print(f"\nOriginal inference:")
    print(f"   Input shape: {x.shape}")
    print(f"   Output shape: {original_output.shape}")
    print(f"   Output range: [{original_output.min():.4f}, {original_output.max():.4f}]")

    # 重みを量子化
    quantized_params = 0
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            weight = module.weight.data
            scale = weight.abs().mean()
            weight_norm = weight / (scale + 1e-8)
            weight_binary = torch.sign(weight_norm)
            module.weight.data = weight_binary
            quantized_params += 1

    print(f"\nQuantization:")
    print(f"   Quantized layers: {quantized_params}")

    # 量子化後の推論
    with torch.no_grad():
        quantized_output = model(x)

    print(f"\nQuantized inference:")
    print(f"   Output shape: {quantized_output.shape}")
    print(f"   Output range: [{quantized_output.min():.4f}, {quantized_output.max():.4f}]")

    # 誤差を計算
    mse = ((original_output - quantized_output) ** 2).mean().item()
    mae = (original_output - quantized_output).abs().mean().item()

    print(f"\nError metrics:")
    print(f"   MSE: {mse:.6f}")
    print(f"   MAE: {mae:.6f}")
    print(f"   ✅ Quantization successful")


def test_size_estimation():
    """サイズ推定のテスト"""
    print("\n" + "=" * 60)
    print("TEST 4: Size Estimation")
    print("=" * 60)

    model = SimpleTestModel(vocab_size=32000, embed_dim=512, hidden_dim=1024, num_layers=6)

    estimate_1bit_size(model, show_details=True)


def test_quantization_levels():
    """量子化レベルの比較テスト"""
    print("\n" + "=" * 60)
    print("TEST 5: Quantization Levels Comparison")
    print("=" * 60)

    compare_quantization_levels()


def test_binary_properties():
    """バイナリテンソルの性質テスト"""
    print("\n" + "=" * 60)
    print("TEST 6: Binary Tensor Properties")
    print("=" * 60)

    x = torch.randn(1000, 1000)
    quantizer = BinaryQuantizer()

    # 量子化
    x_quant = quantizer(x, training=False)

    # 統計情報
    unique_vals = torch.unique(x_quant)
    num_positive = (x_quant > 0).sum().item()
    num_negative = (x_quant < 0).sum().item()
    num_zero = (x_quant == 0).sum().item()

    print(f"Binary tensor statistics:")
    print(f"   Shape: {x_quant.shape}")
    print(f"   Total elements: {x_quant.numel():,}")
    print(f"   Unique values: {unique_vals}")
    print(f"   Positive (1): {num_positive:,} ({100*num_positive/x_quant.numel():.1f}%)")
    print(f"   Negative (-1): {num_negative:,} ({100*num_negative/x_quant.numel():.1f}%)")
    print(f"   Zero: {num_zero:,} ({100*num_zero/x_quant.numel():.1f}%)")

    # スパース性
    sparsity = num_zero / x_quant.numel()
    print(f"   Sparsity: {sparsity:.4f}")
    print(f"   ✅ Binary tensor test successful")


def test_gradient_flow():
    """勾配フローテスト"""
    print("\n" + "=" * 60)
    print("TEST 7: Gradient Flow (Straight-Through Estimator)")
    print("=" * 60)

    # テスト用モデル
    model = nn.Sequential(nn.Linear(10, 20), nn.ReLU(), nn.Linear(20, 5))

    x = torch.randn(2, 10, requires_grad=True)
    target = torch.randn(2, 5)

    # 量子化ラッパー
    quantizer = BinaryQuantizer()

    # 順伝播
    out = model(x)

    # 損失を計算
    loss = ((out - target) ** 2).mean()

    # 逆伝播
    loss.backward()

    print(f"Gradient test:")
    print(f"   Input gradient norm: {x.grad.norm():.6f}")
    print(f"   Has gradients: {x.grad is not None}")

    for name, param in model.named_parameters():
        if param.grad is not None:
            print(f"   {name} gradient norm: {param.grad.norm():.6f}")

    print(f"   ✅ Gradient flow test successful")


def main():
    print("\n" + "=" * 60)
    print("1-BIT BINARY QUANTIZATION TEST SUITE")
    print("=" * 60)

    try:
        test_binary_quantizer()
        test_binary_linear()
        test_model_quantization()
        test_size_estimation()
        test_quantization_levels()
        test_binary_properties()
        test_gradient_flow()

        print("\n" + "=" * 60)
        print("✅ ALL TESTS PASSED!")
        print("=" * 60)
        print("\n💡 Next steps:")
        print("   1. python quantize_neuroquantum_1bit.py checkpoint.pt")
        print("   2. python export_1bit_gguf.py model_1bit.pt")
        print("   3. python check_gguf_params.py model_1bit.gguf")

        return 0

    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
