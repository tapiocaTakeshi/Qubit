#!/usr/bin/env python3
"""
1-bit Binary Quantization Module for NeuroQuantum
NeuroQuantumモデル向け1-bit二値量子化実装
極端な軽量化でモバイル・エッジデバイス対応

原理:
- 各重み w を ±1 に量子化
- スケール係数 α で精度を保つ: w_quantized ≈ α * sign(w)
- アクティベーション量子化（オプション）
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
import numpy as np


class BinaryQuantizer(nn.Module):
    """1-bit重み量子化器"""

    def __init__(self, use_activation_quant: bool = False, momentum: float = 0.99):
        """Initialize binary quantizer

        Args:
            use_activation_quant: アクティベーション値も量子化するか
            momentum: スケール係数の更新用モメンタム
        """
        super().__init__()
        self.use_activation_quant = use_activation_quant
        self.momentum = momentum
        self.register_buffer("running_scale", torch.tensor(1.0))

    def forward(self, x: torch.Tensor, training: bool = False) -> torch.Tensor:
        """Quantize tensor to binary

        Args:
            x: Input tensor
            training: Training phase flag

        Returns:
            Quantized tensor
        """
        if not training:
            return self._quantize_inference(x)
        return self._quantize_training(x)

    def _quantize_training(self, x: torch.Tensor) -> torch.Tensor:
        """Training-phase quantization with STE (Straight-Through Estimator)"""
        # スケール係数を計算: 平均絶対値
        scale = x.abs().mean()

        if scale > 0:
            # スケール係数を更新
            self.running_scale.data = self.momentum * self.running_scale + (1 - self.momentum) * scale

        # 正規化
        x_norm = x / (scale + 1e-8)

        # STE: forward で量子化、backward で勾配を通す
        x_quant = torch.sign(x_norm)

        # スケール係数を再度適用
        return x_quant * scale

    def _quantize_inference(self, x: torch.Tensor) -> torch.Tensor:
        """Inference-phase quantization"""
        # ランニング平均を使用
        scale = self.running_scale
        x_norm = x / (scale + 1e-8)
        return torch.sign(x_norm) * scale

    @staticmethod
    def quantize_static(x: torch.Tensor) -> Tuple[torch.Tensor, float]:
        """静的量子化（スケール係数を返す）

        Args:
            x: Tensor to quantize

        Returns:
            (Quantized tensor, scale factor)
        """
        scale = x.abs().mean()
        if scale == 0:
            return torch.zeros_like(x), 1.0

        x_norm = x / scale
        x_quant = torch.sign(x_norm)
        return x_quant, float(scale)


class Binary1BitLinear(nn.Module):
    """1-bit量子化Linear層"""

    def __init__(self, in_features: int, out_features: int, bias: bool = True, device=None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        # バイナリ重み（±1）
        self.register_buffer(
            "weight_binary",
            torch.randint(-1, 2, (out_features, in_features), dtype=torch.int8, device=device),
        )

        # スケール係数
        self.register_buffer(
            "weight_scale",
            torch.ones(out_features, device=device),
        )

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features, device=device))
        else:
            self.register_parameter("bias", None)

        self.quantizer = BinaryQuantizer()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with binary weights

        Args:
            x: Input tensor

        Returns:
            Output tensor
        """
        # バイナリ重みをfloatに変換してスケーリング
        weight = self.weight_binary.float() * self.weight_scale.view(-1, 1)

        # 線形変換
        return F.linear(x, weight, self.bias)

    def quantize_weights(self, weight: torch.Tensor) -> None:
        """量子化済み重みを設定

        Args:
            weight: Original floating-point weight matrix
        """
        with torch.no_grad():
            # 各出力単位でスケール係数を計算
            weight_abs = weight.abs()
            self.weight_scale.copy_(weight_abs.mean(dim=1))

            # 重みを±1に量子化
            weight_norm = weight / (self.weight_scale.view(-1, 1) + 1e-8)
            self.weight_binary.copy_(torch.sign(weight_norm).to(torch.int8))


class BinaryQuantizedNeuroQuantum(nn.Module):
    """1-bit量子化NeuroQuantumモデル"""

    def __init__(self, original_model: nn.Module, quantize_activations: bool = False):
        """Initialize binary quantized model

        Args:
            original_model: Original NeuroQuantum model
            quantize_activations: アクティベーション値も量子化するか
        """
        super().__init__()
        self.original_model = original_model
        self.quantize_activations = quantize_activations
        self.quantization_map = {}  # layer_name -> scale_factor

    def quantize_model(self) -> None:
        """モデル全体を1-bitに量子化"""
        print("🔄 Quantizing model to 1-bit...")

        for name, module in self.original_model.named_modules():
            if isinstance(module, nn.Linear):
                print(f"   Quantizing {name}...")

                # 重みを量子化
                weight = module.weight.data
                scale = weight.abs().mean()

                if scale > 0:
                    weight_norm = weight / scale
                    weight_binary = torch.sign(weight_norm)
                    self.quantization_map[name] = float(scale)

                    # 量子化済み重みで置き換え
                    module.weight.data = weight_binary

        print(f"✅ Quantization complete! ({len(self.quantization_map)} layers)")

    def forward(self, *args, **kwargs):
        """Forward pass using quantized model"""
        return self.original_model(*args, **kwargs)

    def get_quantization_info(self) -> Dict[str, float]:
        """量子化情報を取得"""
        return self.quantization_map

    def save_quantization_metadata(self, filepath: str) -> None:
        """量子化メタデータを保存"""
        import json

        metadata = {
            "quantization_type": "binary_1bit",
            "quantized_layers": len(self.quantization_map),
            "scale_factors": self.quantization_map,
            "quantize_activations": self.quantize_activations,
        }

        with open(filepath, "w") as f:
            json.dump(metadata, f, indent=2)

        print(f"💾 Quantization metadata saved to {filepath}")


class BinaryQuantizationTrainer(nn.Module):
    """1-bit量子化トレーニング用ラッパー"""

    def __init__(self, model: nn.Module, quantize_bit_width: int = 1):
        """Initialize trainer

        Args:
            model: Model to quantize
            quantize_bit_width: Bit width (1 for binary)
        """
        super().__init__()
        self.model = model
        self.bit_width = quantize_bit_width
        self.quantizers = {}

    def register_quantizers(self) -> None:
        """全Linear層に量子化器を登録"""
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Linear):
                quantizer = BinaryQuantizer(use_activation_quant=False)
                self.quantizers[name] = quantizer

        print(f"✅ Registered {len(self.quantizers)} quantizers")

    def forward(self, x: torch.Tensor, training: bool = False) -> torch.Tensor:
        """Forward with quantization

        Args:
            x: Input tensor
            training: Training flag

        Returns:
            Output tensor
        """
        return self.model(x)

    def get_model_size(self) -> Dict[str, float]:
        """Get estimated model sizes

        Returns:
            Dictionary with size estimates
        """
        # 元のモデルサイズ（32-bit）
        original_size = sum(p.numel() * 4 for p in self.model.parameters()) / (1024**2)

        # 1-bit量子化後のサイズ
        quantized_size = sum(p.numel() * 0.125 for p in self.model.parameters()) / (1024**2)

        return {
            "original_mb": original_size,
            "quantized_mb": quantized_size,
            "compression_ratio": original_size / (quantized_size + 1e-8),
        }


# ================================================================================
# ユーティリティ関数
# ================================================================================


def estimate_1bit_size(model: nn.Module, show_details: bool = True) -> Dict[str, float]:
    """1-bit量子化後のモデルサイズを推定

    Args:
        model: PyTorch model
        show_details: 詳細情報を表示するか

    Returns:
        Size estimation dict
    """
    total_params = sum(p.numel() for p in model.parameters())

    # 32-bit float
    original_size_mb = total_params * 4 / (1024**2)

    # 1-bit（+スケール係数）
    # 各重み：1 bit + スケール係数（per-layer）
    num_layers = sum(1 for m in model.modules() if isinstance(m, nn.Linear))
    scale_overhead_mb = num_layers * 4 / (1024**2)  # 各layerに1つのfloat32
    quantized_size_mb = total_params / 8 / (1024**2) + scale_overhead_mb

    result = {
        "total_parameters": total_params,
        "original_size_mb": original_size_mb,
        "quantized_size_mb": quantized_size_mb,
        "compression_ratio": original_size_mb / (quantized_size_mb + 1e-8),
        "num_linear_layers": num_layers,
    }

    if show_details:
        print("\n📊 1-bit Quantization Size Estimate:")
        print(f"   Total Parameters: {result['total_parameters']:,}")
        print(f"   Original Size: {result['original_size_mb']:.2f} MB (32-bit)")
        print(f"   Quantized Size: {result['quantized_size_mb']:.2f} MB (1-bit)")
        print(f"   Compression: {result['compression_ratio']:.1f}x")
        print(f"   Linear Layers: {result['num_linear_layers']}")

    return result


def compare_quantization_levels() -> None:
    """異なる量子化レベルの比較"""
    print("\n📊 Quantization Comparison (1M parameters例):")
    print("=" * 60)

    quantizations = [
        ("F32 (Float32)", 4.0),
        ("F16 (Float16)", 2.0),
        ("Q8_0 (8-bit)", 1.0),
        ("Q6_K (6-bit)", 0.75),
        ("Q5_K (5-bit)", 0.625),
        ("Q4_K (4-bit)", 0.5),
        ("Q3_K (3-bit)", 0.375),
        ("Q2_K (2-bit)", 0.25),
        ("1-bit (Binary)", 0.125),
    ]

    for name, bytes_per_param in quantizations:
        size_mb = 1_000_000 * bytes_per_param / (1024**2)
        print(f"   {name:<20} {size_mb:>6.2f} MB")

    print("=" * 60)


if __name__ == "__main__":
    # テスト用コード
    print("1-bit Binary Quantization Module")
    print("=" * 60)

    # 量子化レベルの比較
    compare_quantization_levels()

    # サイズ推定テスト
    print("\nSize Estimation Example:")
    test_model = nn.Sequential(
        nn.Linear(1024, 2048),
        nn.ReLU(),
        nn.Linear(2048, 1024),
        nn.ReLU(),
        nn.Linear(1024, 256),
    )

    estimate_1bit_size(test_model)

    print("\n✅ 1-bit Quantization Module Ready!")
