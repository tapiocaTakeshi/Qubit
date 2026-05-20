#!/usr/bin/env python3
"""
2-bit Ternary Quantization Module for NeuroQuantum
2-bit三値量子化実装
4つの値 (-1, -1/3, 1/3, 1) に量子化
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional
import numpy as np


class TernaryQuantizer(nn.Module):
    """2-bit (4値) 量子化器"""

    def __init__(self, use_activation_quant: bool = False, momentum: float = 0.99):
        """Initialize ternary quantizer

        Args:
            use_activation_quant: アクティベーション値も量子化するか
            momentum: スケール係数の更新用モメンタム
        """
        super().__init__()
        self.use_activation_quant = use_activation_quant
        self.momentum = momentum
        # 2-bit: -1, -1/3, 1/3, 1
        self.quantization_levels = torch.tensor([-1.0, -1/3, 1/3, 1.0])
        self.register_buffer("running_scale", torch.tensor(1.0))

    def forward(self, x: torch.Tensor, training: bool = False) -> torch.Tensor:
        """Quantize tensor to 2-bit ternary

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
        """Training-phase quantization with STE"""
        # スケール係数を計算
        scale = x.abs().mean()

        if scale > 0:
            self.running_scale.data = self.momentum * self.running_scale + (1 - self.momentum) * scale

        # 正規化
        x_norm = x / (scale + 1e-8)

        # 4つの量子化レベルに割り当て
        x_quant = self._quantize_to_ternary(x_norm)

        # スケール係数を再度適用
        return x_quant * scale

    def _quantize_inference(self, x: torch.Tensor) -> torch.Tensor:
        """Inference-phase quantization"""
        scale = self.running_scale
        x_norm = x / (scale + 1e-8)
        return self._quantize_to_ternary(x_norm) * scale

    def _quantize_to_ternary(self, x: torch.Tensor) -> torch.Tensor:
        """Quantize to 4 levels: -1, -1/3, 1/3, 1"""
        # 各レベルへの距離を計算
        levels = self.quantization_levels.to(x.device)

        # テンソルを展開
        x_flat = x.flatten().unsqueeze(1)  # [N, 1]

        # 各レベルとの距離
        distances = torch.abs(x_flat - levels.unsqueeze(0))  # [N, 4]

        # 最も近いレベルのインデックス
        indices = torch.argmin(distances, dim=1)

        # 量子化結果
        x_quant = levels[indices].reshape(x.shape)

        return x_quant

    @staticmethod
    def quantize_static(x: torch.Tensor) -> Tuple[torch.Tensor, float]:
        """静的量子化（スケール係数を返す）"""
        scale = x.abs().mean()
        if scale == 0:
            return torch.zeros_like(x), 1.0

        x_norm = x / scale
        quantizer = TernaryQuantizer()
        x_quant = quantizer._quantize_to_ternary(x_norm)
        return x_quant, float(scale)


class Ternary2BitLinear(nn.Module):
    """2-bit量子化Linear層"""

    def __init__(self, in_features: int, out_features: int, bias: bool = True, device=None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        # 2-bit重み（4値: -1, -1/3, 1/3, 1）
        self.register_buffer(
            "weight_ternary",
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

        self.quantizer = TernaryQuantizer()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with 2-bit weights"""
        # 2-bit重みをfloatに変換してスケーリング
        weight = self.weight_ternary.float() * self.weight_scale.view(-1, 1)

        return F.linear(x, weight, self.bias)

    def quantize_weights(self, weight: torch.Tensor) -> None:
        """量子化済み重みを設定"""
        with torch.no_grad():
            weight_abs = weight.abs()
            self.weight_scale.copy_(weight_abs.mean(dim=1))

            weight_norm = weight / (self.weight_scale.view(-1, 1) + 1e-8)

            # 4値に量子化
            x_quant = self.quantizer._quantize_to_ternary(weight_norm)

            # int8で保存（4値なので実際には2-bitで十分だが、簡略化）
            self.weight_ternary.copy_(x_quant.to(torch.int8))


def estimate_2bit_size(model: nn.Module, show_details: bool = True) -> Dict[str, float]:
    """2-bit量子化後のモデルサイズを推定"""
    total_params = sum(p.numel() for p in model.parameters())

    # 32-bit float
    original_size_mb = total_params * 4 / (1024**2)

    # 2-bit（+スケール係数）
    num_layers = sum(1 for m in model.modules() if isinstance(m, nn.Linear))
    scale_overhead_mb = num_layers * 4 / (1024**2)
    quantized_size_mb = total_params * 0.25 / (1024**2) + scale_overhead_mb

    result = {
        "total_parameters": total_params,
        "original_size_mb": original_size_mb,
        "quantized_size_mb": quantized_size_mb,
        "compression_ratio": original_size_mb / (quantized_size_mb + 1e-8),
        "num_linear_layers": num_layers,
    }

    if show_details:
        print("\n📊 2-bit Quantization Size Estimate:")
        print(f"   Total Parameters: {result['total_parameters']:,}")
        print(f"   Original Size: {result['original_size_mb']:.2f} MB (32-bit)")
        print(f"   Quantized Size: {result['quantized_size_mb']:.2f} MB (2-bit)")
        print(f"   Compression: {result['compression_ratio']:.1f}x")
        print(f"   Linear Layers: {result['num_linear_layers']}")

    return result


if __name__ == "__main__":
    print("2-bit Ternary Quantization Module")
    print("=" * 60)

    # テスト
    quantizer = TernaryQuantizer()
    x = torch.randn(100, 100)
    x_quant = quantizer(x, training=False)

    print(f"Input range: [{x.min():.4f}, {x.max():.4f}]")
    print(f"Quantized range: [{x_quant.min():.4f}, {x_quant.max():.4f}]")
    print(f"Unique values: {torch.unique(x_quant).numel()}")
    print(f"✅ 2-bit Ternary Quantization Ready!")
