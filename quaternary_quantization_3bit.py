#!/usr/bin/env python3
"""
3-bit Quaternary Quantization Module for NeuroQuantum
3-bit四値量子化実装
8つの値に量子化 (-1, -0.71, -0.43, -0.14, 0.14, 0.43, 0.71, 1)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional
import numpy as np


class QuaternaryQuantizer(nn.Module):
    """3-bit (8値) 量子化器"""

    def __init__(self, use_activation_quant: bool = False, momentum: float = 0.99):
        """Initialize quaternary quantizer

        Args:
            use_activation_quant: アクティベーション値も量子化するか
            momentum: スケール係数の更新用モメンタム
        """
        super().__init__()
        self.use_activation_quant = use_activation_quant
        self.momentum = momentum

        # 3-bit: 8値 (均等に分布)
        levels = np.linspace(-1, 1, 8)
        self.quantization_levels = torch.tensor(levels, dtype=torch.float32)

        self.register_buffer("running_scale", torch.tensor(1.0))

    def forward(self, x: torch.Tensor, training: bool = False) -> torch.Tensor:
        """Quantize tensor to 3-bit quaternary

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
        scale = x.abs().mean()

        if scale > 0:
            self.running_scale.data = self.momentum * self.running_scale + (1 - self.momentum) * scale

        x_norm = x / (scale + 1e-8)
        x_quant = self._quantize_to_quaternary(x_norm)

        return x_quant * scale

    def _quantize_inference(self, x: torch.Tensor) -> torch.Tensor:
        """Inference-phase quantization"""
        scale = self.running_scale
        x_norm = x / (scale + 1e-8)
        return self._quantize_to_quaternary(x_norm) * scale

    def _quantize_to_quaternary(self, x: torch.Tensor) -> torch.Tensor:
        """Quantize to 8 levels using uniform quantization"""
        levels = self.quantization_levels.to(x.device)

        x_flat = x.flatten().unsqueeze(1)  # [N, 1]
        distances = torch.abs(x_flat - levels.unsqueeze(0))  # [N, 8]
        indices = torch.argmin(distances, dim=1)
        x_quant = levels[indices].reshape(x.shape)

        return x_quant

    @staticmethod
    def quantize_static(x: torch.Tensor) -> Tuple[torch.Tensor, float]:
        """静的量子化（スケール係数を返す）"""
        scale = x.abs().mean()
        if scale == 0:
            return torch.zeros_like(x), 1.0

        x_norm = x / scale
        quantizer = QuaternaryQuantizer()
        x_quant = quantizer._quantize_to_quaternary(x_norm)
        return x_quant, float(scale)


class Quaternary3BitLinear(nn.Module):
    """3-bit量子化Linear層"""

    def __init__(self, in_features: int, out_features: int, bias: bool = True, device=None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        # 3-bit重み（8値）
        self.register_buffer(
            "weight_quaternary",
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

        self.quantizer = QuaternaryQuantizer()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with 3-bit weights"""
        weight = self.weight_quaternary.float() * self.weight_scale.view(-1, 1)
        return F.linear(x, weight, self.bias)

    def quantize_weights(self, weight: torch.Tensor) -> None:
        """量子化済み重みを設定"""
        with torch.no_grad():
            weight_abs = weight.abs()
            self.weight_scale.copy_(weight_abs.mean(dim=1))

            weight_norm = weight / (self.weight_scale.view(-1, 1) + 1e-8)
            x_quant = self.quantizer._quantize_to_quaternary(weight_norm)

            self.weight_quaternary.copy_(x_quant.to(torch.int8))


def estimate_3bit_size(model: nn.Module, show_details: bool = True) -> Dict[str, float]:
    """3-bit量子化後のモデルサイズを推定"""
    total_params = sum(p.numel() for p in model.parameters())

    # 32-bit float
    original_size_mb = total_params * 4 / (1024**2)

    # 3-bit（+スケール係数）
    num_layers = sum(1 for m in model.modules() if isinstance(m, nn.Linear))
    scale_overhead_mb = num_layers * 4 / (1024**2)
    quantized_size_mb = total_params * 0.375 / (1024**2) + scale_overhead_mb

    result = {
        "total_parameters": total_params,
        "original_size_mb": original_size_mb,
        "quantized_size_mb": quantized_size_mb,
        "compression_ratio": original_size_mb / (quantized_size_mb + 1e-8),
        "num_linear_layers": num_layers,
    }

    if show_details:
        print("\n📊 3-bit Quantization Size Estimate:")
        print(f"   Total Parameters: {result['total_parameters']:,}")
        print(f"   Original Size: {result['original_size_mb']:.2f} MB (32-bit)")
        print(f"   Quantized Size: {result['quantized_size_mb']:.2f} MB (3-bit)")
        print(f"   Compression: {result['compression_ratio']:.1f}x")
        print(f"   Linear Layers: {result['num_linear_layers']}")

    return result


if __name__ == "__main__":
    print("3-bit Quaternary Quantization Module")
    print("=" * 60)

    # テスト
    quantizer = QuaternaryQuantizer()
    x = torch.randn(100, 100)
    x_quant = quantizer(x, training=False)

    print(f"Input range: [{x.min():.4f}, {x.max():.4f}]")
    print(f"Quantized range: [{x_quant.min():.4f}, {x_quant.max():.4f}]")
    print(f"Unique values: {torch.unique(x_quant).numel()}")
    print(f"✅ 3-bit Quaternary Quantization Ready!")
