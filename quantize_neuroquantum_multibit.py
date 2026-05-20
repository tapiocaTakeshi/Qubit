#!/usr/bin/env python3
"""
マルチビット量子化スクリプト
NeuroQuantumを1-bit、2-bit、3-bitに量子化
精度と圧縮率のトレードオフを選択可能
"""

import torch
import torch.nn as nn
import json
import sys
from pathlib import Path
from typing import Dict, Optional, Tuple

sys.path.insert(0, str(Path(__file__).parent))

from neuroquantum_layered import NeuroQuantum, NeuroQuantumConfig
from binary_quantization_1bit import BinaryQuantizer
from ternary_quantization_2bit import TernaryQuantizer
from quaternary_quantization_3bit import QuaternaryQuantizer


class NeuroQuantumMultiBitQuantizer:
    """NeuroQuantumをマルチビット量子化する"""

    QUANTIZERS = {
        1: ("binary", BinaryQuantizer),
        2: ("ternary", TernaryQuantizer),
        3: ("quaternary", QuaternaryQuantizer),
    }

    def __init__(self, model: NeuroQuantum, bit_width: int = 2, device: str = "cpu"):
        """Initialize quantizer

        Args:
            model: NeuroQuantum model to quantize
            bit_width: Bit width (1, 2, or 3)
            device: Device to use
        """
        if bit_width not in self.QUANTIZERS:
            raise ValueError(f"Unsupported bit_width: {bit_width}. Choose from {list(self.QUANTIZERS.keys())}")

        self.model = model.to(device)
        self.bit_width = bit_width
        self.device = device
        self.quantizer_name, self.quantizer_class = self.QUANTIZERS[bit_width]
        self.original_model_size = self._get_model_size()
        self.quantization_stats = {}

    def _get_model_size(self) -> float:
        """Get model size in MB"""
        total_params = sum(p.numel() for p in self.model.parameters())
        return total_params * 4 / (1024**2)

    def quantize_weights(self) -> Dict:
        """重みをマルチビットに量子化"""
        print(f"🔄 Quantizing NeuroQuantum weights to {self.bit_width}-bit...")
        print("=" * 70)

        stats = {
            "bit_width": self.bit_width,
            "total_layers": 0,
            "quantized_layers": 0,
            "layer_scales": {},
        }

        quantizer = self.quantizer_class()

        for name, module in self.model.named_modules():
            if isinstance(module, nn.Linear):
                stats["total_layers"] += 1

                weight = module.weight.data
                original_norm = weight.norm().item()

                with torch.no_grad():
                    scale = weight.abs().mean()

                    if scale > 1e-8:
                        weight_norm = weight / scale
                        weight_quantized = quantizer._quantize_training(weight_norm)

                        stats["layer_scales"][name] = float(scale)
                        module.weight.data = weight_quantized
                        stats["quantized_layers"] += 1

                        new_norm = weight_quantized.norm().item()

                        print(f"   ✓ {name}")
                        print(f"      Shape: {weight.shape}")
                        print(f"      Scale: {scale:.6f}")
                        print(f"      Unique values: {torch.unique(weight_quantized).numel()}")

        print("=" * 70)
        print(
            f"\n✅ Quantization Summary:"
            f"\n   Bit Width: {self.bit_width}-bit"
            f"\n   Total Layers: {stats['total_layers']}"
            f"\n   Quantized: {stats['quantized_layers']}"
        )

        self.quantization_stats = stats
        return stats

    def get_size_reduction(self) -> Dict:
        """量子化によるサイズ削減を計算"""
        total_params = sum(p.numel() for p in self.model.parameters())

        # ビット数からバイト数を計算
        bytes_per_param = self.bit_width / 8

        # スケール係数のオーバーヘッド
        num_layers = sum(1 for m in self.model.modules() if isinstance(m, nn.Linear))
        scale_overhead_mb = num_layers * 4 / (1024**2)

        quantized_size_mb = (total_params * bytes_per_param) / (1024**2) + scale_overhead_mb

        reduction = {
            "bit_width": self.bit_width,
            "original_mb": self.original_model_size,
            "quantized_mb": quantized_size_mb,
            "compression_ratio": self.original_model_size / (quantized_size_mb + 1e-8),
            "reduction_percentage": (1 - quantized_size_mb / self.original_model_size) * 100,
        }

        print("\n💾 Size Reduction:")
        print(f"   Original: {reduction['original_mb']:.2f} MB")
        print(f"   Quantized: {reduction['quantized_mb']:.2f} MB")
        print(f"   Compression: {reduction['compression_ratio']:.1f}x")
        print(f"   Reduction: {reduction['reduction_percentage']:.1f}%")

        return reduction

    def save_quantized_model(self, output_path: str, metadata: Dict = None) -> None:
        """量子化済みモデルを保存"""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        print(f"\n💾 Saving {self.bit_width}-bit quantized model to {output_path}...")

        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "quantization_type": f"{self.quantizer_name}_{self.bit_width}bit",
                "bit_width": self.bit_width,
                "model_config": getattr(self.model, "config", None),
                "metadata": metadata or {},
            },
            output_path,
        )

        metadata_path = output_path.with_suffix(".json")
        with open(metadata_path, "w") as f:
            json.dump(
                {
                    "quantization_type": f"{self.quantizer_name}_{self.bit_width}bit",
                    "bit_width": self.bit_width,
                    "quantization_stats": self.quantization_stats,
                    "model_size_mb": self.original_model_size,
                    **(metadata or {}),
                },
                f,
                indent=2,
            )

        print(f"✅ Model saved!")
        print(f"   Checkpoint: {output_path}")
        print(f"   Metadata: {metadata_path}")


def compare_quantizations(model_path: str, model_size: str = "2-bit"):
    """異なるビット幅の量子化を比較"""
    print("\n" + "=" * 70)
    print("MULTI-BIT QUANTIZATION COMPARISON")
    print("=" * 70)

    # モデルをロード
    config_map = {
        "1-bit": {
            "vocab_size": 8000,
            "embed_dim": 256,
            "hidden_dim": 512,
            "num_heads": 4,
            "num_layers": 3,
            "max_seq_len": 512,
        },
        "2-bit": {
            "vocab_size": 32000,
            "embed_dim": 512,
            "hidden_dim": 1024,
            "num_heads": 8,
            "num_layers": 6,
            "max_seq_len": 2048,
        },
        "3-bit": {
            "vocab_size": 32000,
            "embed_dim": 768,
            "hidden_dim": 2048,
            "num_heads": 12,
            "num_layers": 12,
            "max_seq_len": 4096,
        },
    }

    checkpoint = torch.load(model_path, map_location="cpu", weights_only=True)
    if isinstance(checkpoint, dict):
        if "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        else:
            state_dict = checkpoint
    else:
        state_dict = checkpoint

    # チェックポイントから実際の max_seq_len を推論
    config_dict = config_map[model_size]
    if "position_embedding.weight" in state_dict:
        actual_max_seq_len = state_dict["position_embedding.weight"].shape[0]
        config_dict["max_seq_len"] = actual_max_seq_len

    config = NeuroQuantumConfig(**config_dict)

    # 比較テーブルのヘッダー
    print("\n📊 Quantization Comparison:")
    print(f"{'Bit Width':<12} {'Size (MB)':<15} {'Compression':<15} {'Reduction %':<15}")
    print("-" * 60)

    # オリジナル
    total_params = sum(state_dict[k].numel() if isinstance(state_dict[k], torch.Tensor) else np.prod(state_dict[k].shape) for k in state_dict)
    original_size = total_params * 4 / (1024**2)
    print(f"{'F32':<12} {original_size:<15.2f} {'1.0x':<15} {'0.0%':<15}")

    # マルチビット
    for bit_width in [1, 2, 3]:
        bytes_per_param = bit_width / 8
        quantized_size = (total_params * bytes_per_param) / (1024**2)
        compression = original_size / quantized_size
        reduction = (1 - quantized_size / original_size) * 100

        print(f"{bit_width}-bit{'':<6} {quantized_size:<15.2f} {compression:<15.1f}x {reduction:<15.1f}%")

    print("-" * 60)


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Multi-bit quantization for NeuroQuantum (1-bit, 2-bit, 3-bit)"
    )
    parser.add_argument("input_checkpoint", help="Input PyTorch checkpoint (.pt)")
    parser.add_argument(
        "--bit-width",
        "-b",
        type=int,
        default=2,
        choices=[1, 2, 3],
        help="Bit width for quantization (default: 2)",
    )
    parser.add_argument(
        "--output",
        "-o",
        help="Output checkpoint (default: auto-generated)",
    )
    parser.add_argument(
        "--model-size",
        default="2-bit",
        choices=["1-bit", "2-bit", "3-bit"],
        help="Model size (default: 2-bit)",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        choices=["cpu", "cuda"],
        help="Device to use",
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Compare all bit-widths",
    )

    args = parser.parse_args()

    # 比較モードの場合
    if args.compare:
        compare_quantizations(args.input_checkpoint, args.model_size)
        return

    # モデル設定
    config_map = {
        "1-bit": {
            "vocab_size": 8000,
            "embed_dim": 256,
            "hidden_dim": 512,
            "num_heads": 4,
            "num_layers": 3,
            "max_seq_len": 512,
        },
        "2-bit": {
            "vocab_size": 32000,
            "embed_dim": 512,
            "hidden_dim": 1024,
            "num_heads": 8,
            "num_layers": 6,
            "max_seq_len": 2048,
        },
        "3-bit": {
            "vocab_size": 32000,
            "embed_dim": 768,
            "hidden_dim": 2048,
            "num_heads": 12,
            "num_layers": 12,
            "max_seq_len": 4096,
        },
    }

    # チェックポイントをロード
    print(f"📥 Loading model from {args.input_checkpoint}...")
    checkpoint = torch.load(args.input_checkpoint, map_location=args.device)

    if isinstance(checkpoint, dict):
        if "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        else:
            state_dict = checkpoint
    else:
        state_dict = checkpoint

    # チェックポイントから実際の max_seq_len を推論
    config_dict = config_map[args.model_size]
    if "position_embedding.weight" in state_dict:
        actual_max_seq_len = state_dict["position_embedding.weight"].shape[0]
        config_dict["max_seq_len"] = actual_max_seq_len

    config = NeuroQuantumConfig(**config_dict)

    # モデルを初期化
    model = NeuroQuantum(config=config).to(args.device)
    model.load_state_dict(state_dict, strict=False)
    model.eval()

    # 量子化を実行
    quantizer = NeuroQuantumMultiBitQuantizer(model, bit_width=args.bit_width, device=args.device)

    print("\n" + "=" * 70)
    print(f"NEURO QUANTUM {args.bit_width}-BIT QUANTIZATION")
    print("=" * 70)

    quantizer.quantize_weights()
    quantizer.get_size_reduction()

    # 出力ファイル名を決定
    if not args.output:
        input_stem = Path(args.input_checkpoint).stem
        args.output = f"{input_stem}_{args.bit_width}bit_quantized.pt"

    quantizer.save_quantized_model(
        args.output,
        metadata={
            "model_size": args.model_size,
            "config": config_dict,
        },
    )

    print(f"\n{'=' * 70}")
    print("✨ Quantization Complete!")
    print(f"{'=' * 70}")
    print(f"💡 Next steps:")
    print(f"   1. Export to GGUF: python export_multibit_gguf.py {args.output} --bit-width {args.bit_width}")
    print(f"   2. Validate: python validate_gguf_metadata.py model.gguf")
    print(f"   3. Deploy: Use in applications")


if __name__ == "__main__":
    main()
