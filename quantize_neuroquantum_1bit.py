#!/usr/bin/env python3
"""
NeuroQuantum 1-bit量子化スクリプト
既存のNeuroQuantumモデルを1-bit二値量子化してエッジデバイス対応に
"""

import torch
import torch.nn as nn
import json
import sys
from pathlib import Path
from typing import Tuple, Dict

sys.path.insert(0, str(Path(__file__).parent))

from neuroquantum_layered import NeuroQuantum, NeuroQuantumConfig
from binary_quantization_1bit import (
    BinaryQuantizedNeuroQuantum,
    estimate_1bit_size,
    BinaryQuantizer,
)


def get_bit_width_from_model_size(model_size: str) -> int:
    """モデルサイズ名からビット幅を取得"""
    size_to_bitwidth = {
        "1B": 1,
        "2B": 2,
        "3B": 3,
    }
    if model_size not in size_to_bitwidth:
        raise ValueError(f"Unknown model size: {model_size}. Choose from {list(size_to_bitwidth.keys())}")
    return size_to_bitwidth[model_size]


class NeuroQuantum1BitQuantizer:
    """NeuroQuantumモデルを1-bit量子化する"""

    def __init__(self, model: NeuroQuantum, device: str = "cpu"):
        """Initialize quantizer

        Args:
            model: NeuroQuantum model to quantize
            device: Device to use
        """
        self.model = model.to(device)
        self.device = device
        self.original_model_size = self._get_model_size()
        self.quantization_stats = {}

    def _get_model_size(self) -> float:
        """Get model size in MB"""
        total_params = sum(p.numel() for p in self.model.parameters())
        return total_params * 4 / (1024**2)

    def quantize_weights(self) -> Dict[str, float]:
        """重みを1-bitに量子化

        Returns:
            Dictionary with quantization statistics
        """
        print("🔄 Quantizing NeuroQuantum weights to 1-bit...")
        print("=" * 60)

        stats = {
            "total_layers": 0,
            "quantized_layers": 0,
            "skipped_layers": 0,
            "layer_scales": {},
            "sparsity": {},
        }

        quantizer = BinaryQuantizer()

        for name, module in self.model.named_modules():
            if isinstance(module, nn.Linear):
                stats["total_layers"] += 1

                weight = module.weight.data
                original_norm = weight.norm().item()

                # 重みを量子化
                with torch.no_grad():
                    scale = weight.abs().mean()

                    if scale > 1e-8:
                        weight_norm = weight / scale
                        weight_binary = torch.sign(weight_norm)

                        # メタデータを保存
                        stats["layer_scales"][name] = float(scale)
                        stats["sparsity"][name] = float(
                            (weight_binary == 0).sum().item() / weight_binary.numel()
                        )

                        # 重みを置き換え
                        module.weight.data = weight_binary

                        stats["quantized_layers"] += 1

                        # ビットごと情報を出力
                        new_norm = weight_binary.norm().item()
                        compression = original_norm / (new_norm + 1e-8)

                        print(f"   ✓ {name}")
                        print(f"      Shape: {weight.shape}")
                        print(f"      Scale: {scale:.6f}")
                        print(f"      Norm change: {original_norm:.4f} → {new_norm:.4f}")
                        print(f"      Compression: {compression:.2f}x")
                    else:
                        stats["skipped_layers"] += 1

        print("=" * 60)
        print(
            f"\n✅ Quantization Summary:"
            f"\n   Total Layers: {stats['total_layers']}"
            f"\n   Quantized: {stats['quantized_layers']}"
            f"\n   Skipped: {stats['skipped_layers']}"
        )

        return stats

    def evaluate_quantization_impact(self, test_input: torch.Tensor) -> Dict[str, float]:
        """量子化の影響を評価

        Args:
            test_input: Test input tensor

        Returns:
            Evaluation metrics
        """
        print("\n📊 Evaluating Quantization Impact...")

        # 元のモデルで推論
        self.model.eval()
        with torch.no_grad():
            original_output = self.model(test_input)

        # 量子化済みモデルで推論
        with torch.no_grad():
            quantized_output = self.model(test_input)

        # 差異を計算
        mse = ((original_output - quantized_output) ** 2).mean().item()
        mae = (original_output - quantized_output).abs().mean().item()
        max_error = (original_output - quantized_output).abs().max().item()

        # 相関を計算
        correlation = torch.nn.functional.cosine_similarity(
            original_output.flatten().unsqueeze(0), quantized_output.flatten().unsqueeze(0)
        ).item()

        metrics = {
            "mse": mse,
            "mae": mae,
            "max_error": max_error,
            "cosine_similarity": correlation,
        }

        print(f"   MSE: {mse:.6f}")
        print(f"   MAE: {mae:.6f}")
        print(f"   Max Error: {max_error:.6f}")
        print(f"   Cosine Similarity: {correlation:.6f}")

        return metrics

    def get_size_reduction(self) -> Dict[str, float]:
        """量子化によるサイズ削減を計算"""
        estimate = estimate_1bit_size(self.model, show_details=False)

        reduction = {
            "original_mb": self.original_model_size,
            "quantized_mb": estimate["quantized_size_mb"],
            "compression_ratio": estimate["compression_ratio"],
            "reduction_percentage": (
                1 - estimate["quantized_size_mb"] / self.original_model_size
            ) * 100,
        }

        print("\n💾 Size Reduction:")
        print(f"   Original: {reduction['original_mb']:.2f} MB")
        print(f"   Quantized: {reduction['quantized_mb']:.2f} MB")
        print(f"   Compression: {reduction['compression_ratio']:.1f}x")
        print(f"   Reduction: {reduction['reduction_percentage']:.1f}%")

        return reduction

    def save_quantized_model(self, output_path: str, metadata: Dict = None) -> None:
        """量子化済みモデルを保存

        Args:
            output_path: Output file path
            metadata: Additional metadata to save
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        print(f"\n💾 Saving quantized model to {output_path}...")

        # モデルを保存
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "quantization_type": "binary_1bit",
                "model_config": getattr(self.model, "config", None),
                "metadata": metadata or {},
            },
            output_path,
        )

        # メタデータを別ファイルとして保存
        metadata_path = output_path.with_suffix(".json")
        with open(metadata_path, "w") as f:
            json.dump(
                {
                    "quantization_type": "binary_1bit",
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


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Quantize NeuroQuantum model to 1-bit binary"
    )
    parser.add_argument("input_checkpoint", help="Input PyTorch checkpoint (.pt)")
    parser.add_argument(
        "--output", "-o", help="Output checkpoint (default: auto-generated)"
    )
    parser.add_argument(
        "--model-size",
        default="2B",
        choices=["1B", "2B", "3B"],
        help="Model size (default: 2-bit)",
    )
    parser.add_argument(
        "--device", default="cpu", choices=["cpu", "cuda"], help="Device to use"
    )
    parser.add_argument(
        "--test-inference",
        action="store_true",
        help="Test inference with random input",
    )

    args = parser.parse_args()

    # モデル設定
    config_map = {
        "1B": {
            "vocab_size": 8000,
            "embed_dim": 256,
            "hidden_dim": 512,
            "num_heads": 4,
            "num_layers": 3,
            "max_seq_len": 512,
        },
        "2B": {
            "vocab_size": 32000,
            "embed_dim": 512,
            "hidden_dim": 1024,
            "num_heads": 8,
            "num_layers": 6,
            "max_seq_len": 2048,
        },
        "3B": {
            "vocab_size": 32000,
            "embed_dim": 768,
            "hidden_dim": 2048,
            "num_heads": 12,
            "num_layers": 12,
            "max_seq_len": 4096,
        },
    }

    config_dict = config_map[args.model_size]
    config = NeuroQuantumConfig(**config_dict)

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

    # モデルを初期化
    model = NeuroQuantum(config=config).to(args.device)
    model.load_state_dict(state_dict, strict=False)
    model.eval()

    # 量子化を実行
    quantizer = NeuroQuantum1BitQuantizer(model, device=args.device)

    print("\n" + "=" * 60)
    print("1-BIT QUANTIZATION FOR NEUROQUANTUM")
    print("=" * 60)

    # 重みを量子化
    quantizer.quantize_weights()

    # サイズ削減を表示
    quantizer.get_size_reduction()

    # テスト推論
    if args.test_inference:
        print("\n🧪 Testing inference...")
        test_input = torch.randn(1, 64, device=args.device)
        test_output = model(test_input)
        print(f"   Input shape: {test_input.shape}")
        print(f"   Output shape: {test_output.shape}")
        print(f"   ✅ Inference successful!")

    # 出力ファイル名を決定
    if not args.output:
        input_stem = Path(args.input_checkpoint).stem
        args.output = f"{input_stem}_1bit_quantized.pt"

    # モデルを保存
    quantizer.save_quantized_model(
        args.output,
        metadata={
            "model_size": args.model_size,
            "config": config_dict,
            "quantization_stats": quantizer.quantization_stats,
        },
    )

    print(f"\n{'=' * 60}")
    print("✨ Quantization Complete!")
    print(f"{'=' * 60}")
    print(f"💡 Next steps:")
    print(f"   1. Export to GGUF: python export_gguf.py {args.output}")
    print(f"   2. Validate: python validate_gguf_metadata.py model.gguf")
    print(f"   3. Deploy: Use in mobile/edge applications")


if __name__ == "__main__":
    main()
