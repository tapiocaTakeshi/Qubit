#!/usr/bin/env python3
"""
1-bit量子化モデルのGGUFエクスポーター
極端に軽量化されたNeuroQuantumモデルをGGUF形式で保存
モバイル・エッジデバイス用
"""

import torch
import numpy as np
import json
from pathlib import Path
from typing import Dict, Optional
from datetime import datetime

try:
    from gguf import GGUFWriter
except ImportError:
    print("ERROR: gguf module not found. Install with: pip install gguf")
    GGUFWriter = None


class BinaryGGUFExporter:
    """1-bit量子化モデルをGGUF形式でエクスポート"""

    # 1-bit GGUF用パラメータ
    DEFAULT_GGUF_PARAMS = {
        "n_ctx": 512,  # モバイル向けに小さめ
        "n_batch": 32,  # バッチサイズも削減
        "n_ubatch": 32,
        "n_threads": 2,  # エッジデバイス対応
        "n_gpu_layers": 0,  # CPU のみ
        "cache_type_k": "f16",
        "cache_type_v": "f16",
    }

    def __init__(self, output_dir: str = "gguf_models_1bit"):
        """Initialize exporter

        Args:
            output_dir: Output directory
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)

    def export_checkpoint_to_gguf(
        self,
        checkpoint_path: str,
        output_file: str,
        model_name: str = "NeuroQuantum-1bit",
        model_size: str = "medium",
        gguf_params: Optional[Dict] = None,
    ) -> bool:
        """1-bit量子化チェックポイントをGGUFにエクスポート

        Args:
            checkpoint_path: Input .pt checkpoint path
            output_file: Output GGUF file path
            model_name: Model name
            model_size: Model size (small/medium/large)
            gguf_params: GGUF runtime parameters

        Returns:
            Success flag
        """
        if not GGUFWriter:
            print("ERROR: gguf module not available")
            return False

        # パラメータを設定
        gguf_params = gguf_params or self.DEFAULT_GGUF_PARAMS.copy()

        print(f"📥 Loading checkpoint from {checkpoint_path}...")

        try:
            checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        except Exception as e:
            checkpoint = torch.load(checkpoint_path, map_location="cpu")

        # 状態辞書を抽出
        if isinstance(checkpoint, dict):
            if "model_state_dict" in checkpoint:
                state_dict = checkpoint["model_state_dict"]
                metadata = checkpoint.get("metadata", {})
            else:
                state_dict = checkpoint
                metadata = {}
        else:
            state_dict = checkpoint
            metadata = {}

        print(f"📝 Writing GGUF to {output_file}...")

        try:
            writer = GGUFWriter(output_file, "neuroquantum")

            # メタデータを追加
            writer.add_name(f"{model_name} {model_size.capitalize()}")
            writer.add_description(
                f"1-bit Binary Quantized {model_name} for Mobile/Edge Devices"
            )
            writer.add_version("1.0")
            writer.add_author("tapiocaTakeshi - Qubit Project")
            writer.add_url("https://github.com/tapiocatakeshi/qubit")

            # モデル固有のメタデータ
            writer.add_string("model.size", model_size)
            writer.add_string("model.architecture", "neuroquantum_1bit")
            writer.add_string("model.quantization", "binary_1bit")
            writer.add_string("model.created", datetime.now().isoformat())

            # 量子化情報
            writer.add_bool("model.is_quantized", True)
            writer.add_bool("model.is_binary", True)
            writer.add_bool("model.is_quantum", False)
            writer.add_int32("model.bit_width", 1)

            # GGUF ランタイムパラメータ
            writer.add_int32("llm.context_length", gguf_params.get("n_ctx", 512))
            writer.add_int32("llm.batch_size", gguf_params.get("n_batch", 32))
            writer.add_int32("llm.ubatch_size", gguf_params.get("n_ubatch", 32))
            writer.add_int32("llm.threads", gguf_params.get("n_threads", 2))
            writer.add_int32("llm.gpu_layers", gguf_params.get("n_gpu_layers", 0))
            writer.add_string("llm.cache_type_k", gguf_params.get("cache_type_k", "f16"))
            writer.add_string("llm.cache_type_v", gguf_params.get("cache_type_v", "f16"))

            # GGUF パラメータを JSON で保存
            writer.add_string("model.gguf_params", json.dumps(gguf_params))

            # 量子化メタデータを保存
            quant_metadata = {
                "quantization_type": "binary_1bit",
                "bit_width": 1,
                "device_target": "mobile_edge",
                "approximated_size_mb": self._estimate_quantized_size(state_dict),
                "compression_ratio": self._estimate_compression_ratio(state_dict),
                "runtime_params": gguf_params,
                "original_metadata": metadata,
            }
            writer.add_string("model.quantization_metadata", json.dumps(quant_metadata))

            # テンソルを追加（バイナリ形式）
            tensor_count = 0
            total_params = 0
            binary_tensor_count = 0

            print("📦 Adding tensors...")

            for name, tensor in state_dict.items():
                # テンソルを numpy に変換
                if isinstance(tensor, torch.Tensor):
                    data = np.ascontiguousarray(tensor.float().cpu().numpy())
                else:
                    data = tensor

                # バイナリテンソルのフラグ
                is_binary = data.max() <= 1 and data.min() >= -1

                if is_binary:
                    binary_tensor_count += 1

                # テンソルを追加
                writer.add_tensor(name, data)
                tensor_count += 1
                total_params += tensor.numel() if isinstance(tensor, torch.Tensor) else np.prod(data.shape)

            # ファイルに書き込み
            writer.write_header_to_file()
            writer.write_kv_data_to_file()
            writer.write_tensors_to_file()
            writer.close()

            # ファイル情報を出力
            file_size_mb = Path(output_file).stat().st_size / (1024**2)

            print(f"\n✅ GGUF Export Successful!")
            print(f"{'=' * 60}")
            print(f"   Tensors: {tensor_count}")
            print(f"   Binary Tensors: {binary_tensor_count}")
            print(f"   Total Parameters: {total_params:,}")
            print(f"   File Size: {file_size_mb:.2f} MB")
            print(f"   Output: {output_file}")
            print(f"{'=' * 60}")

            return True

        except Exception as e:
            print(f"❌ Export failed: {e}")
            import traceback

            traceback.print_exc()
            return False

    @staticmethod
    def _estimate_quantized_size(state_dict: Dict) -> float:
        """量子化後のサイズを推定（MB）"""
        total_params = sum(
            t.numel() if isinstance(t, torch.Tensor) else np.prod(t.shape) for t in state_dict.values()
        )
        # 1-bit + 少量のスケール係数
        return total_params / 8 / (1024**2)

    @staticmethod
    def _estimate_compression_ratio(state_dict: Dict) -> float:
        """圧縮率を推定"""
        total_params = sum(
            t.numel() if isinstance(t, torch.Tensor) else np.prod(t.shape) for t in state_dict.values()
        )
        original_size = total_params * 4  # 32-bit
        quantized_size = total_params / 8  # 1-bit
        return original_size / quantized_size


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Export 1-bit quantized model to GGUF format for mobile/edge"
    )
    parser.add_argument("input_checkpoint", help="Input 1-bit quantized checkpoint (.pt)")
    parser.add_argument("--output", "-o", help="Output GGUF file (default: auto-generated)")
    parser.add_argument("--model-name", "-n", default="NeuroQuantum-1bit", help="Model name")
    parser.add_argument(
        "--model-size",
        "-s",
        default="medium",
        choices=["small", "medium", "large"],
        help="Model size",
    )
    parser.add_argument(
        "--output-dir",
        "-d",
        default="gguf_models_1bit",
        help="Output directory (default: gguf_models_1bit)",
    )
    parser.add_argument(
        "--gguf-params",
        help="GGUF runtime parameters as JSON",
    )

    args = parser.parse_args()

    # GGUF パラメータをパース
    gguf_params = None
    if args.gguf_params:
        try:
            gguf_params = json.loads(args.gguf_params)
        except json.JSONDecodeError as e:
            print(f"ERROR: Invalid GGUF parameters: {e}")
            return False

    # 出力ファイル名を決定
    if not args.output:
        input_stem = Path(args.input_checkpoint).stem
        output_dir = Path(args.output_dir)
        output_dir.mkdir(exist_ok=True, parents=True)
        args.output = str(output_dir / f"{input_stem}_1bit.gguf")

    # エクスポート
    print("🚀 1-bit GGUF Exporter")
    print(f"{'=' * 60}")
    print(f"   Input: {args.input_checkpoint}")
    print(f"   Output: {args.output}")
    print(f"   Model: {args.model_name} ({args.model_size})")
    print(f"{'=' * 60}\n")

    exporter = BinaryGGUFExporter(output_dir=args.output_dir)
    success = exporter.export_checkpoint_to_gguf(
        args.input_checkpoint,
        args.output,
        model_name=args.model_name,
        model_size=args.model_size,
        gguf_params=gguf_params,
    )

    if success:
        print(f"\n✨ Export Complete!")
        print(f"💾 Model saved to: {args.output}")
        print(f"\n📱 Ready for mobile/edge deployment!")
    else:
        print("\n❌ Export failed!")
        return False


if __name__ == "__main__":
    main()
