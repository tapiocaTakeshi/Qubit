#!/usr/bin/env python3
"""
QBNN to GGUF Converter - Preserves Quantum Characteristics
量子ビットニューラルネットワーク (QBNN) を llama.cpp 互換の GGUF 形式に変換
量子特性を保持しながら変換する

特徴:
- APQB (量子状態パラメータ) の保存
- エンタングルメント情報の保持
- 量子相関行列情報の記録
- llama.cpp との互換性維持
"""

import torch
import numpy as np
import json
import os
from pathlib import Path
from typing import Dict, Optional, Tuple, Any
from datetime import datetime
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

try:
    from gguf import GGUFWriter, GGMLQuantizationType
except ImportError:
    print("ERROR: gguf module not available. Install with: pip install gguf")
    GGUFWriter = None
    GGMLQuantizationType = None


class QBNNQuantumMetadata:
    """QBNN 量子メタデータ管理"""

    def __init__(self):
        self.layer_quantum_info = {}
        self.entanglement_strength = {}
        self.theta_ranges = {}
        self.constraint_violations = {}

    def add_layer_info(self, layer_name: str, layer_dict: Dict[str, torch.Tensor]):
        """レイヤーの量子情報を抽出・保存"""
        info = {
            "has_quantum_corr": any("quantum_corr" in k for k in layer_dict.keys()),
            "has_entangle": any("entangle" in k for k in layer_dict.keys()),
            "theta_count": sum(1 for k in layer_dict.keys() if "theta" in k),
            "tensor_count": len(layer_dict),
        }
        self.layer_quantum_info[layer_name] = info

    def add_entanglement_strength(self, layer_name: str, strength: float):
        """エンタングルメント強度を記録"""
        self.entanglement_strength[layer_name] = float(strength)

    def to_dict(self) -> Dict[str, Any]:
        """メタデータを辞書形式で返す"""
        return {
            "quantum_layers": self.layer_quantum_info,
            "entanglement_strengths": self.entanglement_strength,
            "theta_ranges": self.theta_ranges,
            "constraint_violations": self.constraint_violations,
        }


class QBNNToGGUFConverter:
    """QBNN から GGUF への変換"""

    # Default GGUF runtime parameters
    DEFAULT_GGUF_PARAMS = {
        "n_ctx": 512,
        "n_batch": 64,
        "n_ubatch": 64,
        "n_threads": 4,
        "n_gpu_layers": 0,
        "cache_type_k": "f16",
        "cache_type_v": "f16"
    }

    def __init__(self, output_dir: str = "gguf_models", device: str = "cpu", gguf_params: Dict = None):
        """初期化

        Args:
            output_dir: 出力ディレクトリ
            device: 計算デバイス (cpu/cuda)
            gguf_params: GGUF runtime parameters (uses defaults if not provided)
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        self.device = device
        self.quantum_metadata = QBNNQuantumMetadata()
        self.gguf_params = gguf_params or self.DEFAULT_GGUF_PARAMS.copy()

    def extract_quantum_characteristics(
        self,
        state_dict: Dict[str, torch.Tensor]
    ) -> Tuple[Dict[str, Any], Dict[str, torch.Tensor]]:
        """状態辞書から量子特性を抽出

        Args:
            state_dict: モデルの状態辞書

        Returns:
            (量子特性情報, フィルタ済み状態辞書)
        """
        quantum_features = {
            "has_quantum_correlation": False,
            "has_entanglement": False,
            "apqb_theta_params": [],
            "entangle_layers": [],
            "layer_analysis": {},
        }

        filtered_state = {}

        for name, tensor in state_dict.items():
            # 量子相関行列を検出
            if "quantum_corr" in name:
                quantum_features["has_quantum_correlation"] = True
                quantum_features["layer_analysis"][name] = {
                    "type": "quantum_correlation",
                    "shape": list(tensor.shape),
                    "dtype": str(tensor.dtype),
                }

            # エンタングルメント層を検出
            if "entangle" in name:
                quantum_features["has_entanglement"] = True
                if name not in quantum_features["entangle_layers"]:
                    quantum_features["entangle_layers"].append(name)
                quantum_features["layer_analysis"][name] = {
                    "type": "entanglement",
                    "shape": list(tensor.shape),
                }

            # APQB theta パラメータを検出
            if "theta" in name:
                quantum_features["apqb_theta_params"].append(name)
                quantum_features["layer_analysis"][name] = {
                    "type": "apqb_theta",
                    "shape": list(tensor.shape),
                    "value_range": [float(tensor.min()), float(tensor.max())],
                }

            filtered_state[name] = tensor

        return quantum_features, filtered_state

    def add_quantum_metadata_to_gguf(
        self,
        writer: "GGUFWriter",
        quantum_features: Dict[str, Any],
        model_name: str = "QBNN"
    ):
        """GGUF ファイルに量子メタデータを追加

        Args:
            writer: GGUFWriter インスタンス
            quantum_features: 量子特性情報
            model_name: モデル名
        """
        # 基本的な量子情報
        writer.add_bool("model.is_quantum", True)
        writer.add_bool("model.has_quantum_correlation",
                       quantum_features["has_quantum_correlation"])
        writer.add_bool("model.has_entanglement",
                       quantum_features["has_entanglement"])

        # APQB パラメータ情報
        if quantum_features["apqb_theta_params"]:
            writer.add_int32("model.apqb_theta_count",
                            len(quantum_features["apqb_theta_params"]))

        # エンタングルメント層情報
        if quantum_features["entangle_layers"]:
            writer.add_int32("model.entangle_layer_count",
                            len(quantum_features["entangle_layers"]))

        # メタデータを JSON で保存
        metadata_json = json.dumps({
            "model_name": model_name,
            "quantum_type": "QBNN",
            "has_quantum_correlation": quantum_features["has_quantum_correlation"],
            "has_entanglement": quantum_features["has_entanglement"],
            "apqb_theta_parameters": quantum_features["apqb_theta_params"],
            "entanglement_layers": quantum_features["entangle_layers"],
            "layer_analysis": quantum_features["layer_analysis"],
            "generated_at": datetime.now().isoformat(),
        })
        writer.add_string("model.quantum_metadata", metadata_json)

    def convert_to_gguf(
        self,
        pt_file: str,
        gguf_file: str,
        model_name: str = "Qubit-QBNN",
        model_size: str = "unknown",
        quantization: str = "Q4_K_M",
        preserve_quantum: bool = True,
        gguf_params: Dict = None
    ) -> bool:
        """PyTorch モデルを GGUF 形式に変換

        Args:
            pt_file: 入力 .pt ファイル
            gguf_file: 出力 .gguf ファイル
            model_name: モデル名
            model_size: モデルサイズ (small/medium/large)
            quantization: 量子化タイプ (Q4_K_M/Q5_K_M/F32 など)
            preserve_quantum: 量子特性を保持するか
            gguf_params: GGUF runtime parameters (uses defaults if not provided)

        Returns:
            成功したら True、失敗したら False
        """
        if gguf_params:
            self.gguf_params = gguf_params
        if not GGUFWriter:
            print("ERROR: gguf module not available")
            return False

        try:
            print(f"📥 Loading {pt_file}...")
            checkpoint = torch.load(pt_file, map_location="cpu", weights_only=True)

            if isinstance(checkpoint, dict):
                if "model_state_dict" in checkpoint:
                    state_dict = checkpoint["model_state_dict"]
                elif "model_state" in checkpoint:
                    state_dict = checkpoint["model_state"]
                else:
                    state_dict = checkpoint
            else:
                state_dict = checkpoint

            print(f"✨ Extracting quantum characteristics...")
            quantum_features, filtered_state = self.extract_quantum_characteristics(state_dict)

            print(f"📝 Writing GGUF to {gguf_file}...")
            writer = GGUFWriter(gguf_file, "qbnn")

            # メタデータを追加
            writer.add_name(f"{model_name} {model_size.capitalize()}")
            writer.add_description(
                f"{model_name} {model_size.capitalize()} with Quantum Characteristics "
                f"(QBNN) by tapiocaTakeshi"
            )
            writer.add_version("1.0")
            writer.add_author("tapiocaTakeshi")
            writer.add_url("https://github.com/tapiocatakeshi/qubit")
            writer.add_source_url("https://github.com/tapiocatakeshi/qubit")

            # モデル固有のメタデータ
            writer.add_string("model.size", model_size)
            writer.add_string("model.architecture", "qbnn")
            writer.add_string("model.quantization", quantization)
            writer.add_string("model.created", datetime.now().isoformat())

            # Add GGUF runtime parameters (used by llama.cpp and compatible loaders)
            writer.add_int32("llm.context_length", self.gguf_params.get("n_ctx", 512))
            writer.add_int32("llm.batch_size", self.gguf_params.get("n_batch", 64))
            writer.add_int32("llm.ubatch_size", self.gguf_params.get("n_ubatch", 64))
            writer.add_int32("llm.threads", self.gguf_params.get("n_threads", 4))
            writer.add_int32("llm.gpu_layers", self.gguf_params.get("n_gpu_layers", 0))
            writer.add_string("llm.cache_type_k", self.gguf_params.get("cache_type_k", "f16"))
            writer.add_string("llm.cache_type_v", self.gguf_params.get("cache_type_v", "f16"))

            # Save all GGUF parameters as JSON for reference
            writer.add_string("model.gguf_params", json.dumps(self.gguf_params))

            # 量子特性メタデータを追加
            if preserve_quantum:
                self.add_quantum_metadata_to_gguf(writer, quantum_features, model_name)

            # 量子化マッピング
            quantization_map = {
                "Q4_K_M": GGMLQuantizationType.Q4_K if GGMLQuantizationType else None,
                "Q4_K_S": GGMLQuantizationType.Q4_K if GGMLQuantizationType else None,
                "Q5_K_M": GGMLQuantizationType.Q5_K if GGMLQuantizationType else None,
                "Q5_K_S": GGMLQuantizationType.Q5_K if GGMLQuantizationType else None,
                "Q6_K": GGMLQuantizationType.Q6_K if GGMLQuantizationType else None,
                "Q8_0": GGMLQuantizationType.Q8_0 if GGMLQuantizationType else None,
                "F32": None,
                "F16": None,
            }

            quant_type = quantization_map.get(quantization)

            tensor_count = 0
            total_params = 0
            quantum_tensor_count = 0

            print(f"📦 Adding tensors ({len(filtered_state)} total)...")

            for name, tensor in filtered_state.items():
                data = np.ascontiguousarray(tensor.float().detach().cpu().numpy())

                # 量子特性のあるテンソルはスキップ、保存（量子化しない）
                is_quantum = any(q in name for q in
                               ["quantum_corr", "entangle", "theta"])

                should_quantize = (
                    quant_type is not None
                    and not any(pattern in name
                              for pattern in ["embed", "norm", "bias"])
                    and len(data.shape) >= 2
                    and data.shape[-1] >= 256
                    and not is_quantum  # 量子テンソルは量子化しない
                )

                if should_quantize:
                    writer.add_tensor(name, data, raw_dtype=quant_type)
                else:
                    writer.add_tensor(name, data)
                    if is_quantum:
                        quantum_tensor_count += 1

                tensor_count += 1
                total_params += tensor.numel()

            writer.write_header_to_file()
            writer.write_kv_data_to_file()
            writer.write_tensors_to_file()
            writer.close()

            file_size_mb = os.path.getsize(gguf_file) / (1024 * 1024)
            print(f"✅ Successfully exported {tensor_count} tensors")
            print(f"   - Total parameters: {total_params:,}")
            print(f"   - Quantum tensors preserved: {quantum_tensor_count}")
            print(f"   - File size: {file_size_mb:.2f}MB")
            print(f"   - Quantization: {quantization}")

            if preserve_quantum:
                print(f"   - Quantum correlation: {'✓' if quantum_features['has_quantum_correlation'] else '✗'}")
                print(f"   - Entanglement layers: {len(quantum_features['entangle_layers'])}")
                print(f"   - APQB theta params: {len(quantum_features['apqb_theta_params'])}")

            return True

        except Exception as e:
            print(f"❌ Error converting {pt_file}: {e}")
            import traceback
            traceback.print_exc()
            return False

    def generate_gguf_from_checkpoint(
        self,
        checkpoint_file: str,
        gguf_file: str,
        model_name: str = "Qubit-QBNN",
        model_size: str = "unknown",
        quantization: str = "Q4_K_M"
    ) -> bool:
        """チェックポイントから GGUF を生成

        Args:
            checkpoint_file: .pt チェックポイントファイル
            gguf_file: 出力 .gguf ファイル
            model_name: モデル名
            model_size: モデルサイズ
            quantization: 量子化タイプ

        Returns:
            成功したら True
        """
        return self.convert_to_gguf(
            checkpoint_file,
            gguf_file,
            model_name=model_name,
            model_size=model_size,
            quantization=quantization,
            preserve_quantum=True
        )


def main():
    """メインエントリーポイント"""
    import argparse

    parser = argparse.ArgumentParser(
        description="Convert QBNN PyTorch model to GGUF format with quantum characteristics preservation"
    )
    parser.add_argument("input_file",
                       help="Input PyTorch checkpoint file (.pt)")
    parser.add_argument("--output-file", "-o",
                       help="Output GGUF file (default: auto-generated)")
    parser.add_argument("--model-name", "-n", default="Qubit-QBNN",
                       help="Model name (default: Qubit-QBNN)")
    parser.add_argument("--model-size", "-s", default="unknown",
                       choices=["small", "medium", "large", "unknown"],
                       help="Model size (default: unknown)")
    parser.add_argument("--quantization", "-q", default="Q4_K_M",
                       choices=["Q4_K_M", "Q4_K_S", "Q5_K_M", "Q5_K_S",
                               "Q6_K", "Q8_0", "F32", "F16"],
                       help="Quantization type (default: Q4_K_M)")
    parser.add_argument("--output-dir", "-d", default="gguf_models",
                       help="Output directory (default: gguf_models)")
    parser.add_argument("--device", default="cpu",
                       choices=["cpu", "cuda"],
                       help="Device for processing (default: cpu)")
    parser.add_argument("--preserve-quantum", action="store_true", default=True,
                       help="Preserve quantum characteristics (default: True)")
    parser.add_argument("--gguf-params",
                       type=str,
                       help="GGUF runtime parameters as JSON (e.g., '{\"n_ctx\": 2048, \"n_gpu_layers\": 10}')")

    args = parser.parse_args()

    # Parse GGUF parameters if provided
    gguf_params = None
    if args.gguf_params:
        try:
            gguf_params = json.loads(args.gguf_params)
            print(f"   Custom GGUF Parameters: {gguf_params}")
        except json.JSONDecodeError as e:
            print(f"❌ Error parsing GGUF parameters: {e}")
            exit(1)

    # 出力ファイル名を自動生成
    if not args.output_file:
        input_stem = Path(args.input_file).stem
        args.output_file = f"{args.output_dir}/{input_stem}_{args.quantization}.gguf"

    print("🚀 QBNN to GGUF Converter")
    print(f"   Input: {args.input_file}")
    print(f"   Output: {args.output_file}")
    print(f"   Model: {args.model_name} ({args.model_size})")
    print(f"   Quantization: {args.quantization}")
    print(f"   Preserve Quantum: {args.preserve_quantum}\n")

    converter = QBNNToGGUFConverter(output_dir=args.output_dir, device=args.device, gguf_params=gguf_params)
    success = converter.convert_to_gguf(
        args.input_file,
        args.output_file,
        model_name=args.model_name,
        model_size=args.model_size,
        quantization=args.quantization,
        preserve_quantum=args.preserve_quantum,
        gguf_params=gguf_params
    )

    if success:
        print(f"\n✨ Conversion complete!")
        print(f"📂 Output: {args.output_file}")
    else:
        print(f"\n❌ Conversion failed!")
        exit(1)


if __name__ == "__main__":
    main()
