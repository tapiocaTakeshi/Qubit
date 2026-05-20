#!/usr/bin/env python3
"""
GGUF Client Implementation Examples
QubitモデルをGGUFから正しく読み込むクライアント実装例
"""

import json
from pathlib import Path
from typing import Dict, Any, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class QubitGGUFClient:
    """Qubit GGUF モデルをロードするクライアント"""

    def __init__(self, model_path: str):
        """Initialize client

        Args:
            model_path: Path to GGUF model file
        """
        self.model_path = Path(model_path)
        self.metadata = {}
        self.model = None
        self.runtime_params = {}

    def load_metadata(self) -> Dict[str, Any]:
        """Load GGUF metadata without loading the full model"""
        try:
            from gguf import GGUFReader
        except ImportError:
            logger.error("gguf module not found. Install with: pip install gguf")
            raise

        logger.info(f"Loading metadata from {self.model_path}")

        reader = GGUFReader(str(self.model_path))

        # モデル情報
        self.metadata["architecture"] = self._get_string_field(reader, "model.architecture", "unknown")
        self.metadata["size"] = self._get_string_field(reader, "model.size", "unknown")
        self.metadata["quantization"] = self._get_string_field(reader, "model.quantization", "unknown")

        # ランタイムパラメータ
        self.runtime_params = {
            "n_ctx": self._get_int_field(reader, "llm.context_length", 512),
            "n_batch": self._get_int_field(reader, "llm.batch_size", 64),
            "n_ubatch": self._get_int_field(reader, "llm.ubatch_size", 64),
            "n_threads": self._get_int_field(reader, "llm.threads", 4),
            "n_gpu_layers": self._get_int_field(reader, "llm.gpu_layers", 0),
            "cache_type_k": self._get_string_field(reader, "llm.cache_type_k", "f16"),
            "cache_type_v": self._get_string_field(reader, "llm.cache_type_v", "f16"),
        }

        # GGUF パラメータ（JSON）
        gguf_params_str = self._get_string_field(reader, "model.gguf_params", None)
        if gguf_params_str:
            try:
                self.metadata["gguf_params"] = json.loads(gguf_params_str)
            except json.JSONDecodeError:
                logger.warning("Failed to parse GGUF parameters JSON")

        # 量子特性
        self.metadata["is_quantum"] = self._get_bool_field(reader, "model.is_quantum", False)
        if self.metadata["is_quantum"]:
            self.metadata["has_quantum_correlation"] = self._get_bool_field(
                reader, "model.has_quantum_correlation", False
            )
            self.metadata["has_entanglement"] = self._get_bool_field(
                reader, "model.has_entanglement", False
            )
            self.metadata["apqb_theta_count"] = self._get_int_field(reader, "model.apqb_theta_count", 0)

        logger.info(f"Metadata loaded: {json.dumps(self.metadata, indent=2)}")
        return self.metadata

    def load_with_llama_cpp(self, override_params: Optional[Dict] = None) -> Any:
        """Load model using llama-cpp-python

        Args:
            override_params: Runtime parameters to override

        Returns:
            Llama model instance
        """
        try:
            from llama_cpp import Llama
        except ImportError:
            logger.error("llama-cpp-python not found. Install with: pip install llama-cpp-python")
            raise

        # メタデータを先に読み込む
        if not self.metadata:
            self.load_metadata()

        # パラメータをセットアップ
        load_params = self.runtime_params.copy()
        if override_params:
            load_params.update(override_params)

        logger.info(f"Loading model with parameters: {load_params}")

        try:
            self.model = Llama(
                model_path=str(self.model_path),
                n_ctx=load_params["n_ctx"],
                n_batch=load_params["n_batch"],
                n_ubatch=load_params["n_ubatch"],
                n_threads=load_params["n_threads"],
                n_gpu_layers=load_params["n_gpu_layers"],
                verbose=True,
            )
            logger.info("✅ Model loaded successfully")
            return self.model
        except RuntimeError as e:
            logger.error(f"Failed to load model: {e}")
            # Fallback: try with GPU layers = 0
            logger.info("Retrying with n_gpu_layers=0...")
            try:
                self.model = Llama(
                    model_path=str(self.model_path),
                    n_ctx=load_params["n_ctx"],
                    n_batch=load_params["n_batch"],
                    n_threads=load_params["n_threads"],
                    n_gpu_layers=0,  # CPU only
                    verbose=True,
                )
                logger.info("✅ Model loaded in CPU-only mode")
                return self.model
            except Exception as e2:
                logger.error(f"Failed even in CPU mode: {e2}")
                raise

    def load_with_pytorch(self, checkpoint_path: str) -> Any:
        """Load as PyTorch model (if .pt file available)

        Args:
            checkpoint_path: Path to PyTorch checkpoint (.pt file)

        Returns:
            PyTorch model instance
        """
        try:
            import torch
            from neuroquantum_layered import NeuroQuantum, NeuroQuantumConfig
        except ImportError:
            logger.error("PyTorch or NeuroQuantum not available")
            raise

        logger.info(f"Loading PyTorch model from {checkpoint_path}")

        # チェックポイントをロード
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)

        if isinstance(checkpoint, dict):
            if "model_state_dict" in checkpoint:
                state_dict = checkpoint["model_state_dict"]
            elif "model_state" in checkpoint:
                state_dict = checkpoint["model_state"]
            else:
                state_dict = checkpoint
        else:
            state_dict = checkpoint

        # モデルをセットアップ
        config = NeuroQuantumConfig(
            vocab_size=32000,
            embed_dim=768,
            hidden_dim=2048,
            num_heads=12,
            num_layers=12,
        )
        self.model = NeuroQuantum(config=config)
        self.model.load_state_dict(state_dict)
        self.model.eval()

        logger.info("✅ PyTorch model loaded")
        return self.model

    def generate(self, prompt: str, max_tokens: int = 100, **kwargs) -> str:
        """Generate text using loaded model

        Args:
            prompt: Input prompt
            max_tokens: Maximum tokens to generate
            **kwargs: Additional parameters

        Returns:
            Generated text
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_with_llama_cpp() first")

        if hasattr(self.model, '__call__'):  # llama_cpp.Llama
            result = self.model(prompt, max_tokens=max_tokens, **kwargs)
            return result["choices"][0]["text"]
        else:
            raise NotImplementedError("Generate not implemented for this model type")

    def get_info(self) -> Dict[str, Any]:
        """Get model information"""
        info = {
            "path": str(self.model_path),
            "file_size_mb": self.model_path.stat().st_size / (1024**2),
            "metadata": self.metadata,
            "runtime_params": self.runtime_params,
        }
        return info

    @staticmethod
    def _get_string_field(reader, field_name: str, default: str = None) -> str:
        """Safely get string field from GGUF"""
        try:
            return reader.get_field(field_name).strings[0]
        except (KeyError, IndexError, AttributeError):
            return default

    @staticmethod
    def _get_int_field(reader, field_name: str, default: int = 0) -> int:
        """Safely get int field from GGUF"""
        try:
            return reader.get_field(field_name).ints[0]
        except (KeyError, IndexError, AttributeError):
            return default

    @staticmethod
    def _get_bool_field(reader, field_name: str, default: bool = False) -> bool:
        """Safely get bool field from GGUF"""
        try:
            return reader.get_field(field_name).bools[0]
        except (KeyError, IndexError, AttributeError):
            return default


# ================================================================================
# 使用例
# ================================================================================

def example_basic_loading():
    """基本的なモデルローディング例"""
    print("\n=== Example 1: Basic Loading ===")

    client = QubitGGUFClient("neuroquantum_medium_Q4_K_M.gguf")

    # メタデータを確認
    metadata = client.load_metadata()
    print(f"Architecture: {metadata['architecture']}")
    print(f"Size: {metadata['size']}")
    print(f"Quantization: {metadata['quantization']}")

    # モデルをロード
    model = client.load_with_llama_cpp()

    # 推論実行
    output = client.generate("こんにちは、元気ですか？", max_tokens=50)
    print(f"Generated: {output}")


def example_custom_parameters():
    """カスタムパラメータでロード"""
    print("\n=== Example 2: Custom Parameters ===")

    client = QubitGGUFClient("model.gguf")
    client.load_metadata()

    # パラメータをオーバーライド
    custom_params = {
        "n_gpu_layers": 10,  # GPU で 10 レイヤーを実行
        "n_batch": 128,      # バッチサイズを増加
        "n_threads": 8,      # スレッド数を増加
    }

    model = client.load_with_llama_cpp(override_params=custom_params)
    print(f"Model loaded with custom parameters: {custom_params}")


def example_error_handling():
    """エラーハンドリング例"""
    print("\n=== Example 3: Error Handling ===")

    client = QubitGGUFClient("model.gguf")

    try:
        metadata = client.load_metadata()
        print(f"✅ Metadata loaded successfully")

        # GPU メモリ不足を検出して自動的に CPU にフォールバック
        model = client.load_with_llama_cpp()

    except RuntimeError as e:
        logger.error(f"Failed to load model: {e}")
        # PyTorch フォールバック
        try:
            model = client.load_with_pytorch("checkpoint.pt")
            logger.info("Loaded PyTorch checkpoint instead")
        except Exception as e2:
            logger.error(f"All loading methods failed: {e2}")


def example_info_display():
    """モデル情報表示例"""
    print("\n=== Example 4: Model Info Display ===")

    client = QubitGGUFClient("model.gguf")
    client.load_metadata()

    info = client.get_info()
    print(json.dumps(info, indent=2, default=str))


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        model_path = sys.argv[1]
        print(f"Loading {model_path}...")

        client = QubitGGUFClient(model_path)
        metadata = client.load_metadata()

        print("\n📋 Metadata:")
        print(json.dumps(metadata, indent=2))

        print("\n⚙️  Runtime Parameters:")
        print(json.dumps(client.runtime_params, indent=2))

        # Try to load
        try:
            model = client.load_with_llama_cpp()
            print("\n✅ Model loaded successfully")

            # Test generation
            prompt = "Hello"
            if metadata.get("is_quantum"):
                print("⚛️  This is a quantum model")

        except Exception as e:
            print(f"\n⚠️  Load failed: {e}")
            print("💡 Try: python check_gguf_params.py --diagnose <model>")

    else:
        print("GGUF Client Implementation Examples")
        print("\nUsage: python examples_gguf_client.py <model.gguf>")
        print("\nExamples in file:")
        print("  - example_basic_loading()")
        print("  - example_custom_parameters()")
        print("  - example_error_handling()")
        print("  - example_info_display()")
