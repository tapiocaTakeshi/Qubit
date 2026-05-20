#!/usr/bin/env python3
"""
Save QBNN model to HuggingFace format for llama.cpp conversion

このスクリプトは、Qubit の QBNN モデルを HuggingFace 互換形式で保存します。
llama.cpp の convert_hf_to_gguf.py で直接変換可能になります。
"""

import json
import torch
from pathlib import Path
from typing import Dict, Optional
import shutil


class QBNNHFModelSaver:
    """QBNN モデルを HuggingFace 形式で保存"""

    def __init__(self, output_dir: str = "qbnn_hf_model"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)

    def create_config(
        self,
        hidden_size: int = 768,
        intermediate_size: int = 3072,
        num_hidden_layers: int = 12,
        num_attention_heads: int = 12,
        vocab_size: int = 256000,
        max_position_embeddings: int = 2048,
        theta_dim: int = 768,
        entangle_strength: float = 0.5,
        use_entanglement: bool = True,
        model_id: str = "qbnn-model"
    ) -> Dict:
        """HuggingFace 互換 config.json を生成"""
        config = {
            "architectures": ["QBNNForCausalLM"],
            "model_type": "qbnn",
            "hidden_size": hidden_size,
            "intermediate_size": intermediate_size,
            "num_hidden_layers": num_hidden_layers,
            "num_attention_heads": num_attention_heads,
            "vocab_size": vocab_size,
            "max_position_embeddings": max_position_embeddings,
            "torch_dtype": "float32",
            "transformers_version": "4.40.0",

            # QBNN固有パラメータ
            "theta_dim": theta_dim,
            "entangle_strength": entangle_strength,
            "use_entanglement": use_entanglement,

            # Quantum metadata
            "quantum_info": {
                "model_type": "QBNN",
                "architecture": "Entangled Quantum Bit Neural Network",
                "has_apqb": True,
                "has_entanglement": use_entanglement,
                "theta_parameters": f"dim={theta_dim}",
            }
        }
        return config

    def create_tokenizer_config(
        self,
        vocab_size: int = 256000,
        bos_token_id: int = 1,
        eos_token_id: int = 2,
    ) -> Dict:
        """tokenizer_config.json を生成"""
        return {
            "bos_token": "<s>",
            "chat_template": "[INST] {prompt} [/INST]",
            "eos_token": "</s>",
            "model_max_length": 2048,
            "tokenizer_class": "PreTrainedTokenizerFast",
            "unk_token": "<unk>",
        }

    def save_config_files(self, config: Dict, tokenizer_config: Dict):
        """config.json と tokenizer_config.json を保存"""
        config_path = self.output_dir / "config.json"
        with open(config_path, "w") as f:
            json.dump(config, f, indent=2)
        print(f"✓ Saved config.json to {config_path}")

        tokenizer_config_path = self.output_dir / "tokenizer_config.json"
        with open(tokenizer_config_path, "w") as f:
            json.dump(tokenizer_config, f, indent=2)
        print(f"✓ Saved tokenizer_config.json to {tokenizer_config_path}")

    def copy_tokenizer_files(
        self,
        tokenizer_model_path: Optional[str] = None,
        tokenizer_vocab_path: Optional[str] = None
    ):
        """トークナイザーファイルをコピー"""
        if tokenizer_model_path:
            src = Path(tokenizer_model_path)
            if src.exists():
                dst = self.output_dir / "tokenizer.model"
                shutil.copy(src, dst)
                print(f"✓ Copied tokenizer.model to {dst}")

        if tokenizer_vocab_path:
            src = Path(tokenizer_vocab_path)
            if src.exists():
                dst = self.output_dir / "tokenizer.vocab"
                shutil.copy(src, dst)
                print(f"✓ Copied tokenizer.vocab to {dst}")

    def save_model_weights(self, model_state_dict: Dict[str, torch.Tensor]):
        """モデルの重みを safetensors 形式で保存"""
        try:
            from safetensors.torch import save_file
            weights_path = self.output_dir / "model.safetensors"
            save_file(model_state_dict, weights_path)
            print(f"✓ Saved model weights to {weights_path}")
        except ImportError:
            # safetensors がない場合は PyTorch 形式で保存
            weights_path = self.output_dir / "pytorch_model.bin"
            torch.save(model_state_dict, weights_path)
            print(f"✓ Saved model weights to {weights_path} (PyTorch format)")

    def save_model(
        self,
        model_state_dict: Dict[str, torch.Tensor],
        model_config: Optional[Dict] = None,
        tokenizer_model_path: Optional[str] = None,
        tokenizer_vocab_path: Optional[str] = None,
    ) -> Path:
        """QBNN モデルを HuggingFace 形式で完全に保存

        Args:
            model_state_dict: モデルの状態辞書
            model_config: モデル設定（デフォルトなら自動生成）
            tokenizer_model_path: tokenizer.model ファイルへのパス
            tokenizer_vocab_path: tokenizer.vocab ファイルへのパス

        Returns:
            保存先ディレクトリのパス
        """
        print(f"💾 Saving QBNN model to HuggingFace format in {self.output_dir}...")

        # config 生成
        if model_config is None:
            config = self.create_config()
        else:
            config = model_config

        # tokenizer config 生成
        tokenizer_config = self.create_tokenizer_config(
            vocab_size=config.get("vocab_size", 256000)
        )

        # ファイル保存
        self.save_config_files(config, tokenizer_config)
        self.copy_tokenizer_files(tokenizer_model_path, tokenizer_vocab_path)
        self.save_model_weights(model_state_dict)

        print(f"\n✅ QBNN model saved to {self.output_dir}")
        print(f"   Files: config.json, tokenizer_config.json, model.safetensors")
        print(f"\n📝 To convert to GGUF format:")
        print(f"   python convert_hf_to_gguf.py {self.output_dir}")

        return self.output_dir


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Save QBNN model to HuggingFace format")
    parser.add_argument("--model-path", type=str, help="Path to .pt model file")
    parser.add_argument("--output-dir", type=str, default="qbnn_hf_model",
                        help="Output directory for HuggingFace format model")
    parser.add_argument("--tokenizer-model", type=str, default="neuroq_tokenizer.model",
                        help="Path to tokenizer.model file")
    parser.add_argument("--hidden-size", type=int, default=768)
    parser.add_argument("--num-layers", type=int, default=12)
    parser.add_argument("--vocab-size", type=int, default=256000)
    parser.add_argument("--theta-dim", type=int, default=768)

    args = parser.parse_args()

    saver = QBNNHFModelSaver(args.output_dir)

    # モデル設定
    config = saver.create_config(
        hidden_size=args.hidden_size,
        num_hidden_layers=args.num_layers,
        vocab_size=args.vocab_size,
        theta_dim=args.theta_dim,
    )

    # モデルの重みをロード
    if args.model_path:
        checkpoint = torch.load(args.model_path, map_location="cpu")
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        else:
            state_dict = checkpoint
    else:
        print("⚠️  No model path provided. Creating empty model structure.")
        state_dict = {}

    # HuggingFace 形式で保存
    saver.save_model(
        state_dict,
        model_config=config,
        tokenizer_model_path=args.tokenizer_model,
    )
