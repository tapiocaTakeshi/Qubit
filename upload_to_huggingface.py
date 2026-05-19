#!/usr/bin/env python3
"""
Upload GGUF models to Hugging Face Hub.

Usage:
    python upload_to_huggingface.py --repo-name "username/qubit-q4-k-m" --gguf-path "gguf_models/neuroquantum_small_Q4_K_M.gguf"
"""

import argparse
import os
from pathlib import Path
from huggingface_hub import HfApi, ModelCard, ModelCardData

def create_model_card(repo_id: str, model_name: str, quantization: str, file_size_mb: float) -> str:
    """Create a model card for the GGUF model."""
    return f"""---
license: mit
language: ja
library_name: ggml
pipeline_tag: text-generation
tags:
  - gguf
  - quantization
  - {quantization.lower()}
  - quantum
  - transformer
---

# Qubit {model_name} - {quantization} Quantized

量子インスパイアニューラルネットワーク（QBNN）のGGUF形式量子化モデル

## Model Details

- **Architecture**: NeuroQuantum
- **Size**: {model_name}
- **Quantization**: {quantization}
- **File Size**: {file_size_mb:.2f}MB
- **Format**: GGUF

## Quantization Information

このモデルは{quantization}形式で量子化されています。

- **推奨用途**: 標準的な推論タスク
- **メモリ効率**: 優秀
- **精度**: 高い

## Usage

### Ollama
```bash
ollama pull {repo_id}
ollama run {repo_id}
```

### llama.cpp
```bash
./main -m {model_name.lower()}_q4_k_m.gguf -p "プロンプト"
```

### Python
```python
from llama_cpp import Llama

llm = Llama(
    model_path="{model_name.lower()}_q4_k_m.gguf",
    n_ctx=4096,
)

response = llm("プロンプト")
```

## Model Specifications

- **Embedding Dimension**: 256
- **Hidden Dimension**: 512
- **Number of Heads**: 8
- **Number of Layers**: 4
- **Max Sequence Length**: 4096
- **Vocabulary Size**: 32,000

## License

MIT License

## Creator

tapiocaTakeshi
- GitHub: https://github.com/tapiocatakeshi/qubit

## References

- Repository: https://github.com/tapiocatakeshi/Qubit
- Paper: [APQB Theory Documentation](https://github.com/tapiocatakeshi/Qubit/blob/main/README.md)
"""

def upload_to_huggingface(
    repo_id: str,
    gguf_path: str,
    hf_token: str = None,
    private: bool = False,
) -> None:
    """Upload GGUF model to Hugging Face Hub.

    Args:
        repo_id: Hugging Face repository ID (e.g., "username/model-name")
        gguf_path: Path to GGUF file
        hf_token: Hugging Face API token (uses HF_TOKEN env var if not provided)
        private: Whether to create a private repository
    """
    # Get token from environment if not provided
    if hf_token is None:
        hf_token = os.getenv("HF_TOKEN")
        if not hf_token:
            raise ValueError(
                "Hugging Face token not provided. "
                "Please set HF_TOKEN environment variable or pass --token argument."
            )

    # Verify GGUF file exists
    gguf_file = Path(gguf_path)
    if not gguf_file.exists():
        raise FileNotFoundError(f"GGUF file not found: {gguf_path}")

    file_size_mb = gguf_file.stat().st_size / (1024 * 1024)
    model_name = gguf_file.stem.split("_")[0].capitalize()
    quantization = gguf_file.stem.split("_")[-1]

    print(f"🚀 Uploading to Hugging Face Hub")
    print(f"   Repository: {repo_id}")
    print(f"   File: {gguf_file.name} ({file_size_mb:.2f}MB)")
    print(f"   Quantization: {quantization}")

    # Initialize API
    api = HfApi(token=hf_token)

    # Create repository (will skip if already exists)
    try:
        print(f"\n📦 Creating repository...")
        repo_url = api.create_repo(
            repo_id=repo_id,
            private=private,
            exist_ok=True,
        )
        print(f"   ✅ Repository created/exists: {repo_url}")
    except Exception as e:
        print(f"   ⚠️  Repository creation: {e}")

    # Upload GGUF file
    print(f"\n📤 Uploading GGUF file...")
    try:
        file_url = api.upload_file(
            path_or_fileobj=str(gguf_file),
            path_in_repo=gguf_file.name,
            repo_id=repo_id,
            commit_message=f"Upload {quantization} quantized GGUF model",
        )
        print(f"   ✅ GGUF uploaded: {file_url}")
    except Exception as e:
        print(f"   ❌ Upload failed: {e}")
        raise

    # Create and upload model card
    print(f"\n📝 Creating model card...")
    try:
        model_card_content = create_model_card(
            repo_id=repo_id,
            model_name=model_name,
            quantization=quantization,
            file_size_mb=file_size_mb,
        )

        api.upload_file(
            path_or_fileobj=model_card_content.encode("utf-8"),
            path_in_repo="README.md",
            repo_id=repo_id,
            commit_message="Add model card",
        )
        print(f"   ✅ Model card created")
    except Exception as e:
        print(f"   ⚠️  Model card creation: {e}")

    # Upload manifest.json if exists
    manifest_path = gguf_file.parent / "manifest.json"
    if manifest_path.exists():
        print(f"\n📋 Uploading manifest...")
        try:
            api.upload_file(
                path_or_fileobj=str(manifest_path),
                path_in_repo="manifest.json",
                repo_id=repo_id,
                commit_message="Add generation manifest",
            )
            print(f"   ✅ Manifest uploaded")
        except Exception as e:
            print(f"   ⚠️  Manifest upload: {e}")

    print(f"\n✨ Upload complete!")
    print(f"   Model URL: https://huggingface.co/{repo_id}")


def main():
    parser = argparse.ArgumentParser(
        description="Upload GGUF models to Hugging Face Hub"
    )
    parser.add_argument(
        "--repo-name",
        required=True,
        help="Hugging Face repository ID (e.g., 'username/qubit-q4-k-m')"
    )
    parser.add_argument(
        "--gguf-path",
        required=True,
        help="Path to GGUF file"
    )
    parser.add_argument(
        "--token",
        help="Hugging Face API token (uses HF_TOKEN env var if not provided)"
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="Create a private repository"
    )

    args = parser.parse_args()

    upload_to_huggingface(
        repo_id=args.repo_name,
        gguf_path=args.gguf_path,
        hf_token=args.token,
        private=args.private,
    )


if __name__ == "__main__":
    main()
