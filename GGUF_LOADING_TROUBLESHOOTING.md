# GGUF Model Loading Troubleshooting Guide
# GGUFモデルロード問題のトラブルシューティングガイド

## 概要

このガイドは、GGUF形式で保存されたQubit（NeuroQuantumおよびQBNN）モデルがクライアントアプリケーション（PocketPalなど）で読み込めない場合の対処方法を説明します。

## よくある問題と原因

### 問題1: 「Failed to load model」エラー

**原因**: llama.cpp（およびその互換実装）は特定のアーキテクチャのみをサポートしています。
Qubitの「neuroquantum」や「qbnn」といったカスタムアーキテクチャは、デフォルトではサポートされていません。

**症状**:
```
Error: Failed to load model
Architecture "neuroquantum" not supported
```

**解決方法**:

#### オプション1: カスタムローダーを使用（推奨）
Python + llama-cpp-python でカスタムアーキテクチャをサポート：

```python
from llama_cpp import Llama

model = Llama(
    model_path="neuroquantum_medium_Q4_K_M.gguf",
    n_gpu_layers=-1,  # GPU に読み込み
    n_threads=4,
    verbose=True  # ロード問題を診断
)
```

#### オプション2: Hugging Face transformers + bitsandbytes
```python
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

model = AutoModelForCausalLM.from_pretrained(
    "path/to/qubit",
    torch_dtype=torch.float16,
    device_map="auto",
    load_in_4bit=True  # 4-bit quantization
)
```

#### オプション3: Flask/FastAPI サーバー
GGUF ではなく PyTorch チェックポイント（.pt）を直接ロード：

```python
from flask import Flask, request, jsonify
import torch
from neuroquantum_layered import NeuroQuantum, NeuroQuantumConfig

app = Flask(__name__)
model = NeuroQuantum(config=NeuroQuantumConfig(...))
model.load_state_dict(torch.load("checkpoint.pt"))
model.eval()

@app.route("/generate", methods=["POST"])
def generate():
    prompt = request.json["prompt"]
    with torch.no_grad():
        output = model.generate(prompt, max_length=100)
    return {"text": output}
```

### 問題2: メタデータが読み込まれない

**症状**:
- GGUF ファイルに `n_gpu_layers=0` と記録されているのに、アプリが `n_gpu_layers=99` を使用
- コンテキスト長やバッチサイズが反映されない

**原因**:
1. アプリケーションが GGUF メタデータを読み込んでいない
2. メタデータキーが標準仕様と異なる
3. アプリケーションがハードコードされたデフォルト値を使用している

**解決方法**:

Qubit の GGUF ファイルには以下のメタデータが含まれています：

```
llm.context_length    → n_ctx (512)
llm.batch_size        → n_batch (64)
llm.ubatch_size       → n_ubatch (64)
llm.threads           → n_threads (4)
llm.gpu_layers        → n_gpu_layers (0)
llm.cache_type_k      → cache_type_k ("f16")
llm.cache_type_v      → cache_type_v ("f16")
model.gguf_params     → すべてのパラメータを JSON 形式で保存
```

クライアントアプリケーションは以下のように読み込む必要があります：

```python
from gguf import GGUFReader

reader = GGUFReader("model.gguf")

# メタデータを読み込み
params = {
    "n_ctx": reader.get_field("llm.context_length").ints[0],
    "n_batch": reader.get_field("llm.batch_size").ints[0],
    "n_gpu_layers": reader.get_field("llm.gpu_layers").ints[0],
    # ...
}
```

### 問題3: メモリ不足（OOM）エラー

**症状**:
```
RuntimeError: CUDA out of memory
```

**原因**:
- `n_gpu_layers` の値が高すぎる
- バッチサイズが大きすぎる
- GPU メモリが不足している

**解決方法**:

GGUF ファイルの設定を確認：
```bash
# Qubit GGUF チェッカー（後述のスクリプト参照）
python check_gguf_params.py neuroquantum_medium_Q4_K_M.gguf
```

設定を調整：
```python
from llama_cpp import Llama

model = Llama(
    model_path="model.gguf",
    n_gpu_layers=10,  # デフォルト (n_gpu_layers=0) から調整
    n_batch=32,       # バッチサイズを削減
    n_threads=2,      # スレッド数を削減
    verbose=True
)
```

### 問題4: アーキテクチャ不一致警告

**症状**:
```
Warning: Model architecture "qbnn" is not recognized by llama.cpp
```

**原因**: llama.cpp は "llama", "mistral", "gemma" などの標準アーキテクチャのみをサポート

**解決方法**:
- これは警告です。機能は失われませんが、自動最適化が適用されない可能性があります
- Qubit 専用のローダーを使用するか、PyTorch チェックポイント形式を使用してください

## 診断ツール

### GGUF パラメータチェッカー

```python
# check_gguf_params.py
import json
from pathlib import Path

def check_gguf_file(gguf_path: str):
    """GGUF ファイルのパラメータを確認"""
    try:
        from gguf import GGUFReader
    except ImportError:
        print("ERROR: pip install gguf")
        return

    reader = GGUFReader(gguf_path)
    
    print(f"\n📋 GGUF File: {gguf_path}")
    print(f"   Size: {Path(gguf_path).stat().st_size / (1024**3):.2f} GB")
    
    # モデル情報
    print("\n🏗️  Model Info:")
    fields = [
        ("model.architecture", "Architecture"),
        ("model.size", "Size"),
        ("model.quantization", "Quantization"),
    ]
    
    for field_name, label in fields:
        try:
            value = reader.get_field(field_name).strings[0]
            print(f"   {label}: {value}")
        except:
            pass
    
    # ランタイムパラメータ
    print("\n⚙️  Runtime Parameters:")
    params = [
        ("llm.context_length", "Context Length"),
        ("llm.batch_size", "Batch Size"),
        ("llm.gpu_layers", "GPU Layers"),
        ("llm.threads", "Threads"),
    ]
    
    for field_name, label in params:
        try:
            value = reader.get_field(field_name).ints[0]
            print(f"   {label}: {value}")
        except:
            print(f"   {label}: (not found)")
    
    # 量子パラメータ（QBNN の場合）
    print("\n⚛️  Quantum Parameters:")
    quantum_fields = [
        ("model.is_quantum", "Is Quantum"),
        ("model.has_quantum_correlation", "Has Quantum Correlation"),
        ("model.has_entanglement", "Has Entanglement"),
        ("model.apqb_theta_count", "APQB Theta Count"),
    ]
    
    for field_name, label in quantum_fields:
        try:
            value = reader.get_field(field_name)
            if hasattr(value, 'bools'):
                result = value.bools[0]
            elif hasattr(value, 'ints'):
                result = value.ints[0]
            else:
                result = value
            print(f"   {label}: {result}")
        except:
            pass
    
    # GGUF パラメータの詳細（JSON）
    print("\n📊 Detailed Parameters (JSON):")
    try:
        gguf_params_str = reader.get_field("model.gguf_params").strings[0]
        gguf_params = json.loads(gguf_params_str)
        print(json.dumps(gguf_params, indent=2))
    except:
        print("   (not found)")

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        check_gguf_file(sys.argv[1])
    else:
        print("Usage: python check_gguf_params.py <model.gguf>")
```

使用方法：
```bash
python check_gguf_params.py neuroquantum_medium_Q4_K_M.gguf
```

## ベストプラクティス

### 1. GGUF ファイル作成時

```python
from generate_gguf_models import GGUFModelGenerator

generator = GGUFModelGenerator(
    gguf_params={
        "n_ctx": 2048,      # コンテキスト長を調整
        "n_batch": 512,     # バッチサイズ
        "n_ubatch": 512,    
        "n_threads": 4,
        "n_gpu_layers": 0,  # デフォルトは CPU のみ
        "cache_type_k": "f16",
        "cache_type_v": "f16"
    }
)

generator.pt_to_gguf(
    "model.pt",
    "model.gguf",
    architecture="neuroquantum"
)
```

### 2. GGUF ファイル読み込み時（クライアント側）

```python
# 1. メタデータを読み込む
from gguf import GGUFReader
reader = GGUFReader("model.gguf")
n_gpu_layers = reader.get_field("llm.gpu_layers").ints[0]

# 2. パラメータを設定してロード
from llama_cpp import Llama
model = Llama(
    model_path="model.gguf",
    n_gpu_layers=n_gpu_layers,  # メタデータから読み込み
    n_batch=64,
    verbose=False
)

# 3. 推論実行
output = model("Hello", max_tokens=100)
```

### 3. エラーハンドリング

```python
def safe_load_gguf(model_path: str):
    """GGUF ファイルを安全にロード"""
    import logging
    from llama_cpp import Llama
    
    logging.basicConfig(level=logging.DEBUG)
    logger = logging.getLogger(__name__)
    
    try:
        logger.info(f"Loading {model_path}...")
        
        # メタデータを先に確認
        from gguf import GGUFReader
        reader = GGUFReader(model_path)
        arch = reader.get_field("model.architecture").strings[0]
        logger.warning(f"Architecture: {arch}")
        
        # n_gpu_layers を段階的に削減して試す
        for gpu_layers in [32, 16, 8, 0]:
            try:
                logger.info(f"Trying with n_gpu_layers={gpu_layers}...")
                model = Llama(
                    model_path=model_path,
                    n_gpu_layers=gpu_layers,
                    verbose=True
                )
                logger.info("✅ Successfully loaded!")
                return model
            except RuntimeError as e:
                logger.warning(f"Failed with {gpu_layers}: {e}")
                continue
        
        raise RuntimeError("Failed to load model with any configuration")
        
    except Exception as e:
        logger.error(f"❌ Load failed: {e}")
        raise
```

## 推奨事項

### モデル配布形式

Qubit モデルを配布する際は、以下の形式を含めることをお勧めします：

```
models/
├── neuroquantum_medium.pt          # PyTorch チェックポイント（完全性重視）
├── neuroquantum_medium_Q4_K_M.gguf # GGUF (llama.cpp 互換)
└── config.json                      # メタデータ
```

### config.json の例

```json
{
  "architecture": "neuroquantum",
  "size": "medium",
  "parameters": {
    "vocab_size": 32000,
    "embed_dim": 768,
    "num_layers": 12
  },
  "gguf_runtime": {
    "n_ctx": 2048,
    "n_batch": 512,
    "n_gpu_layers": 0,
    "n_threads": 4
  },
  "quantization": "Q4_K_M",
  "recommended_loader": "llama-cpp-python",
  "huggingface_url": "https://huggingface.co/tapiocatakeshi/qubit"
}
```

### アプリケーション側の対応

PocketPal などのアプリケーションは：

1. GGUF メタデータを読み込む
2. 不明なアーキテクチャの場合は警告を表示
3. llm.* フィールドから設定を読み込む
4. PyTorch ローダーへのフォールバック機能を実装

## 参考資料

- [GGUF 仕様](https://github.com/ggerganov/ggml/blob/master/docs/gguf.md)
- [llama.cpp](https://github.com/ggerganov/llama.cpp)
- [llama-cpp-python](https://github.com/abetlen/llama-cpp-python)
- [Hugging Face Transformers](https://huggingface.co/docs/transformers/)

## サポート

問題が解決しない場合：

1. 上記の診断ツールを実行: `python check_gguf_params.py <model.gguf>`
2. ログ出力を確認: `verbose=True` に設定
3. GitHub Issue を報告: https://github.com/tapiocatakeshi/qubit/issues

出力内容を含めて報告してください。
