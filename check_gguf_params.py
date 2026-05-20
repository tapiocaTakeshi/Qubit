#!/usr/bin/env python3
"""
GGUF パラメータ確認スクリプト
Qubit GGUF ファイルのメタデータとランタイムパラメータを確認
"""

import json
from pathlib import Path
from typing import Dict, Any, Optional


def check_gguf_file(gguf_path: str, verbose: bool = True) -> Optional[Dict[str, Any]]:
    """GGUF ファイルのパラメータを確認

    Args:
        gguf_path: GGUF ファイルのパス
        verbose: 詳細出力フラグ

    Returns:
        パラメータ情報の辞書、またはロード失敗時は None
    """
    try:
        from gguf import GGUFReader
    except ImportError:
        print("❌ ERROR: gguf module not found. Install with: pip install gguf")
        return None

    file_path = Path(gguf_path)
    if not file_path.exists():
        print(f"❌ ERROR: File not found: {gguf_path}")
        return None

    try:
        reader = GGUFReader(gguf_path)
    except Exception as e:
        print(f"❌ ERROR: Failed to read GGUF file: {e}")
        return None

    info = {}

    if verbose:
        print(f"\n{'='*60}")
        print(f"📋 GGUF File: {gguf_path}")
        print(f"{'='*60}")

        file_size_gb = file_path.stat().st_size / (1024**3)
        print(f"   File Size: {file_size_gb:.2f} GB")

    # モデル情報
    model_info = {}
    fields = [
        ("model.architecture", "Architecture"),
        ("model.size", "Size"),
        ("model.quantization", "Quantization"),
        ("model.created", "Created"),
    ]

    if verbose:
        print("\n🏗️  Model Info:")

    for field_name, label in fields:
        try:
            value = reader.get_field(field_name).strings[0]
            model_info[label.lower()] = value
            if verbose:
                print(f"   {label}: {value}")
        except (KeyError, IndexError, AttributeError):
            pass

    # ランタイムパラメータ
    runtime_params = {}
    params = [
        ("llm.context_length", "Context Length", "n_ctx"),
        ("llm.batch_size", "Batch Size", "n_batch"),
        ("llm.ubatch_size", "UBatch Size", "n_ubatch"),
        ("llm.threads", "Threads", "n_threads"),
        ("llm.gpu_layers", "GPU Layers", "n_gpu_layers"),
    ]

    if verbose:
        print("\n⚙️  Runtime Parameters:")

    for field_name, label, key in params:
        try:
            value = reader.get_field(field_name).ints[0]
            runtime_params[key] = value
            if verbose:
                print(f"   {label}: {value}")
        except (KeyError, IndexError, AttributeError):
            if verbose:
                print(f"   {label}: (not found - using default)")

    # キャッシュ型
    cache_types = {}
    cache_fields = [
        ("llm.cache_type_k", "Cache Type K"),
        ("llm.cache_type_v", "Cache Type V"),
    ]

    for field_name, label in cache_fields:
        try:
            value = reader.get_field(field_name).strings[0]
            cache_types[label.lower()] = value
            if verbose:
                print(f"   {label}: {value}")
        except (KeyError, IndexError, AttributeError):
            pass

    # 量子パラメータ（QBNN の場合）
    quantum_info = {}
    quantum_fields = [
        ("model.is_quantum", "Is Quantum", "bools"),
        ("model.has_quantum_correlation", "Has Quantum Correlation", "bools"),
        ("model.has_entanglement", "Has Entanglement", "bools"),
        ("model.apqb_theta_count", "APQB Theta Count", "ints"),
        ("model.entangle_layer_count", "Entangle Layer Count", "ints"),
    ]

    if verbose:
        print("\n⚛️  Quantum Parameters:")

    has_quantum = False
    for field_name, label, field_type in quantum_fields:
        try:
            field = reader.get_field(field_name)
            if field_type == "bools" and hasattr(field, 'bools'):
                value = field.bools[0]
                if value:
                    has_quantum = True
            elif field_type == "ints" and hasattr(field, 'ints'):
                value = field.ints[0]
                has_quantum = True
            else:
                continue
            quantum_info[label.lower()] = value
            if verbose:
                print(f"   {label}: {value}")
        except (KeyError, IndexError, AttributeError):
            pass

    if not has_quantum and verbose:
        print("   (None - Standard model)")

    # GGUF パラメータの詳細（JSON）
    gguf_params_json = None
    if verbose:
        print("\n📊 Detailed Parameters (JSON):")

    try:
        gguf_params_str = reader.get_field("model.gguf_params").strings[0]
        gguf_params_json = json.loads(gguf_params_str)
        if verbose:
            print(json.dumps(gguf_params_json, indent=2))
    except (KeyError, IndexError, AttributeError, json.JSONDecodeError):
        if verbose:
            print("   (not found)")

    # 量子メタデータの詳細
    quantum_metadata = None
    if has_quantum and verbose:
        print("\n⚛️  Quantum Metadata (JSON):")

    try:
        quantum_meta_str = reader.get_field("model.quantum_metadata").strings[0]
        quantum_metadata = json.loads(quantum_meta_str)
        if verbose:
            print(json.dumps(quantum_metadata, indent=2))
    except (KeyError, IndexError, AttributeError, json.JSONDecodeError):
        if has_quantum and verbose:
            print("   (not found)")

    if verbose:
        print(f"\n{'='*60}\n")

    # 結果をまとめる
    info = {
        "file": str(file_path),
        "file_size_gb": file_size_gb,
        "model": model_info,
        "runtime": runtime_params,
        "cache": cache_types,
        "quantum": quantum_info if has_quantum else {},
        "gguf_params": gguf_params_json,
        "quantum_metadata": quantum_metadata,
    }

    return info


def diagnose_gguf_compatibility(gguf_path: str) -> None:
    """GGUF ファイルの互換性を診断"""

    print("\n🔍 Compatibility Diagnosis")
    print("="*60)

    try:
        from gguf import GGUFReader
        reader = GGUFReader(gguf_path)
    except Exception as e:
        print(f"❌ Cannot read GGUF file: {e}")
        return

    # アーキテクチャ確認
    try:
        arch = reader.get_field("model.architecture").strings[0]
        print(f"\n📦 Architecture: {arch}")

        supported_archs = ["llama", "mistral", "gemma", "phi", "qwen"]
        if arch.lower() in supported_archs:
            print(f"   ✅ Standard llama.cpp supported architecture")
        else:
            print(f"   ⚠️  Custom architecture - may not work with standard llama.cpp")
            print(f"   💡 Recommendation: Use llama-cpp-python or PyTorch loader")
    except:
        pass

    # GPU レイヤー確認
    try:
        n_gpu_layers = reader.get_field("llm.gpu_layers").ints[0]
        print(f"\n🎮 GPU Configuration:")
        print(f"   GPU Layers: {n_gpu_layers}")

        if n_gpu_layers > 0:
            print(f"   ⚠️  This model expects GPU acceleration")
            print(f"   💡 Ensure CUDA is properly installed if using GPU")
        else:
            print(f"   ✅ CPU-only mode - no GPU required")
    except:
        pass

    # メモリ要件推定
    file_size = Path(gguf_path).stat().st_size / (1024**3)
    print(f"\n💾 Memory Requirements:")
    print(f"   Model Size: {file_size:.2f} GB")
    print(f"   Estimated RAM: {file_size * 1.2:.2f} GB (with overhead)")

    if file_size > 8:
        print(f"   ⚠️  Large model - 16+ GB RAM recommended")
    elif file_size > 4:
        print(f"   ℹ️  8+ GB RAM recommended")
    else:
        print(f"   ✅ 4+ GB RAM should be sufficient")

    # コンテキスト長確認
    try:
        n_ctx = reader.get_field("llm.context_length").ints[0]
        print(f"\n📝 Context Configuration:")
        print(f"   Max Context Length: {n_ctx}")

        estimated_vram = (n_ctx * 4) / (1024**2)  # Rough estimate
        print(f"   Estimated VRAM/RAM for max context: ~{estimated_vram:.0f} MB")
    except:
        pass

    print(f"\n{'='*60}\n")


def main():
    import sys

    if len(sys.argv) < 2:
        print("GGUF Parameter Checker for Qubit Models")
        print("\nUsage: python check_gguf_params.py <model.gguf> [--diagnose]")
        print("\nOptions:")
        print("  --diagnose    Run compatibility diagnosis")
        print("  --json        Output as JSON (no headers)")
        sys.exit(1)

    gguf_path = sys.argv[1]
    diagnose = "--diagnose" in sys.argv
    json_output = "--json" in sys.argv

    info = check_gguf_file(gguf_path, verbose=not json_output)

    if info is None:
        sys.exit(1)

    if json_output:
        print(json.dumps(info, indent=2))

    if diagnose:
        diagnose_gguf_compatibility(gguf_path)


if __name__ == "__main__":
    main()
