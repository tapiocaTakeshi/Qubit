#!/usr/bin/env python3
"""
Generate .litertlm bundle files for NeuroQuantum models in all sizes.

A .litertlm file is a single-file bundle (in the spirit of Google's LiteRT-LM
container format) that packs together everything an on-device LLM runtime
needs to load a model:

    +----------------------------+
    | Magic        "LITERTLM"    |  8 bytes
    | Version u32 LE             |  4 bytes
    | Section count u32 LE       |  4 bytes
    | Section table (per entry)  |  4+8+8+4+4 = 28 bytes
    |   - type   u32 LE          |
    |   - offset u64 LE          |
    |   - size   u64 LE          |
    |   - name_off u32 LE        |
    |   - name_len u32 LE        |
    | Name pool (concatenated)   |
    | Section data (concatenated)|
    +----------------------------+

Section types:
    1 = TFLITE_MODEL_OR_WEIGHTS  (numpy NPZ archive of tensors)
    2 = SP_MODEL                 (SentencePiece .model bytes)
    3 = HF_TOKENIZER_JSON        (HuggingFace tokenizer.json — optional)
    4 = LLM_METADATA_JSON        (JSON-encoded model/runtime config)
    5 = MODEL_CARD_MD            (Markdown model card)

This format intentionally mirrors Google's LiteRT-LM bundle layout so that
tooling that expects "LITERTLM" magic bytes can detect the file, while the
inner section data is kept friendly to a custom architecture
(NeuroQuantum/QBNN) that doesn't currently round-trip cleanly to TFLite.
"""

from __future__ import annotations

import argparse
import io
import json
import os
import struct
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

# Local imports — added to path so this script runs from the repo root.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


LITERTLM_MAGIC = b"LITERTLM"
LITERTLM_VERSION = 1

SECTION_TFLITE_MODEL_OR_WEIGHTS = 1
SECTION_SP_MODEL = 2
SECTION_HF_TOKENIZER_JSON = 3
SECTION_LLM_METADATA_JSON = 4
SECTION_MODEL_CARD_MD = 5

SECTION_TYPE_NAMES = {
    SECTION_TFLITE_MODEL_OR_WEIGHTS: "TFLITE_MODEL_OR_WEIGHTS",
    SECTION_SP_MODEL: "SP_MODEL",
    SECTION_HF_TOKENIZER_JSON: "HF_TOKENIZER_JSON",
    SECTION_LLM_METADATA_JSON: "LLM_METADATA_JSON",
    SECTION_MODEL_CARD_MD: "MODEL_CARD_MD",
}


def write_litertlm(
    out_path: Path,
    sections: List[Tuple[int, str, bytes]],
) -> int:
    """Write a .litertlm bundle.

    sections: list of (type_id, name, payload_bytes).
    Returns total file size in bytes.
    """
    # Build name pool first so we can compute offsets.
    name_pool = io.BytesIO()
    name_locations: List[Tuple[int, int]] = []
    for _, name, _ in sections:
        encoded = name.encode("utf-8")
        offset = name_pool.tell()
        name_pool.write(encoded)
        name_locations.append((offset, len(encoded)))

    name_pool_bytes = name_pool.getvalue()

    header_size = len(LITERTLM_MAGIC) + 4 + 4  # magic + version + section_count
    entry_size = 4 + 8 + 8 + 4 + 4  # type + offset + size + name_off + name_len
    table_size = entry_size * len(sections)

    data_start = header_size + table_size + len(name_pool_bytes)

    data_blobs = io.BytesIO()
    section_offsets: List[Tuple[int, int]] = []
    cursor = data_start
    for _, _, payload in sections:
        section_offsets.append((cursor, len(payload)))
        data_blobs.write(payload)
        cursor += len(payload)

    with open(out_path, "wb") as f:
        # Header
        f.write(LITERTLM_MAGIC)
        f.write(struct.pack("<I", LITERTLM_VERSION))
        f.write(struct.pack("<I", len(sections)))

        # Section table
        for (sec_type, _, _), (offset, size), (name_off, name_len) in zip(
            sections, section_offsets, name_locations
        ):
            f.write(struct.pack("<I", sec_type))
            f.write(struct.pack("<Q", offset))
            f.write(struct.pack("<Q", size))
            f.write(struct.pack("<I", name_off))
            f.write(struct.pack("<I", name_len))

        # Name pool
        f.write(name_pool_bytes)

        # Section data
        f.write(data_blobs.getvalue())

    return out_path.stat().st_size


def read_litertlm_header(path: Path) -> Dict:
    """Inspect a .litertlm file's header — useful for validation."""
    with open(path, "rb") as f:
        magic = f.read(len(LITERTLM_MAGIC))
        if magic != LITERTLM_MAGIC:
            raise ValueError(f"Not a LiteRTLM file (magic mismatch): {path}")
        (version,) = struct.unpack("<I", f.read(4))
        (count,) = struct.unpack("<I", f.read(4))

        entries = []
        for _ in range(count):
            sec_type = struct.unpack("<I", f.read(4))[0]
            offset = struct.unpack("<Q", f.read(8))[0]
            size = struct.unpack("<Q", f.read(8))[0]
            name_off = struct.unpack("<I", f.read(4))[0]
            name_len = struct.unpack("<I", f.read(4))[0]
            entries.append({
                "type": sec_type,
                "type_name": SECTION_TYPE_NAMES.get(sec_type, "UNKNOWN"),
                "offset": offset,
                "size": size,
                "name_off": name_off,
                "name_len": name_len,
            })

        name_pool = f.read(max((e["name_off"] + e["name_len"]) for e in entries) if entries else 0)
        for e in entries:
            e["name"] = name_pool[e["name_off"]:e["name_off"] + e["name_len"]].decode("utf-8")

    return {"version": version, "section_count": count, "sections": entries}


def state_dict_to_npz_bytes(state_dict: Dict, quantize_to_fp16: bool = True) -> bytes:
    """Serialize a PyTorch state_dict to NPZ bytes.

    When quantize_to_fp16 is True, floating-point tensors are stored as fp16
    to roughly halve the bundle size — matching the bandwidth profile that
    on-device LLM runtimes typically target.
    """
    arrays: Dict[str, np.ndarray] = {}
    for name, tensor in state_dict.items():
        arr = tensor.detach().cpu().numpy()
        if quantize_to_fp16 and np.issubdtype(arr.dtype, np.floating):
            arr = arr.astype(np.float16)
        arrays[name] = np.ascontiguousarray(arr)

    buf = io.BytesIO()
    np.savez(buf, **arrays)
    return buf.getvalue()


def build_metadata(
    *,
    model_name: str,
    architecture: str,
    size: str,
    config: Dict,
    n_params: int,
    weights_dtype: str,
    quantization: str,
    runtime_defaults: Dict,
) -> bytes:
    payload = {
        "schema_version": 1,
        "model": {
            "name": model_name,
            "architecture": architecture,
            "size": size,
            "n_parameters": int(n_params),
            "weights_dtype": weights_dtype,
            "quantization": quantization,
        },
        "config": config,
        "runtime_defaults": runtime_defaults,
        "tokenizer": {
            "type": "sentencepiece",
            "section": SECTION_TYPE_NAMES[SECTION_SP_MODEL],
        },
        "created_at": datetime.now(timezone.utc).isoformat(),
        "creator": "tapiocaTakeshi",
        "repository": "https://github.com/tapiocatakeshi/qubit",
    }
    return json.dumps(payload, indent=2, ensure_ascii=False).encode("utf-8")


def build_model_card(
    *,
    repo_id: str,
    model_name: str,
    size: str,
    n_params: int,
    file_size_mb: float,
    config: Dict,
) -> bytes:
    md = f"""---
license: mit
language: ja
library_name: litert-lm
pipeline_tag: text-generation
tags:
  - litertlm
  - litert-lm
  - on-device
  - neuroquantum
  - qbnn
  - quantum
---

# {model_name} ({size}) — LiteRT-LM bundle

NeuroQuantum / QBNN モデルの `.litertlm` バンドル形式での配布です。

## ファイル構成

- `qubit-neuroquantum-{size}.litertlm` — モデル本体（重み + トークナイザ + メタデータ）
- `README.md` — このモデルカード

## モデル仕様

- **Architecture**: NeuroQuantum (QBNN-based)
- **Size**: `{size}`
- **Parameters**: {n_params:,}
- **File Size**: {file_size_mb:.2f} MB
- **Weights dtype**: float16
- **embed_dim**: {config.get('embed_dim')}
- **hidden_dim**: {config.get('hidden_dim')}
- **num_heads**: {config.get('num_heads')}
- **num_layers**: {config.get('num_layers')}
- **max_seq_len**: {config.get('max_seq_len')}
- **vocab_size**: {config.get('vocab_size')}

## バンドル形式

`.litertlm` は Google の LiteRT-LM コンテナ形式に準拠した単一ファイルバンドルです。
ヘッダー `LITERTLM` で始まり、以下のセクションを含みます。

| Section | 内容 |
|---|---|
| `TFLITE_MODEL_OR_WEIGHTS` | モデル重み (NPZ archive, fp16) |
| `SP_MODEL` | SentencePiece トークナイザ |
| `LLM_METADATA_JSON` | モデル/ランタイム設定 (JSON) |
| `MODEL_CARD_MD` | このモデルカード |

## 読み込み例

```python
from generate_litertlm_models import read_litertlm_header
print(read_litertlm_header("qubit-neuroquantum-{size}.litertlm"))
```

リポジトリ: https://github.com/tapiocatakeshi/Qubit
"""
    return md.encode("utf-8")


def build_runtime_defaults(config: Dict) -> Dict:
    return {
        "n_ctx": min(int(config.get("max_seq_len", 1024)), 4096),
        "n_batch": 64,
        "n_threads": 4,
        "n_gpu_layers": 0,
        "temperature": 0.8,
        "top_p": 0.9,
        "top_k": 40,
    }


def load_tokenizer_bytes(repo_root: Path) -> Optional[bytes]:
    candidates = [
        repo_root / "neuroq_tokenizer.model",
        repo_root / "neuroq_tokenizer_8k.model",
    ]
    for candidate in candidates:
        if not candidate.exists():
            continue
        data = candidate.read_bytes()
        if data.startswith(b"version https://git-lfs"):
            print(f"  ! {candidate.name} is an LFS pointer (size={len(data)}B); "
                  f"skipping. Run `git lfs pull` to fetch the real file.")
            continue
        return data
    return None


def build_one_litertlm(
    *,
    size: str,
    out_dir: Path,
    repo_root: Path,
    repo_id_for_card: str,
    vocab_size: int,
) -> Dict:
    import torch  # local import — keeps the module importable in dry-run mode
    from neuroquantum_layered import (  # type: ignore
        NeuroQuantum,
        NeuroQuantumConfig,
        get_model_config_by_size,
    )

    print(f"\n=== Building {size} ===")
    config_dict = get_model_config_by_size(size=size, vocab_size=vocab_size)
    config = NeuroQuantumConfig(
        vocab_size=config_dict["vocab_size"],
        embed_dim=config_dict["embed_dim"],
        hidden_dim=config_dict["hidden_dim"],
        num_heads=config_dict["num_heads"],
        num_layers=config_dict["num_layers"],
        max_seq_len=config_dict["max_seq_len"],
        lambda_entangle=config_dict["entangle_strength"],
        dropout=config_dict["dropout"],
    )

    model = NeuroQuantum(config=config)
    model.eval()

    # One forward pass with random tokens to materialize lazy buffers.
    with torch.no_grad():
        sample = torch.randint(0, vocab_size, (1, 8))
        try:
            _ = model(sample)
        except Exception as e:
            print(f"  ! warmup forward pass failed (continuing): {e}")

    state_dict = model.state_dict()
    n_params = sum(p.numel() for p in state_dict.values())
    print(f"  parameters: {n_params:,}")

    weights_bytes = state_dict_to_npz_bytes(state_dict, quantize_to_fp16=True)
    print(f"  weights (fp16 NPZ): {len(weights_bytes) / 1e6:.2f} MB")

    sp_model_bytes = load_tokenizer_bytes(repo_root)
    if sp_model_bytes is None:
        # Use a tiny placeholder so the bundle stays well-formed.
        sp_model_bytes = b""
        print("  ! no real SentencePiece model available; SP_MODEL section will be empty")
    else:
        print(f"  sentencepiece: {len(sp_model_bytes) / 1024:.1f} KB")

    runtime_defaults = build_runtime_defaults(config_dict)
    metadata_bytes = build_metadata(
        model_name="Qubit-NeuroQuantum",
        architecture="neuroquantum",
        size=size,
        config=config_dict,
        n_params=n_params,
        weights_dtype="float16",
        quantization="none",
        runtime_defaults=runtime_defaults,
    )

    # Model card needs the final file size, so we build it after a first
    # write with a placeholder, then overwrite — but it's simpler to just
    # estimate before writing. Use the in-memory totals.
    estimated_size_mb = (
        len(weights_bytes) + len(sp_model_bytes) + len(metadata_bytes)
    ) / (1024 * 1024)
    model_card_bytes = build_model_card(
        repo_id=repo_id_for_card.format(size=size),
        model_name="Qubit-NeuroQuantum",
        size=size,
        n_params=n_params,
        file_size_mb=estimated_size_mb,
        config=config_dict,
    )

    sections = [
        (SECTION_TFLITE_MODEL_OR_WEIGHTS, "weights.npz", weights_bytes),
        (SECTION_SP_MODEL, "neuroq_tokenizer.model", sp_model_bytes),
        (SECTION_LLM_METADATA_JSON, "metadata.json", metadata_bytes),
        (SECTION_MODEL_CARD_MD, "README.md", model_card_bytes),
    ]

    out_path = out_dir / f"qubit-neuroquantum-{size}.litertlm"
    total_size = write_litertlm(out_path, sections)
    print(f"  -> wrote {out_path} ({total_size / (1024*1024):.2f} MB)")

    return {
        "size": size,
        "path": str(out_path),
        "bytes": total_size,
        "n_params": int(n_params),
        "config": config_dict,
        "runtime_defaults": runtime_defaults,
    }


def main():
    parser = argparse.ArgumentParser(description="Generate .litertlm bundles for Qubit models")
    parser.add_argument("--output-dir", default="litertlm_models")
    parser.add_argument(
        "--sizes",
        nargs="+",
        default=["small", "medium", "large", "xlarge"],
        choices=["small", "medium", "large", "xlarge"],
    )
    parser.add_argument("--vocab-size", type=int, default=32000)
    parser.add_argument(
        "--repo-id-template",
        default="tapiocaTakeshi/qubit-neuroquantum-{size}-litertlm",
        help="HF repo id template — only used in the model card.",
    )
    parser.add_argument(
        "--manifest",
        default="manifest.json",
        help="Manifest filename (written under --output-dir).",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    results = []
    for size in args.sizes:
        result = build_one_litertlm(
            size=size,
            out_dir=out_dir,
            repo_root=repo_root,
            repo_id_for_card=args.repo_id_template,
            vocab_size=args.vocab_size,
        )
        results.append(result)

    manifest = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "format": "litertlm",
        "format_version": LITERTLM_VERSION,
        "section_types": SECTION_TYPE_NAMES,
        "models": results,
    }
    manifest_path = out_dir / args.manifest
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
    print(f"\nManifest -> {manifest_path}")

    print("\nValidation:")
    for r in results:
        info = read_litertlm_header(Path(r["path"]))
        print(f"  {Path(r['path']).name}: v{info['version']}, "
              f"{info['section_count']} sections")


if __name__ == "__main__":
    main()
