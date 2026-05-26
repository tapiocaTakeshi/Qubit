#!/usr/bin/env python3
"""
Upload .litertlm bundles to Hugging Face Hub.

Walks the output directory of generate_litertlm_models.py and uploads each
.litertlm file to its own HF repo, named per the --repo-id-template.

Usage:
    HF_TOKEN=hf_... python upload_litertlm_to_huggingface.py \\
        --litertlm-dir litertlm_models \\
        --repo-id-template "tapiocaTakeshi/qubit-neuroquantum-{size}-litertlm"
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Optional

from huggingface_hub import HfApi


def extract_size_from_filename(path: Path) -> Optional[str]:
    m = re.match(r"qubit-neuroquantum-(small|medium|large|xlarge)\.litertlm$", path.name)
    return m.group(1) if m else None


def upload_one(
    *,
    api: HfApi,
    litertlm_path: Path,
    repo_id: str,
    private: bool,
    readme_text: Optional[str],
) -> str:
    print(f"\n--- {litertlm_path.name} -> {repo_id} ---")
    api.create_repo(repo_id=repo_id, repo_type="model", private=private, exist_ok=True)

    print(f"  uploading {litertlm_path.name} "
          f"({litertlm_path.stat().st_size / (1024*1024):.2f} MB)")
    api.upload_file(
        path_or_fileobj=str(litertlm_path),
        path_in_repo=litertlm_path.name,
        repo_id=repo_id,
        repo_type="model",
        commit_message=f"Add {litertlm_path.name}",
    )

    if readme_text:
        print("  uploading README.md")
        api.upload_file(
            path_or_fileobj=readme_text.encode("utf-8"),
            path_in_repo="README.md",
            repo_id=repo_id,
            repo_type="model",
            commit_message="Add model card",
        )

    return f"https://huggingface.co/{repo_id}"


def build_readme(size: str, manifest_entry: dict) -> str:
    cfg = manifest_entry.get("config", {})
    n_params = manifest_entry.get("n_params", 0)
    file_size_mb = manifest_entry.get("bytes", 0) / (1024 * 1024)
    runtime = manifest_entry.get("runtime_defaults", {})
    return f"""---
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
---

# Qubit NeuroQuantum ({size}) — LiteRT-LM bundle

NeuroQuantum / QBNN モデルを Google LiteRT-LM 互換のバンドル形式
(`.litertlm`) で配布しています。

## ファイル

- `qubit-neuroquantum-{size}.litertlm` — モデルバンドル ({file_size_mb:.2f} MB)
  - `TFLITE_MODEL_OR_WEIGHTS` セクション: モデル重み (fp16 NPZ archive)
  - `SP_MODEL` セクション: SentencePiece トークナイザ
  - `LLM_METADATA_JSON` セクション: モデル / ランタイム設定
  - `MODEL_CARD_MD` セクション: モデルカード

## モデル仕様

| Field | Value |
|---|---|
| Architecture | NeuroQuantum |
| Size | `{size}` |
| Parameters | {n_params:,} |
| embed_dim | {cfg.get('embed_dim')} |
| hidden_dim | {cfg.get('hidden_dim')} |
| num_heads | {cfg.get('num_heads')} |
| num_layers | {cfg.get('num_layers')} |
| max_seq_len | {cfg.get('max_seq_len')} |
| vocab_size | {cfg.get('vocab_size')} |

## 推奨ランタイム設定

```json
{json.dumps(runtime, indent=2)}
```

## ロード例

```python
import struct, json
from pathlib import Path

MAGIC = b"LITERTLM"
with open("qubit-neuroquantum-{size}.litertlm", "rb") as f:
    assert f.read(8) == MAGIC
    version, count = struct.unpack("<II", f.read(8))
    # ... see generate_litertlm_models.py:read_litertlm_header
```

完全なパーサ実装と生成スクリプトは
[tapiocatakeshi/Qubit](https://github.com/tapiocatakeshi/Qubit) の
`generate_litertlm_models.py` を参照してください。

## License

MIT
"""


def main():
    parser = argparse.ArgumentParser(description="Upload .litertlm files to HF Hub")
    parser.add_argument("--litertlm-dir", default="litertlm_models")
    parser.add_argument(
        "--repo-id-template",
        default="tapiocaTakeshi/qubit-neuroquantum-{size}-litertlm",
    )
    parser.add_argument("--private", action="store_true")
    parser.add_argument(
        "--token",
        default=None,
        help="HF token. Falls back to $HF_TOKEN.",
    )
    args = parser.parse_args()

    token = args.token or os.getenv("HF_TOKEN")
    if not token:
        print("ERROR: no HF token provided (pass --token or set HF_TOKEN)",
              file=sys.stderr)
        sys.exit(2)

    src = Path(args.litertlm_dir)
    if not src.is_dir():
        print(f"ERROR: --litertlm-dir not found: {src}", file=sys.stderr)
        sys.exit(2)

    manifest_path = src / "manifest.json"
    manifest = {}
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
    entries_by_size = {m["size"]: m for m in manifest.get("models", [])}

    files = sorted(src.glob("qubit-neuroquantum-*.litertlm"))
    if not files:
        print(f"ERROR: no .litertlm files in {src}", file=sys.stderr)
        sys.exit(2)

    api = HfApi(token=token)
    results = []
    for f in files:
        size = extract_size_from_filename(f)
        if size is None:
            print(f"  ! skipping (unrecognized name): {f.name}")
            continue
        repo_id = args.repo_id_template.format(size=size)
        readme = build_readme(size, entries_by_size.get(size, {}))
        try:
            url = upload_one(
                api=api,
                litertlm_path=f,
                repo_id=repo_id,
                private=args.private,
                readme_text=readme,
            )
            results.append({"size": size, "repo": repo_id, "url": url, "ok": True})
        except Exception as e:
            print(f"  ! upload failed: {e}")
            results.append({"size": size, "repo": repo_id, "ok": False, "error": str(e)})

    print("\nSummary:")
    for r in results:
        status = "OK " if r["ok"] else "FAIL"
        print(f"  [{status}] {r['size']:<6} {r['repo']}")
        if r["ok"]:
            print(f"           {r['url']}")
        else:
            print(f"           error: {r.get('error')}")

    if not all(r["ok"] for r in results):
        sys.exit(1)


if __name__ == "__main__":
    main()
