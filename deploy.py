“””
neuroQ デプロイスクリプト
環境変数 HF_TOKEN を使って HuggingFace Space にファイルをアップロードします

使い方:
export HF_TOKEN=hf_xxxx   # Mac/Linux
set HF_TOKEN=hf_xxxx      # Windows
python deploy.py
“””

import os
import sys
from pathlib import Path

# ── 設定 ──────────────────────────────────────

REPO_ID   = “tapiocaTakeshi/neuroQ”
REPO_TYPE = “space”
FILES     = [“app.py”, “requirements.txt”]

# ──────────────────────────────────────────────

def main():
# トークン取得
token = os.environ.get(“HF_TOKEN”) or os.environ.get(“HUGGING_FACE_HUB_TOKEN”)
if not token:
print(“❌ 環境変数 HF_TOKEN が設定されていません。”)
print()
print(“設定方法:”)
print(”  Mac/Linux: export HF_TOKEN=hf_xxxx”)
print(”  Windows:   set HF_TOKEN=hf_xxxx”)
sys.exit(1)

```
print(f"✅ トークン確認: {token[:8]}...")
print(f"📦 デプロイ先: {REPO_ID}")
print()

try:
    from huggingface_hub import HfApi
except ImportError:
    print("❌ huggingface_hub が未インストールです。")
    print("   pip install huggingface_hub")
    sys.exit(1)

api = HfApi()

for filename in FILES:
    path = Path(filename)
    if not path.exists():
        print(f"⚠️  {filename} が見つかりません（スキップ）")
        continue

    print(f"⬆️  {filename} をアップロード中...")
    try:
        api.upload_file(
            path_or_fileobj=str(path),
            path_in_repo=filename,
            repo_id=REPO_ID,
            repo_type=REPO_TYPE,
            token=token,
            commit_message=f"Deploy: update {filename}",
        )
        print(f"   ✅ {filename} 完了")
    except Exception as e:
        print(f"   ❌ エラー: {e}")
        sys.exit(1)

print()
print("🎉 デプロイ完了！")
print(f"👉 https://huggingface.co/spaces/{REPO_ID}")
```

if **name** == “**main**”:
main()