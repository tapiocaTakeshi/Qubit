from huggingface_hub import HfApi

api   = HfApi()
token = "新しいhf_トークン"  # ← ここだけ変える
repo  = "tapiocaTakeshi/neuroQ"

for filename in ["app.py", "requirements.txt"]:
    api.upload_file(
        path_or_fileobj=filename,
        path_in_repo=filename,
        repo_id=repo,
        repo_type="space",
        token=token,
        commit_message="Add dataset training UI",
    )
    print(f"✅ {filename} アップロード完了")

print("🎉 デプロイ完了！")
