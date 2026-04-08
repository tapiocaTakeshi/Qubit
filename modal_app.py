"""
Modal Serverless Deployment for NeuroQ QBNN
NeuroQuantum Transformer - Japanese Language Model

Modal.com上でサーバーレスGPU推論・学習エンドポイントを提供します。

Usage:
    # ローカルテスト
    modal serve modal_app.py

    # デプロイ
    modal deploy modal_app.py

    # 推論テスト
    curl -X POST https://<your-app>.modal.run/inference \
         -H "Content-Type: application/json" \
         -d '{"prompt": "こんにちは", "parameters": {"max_new_tokens": 100}}'
"""

import os
import modal

# ==============================================================
# Modal Image Definition
# ==============================================================

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git")
    .pip_install(
        "torch>=2.4.0",
        "numpy>=1.24.0",
        "sentencepiece>=0.1.99",
        "huggingface-hub>=0.30.0,<1.0",
        "datasets>=2.18.0",
        "accelerate>=1.1.0",
        "fastapi[standard]",
    )
    .add_local_file("neuroquantum_layered.py", "/app/neuroquantum_layered.py", copy=True)
    .add_local_file("handler.py", "/app/handler.py", copy=True)
    .add_local_file("dataset_utils.py", "/app/dataset_utils.py", copy=True)
    .add_local_file("progress_logger.py", "/app/progress_logger.py", copy=True)
    .add_local_file("training_history.json", "/app/training_history.json", copy=True)
    .add_local_file("train_tokenizer.py", "/app/train_tokenizer.py", copy=True)
    .run_commands("cd /app && python train_tokenizer.py 8000 /app 20000")
)

# ==============================================================
# Modal App
# ==============================================================

app = modal.App(name="qubit_ai", image=image)

# チェックポイント永続化用 Volume
volume = modal.Volume.from_name("neuroq-checkpoints", create_if_missing=True)
VOLUME_PATH = "/data/checkpoints"


@app.cls(
    gpu="A100-80GB",
    timeout=86400,
    scaledown_window=120,
    volumes={VOLUME_PATH: volume},
)
@modal.concurrent(max_inputs=4)
class NeuroQService:
    """Modal serverless class for NeuroQ inference and training."""

    @modal.enter()
    def load_model(self):
        """コンテナ起動時にモデルをロード（コールドスタート時1回のみ）"""
        import sys
        import shutil
        import time
        sys.path.insert(0, "/app")

        # MODAL_VOLUME_PATH を設定して handler が Volume を認識できるようにする
        os.environ["MODAL_VOLUME_PATH"] = VOLUME_PATH

        from handler import EndpointHandler

        print("[modal] ========== Container Startup ==========")

        # Volume を最新状態にリフレッシュ（他コンテナの commit を反映）
        reload_start = time.time()
        volume.reload()
        reload_elapsed = time.time() - reload_start
        print(f"[modal] Volume reloaded ({reload_elapsed:.1f}s)")

        # Volume内のファイル一覧をログ出力
        if os.path.isdir(VOLUME_PATH):
            volume_files = os.listdir(VOLUME_PATH)
            print(f"[modal] Volume contents ({VOLUME_PATH}): {volume_files}")
            for vf in volume_files:
                vf_path = os.path.join(VOLUME_PATH, vf)
                if os.path.isfile(vf_path):
                    size_mb = os.path.getsize(vf_path) / (1024 * 1024)
                    print(f"[modal]   {vf}: {size_mb:.2f}MB")
        else:
            print(f"[modal] Volume path not found: {VOLUME_PATH}")

        # Volume内のチェックポイントを /app にコピー（全候補ファイルを探索）
        checkpoint_names = [
            "qbnn_checkpoint.pt",
            "neuroq_checkpoint.pt",
            "checkpoint.pt",
            "model.pt",
        ]
        restored_ckpt = False
        for name in checkpoint_names:
            volume_ckpt = os.path.join(VOLUME_PATH, name)
            app_ckpt = os.path.join("/app", name)
            if os.path.exists(volume_ckpt) and not os.path.exists(app_ckpt):
                copy_start = time.time()
                shutil.copy2(volume_ckpt, app_ckpt)
                copy_elapsed = time.time() - copy_start
                size_mb = os.path.getsize(app_ckpt) / (1024 * 1024)
                print(
                    f"[modal] Restored checkpoint: {name} "
                    f"({size_mb:.1f}MB, {copy_elapsed:.1f}s)"
                )
                restored_ckpt = True

        # Volume内のトークナイザーを /app にコピー（学習時のトークナイザーを優先）
        for tok_name in ["neuroq_tokenizer.model", "neuroq_tokenizer_8k.model"]:
            volume_tok = os.path.join(VOLUME_PATH, tok_name)
            app_tok = os.path.join("/app", tok_name)
            if os.path.exists(volume_tok):
                shutil.copy2(volume_tok, app_tok)
                size_kb = os.path.getsize(app_tok) / 1024
                print(f"[modal] Restored tokenizer: {tok_name} ({size_kb:.1f}KB)")

        if not restored_ckpt:
            print("[modal] No checkpoint found in volume, starting fresh")

        self.handler = EndpointHandler(path="/app")
        print(f"[modal] Model loaded (checkpoint={self.handler.ckpt_path})")
        print("[modal] ========== Startup Complete ==========")

    def _process(self, job_input: dict) -> dict:
        """EndpointHandler形式に変換して実行"""
        data = {}

        if "action" in job_input:
            data["action"] = job_input["action"]

        if "prompt" in job_input:
            data["inputs"] = job_input["prompt"]
        elif "inputs" in job_input:
            data["inputs"] = job_input["inputs"]

        if "parameters" in job_input:
            data["parameters"] = job_input["parameters"]

        for key in ("qa_pairs", "dataset_ids", "epochs", "lr", "batch_size",
                     "mode", "num_chunks", "resume"):
            if key in job_input:
                data.setdefault("parameters", {})[key] = job_input[key]

        result = self.handler(data)

        if isinstance(result, list) and len(result) == 1:
            return result[0]
        return result

    def _sync_checkpoint(self):
        """学習後にチェックポイントとトークナイザーをVolumeに保存（詳細ログ付き）"""
        import shutil
        import glob
        import time

        sync_start = time.time()
        synced_files = []
        total_bytes = 0

        print("[modal] ========== Volume Sync Start ==========")
        print(f"[modal] Volume path: {VOLUME_PATH}")

        # チェックポイント同期
        for pattern in ["neuroq_checkpoint.pt", "qbnn_checkpoint.pt"]:
            for src in glob.glob(f"/app/{pattern}"):
                dst = os.path.join(VOLUME_PATH, os.path.basename(src))
                src_size = os.path.getsize(src)
                copy_start = time.time()
                shutil.copy2(src, dst)
                copy_elapsed = time.time() - copy_start
                size_mb = src_size / (1024 * 1024)
                total_bytes += src_size
                synced_files.append(os.path.basename(src))
                print(
                    f"[modal] Checkpoint synced: {os.path.basename(src)} "
                    f"({size_mb:.1f}MB, {copy_elapsed:.1f}s)"
                )

        if not synced_files:
            print("[modal] WARNING: No checkpoint files found in /app/")

        # トークナイザー同期（イメージ再ビルド時のトークナイザー不一致を防止）
        for tok_name in ["neuroq_tokenizer.model", "neuroq_tokenizer_8k.model"]:
            tok_src = os.path.join("/app", tok_name)
            if os.path.isfile(tok_src):
                tok_dst = os.path.join(VOLUME_PATH, tok_name)
                tok_size = os.path.getsize(tok_src)
                shutil.copy2(tok_src, tok_dst)
                size_kb = tok_size / 1024
                total_bytes += tok_size
                synced_files.append(tok_name)
                print(f"[modal] Tokenizer synced: {tok_name} ({size_kb:.1f}KB)")

        # Volume commit
        commit_start = time.time()
        volume.commit()
        commit_elapsed = time.time() - commit_start

        total_elapsed = time.time() - sync_start
        total_mb = total_bytes / (1024 * 1024)
        print(f"[modal] Volume commit: {commit_elapsed:.1f}s")
        print(
            f"[modal] Sync complete: {len(synced_files)} files, "
            f"{total_mb:.1f}MB total, {total_elapsed:.1f}s"
        )
        print(f"[modal] Synced files: {synced_files}")
        print("[modal] ========== Volume Sync End ==========")

    @modal.fastapi_endpoint(method="POST")
    def inference(self, request: dict):
        """推論エンドポイント"""
        request.setdefault("action", "inference")
        return self._process(request)

    @modal.fastapi_endpoint(method="POST")
    def train(self, request: dict):
        """学習エンドポイント"""
        request.setdefault("action", "train")
        result = self._process(request)
        self._sync_checkpoint()
        print(f"[modal] Train complete. Result status: {result.get('status', 'unknown') if isinstance(result, dict) else 'ok'}")
        return result

    @modal.fastapi_endpoint(method="POST")
    def train_qa(self, request: dict):
        """QA学習エンドポイント"""
        request.setdefault("action", "train_qa")
        result = self._process(request)
        self._sync_checkpoint()
        print(f"[modal] Train QA complete. Result status: {result.get('status', 'unknown') if isinstance(result, dict) else 'ok'}")
        return result

    @modal.fastapi_endpoint(method="GET")
    def status(self):
        """モデルステータス"""
        return self._process({"action": "status"})

    @modal.method()
    def run(self, job_input: dict):
        """汎用メソッド（Modal SDK経由で呼び出し）"""
        action = job_input.get("action", "inference")
        print(f"[modal] run() called with action={action}")

        result = self._process(job_input)

        if action.startswith("train"):
            self._sync_checkpoint()
            print(f"[modal] run() training complete. Result status: {result.get('status', 'unknown') if isinstance(result, dict) else 'ok'}")

        return result
