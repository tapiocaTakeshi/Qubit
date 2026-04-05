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
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("git")
    .pip_install(
        "torch>=2.0.0",
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
    gpu="T4",
    timeout=600,
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
        sys.path.insert(0, "/app")

        from handler import EndpointHandler

        # Volume内のチェックポイントを /app にコピー
        volume_ckpt = os.path.join(VOLUME_PATH, "neuroq_checkpoint.pt")
        app_ckpt = "/app/neuroq_checkpoint.pt"
        if os.path.exists(volume_ckpt) and not os.path.exists(app_ckpt):
            import shutil
            shutil.copy2(volume_ckpt, app_ckpt)
            print(f"[modal] Restored checkpoint from volume: {volume_ckpt}")

        self.handler = EndpointHandler(path="/app")
        print(f"[modal] Model loaded (checkpoint={self.handler.ckpt_path})")

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
        """学習後にチェックポイントをVolumeに保存"""
        import shutil
        import glob

        for pattern in ["neuroq_checkpoint.pt", "qbnn_checkpoint.pt"]:
            for src in glob.glob(f"/app/{pattern}"):
                dst = os.path.join(VOLUME_PATH, os.path.basename(src))
                shutil.copy2(src, dst)
                print(f"[modal] Synced checkpoint to volume: {dst}")

        volume.commit()

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
        return result

    @modal.fastapi_endpoint(method="POST")
    def train_qa(self, request: dict):
        """QA学習エンドポイント"""
        request.setdefault("action", "train_qa")
        result = self._process(request)
        self._sync_checkpoint()
        return result

    @modal.fastapi_endpoint(method="GET")
    def status(self):
        """モデルステータス"""
        return self._process({"action": "status"})

    @modal.method()
    def run(self, job_input: dict):
        """汎用メソッド（Modal SDK経由で呼び出し）"""
        result = self._process(job_input)

        action = job_input.get("action", "inference")
        if action.startswith("train"):
            self._sync_checkpoint()

        return result
