"""
Modal Serverless Deployment for NeuroQ QBNN
NeuroQuantum Transformer - Japanese Language Model

Modal.com上でサーバーレスGPU推論・学習エンドポイントを提供します。
api.py の全エンドポイントを Modal 上で提供します。

Usage:
    # ローカルテスト
    modal serve modal_app.py

    # デプロイ
    modal deploy modal_app.py

    # 推論テスト
    curl -X POST https://<your-app>.modal.run/inference \
         -H "Content-Type: application/json" \
         -d '{"prompt": "こんにちは", "parameters": {"max_new_tokens": 100}}'

Endpoints:
    POST /inference          - テキスト生成
    POST /train              - 一般学習
    POST /train_qa           - QA学習
    POST /train_qa_dataset   - QA形式HFデータセット学習
    POST /train_split        - 分割データセット学習（全チャンク）
    POST /train_split_next   - 次のチャンクのみ学習（タイムアウト防止）
    POST /train_split_reset  - 分割学習セッションリセット
    POST /train_split_learning - 分割学習（モデル分割）
    POST /tts                - テキスト音声合成（Replicate経由）
    POST /reload             - モデル再読み込み
    GET  /status             - モデルステータス
    GET  /train_status       - 学習進捗ステータス
    GET  /train_split_status - 分割学習進捗
    GET  /health             - ヘルスチェック
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
        "replicate>=0.25.0",
    )
    .add_local_file("neuroquantum_layered.py", "/app/neuroquantum_layered.py", copy=True)
    .add_local_file("handler.py", "/app/handler.py", copy=True)
    .add_local_file("dataset_utils.py", "/app/dataset_utils.py", copy=True)
    .add_local_file("progress_logger.py", "/app/progress_logger.py", copy=True)
    .add_local_file("training_history.json", "/app/training_history.json", copy=True)
    .add_local_file("train_tokenizer.py", "/app/train_tokenizer.py", copy=True)
    .add_local_file("split_learning.py", "/app/split_learning.py", copy=True)
    .add_local_file("train_split_learning.py", "/app/train_split_learning.py", copy=True)
    .add_local_file("tts_replicate.py", "/app/tts_replicate.py", copy=True)
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

        # トップレベルのパラメータを parameters に透過的に渡す
        _passthrough_keys = (
            "qa_pairs", "dataset_ids", "dataset_id",
            "epochs", "lr", "batch_size", "grad_accum_steps", "warmup_steps",
            "max_samples_per_dataset", "max_samples",
            "mode", "data_mode", "num_chunks", "samples_per_batch",
            "chunk_index", "start_sample", "end_sample",
            "max_minutes_per_chunk", "max_minutes",
            "epochs_per_chunk", "crafted_repeat",
            "cut_layer", "grad_clip",
            "resume",
        )
        for key in _passthrough_keys:
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

    # ==============================================================
    # 推論エンドポイント
    # ==============================================================

    @modal.fastapi_endpoint(method="POST")
    def inference(self, request: dict):
        """推論エンドポイント"""
        request.setdefault("action", "inference")
        return self._process(request)

    # ==============================================================
    # 学習エンドポイント
    # ==============================================================

    @modal.fastapi_endpoint(method="POST")
    def train(self, request: dict):
        """一般学習エンドポイント"""
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

    @modal.fastapi_endpoint(method="POST")
    def train_qa_dataset(self, request: dict):
        """QA形式HFデータセット学習エンドポイント"""
        request.setdefault("action", "train_qa_dataset")
        result = self._process(request)
        self._sync_checkpoint()
        print(f"[modal] Train QA Dataset complete. Result status: {result.get('status', 'unknown') if isinstance(result, dict) else 'ok'}")
        return result

    @modal.fastapi_endpoint(method="POST")
    def train_split(self, request: dict):
        """分割データセット学習エンドポイント（全チャンク）"""
        request.setdefault("action", "train_split")
        result = self._process(request)
        self._sync_checkpoint()
        print(f"[modal] Train Split complete. Result status: {result.get('status', 'unknown') if isinstance(result, dict) else 'ok'}")
        return result

    @modal.fastapi_endpoint(method="POST")
    def train_split_next(self, request: dict):
        """次の1チャンクだけ学習するエンドポイント（タイムアウト防止用）

        使い方:
          1. POST /train_split_next を呼ぶ → チャンク1を学習
          2. レスポンスの chunks_remaining > 0 なら再度呼ぶ
          3. chunks_remaining == 0 になるまで繰り返す
          4. リセットは POST /train_split_reset を呼ぶ
        """
        request.setdefault("action", "train_split_next")
        result = self._process(request)
        self._sync_checkpoint()
        print(f"[modal] Train Split Next complete. Result: {result if isinstance(result, dict) else 'ok'}")
        return result

    @modal.fastapi_endpoint(method="POST")
    def train_split_reset(self):
        """分割学習セッションリセット（履歴は保持）"""
        return self._process({"action": "split_reset"})

    @modal.fastapi_endpoint(method="POST")
    def train_split_learning(self, request: dict):
        """分割学習（Split Learning）エンドポイント。モデルをカットレイヤーで分割して学習する。"""
        request.setdefault("action", "train_split_learning")
        result = self._process(request)
        self._sync_checkpoint()
        print(f"[modal] Split Learning complete. Result status: {result.get('status', 'unknown') if isinstance(result, dict) else 'ok'}")
        return result

    # ==============================================================
    # ステータス・ユーティリティエンドポイント
    # ==============================================================

    @modal.fastapi_endpoint(method="GET")
    def health(self):
        """ヘルスチェック"""
        handler = self.handler
        model_info = {
            "name": "NeuroQuantum API (Modal)",
            "version": "1.0.0",
            "model": "neuroquantum",
            "checkpoint": getattr(handler, "ckpt_path", None),
            "parameters": sum(p.numel() for p in handler.model.parameters()) if hasattr(handler, "model") and handler.model else 0,
        }
        if hasattr(handler, "config") and handler.config:
            model_info["config"] = handler.config
        return model_info

    @modal.fastapi_endpoint(method="GET")
    def status(self):
        """モデルステータス"""
        return self._process({"action": "status"})

    @modal.fastapi_endpoint(method="GET")
    def train_status(self):
        """学習進捗ステータス"""
        handler = self.handler
        ts = getattr(handler, "training_status", {"running": False, "log": [], "message": "idle"})
        return {"running": ts["running"], "log": ts["log"], "message": ts["message"]}

    @modal.fastapi_endpoint(method="GET")
    def train_split_status(self):
        """分割学習進捗ステータス"""
        return self._process({"action": "split_status"})

    @modal.fastapi_endpoint(method="POST")
    def reload(self):
        """モデル再読み込み（Volumeから最新チェックポイントをロード）"""
        import sys
        import shutil
        import time
        sys.path.insert(0, "/app")

        print("[modal] ========== Model Reload ==========")

        # Volume を最新状態にリフレッシュ
        reload_start = time.time()
        volume.reload()
        reload_elapsed = time.time() - reload_start
        print(f"[modal] Volume reloaded ({reload_elapsed:.1f}s)")

        # Volume内のチェックポイントを /app にコピー
        checkpoint_names = [
            "qbnn_checkpoint.pt",
            "neuroq_checkpoint.pt",
            "checkpoint.pt",
            "model.pt",
        ]
        for name in checkpoint_names:
            volume_ckpt = os.path.join(VOLUME_PATH, name)
            app_ckpt = os.path.join("/app", name)
            if os.path.exists(volume_ckpt):
                shutil.copy2(volume_ckpt, app_ckpt)
                size_mb = os.path.getsize(app_ckpt) / (1024 * 1024)
                print(f"[modal] Restored checkpoint: {name} ({size_mb:.1f}MB)")

        # トークナイザーもコピー
        for tok_name in ["neuroq_tokenizer.model", "neuroq_tokenizer_8k.model"]:
            volume_tok = os.path.join(VOLUME_PATH, tok_name)
            app_tok = os.path.join("/app", tok_name)
            if os.path.exists(volume_tok):
                shutil.copy2(volume_tok, app_tok)

        # handler を再初期化
        from handler import EndpointHandler
        self.handler = EndpointHandler(path="/app")
        print(f"[modal] Model reloaded (checkpoint={self.handler.ckpt_path})")
        print("[modal] ========== Reload Complete ==========")

        return {"status": "reloaded", "checkpoint": self.handler.ckpt_path}

    # ==============================================================
    # TTS（テキスト音声合成）エンドポイント
    # ==============================================================

    @modal.fastapi_endpoint(method="POST")
    def tts(self, request: dict):
        """テキスト音声合成（Replicate経由）"""
        import sys
        sys.path.insert(0, "/app")

        text = request.get("text", "")
        voice_id = request.get("voice_id", "Ashley")

        if not text:
            return {"error": "text is required"}

        try:
            from tts_replicate import text_to_speech
            output = text_to_speech(text=text, voice_id=voice_id)
            return {"status": "ok", "output": output}
        except Exception as e:
            return {"error": str(e)}

    # ==============================================================
    # 汎用メソッド（Modal SDK経由）
    # ==============================================================

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
