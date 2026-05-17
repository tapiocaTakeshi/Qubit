# RunPod Serverless用Dockerfile
# NeuroQ QBNN - NeuroQuantum Transformer
FROM runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04

# 作業ディレクトリの設定
WORKDIR /app

# RunPod用依存パッケージのインストール（modal/replicateは除外）
COPY requirements-runpod.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements-runpod.txt

# Train SentencePiece tokenizer during build (avoids Git LFS issues)
COPY train_tokenizer.py .
RUN python train_tokenizer.py 8000 /app 20000

# コアモデルアーキテクチャ
COPY neuroquantum_layered.py .
COPY qbnn_layered.py .

# ハンドラー & ユーティリティ
COPY handler.py .
COPY runpod_handler.py .
COPY runpod_manager.py .
COPY dataset_utils.py .
COPY progress_logger.py .

# 分割学習（split learning）サポート
COPY split_learning.py .
COPY train_split_learning.py .

# トレーニング履歴
COPY training_history.json .

# 環境変数
ENV PYTHONUNBUFFERED=1
ENV MODEL_DIR=/app
ENV NETWORK_VOLUME_PATH=/runpod-volume

# RunPodのサーバーレスハンドラーを起動
CMD ["python", "-u", "handler.py"]
