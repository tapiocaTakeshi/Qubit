# RunPod Serverless用Dockerfile
# NeuroQ QBNN - NeuroQuantum Transformer
FROM runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04

# 作業ディレクトリの設定
WORKDIR /app

# RunPod用依存パッケージのインストール（modal/replicateは除外）
COPY requirements-runpod.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements-runpod.txt

# Build the tokenizer from the versioned vocabulary.  Image builds cannot
# depend on Hugging Face dataset downloads, and token IDs must stay aligned
# with the checkpoints' embedding table.
COPY build_tokenizer_from_vocab.py neuroq_tokenizer.vocab ./
RUN python build_tokenizer_from_vocab.py \
    --vocab /app/neuroq_tokenizer.vocab \
    --output /app/neuroq_tokenizer.model

# コアモデルアーキテクチャ
COPY neuroquantum_layered.py .
COPY qbnn_layered.py .

# ハンドラー & ユーティリティ
COPY handler.py .
COPY neuroquantum_agent.py neuroquantum_search.py ./
COPY neuroquantum_agent_protocol.py train_agent.py ./
COPY runpod_handler.py .
COPY runpod_manager.py .
COPY dataset_utils.py .
COPY dpo_utils.py .
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
