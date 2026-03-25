# RunPod Serverless用Dockerfile
# NeuroQ QBNN - NeuroQuantum Transformer
FROM runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel-ubuntu22.04

# 作業ディレクトリの設定
WORKDIR /app

# 依存パッケージのリストをコピーしてインストール
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# アプリケーションのコードをコンテナ内にコピー
COPY neuroquantum_layered.py .
COPY handler.py .
COPY runpod_handler.py .
COPY runpod_manager.py .
COPY predict.py .
COPY neuroq_tokenizer.model .
COPY neuroq_tokenizer.vocab .
COPY neuroq_tokenizer_8k.model .
COPY training_history.json .

# 環境変数
ENV PYTHONUNBUFFERED=1
ENV MODEL_DIR=/app

# RunPodのサーバーレスハンドラーを起動
CMD ["python", "-u", "runpod_handler.py"]
