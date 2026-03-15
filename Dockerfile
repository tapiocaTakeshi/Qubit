# PyTorch環境をベースにする（RunPodのようなGPU環境を想定してCUDA対応コンテナを使います）
FROM pytorch/pytorch:2.1.2-cuda11.8-cudnn8-runtime

# 作業ディレクトリの設定
WORKDIR /app

# 必要なシステムパッケージのインストール
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# 依存パッケージのリストをコピーしてインストール
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# アプリケーションのコードをコンテナ内にコピー
COPY . .

# 標準出力のバッファリングを無効化（ログをリアルタイムで見るため）
ENV PYTHONUNBUFFERED=1

# RunPodのサーバーレスハンドラーを起動
CMD ["python", "handler.py"]
