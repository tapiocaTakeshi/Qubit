# RunPod公式ベースイメージを使用（Docker Hub認証不要）
FROM runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel-ubuntu22.04

# 作業ディレクトリの設定
WORKDIR /app

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
