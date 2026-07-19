#!/usr/bin/env bash
# ============================================================
# Qubit (QBNN) を Hugging Face Jobs 上で
# SFT (QA形式) トレーニングするスクリプト
#
# データセット: kunishou/databricks-dolly-15k-ja（日本語QA最適化）
# モデルサイズ: small / medium / large / xlarge （カスタマイズ可能）
#
# 事前準備:
#   1. `huggingface-cli login` でログイン済みであること
#   2. https://huggingface.co/tapiocaTakeshi/Qubit の利用条件に同意済み
#   3. HF_TOKEN を Job シークレットとして設定
#      (書き込み権限: https://huggingface.co/settings/tokens)
#
# 実行方法:
#   chmod +x scripts/train_qubit_hfjobs.sh
#   ./scripts/train_qubit_hfjobs.sh                    # デフォルト (medium)
#   ./scripts/train_qubit_hfjobs.sh small              # small モデル
#   ./scripts/train_qubit_hfjobs.sh large --epochs 10  # large + エポック変更
#   ./scripts/train_qubit_hfjobs.sh medium --no-upload # アップロードなし
#
# 環境変数:
#   HF_TOKEN              Hugging Face API トークン（自動）
#   HF_UPLOAD_REPO        アップロード先リポジトリ（デフォルト: qubit-{size}-sft-q4km）
#   HF_JOBS_FLAVOR        GPU タイプ（デフォルト: a10g-small）
#   HF_JOBS_TIMEOUT       タイムアウト（デフォルト: 6h）
#
# ============================================================

set -euo pipefail

# ========== 設定 ==========
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# デフォルト設定
MODEL_SIZE="medium"
DATASET_ID="kunishou/databricks-dolly-15k-ja"
QUANTIZATION="${QUANTIZATION:-Q4_K_M}"
EPOCHS="${EPOCHS:-20}"
BATCH_SIZE="${BATCH_SIZE:-4}"
GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-4}"
LR="${LR:-3e-5}"
UPLOAD_ENABLED=true
JOBS_FLAVOR="${HF_JOBS_FLAVOR:-a10g-small}"
JOBS_TIMEOUT="${HF_JOBS_TIMEOUT:-6h}"
UPLOAD_REPO=""

# ========== 関数 ==========

log_info() {
    echo "📝 [$(date +'%Y-%m-%d %H:%M:%S')] $*"
}

log_error() {
    echo "❌ [$(date +'%Y-%m-%d %H:%M:%S')] ERROR: $*" >&2
}

log_success() {
    echo "✅ [$(date +'%Y-%m-%d %H:%M:%S')] $*"
}

validate_model_size() {
    case "$1" in
        small|medium|large|xlarge) return 0 ;;
        *)
            log_error "Invalid model size: $1. Must be one of: small, medium, large, xlarge"
            return 1
            ;;
    esac
}

show_usage() {
    cat << 'EOF'
使用方法: ./scripts/train_qubit_hfjobs.sh [MODEL_SIZE] [OPTIONS]

位置指定引数:
  MODEL_SIZE              モデルサイズ (small/medium/large/xlarge)
                          デフォルト: medium

オプション:
  --epochs N              トレーニングエポック数 (デフォルト: 20)
  --batch-size N          バッチサイズ (デフォルト: 4)
  --lr RATE               学習率 (デフォルト: 3e-5)
  --grad-accum N          グラディエント蓄積ステップ (デフォルト: 4)
  --quantization Q        量子化形式 (デフォルト: Q4_K_M)
  --upload-repo REPO      アップロード先リポジトリ ID
  --jobs-flavor FLAVOR    GPU タイプ (デフォルト: a10g-small)
  --jobs-timeout TIME     タイムアウト (デフォルト: 6h)
  --no-upload             Hub へのアップロードをスキップ
  --help                  ヘルプを表示

例:
  # デフォルト設定で実行
  ./scripts/train_qubit_hfjobs.sh

  # small モデルで実行
  ./scripts/train_qubit_hfjobs.sh small

  # large モデル、エポック 10、アップロードなし
  ./scripts/train_qubit_hfjobs.sh large --epochs 10 --no-upload

  # xlarge モデル、高性能 GPU、カスタム学習率
  ./scripts/train_qubit_hfjobs.sh xlarge --jobs-flavor a40-large --lr 1e-4
EOF
}

# ========== 引数解析 ==========

# Handle positional MODEL_SIZE argument first
if [[ $# -gt 0 ]] && ! [[ "$1" =~ ^-- ]]; then
    MODEL_SIZE="$1"
    shift
fi

while [[ $# -gt 0 ]]; do
    case "$1" in
        --epochs)
            EPOCHS="$2"
            shift 2
            ;;
        --batch-size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --lr)
            LR="$2"
            shift 2
            ;;
        --grad-accum)
            GRAD_ACCUM_STEPS="$2"
            shift 2
            ;;
        --quantization)
            QUANTIZATION="$2"
            shift 2
            ;;
        --upload-repo)
            UPLOAD_REPO="$2"
            shift 2
            ;;
        --jobs-flavor)
            JOBS_FLAVOR="$2"
            shift 2
            ;;
        --jobs-timeout)
            JOBS_TIMEOUT="$2"
            shift 2
            ;;
        --no-upload)
            UPLOAD_ENABLED=false
            shift
            ;;
        --help)
            show_usage
            exit 0
            ;;
        *)
            log_error "Unknown option: $1"
            show_usage
            exit 1
            ;;
    esac
done

# Set default upload repo if not specified
if [ -z "$UPLOAD_REPO" ]; then
    UPLOAD_REPO="${HF_UPLOAD_REPO:-tapiocaTakeshi/qubit-${MODEL_SIZE}-sft-q4km}"
fi

# ========== 検証 ==========

validate_model_size "$MODEL_SIZE" || exit 1

log_info "========== Qubit SFT トレーニング（Hugging Face Jobs） =========="
log_info "モデルサイズ:        $MODEL_SIZE"
log_info "データセット:        $DATASET_ID"
log_info "エポック:           $EPOCHS"
log_info "バッチサイズ:       $BATCH_SIZE"
log_info "学習率:             $LR"
log_info "GPU タイプ:         $JOBS_FLAVOR"
log_info "タイムアウト:       $JOBS_TIMEOUT"
if [ "$UPLOAD_ENABLED" = true ]; then
    log_info "アップロード先:     $UPLOAD_REPO"
fi
log_info "=========================================================="

# Hugging Face CLI が利用可能か確認
if ! command -v hf &> /dev/null; then
    log_error "hf CLI が見つかりません。以下をインストール: pip install huggingface_hub"
    exit 1
fi

# HF_TOKEN が設定されているか確認（ローカル実行時）
if [ -z "${HF_TOKEN:-}" ] && ! huggingface-cli whoami &> /dev/null; then
    log_error "認証エラー。以下を実行してください: huggingface-cli login"
    exit 1
fi

log_info "Hugging Face Jobs でトレーニングを開始..."

# ========== トレーニング実行 ==========

hf jobs run \
  --flavor "$JOBS_FLAVOR" \
  --secrets HF_TOKEN \
  --timeout "$JOBS_TIMEOUT" \
  pytorch/pytorch:2.6.0-cuda12.4-cudnn9-devel \
  bash -c "
    set -e

    # ============================================================
    # Job 内での実行スクリプト
    # ============================================================

    log_info() {
        echo \"📝 [\$(date +'%Y-%m-%d %H:%M:%S')] \$*\"
    }

    log_success() {
        echo \"✅ [\$(date +'%Y-%m-%d %H:%M:%S')] \$*\"
    }

    log_error() {
        echo \"❌ [\$(date +'%Y-%m-%d %H:%M:%S')] ERROR: \$*\" >&2
    }

    trap 'log_error \"Job 失敗\"' ERR

    log_info '========== ステップ 1: 依存パッケージをインストール =========='
    pip install --quiet -U huggingface_hub fastapi uvicorn datasets requests tqdm

    log_info '========== ステップ 2: Qubit リポジトリを取得 =========='
    hf download tapiocaTakeshi/Qubit --local-dir /workspace/qubit
    cd /workspace/qubit

    if [ -f requirements.txt ]; then
        log_info 'requirements.txt が見つかりました。インストール中...'
        pip install --quiet -r requirements.txt
    fi

    log_info '========== ステップ 3: API サーバーを起動 =========='
    python api.py &
    API_PID=\$!

    # サーバー起動待ち
    RETRY_COUNT=0
    MAX_RETRIES=30
    while [ \$RETRY_COUNT -lt \$MAX_RETRIES ]; do
        if curl -s http://localhost:8000/docs > /dev/null 2>&1; then
            log_success 'API サーバーの起動を確認'
            break
        fi
        RETRY_COUNT=\$((RETRY_COUNT + 1))
        sleep 2
    done

    if [ \$RETRY_COUNT -eq \$MAX_RETRIES ]; then
        log_error 'API サーバーの起動タイムアウト'
        kill \$API_PID 2>/dev/null || true
        exit 1
    fi

    log_info '========== ステップ 4: SFT (QA形式) トレーニングを開始 =========='

    TRAIN_RESPONSE=\$(curl -s -X POST http://localhost:8000/train/qa \\
      -H 'Content-Type: application/json' \\
      -d '{
        \"dataset_id\": \"$DATASET_ID\",
        \"epochs\": $EPOCHS,
        \"lr\": $LR,
        \"batch_size\": $BATCH_SIZE,
        \"grad_accum_steps\": $GRAD_ACCUM_STEPS,
        \"model_size\": \"$MODEL_SIZE\"
      }')

    log_info \"トレーニングレスポンス: \$TRAIN_RESPONSE\"

    log_info '========== ステップ 5: トレーニングステータスをポーリング =========='

    MAX_POLLS=360  # 最大 3 時間 (30秒 × 360)
    POLL_COUNT=0

    while [ \$POLL_COUNT -lt \$MAX_POLLS ]; do
        STATUS=\$(curl -s http://localhost:8000/train/status)
        echo \"[Poll \$POLL_COUNT] \$STATUS\"

        if echo \"\$STATUS\" | grep -q '\"completed\"'; then
            log_success 'トレーニング完了'
            break
        fi

        if echo \"\$STATUS\" | grep -q '\"error\"'; then
            log_error \"トレーニングエラー: \$STATUS\"
            kill \$API_PID 2>/dev/null || true
            exit 1
        fi

        POLL_COUNT=\$((POLL_COUNT + 1))
        sleep 30
    done

    if [ \$POLL_COUNT -eq \$MAX_POLLS ]; then
        log_error 'トレーニングがタイムアウト'
        kill \$API_PID 2>/dev/null || true
        exit 1
    fi

    log_info '========== ステップ 6: モデルをGGUF形式に変換 =========='

    if python generate_gguf_models.py --sizes $MODEL_SIZE --quantization $QUANTIZATION; then
        log_success 'GGUF 変換完了'
    else
        log_error 'GGUF 変換に失敗'
        kill \$API_PID 2>/dev/null || true
        exit 1
    fi
"

# Jobs 内でのアップロード処理
if [ "$UPLOAD_ENABLED" = true ]; then
    log_info "========== ステップ 7: Hugging Face Hub へアップロード =========="

    # アップロードは別の Jobs で実行（タイムアウト回避）
    hf jobs run \
      --flavor "$JOBS_FLAVOR" \
      --secrets HF_TOKEN \
      --timeout 2h \
      pytorch/pytorch:2.6.0-cuda12.4-cudnn9-devel \
      bash -c "
        set -e

        pip install --quiet -U huggingface_hub

        hf download tapiocaTakeshi/Qubit --local-dir /workspace/qubit
        cd /workspace/qubit

        if [ -f requirements.txt ]; then
            pip install --quiet -r requirements.txt
        fi

        echo '📝 GGUF ファイルをアップロード中...'
        python upload_to_huggingface.py \\
          --repo-name '$UPLOAD_REPO' \\
          --gguf-path 'gguf_models/qbnn_${MODEL_SIZE}_${QUANTIZATION}.gguf'

        echo '✅ アップロード完了'
      " || log_error "アップロード Job に失敗しました"

    log_success "モデルがアップロードされました: $UPLOAD_REPO"
else
    log_info "ステップ 7: Hub へのアップロードをスキップ"
fi

log_success "========== トレーニング完了 =========="
log_info "詳細は Hugging Face Jobs のダッシュボードで確認してください"
log_info "https://huggingface.co/docs/hub/security-tokens"
