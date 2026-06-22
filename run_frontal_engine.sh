#!/bin/bash

###############################################################################
# QBNN Frontal Engine スタンドアロン起動スクリプト
#
# 使用方法:
#   ./run_frontal_engine.sh start      # サーバーを起動
#   ./run_frontal_engine.sh stop       # サーバーを停止
#   ./run_frontal_engine.sh restart    # サーバーを再起動
#   ./run_frontal_engine.sh status     # ステータス確認
#   ./run_frontal_engine.sh logs       # ログ表示
#   ./run_frontal_engine.sh docker-build   # Docker イメージをビルド
#   ./run_frontal_engine.sh docker-run     # Docker コンテナを実行
###############################################################################

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$SCRIPT_DIR"
SERVICE_NAME="qbnn-frontal-engine"
PID_FILE="/tmp/${SERVICE_NAME}.pid"
LOG_FILE="/tmp/${SERVICE_NAME}.log"
PYTHON_CMD="python3"

# 色の定義
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# ロギング関数
log() {
    echo -e "${BLUE}[$(date '+%Y-%m-%d %H:%M:%S')]${NC} $*" | tee -a "$LOG_FILE"
}

log_success() {
    echo -e "${GREEN}✓${NC} $*" | tee -a "$LOG_FILE"
}

log_error() {
    echo -e "${RED}✗ Error:${NC} $*" >&2 | tee -a "$LOG_FILE"
}

log_warning() {
    echo -e "${YELLOW}⚠${NC} Warning: $*" | tee -a "$LOG_FILE"
}

# 依存パッケージをチェック
check_dependencies() {
    log "依存パッケージをチェック中..."

    if ! command -v "$PYTHON_CMD" &> /dev/null; then
        log_error "Python 3 がインストールされていません"
        exit 1
    fi

    log_success "Python: $($PYTHON_CMD --version)"

    # torch チェック
    if ! $PYTHON_CMD -c "import torch" 2>/dev/null; then
        log_warning "PyTorch がインストールされていません。インストール中..."
        pip install --no-cache-dir torch numpy mcp
    fi
}

# サーバーを起動
start_server() {
    log "QBNN Frontal Engine を起動中..."

    if [ -f "$PID_FILE" ]; then
        local old_pid=$(cat "$PID_FILE")
        if kill -0 "$old_pid" 2>/dev/null; then
            log_error "サーバーは既に実行中です (PID: $old_pid)"
            return 1
        else
            log_warning "古いPIDファイルを削除しています"
            rm "$PID_FILE"
        fi
    fi

    check_dependencies

    cd "$PROJECT_ROOT"
    nohup $PYTHON_CMD frontal_engine_mcp_server.py > "$LOG_FILE" 2>&1 &
    local new_pid=$!
    echo "$new_pid" > "$PID_FILE"

    sleep 2

    if kill -0 "$new_pid" 2>/dev/null; then
        log_success "サーバーが起動しました (PID: $new_pid)"
        return 0
    else
        log_error "サーバーの起動に失敗しました"
        cat "$LOG_FILE"
        return 1
    fi
}

# サーバーを停止
stop_server() {
    log "QBNN Frontal Engine を停止中..."

    if [ ! -f "$PID_FILE" ]; then
        log_warning "PIDファイルが見つかりません"
        return 0
    fi

    local pid=$(cat "$PID_FILE")

    if ! kill -0 "$pid" 2>/dev/null; then
        log_warning "プロセスは実行されていません"
        rm "$PID_FILE"
        return 0
    fi

    kill "$pid"
    sleep 2

    if kill -0 "$pid" 2>/dev/null; then
        log_warning "プロセスを強制終了します..."
        kill -9 "$pid"
        sleep 1
    fi

    rm "$PID_FILE"
    log_success "サーバーを停止しました"
}

# ステータスを表示
show_status() {
    echo "╔════════════════════════════════════════════════════════════╗"
    echo "║     QBNN Frontal Engine - Service Status                  ║"
    echo "╚════════════════════════════════════════════════════════════╝"
    echo ""

    if [ ! -f "$PID_FILE" ]; then
        echo -e "Status: ${RED}STOPPED${NC}"
        return 0
    fi

    local pid=$(cat "$PID_FILE")

    if kill -0 "$pid" 2>/dev/null; then
        echo -e "Status: ${GREEN}RUNNING${NC}"
        echo "PID: $pid"
        echo "Process: $(ps -p "$pid" -o comm=)"
        echo ""
        echo "Resource Usage:"
        ps -p "$pid" -o %cpu,%mem,vsz,rss | tail -1
    else
        echo -e "Status: ${RED}STOPPED${NC} (PID file exists but process not running)"
        rm "$PID_FILE"
    fi

    echo ""
    echo "Log File: $LOG_FILE"
    echo "Recent Logs:"
    tail -5 "$LOG_FILE" 2>/dev/null || echo "  (no logs yet)"
}

# ログを表示
show_logs() {
    if [ ! -f "$LOG_FILE" ]; then
        log_warning "ログファイルがまだ作成されていません"
        return
    fi

    echo "════════════════════════════════════════════════════════════"
    echo "QBNN Frontal Engine Logs"
    echo "════════════════════════════════════════════════════════════"
    tail -f "$LOG_FILE"
}

# Docker イメージをビルド
build_docker_image() {
    log "Docker イメージをビルド中..."
    cd "$PROJECT_ROOT"
    docker build -f Dockerfile.mcp -t "$SERVICE_NAME:latest" .
    log_success "Docker イメージのビルドが完了しました"
    docker images | grep "$SERVICE_NAME"
}

# Docker コンテナを実行
run_docker_container() {
    log "Docker コンテナを実行中..."

    docker run -it \
        --name "$SERVICE_NAME" \
        --rm \
        -v "$PROJECT_ROOT:/app" \
        -e PYTHONUNBUFFERED=1 \
        "$SERVICE_NAME:latest"
}

# Docker コンテナを実行（バックグラウンド）
run_docker_daemon() {
    log "Docker コンテナをバックグラウンドで実行中..."

    docker run -d \
        --name "$SERVICE_NAME" \
        -v "$PROJECT_ROOT:/app" \
        -e PYTHONUNBUFFERED=1 \
        --restart unless-stopped \
        "$SERVICE_NAME:latest"

    log_success "Docker コンテナが起動しました"
    docker ps | grep "$SERVICE_NAME"
}

# メイン処理
main() {
    local command="${1:-status}"

    case "$command" in
        start)
            start_server
            ;;
        stop)
            stop_server
            ;;
        restart)
            stop_server || true
            sleep 1
            start_server
            ;;
        status)
            show_status
            ;;
        logs)
            show_logs
            ;;
        docker-build)
            build_docker_image
            ;;
        docker-run)
            run_docker_container
            ;;
        docker-daemon)
            run_docker_daemon
            ;;
        test)
            log "テストを実行中..."
            cd "$PROJECT_ROOT"
            $PYTHON_CMD test_frontal_engine_light.py
            ;;
        *)
            cat << EOF
使用方法: $0 <command>

Commands:
  start              スタンドアロンサーバーを起動
  stop               スタンドアロンサーバーを停止
  restart            スタンドアロンサーバーを再起動
  status             サーバーのステータスを表示
  logs               リアルタイムログを表示
  test               テストを実行
  docker-build       Docker イメージをビルド
  docker-run         Docker コンテナをインタラクティブ実行
  docker-daemon      Docker コンテナをバックグラウンド実行

例:
  $0 start           # サーバーを起動
  $0 status          # ステータス確認
  $0 logs            # ログを表示
  $0 docker-build    # Docker イメージをビルド

EOF
            exit 1
            ;;
    esac
}

main "$@"
