#!/bin/bash

# QBNN Frontal Engine MCP Server - Local Run Script
# MCPサーバーをローカルで実行するためのスクリプト

set -e

echo "=== QBNN Frontal Engine MCP Server ==="
echo ""

# Python環境の確認
if ! command -v python3 &> /dev/null; then
    echo "❌ Error: Python 3 is not installed"
    exit 1
fi

echo "✓ Python version: $(python3 --version)"

# 依存パッケージの確認
echo ""
echo "Checking dependencies..."

# mcpパッケージの確認
if ! python3 -c "import mcp" 2>/dev/null; then
    echo "⚠ Installing mcp package..."
    pip install mcp
fi

# PyTorchの確認
if ! python3 -c "import torch" 2>/dev/null; then
    echo "⚠ Installing torch..."
    pip install torch==2.4.0
fi

# その他の依存パッケージ
for package in numpy sentencepiece huggingface-hub; do
    if ! python3 -c "import ${package}" 2>/dev/null; then
        echo "⚠ Installing ${package}..."
        pip install "${package}"
    fi
done

echo "✓ All dependencies installed"
echo ""

# MCPサーバーの実行
echo "Starting QBNN Frontal Engine MCP Server..."
echo "Listening on stdio..."
echo ""
echo "Press Ctrl+C to stop the server"
echo ""

exec python3 -u frontal_engine_mcp_server.py
