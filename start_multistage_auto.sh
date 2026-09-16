#!/bin/bash
"""
Automated multi-stage training orchestrator launcher.

Sets up environment and runs the automatic stage progression script
with periodic status checks (every 5 minutes by default).

Usage:
  ./start_multistage_auto.sh                    # Run once
  ./start_multistage_auto.sh --daemon 300       # Run every 300s
  ./start_multistage_auto.sh --daemon 300 &     # Background daemon
"""

set -e

# Configuration
RUNPOD_ENDPOINT_ID="${RUNPOD_ENDPOINT_ID:-aic6yigpthbck5}"
RUNPOD_API_KEY="${RUNPOD_API_KEY}"
CHECK_INTERVAL="${1:-300}"  # Default 5 minutes
DAEMON_MODE="${2:-false}"

# Validate credentials
if [ -z "$RUNPOD_API_KEY" ]; then
  echo "❌ Error: RUNPOD_API_KEY not set"
  echo "Set with: export RUNPOD_API_KEY='your-api-key'"
  exit 1
fi

echo "✅ Configuration:"
echo "   Endpoint ID: $RUNPOD_ENDPOINT_ID"
echo "   Check Interval: ${CHECK_INTERVAL}s"
echo "   State File: multistage_state.json"
echo ""

# Run orchestrator
if [ "$DAEMON_MODE" == "--daemon" ]; then
  echo "🔄 Starting daemon mode (checking every ${CHECK_INTERVAL}s)..."
  echo "Press Ctrl+C to stop"
  echo ""

  while true; do
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Running stage orchestrator..."
    python3 runpod_multistage_auto.py || echo "[WARN] Orchestrator returned an error"

    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Waiting ${CHECK_INTERVAL}s until next check..."
    sleep "$CHECK_INTERVAL"
  done
else
  echo "🚀 Running stage orchestrator once..."
  python3 runpod_multistage_auto.py
fi
