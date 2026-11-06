#!/usr/bin/env bash
# 全モデルを ROUND ステップずつ続けて学習し、区切りごとに infer_rnn.py で比べる。
#   QBNN_LM_DATA=... bash rounds.sh [MAX_STEPS] [ROUND] [並列数]
# 各モデルの最新の model_S_<kind>_<steps>.pt から続け、MAX_STEPS に達したモデルは止める。
# 区切りごとの比較は $QBNN_LM_DATA/round_<n>.txt に書く。
set -u
cd "$(dirname "$0")"
D=${QBNN_LM_DATA:-data}
MAX=${1:-6000}; ROUND=${2:-1500}; JOBS=${3:-4}
KINDS="rwkv7a rwkv7 rwkv7c rwkv7d rwkv7e rwkv7b rnn complete"   # 時間のかかる順

latest() { ls "$D"/model_S_"$1"_*.pt 2>/dev/null | sed -E 's/.*_([0-9]+)\.pt$/\1/' | sort -n | tail -1; }

n=0
while :; do
  todo=""
  for k in $KINDS; do
    s=$(latest "$k"); s=${s:-0}
    [ "$s" -lt "$MAX" ] && todo="$todo $k:$s"
  done
  [ -z "$todo" ] && break
  n=$((n + 1))
  echo "round $n:$todo ($(date -u +%H:%M))"
  for js in $todo; do echo "$js"; done | xargs -P "$JOBS" -I{} bash -c \
    'k=${1%%:*}; s=${1##*:}; python scale_exp.py S "$k" '"$ROUND"' 0 1 "$s" >> "'"$D"'/log_${k}.txt" 2>&1' _ {}
  specs=""
  for k in $KINDS; do s=$(latest "$k"); [ -n "$s" ] && specs="$specs,$k@$s"; done
  python infer_rnn.py S "$ROUND" "${specs#,}" > "$D/round_$n.txt" 2>&1
  echo "round $n done ($(date -u +%H:%M))"
done
echo "all done"
