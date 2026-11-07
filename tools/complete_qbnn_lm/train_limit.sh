#!/usr/bin/env bash
# 検証損失の改善が止まるまで、1,500 ステップずつ続けて学習する（scale_exp.py の PREV を使う）。
#   QBNN_LM_DATA=... bash train_limit.sh KIND [SEED] [THREADS] [SIZE] [MIN_GAIN] [MAX_STEPS]
# $QBNN_LM_DATA/model_<SIZE>_<KIND>_<steps>.pt の最新から続ける。止める条件は
#   - 1 区切りの改善が MIN_GAIN（既定 0.01 bits/文字）未満になった
#   - MAX_STEPS（既定 14,400 = Wikipedia 5,900 万文字をほぼ 1 周）に達した
# 途中で止まっても、もう一度実行すれば続きから再開する（区切りの途中は resume_*.pt から）。
set -u
cd "$(dirname "$0")"
D=${QBNN_LM_DATA:-data}
k=$1; s=${2:-0}; th=${3:-2}; size=${4:-S}; gain=${5:-0.01}; max=${6:-14400}; R=1500
val() { python3 -c "
import json
v = None
for l in open('$D/results_long.jsonl'):
    r = json.loads(l)
    if r['size'] == '$size' and r['kind'] == '$k' and r['seed'] == $s and r['steps'] == $1:   # steps は続きの学習でも通算
        v = r['val_bpc']
print(v)"; }
while :; do
  cur=$(ls "$D"/model_${size}_${k}_*.pt 2>/dev/null | sed -E 's/.*_([0-9]+)\.pt$/\1/' | sort -n | tail -1)
  cur=${cur:-0}
  [ "$cur" -ge "$max" ] && { echo "$k stop: reached $cur steps"; break; }
  if [ "$cur" -ge $((2 * R)) ]; then
    a=$(val $((cur - R))); b=$(val "$cur")
    if [ "$a" != None ] && [ "$b" != None ]; then
      echo "$k $cur: $b (improved $(python3 -c "print(round($a - $b, 4))"))"
      python3 -c "import sys; sys.exit(0 if $a - $b < $gain else 1)" && { echo "$k stop: plateau at $cur"; break; }
    fi
  fi
  python scale_exp.py "$size" "$k" $R "$s" "$th" "$cur" >> "$D/log_${k}.txt" 2>&1 || { echo "$k FAILED at $cur"; exit 1; }
done
echo "$k LIMIT_DONE"
