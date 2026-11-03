#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p logs
if [ -n "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits)" ]; then
 echo 'GPU occupied; experiment not launched' >&2
 exit 75
fi
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1 TOKENIZERS_PARALLELISM=false
export OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=4
/tmp/magic17-env-tf5201-WzAQHOGE/venv/bin/python run_study.py --out results > logs/stdout.txt 2> logs/stderr.txt &
inference_pid=$!
echo "$inference_pid" > logs/inference.pid
(
 while kill -0 "$inference_pid" 2>/dev/null; do
  date -u +%FT%TZ
  nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader,nounits
  nvidia-smi --query-gpu=memory.used,utilization.gpu,temperature.gpu --format=csv,noheader,nounits
  sleep 2
 done
) > logs/gpu.txt &
monitor_pid=$!
set +e
wait "$inference_pid"
rc=$?
wait "$monitor_pid"
printf '%s\n' "$rc" > logs/exit-code.txt
exit "$rc"
