#!/usr/bin/env bash
set -u
cd "$(dirname "$0")"
script=$1
protocol=$2
out=$3
label=$4
mkdir -p logs
date -u +%FT%TZ > "logs/$label.start"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
/tmp/magic17-env-tf5201-WzAQHOGE/venv/bin/python "$script" --protocol "$protocol" --out "$out" > "logs/$label.stdout" 2> "logs/$label.stderr" &
inference_pid=$!
echo "$inference_pid" > "logs/$label.pid"
while kill -0 "$inference_pid" 2>/dev/null; do
 date -u +%FT%TZ
 nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader,nounits
 nvidia-smi --query-gpu=memory.used,utilization.gpu,temperature.gpu --format=csv,noheader,nounits
 sleep 10
done > "logs/$label.telemetry" 2>&1 &
monitor_pid=$!
wait "$inference_pid"
rc=$?
echo "$rc" > "logs/$label.exitcode"
wait "$monitor_pid"
date -u +%FT%TZ > "logs/$label.end"
exit "$rc"
