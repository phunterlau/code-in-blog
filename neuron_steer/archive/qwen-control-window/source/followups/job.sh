#!/usr/bin/env bash
set -u
cd /tmp/qwen-neuron-20261028-B7T9yHQx/control-window-followups-v1
mkdir -p logs
date -u +%FT%TZ > logs/run.start
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
/tmp/magic17-env-tf5201-WzAQHOGE/venv/bin/python run_followups.py --protocol protocol.json --out results > logs/run.stdout 2> logs/run.stderr &
inference_pid=$!
echo "$inference_pid" > logs/run.pid
while kill -0 "$inference_pid" 2>/dev/null; do
  date -u +%FT%TZ
  nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader,nounits
  nvidia-smi --query-gpu=memory.used,utilization.gpu,temperature.gpu --format=csv,noheader,nounits
  sleep 10
done > logs/run.telemetry 2>&1 &
monitor_pid=$!
wait "$inference_pid"
rc=$?
echo "$rc" > logs/run.exitcode
wait "$monitor_pid"
date -u +%FT%TZ > logs/run.end
exit "$rc"
