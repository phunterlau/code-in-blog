#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p logs
parent=/home/hliu/work/zero-egg-verification-20261102-v1
# Wait for the authorized parent experiment; never preempt it or other work.
for ((i=0;i<660;i++)); do
 if [ -f "$parent/logs/exit-code.txt" ]; then
  if [ "$(cat "$parent/logs/exit-code.txt")" != 0 ]; then echo 'Parent failed; no follow-up' >&2;exit 76;fi
  if [ -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits)" ]; then break;fi
 fi
 sleep 5
done
if [ -n "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits)" ]; then echo 'No free GPU' >&2;exit 75;fi
if [ ! -f "$parent/results/completion.json" ]; then echo 'Parent incomplete' >&2;exit 77;fi
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=4
/tmp/magic17-env-tf5201-WzAQHOGE/venv/bin/python run_patch.py --out results > logs/stdout.txt 2> logs/stderr.txt &
inference_pid=$!
echo "$inference_pid" > logs/inference.pid
(
 while kill -0 "$inference_pid" 2>/dev/null; do
  date -u +%FT%TZ
  nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader,nounits
  sleep 2
 done
) > logs/gpu.txt &
monitor=$!
set +e
wait "$inference_pid"
rc=$?
wait "$monitor"
printf '%s\n' "$rc" > logs/exit-code.txt
exit "$rc"
