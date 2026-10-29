#!/usr/bin/env bash
set -u
cd "$(dirname "$0")"
phase=$1
out=$2
label=$3
selection=${4:-}
mkdir -p logs
date -u +%FT%TZ > "logs/$label.queue-start"
wait_start=$(date +%s)
while true; do
  gpu_pids=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits) || exit 78
  free_mib=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits) || exit 78
  if [ -z "$gpu_pids" ] && [ "$free_mib" -ge 7700 ]; then break; fi
  now=$(date +%s)
  if [ $((now - wait_start)) -ge 3600 ]; then
    echo GPU_WAIT_TIMEOUT > "logs/$label.queue-failure"
    exit 75
  fi
  date -u +%FT%TZ >> "logs/$label.gpu-wait"
  printf 'free_mib=%s active_pids=%s\n' "$free_mib" "$gpu_pids" >> "logs/$label.gpu-wait"
  sleep 20
done
date -u +%FT%TZ > "logs/$label.start"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
mkdir -p cache/hf cache/torch cache/cuda tmp
export HF_HOME="$PWD/cache/hf" XDG_CACHE_HOME="$PWD/cache" TORCH_HOME="$PWD/cache/torch"
export CUDA_CACHE_PATH="$PWD/cache/cuda" TMPDIR="$PWD/tmp" PYTHONDONTWRITEBYTECODE=1
unset HF_TOKEN HUGGING_FACE_HUB_TOKEN
cmd=(/tmp/magic17-env-tf5201-WzAQHOGE/venv/bin/python run_portable.py --protocol protocol.json --out "$out" --phase "$phase")
if [ -n "$selection" ]; then cmd+=(--selection "$selection"); fi
"${cmd[@]}" > "logs/$label.stdout" 2> "logs/$label.stderr" &
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
