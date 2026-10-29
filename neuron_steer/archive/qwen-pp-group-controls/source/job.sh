#!/usr/bin/env bash
set -u
cd "$(dirname "$0")"
mkdir -p logs
out=${1:-results}
# This follow-up must not overtake the already queued portable-style study.
python3 - <<'PY' || exit 77
import json
from pathlib import Path
p=Path('portable-completion-receipt.json')
s=json.loads(p.read_text())
assert s['status']=='completed_and_reviewed'
assert s['protocol_sha256']=='31c4ed50e358fd26861b5374035f14d5e4b5d08968b1ad6415deb05d9503eb85'
for key in ['calibration_final_seal_sha256','heldout_final_seal_sha256','heldout_analysis_sha256','report_sha256']:
    assert len(s[key])==64
PY
date -u +%FT%TZ > logs/groups.queue-start
wait_start=$(date +%s)
while true; do
  gpu_pids=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits) || exit 78
  free_mib=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits) || exit 78
  if [ -z "$gpu_pids" ] && [ "$free_mib" -ge 7700 ]; then break; fi
  now=$(date +%s)
  if [ $((now - wait_start)) -ge 3600 ]; then
    echo GPU_WAIT_TIMEOUT > logs/groups.queue-failure
    exit 75
  fi
  date -u +%FT%TZ >> logs/groups.gpu-wait
  printf 'free_mib=%s active_pids=%s\n' "$free_mib" "$gpu_pids" >> logs/groups.gpu-wait
  sleep 20
done
date -u +%FT%TZ > logs/groups.start
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
mkdir -p cache/hf cache/torch cache/cuda tmp
export HF_HOME="$PWD/cache/hf" XDG_CACHE_HOME="$PWD/cache" TORCH_HOME="$PWD/cache/torch"
export CUDA_CACHE_PATH="$PWD/cache/cuda" TMPDIR="$PWD/tmp" PYTHONDONTWRITEBYTECODE=1
export OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=4
unset HF_TOKEN HUGGING_FACE_HUB_TOKEN
/tmp/magic17-env-tf5201-WzAQHOGE/venv/bin/python run_groups.py --protocol protocol.json --out "$out" > logs/groups.stdout 2> logs/groups.stderr &
inference_pid=$!
echo "$inference_pid" > logs/groups.pid
while kill -0 "$inference_pid" 2>/dev/null; do
  date -u +%FT%TZ
  nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader,nounits
  nvidia-smi --query-gpu=memory.used,utilization.gpu,temperature.gpu --format=csv,noheader,nounits
  sleep 10
done > logs/groups.telemetry 2>&1 &
monitor_pid=$!
wait "$inference_pid"
rc=$?
echo "$rc" > logs/groups.exitcode
wait "$monitor_pid"
date -u +%FT%TZ > logs/groups.end
exit "$rc"
