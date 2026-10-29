#!/usr/bin/env bash
set -eu
main=/tmp/qwen-neuron-20261028-B7T9yHQx/control-window-followups-v1/logs
while [ ! -f "$main/run.exitcode" ]; do sleep 10; done
[ "$(cat "$main/run.exitcode")" = 0 ] || exit 3
active=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits)
[ -z "$active" ] || exit 4
exec bash /tmp/qwen-neuron-20261028-B7T9yHQx/control-window-eos-v1/job.sh
