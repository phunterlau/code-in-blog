#!/usr/bin/env bash
set -eu
cd "$(dirname "$0")"
while [ ! -f ../logs/main.exitcode ]; do sleep 5; done
if [ "$(cat ../logs/main.exitcode)" != 0 ]; then
  echo 'Main experiment failed; follow-up not launched.' > queue.blocked
  exit 1
fi
bash job.sh run_audit.py protocol.json results genaudit
