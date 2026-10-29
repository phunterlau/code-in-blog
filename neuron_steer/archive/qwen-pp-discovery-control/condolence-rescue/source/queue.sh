#!/usr/bin/env bash
set -eu
cd "$(dirname "$0")"
while [ ! -f ../generation-audit/logs/genaudit.exitcode ]; do sleep 5; done
if [ "$(cat ../generation-audit/logs/genaudit.exitcode)" != 0 ]; then
  echo 'Generation audit failed; rescue not launched.' > queue.blocked
  exit 1
fi
bash job.sh run_rescue.py protocol.json results rescue
