#!/usr/bin/env bash
set -euo pipefail
PROJECT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
ACTIVE=${1:-/root/autodl-tmp/solar_filament_linux}
ACTIVE=$(realpath "$ACTIVE")
PY="$ACTIVE/.envs/solar-routes/bin/python"
cd "$PROJECT"
mkdir -p logs
"$PY" scripts/run_rf_resolution.py --active-project "$ACTIVE" --prepare-only "${@:2}"
nohup "$PY" -u scripts/run_rf_resolution.py --active-project "$ACTIVE" \
  "${@:2}" >> logs/rf_resolution_console.log 2>&1 < /dev/null &
solar_pid=$!
sleep 2
if ! kill -0 "$solar_pid" 2>/dev/null; then
  tail -n 20 logs/rf_resolution_console.log
  echo 'RF runner already exited; inspect the result/log before starting again.'
  exit 1
fi
echo "RF 1536 runner PID=$solar_pid; only this experiment is scheduled."
echo "tail -n 0 -F $PROJECT/logs/rf_resolution_console.log"
