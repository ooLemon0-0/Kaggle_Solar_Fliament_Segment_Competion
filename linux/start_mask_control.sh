#!/usr/bin/env bash
set -euo pipefail
PROJECT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
ACTIVE=$(realpath "${1:-/root/autodl-tmp/solar_filament_linux}")
PY="$ACTIVE/.envs/solar-routes/bin/python"
cd "$PROJECT"
mkdir -p logs
"$PY" scripts/run_mask_control.py --active-project "$ACTIVE" --prepare-only "${@:2}"
nohup "$PY" -u scripts/run_mask_control.py --active-project "$ACTIVE" \
  "${@:2}" >> logs/mask_control_console.log 2>&1 < /dev/null &
solar_pid=$!
sleep 2
if ! kill -0 "$solar_pid" 2>/dev/null; then
  tail -n 20 logs/mask_control_console.log
  echo 'M0 runner exited; inspect the log/result before starting again.'
  exit 1
fi
echo "M0 runner PID=$solar_pid; original mask head only; 1056 input; no next model scheduled."
echo "tail -n 0 -F $PROJECT/logs/mask_control_console.log"
