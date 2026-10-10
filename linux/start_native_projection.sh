#!/usr/bin/env bash
set -euo pipefail
PROJECT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
ACTIVE=$(realpath "${1:-/root/autodl-tmp/solar_filament_linux}")
RF=$(realpath "${2:-/root/autodl-tmp/solar_filament_rf1536}")
PY="$ACTIVE/.envs/solar-routes/bin/python"
cd "$PROJECT"
mkdir -p logs
"$PY" scripts/run_native_projection.py --active-project "$ACTIVE" --rf-project "$RF" --prepare-only "${@:3}"
nohup "$PY" -u scripts/run_native_projection.py --active-project "$ACTIVE" --rf-project "$RF" \
  "${@:3}" >> logs/native_projection_console.log 2>&1 < /dev/null &
solar_pid=$!
sleep 2
if ! kill -0 "$solar_pid" 2>/dev/null; then
  tail -n 25 logs/native_projection_console.log
  echo 'P0 exited; inspect logs before retry.'; exit 1
fi
echo "P0 runner PID=$solar_pid; native2048 / context1536; 12 epochs or 2 training hours; no next job."
echo "tail -n 0 -F $PROJECT/logs/native_projection_console.log"
