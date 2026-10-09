#!/usr/bin/env bash
set -euo pipefail
PROJECT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
ACTIVE=${1:-/root/autodl-tmp/solar_filament_linux}
ACTIVE=$(realpath "$ACTIVE")
PY="$ACTIVE/.envs/solar-routes/bin/python"
cd "$PROJECT"
mkdir -p logs
"$PY" linux/queue_flow_followup.py --active-project "$ACTIVE" --prepare-only
# The waiter is CPU-only; it acquires original GPU/runner locks only after all routes finish.
nohup "$PY" -u linux/queue_flow_followup.py --active-project "$ACTIVE" \
  >> logs/flow_followup_console.log 2>&1 < /dev/null &
echo "Flow follow-up waiter PID=$!; tail -f $PROJECT/logs/flow_followup_console.log"
