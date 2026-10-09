#!/usr/bin/env bash
# Run from the freshly FETCHED blob, BEFORE merging into the pinned checkout.
set -euo pipefail
PROJECT=$(realpath "${1:-/root/autodl-tmp/solar_filament_source_clean}")
ACTIVE=$(realpath "${2:-/root/autodl-tmp/solar_filament_linux}")
PY="$ACTIVE/.envs/solar-routes/bin/python"
cd "$PROJECT"
if [ -n "$(git status --porcelain --untracked-files=no)" ]; then
  echo 'Tracked source modifications exist; refusing automatic merge.' >&2
  exit 1
fi
git merge-base --is-ancestor HEAD origin/linux-autodl
CANCEL_HELPER=$(mktemp /tmp/solar-cancel-waiter.XXXXXX.py)
trap 'rm -f -- "$CANCEL_HELPER"' EXIT
git show origin/linux-autodl:linux/cancel_waiting_followup.py > "$CANCEL_HELPER"
"$PY" "$CANCEL_HELPER" --checkout "$PROJECT"
# The CPU waiter has exited and no follow-up trainer exists. Original jobs are untouched.
git merge --ff-only origin/linux-autodl
git log -1 --oneline
bash linux/start_flow_followup.sh "$ACTIVE" --repair-projection
echo 'Queued: remaining original routes -> projection repair -> Flow v2.'
echo "Monitor: tail -F $PROJECT/logs/flow_followup_console.log $ACTIVE/output/benchmark/routes_v3/image_projection/train_console.log $PROJECT/output/benchmark/flow_followup_v1/flow_v2/train_console.log"
