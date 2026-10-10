#!/usr/bin/env bash
set -euo pipefail
SOURCE=$(realpath "${1:?existing Git checkout required}")
ACTIVE=$(realpath "${2:-/root/autodl-tmp/solar_filament_linux}")
RF=$(realpath "${3:-/root/autodl-tmp/solar_filament_rf1536}")
DEST=${4:-/root/autodl-tmp/solar_filament_native_projection}
COMMIT=$(git -C "$SOURCE" rev-parse refs/remotes/origin/linux-autodl)
if [[ -e "$DEST" ]]; then
  [[ "$(git -C "$DEST" rev-parse HEAD)" == "$COMMIT" ]] || {
    echo 'Existing P0 source is pinned to a different commit; preserve it.'; exit 1;
  }
  [[ -z "$(git -C "$DEST" status --porcelain)" ]] || { echo 'P0 source changed; inspect it.'; exit 1; }
else
  git -C "$SOURCE" worktree add --detach "$DEST" "$COMMIT"
fi
bash "$DEST/linux/start_native_projection.sh" "$ACTIVE" "$RF"
