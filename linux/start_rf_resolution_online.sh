#!/usr/bin/env bash
# Run this freshly fetched script from /tmp; do not merge into the old Flow checkout.
set -euo pipefail
SOURCE=$(realpath "${1:?existing Git checkout required}")
ACTIVE=$(realpath "${2:-/root/autodl-tmp/solar_filament_linux}")
DEST=${3:-/root/autodl-tmp/solar_filament_rf1536}
COMMIT=$(git -C "$SOURCE" rev-parse refs/remotes/origin/linux-autodl)
if [[ -e "$DEST" ]]; then
  [[ "$(git -C "$DEST" rev-parse HEAD)" == "$COMMIT" ]] || {
    echo 'Existing RF checkout is pinned to another commit. Preserve it; do not overwrite or relaunch.'; exit 1;
  }
  [[ -z "$(git -C "$DEST" status --porcelain)" ]] || {
    echo 'RF checkout has source changes; inspect before launch.'; exit 1;
  }
else
  git -C "$SOURCE" worktree add --detach "$DEST" "$COMMIT"
fi
bash "$DEST/linux/start_rf_resolution.sh" "$ACTIVE"
