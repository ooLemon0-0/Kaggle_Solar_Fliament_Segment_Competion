#!/usr/bin/env bash
# Extract this fetched script to /tmp. Existing running checkouts stay pinned.
set -euo pipefail
SOURCE=$(realpath "${1:?existing Git checkout required}")
ACTIVE=$(realpath "${2:-/root/autodl-tmp/solar_filament_linux}")
DEST=${3:-/root/autodl-tmp/solar_filament_mask_control}
COMMIT=$(git -C "$SOURCE" rev-parse refs/remotes/origin/linux-autodl)
if [[ -e "$DEST" ]]; then
  [[ "$(git -C "$DEST" rev-parse HEAD)" == "$COMMIT" ]] || {
    echo 'Existing M0 checkout is pinned to a different commit. Preserve it; do not overwrite.'; exit 1;
  }
  [[ -z "$(git -C "$DEST" status --porcelain)" ]] || {
    echo 'M0 source changed; inspect before launch.'; exit 1;
  }
else
  git -C "$SOURCE" worktree add --detach "$DEST" "$COMMIT"
fi
bash "$DEST/linux/start_mask_control.sh" "$ACTIVE"
