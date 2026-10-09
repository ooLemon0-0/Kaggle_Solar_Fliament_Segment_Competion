#!/usr/bin/env bash
set -euo pipefail
PROJECT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd "$PROJECT"
MODE=${1:-all}
TRAIN_PREFIX=${SOLAR_TRAIN_PREFIX:-"$PROJECT/.envs/solar-routes"}
TOOLS_PREFIX=${SOLAR_TOOLS_PREFIX:-"$PROJECT/.envs/kaggle-tools"}
command -v conda >/dev/null || { echo '找不到 conda：先加载 AutoDL 的 Miniconda，再运行本脚本。'; exit 1; }
mkdir -p work logs
if [[ "$MODE" == tools || "$MODE" == all ]]; then
  [[ -x "$TOOLS_PREFIX/bin/python" ]] || conda create -y -p "$TOOLS_PREFIX" --override-channels -c conda-forge python=3.11 pip
  "$TOOLS_PREFIX/bin/python" -m pip install --upgrade pip
  "$TOOLS_PREFIX/bin/python" -m pip install kaggle
  "$TOOLS_PREFIX/bin/python" -m pip check
  "$TOOLS_PREFIX/bin/python" -m pip freeze > work/kaggle-tools-freeze.txt
fi
if [[ "$MODE" == train || "$MODE" == all ]]; then
  [[ -x "$TRAIN_PREFIX/bin/python" ]] || conda create -y -p "$TRAIN_PREFIX" --override-channels -c conda-forge python=3.11 pip
  "$TRAIN_PREFIX/bin/python" -m pip install --upgrade pip setuptools wheel
  "$TRAIN_PREFIX/bin/python" -m pip install torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cu128
  "$TRAIN_PREFIX/bin/python" -m pip install -r linux/requirements-routes.txt
  "$TRAIN_PREFIX/bin/python" -m pip install --no-deps ./vendor/dinov3_snapshot
  "$TRAIN_PREFIX/bin/python" -m pip check
  "$TRAIN_PREFIX/bin/python" -m pip freeze > work/solar-routes-freeze.txt
  "$TRAIN_PREFIX/bin/python" linux/check_environment.py --mode gpu
fi
[[ "$MODE" == tools || "$MODE" == train || "$MODE" == all ]] || { echo 'usage: bash linux/setup_env.sh tools|train|all'; exit 2; }
echo "训练解释器: $TRAIN_PREFIX/bin/python"
echo "Kaggle CLI: $TOOLS_PREFIX/bin/kaggle"
