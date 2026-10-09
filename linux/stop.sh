#!/usr/bin/env bash
set -euo pipefail
PROJECT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
PY=${SOLAR_TRAIN_PREFIX:-"$PROJECT/.envs/solar-routes"}/bin/python
"$PY" "$PROJECT/linux/control.py" stop
echo '已请求安全暂停：等待当前梯度累积完成并写入 last.pt；验证中则完成本次验证。确认 status.py 无真实训练进程后再关机。'
