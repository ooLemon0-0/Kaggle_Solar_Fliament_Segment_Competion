#!/usr/bin/env bash
set -euo pipefail
PROJECT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd "$PROJECT"
PY=${SOLAR_TRAIN_PREFIX:-"$PROJECT/.envs/solar-routes"}/bin/python
export CUDA_VISIBLE_DEVICES=0 KAGGLE_COOPERATIVE_GPU=0 KAGGLE_GPU_MEMORY_FRACTION=0.85
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export HF_HOME="$PROJECT/output/pretrained/hf" MPLCONFIGDIR="$PROJECT/work/matplotlib"
mkdir -p logs
"$PY" linux/prepare_project.py
"$PY" linux/check_environment.py --mode ready
# A start is an explicit resume authorization; refuse duplicates before clearing stop.
"$PY" linux/control.py start-check
nohup "$PY" -u scripts/run_route_experiments.py >> logs/pipeline_console.log 2>&1 < /dev/null &
echo "Runner PID=$!; inspect with $PY linux/status.py and tail -f logs/pipeline_console.log"
