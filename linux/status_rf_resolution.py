"""One-shot status: actual identity plus progress/metrics/checkpoint activity."""
import json
from pathlib import Path
import time
import psutil

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'output/benchmark/rf_resolution_v1'


def read(path):
    return json.loads(path.read_text(encoding='utf-8')) if path.exists() else {}


state = read(BASE / 'pipeline_status.json')
alive = False
try:
    process = psutil.Process(state['pid'])
    alive = (abs(process.create_time() - state['create_time']) < .01
             and Path(process.cwd()).resolve() == ROOT
             and any(Path(x).name == 'run_rf_resolution.py' for x in process.cmdline()))
except (psutil.Error, KeyError):
    pass
folder = BASE / 'rf_1536'
metrics, done, progress = [read(folder / name) for name in ['best_metrics.json', 'completed.json', 'progress.json']]
files = [folder / 'last.pt', folder / 'best.pt', folder / 'progress.json', ROOT / 'logs/rf_resolution_console.log']
age = min((time.time() - path.stat().st_mtime for path in files if path.exists()), default=None)
stage = state.get('stage', 'not_started')
if not alive and stage in {'initializing', 'preflight_debug', 'training_and_validation', 'reporting'}:
    stage = 'process_missing_inspect_log'
print('RF 1536:', stage, '| real process alive:', alive, '| last activity seconds:', round(age) if age is not None else None)
print('Epoch:', done.get('completed_epochs', progress.get('epoch')), '| sample/image:', progress.get('sample', progress.get('image')),
      '| best epoch:', done.get('best_epoch'), '| PQ/Dice/IoU:', metrics.get('pq'), metrics.get('dice'), metrics.get('iou'))
print('Decision:', read(BASE / 'comparison.json').get('decision'), '| automatic next stage: False')
