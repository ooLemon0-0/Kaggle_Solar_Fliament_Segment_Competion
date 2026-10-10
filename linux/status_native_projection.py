"""One-shot P0 native projection status, with process identity and cached results."""
import json
from pathlib import Path
import time
import psutil

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'output/benchmark/native_projection_v1'

def read(path):
    return json.loads(path.read_text(encoding='utf-8')) if path.exists() else {}

state=read(BASE/'pipeline_status.json');alive=False
try:
    process=psutil.Process(state['pid'])
    alive=(abs(process.create_time()-state['create_time'])<.01
           and Path(process.cwd()).resolve()==ROOT
           and any(Path(part).name=='run_native_projection.py' for part in process.cmdline()))
except (KeyError,psutil.Error):
    pass
folder=BASE/'native_projection'
progress,done,metrics=[read(folder/name) for name in ['progress.json','completed.json','best_metrics.json']]
stage=state.get('stage','not_started')
if not alive and stage in {'initializing','preflight_debug','training_and_validation','reporting'}:
    stage='process_missing_inspect_log'
age=min((time.time()-path.stat().st_mtime for path in
         [folder/'last.pt',folder/'best.pt',folder/'progress.json',ROOT/'logs/native_projection_console.log']
         if path.exists()),default=None)
print('P0:',stage,'| real process alive:',alive,'| last activity seconds:',round(age) if age is not None else None)
print('Epoch:',done.get('completed_epochs',progress.get('epoch')),'| sample/image:',progress.get('sample',progress.get('image')),
      '| best epoch:',done.get('best_epoch',metrics.get('training_epochs')))
print('PQ/Dice/IoU:',metrics.get('pq'),metrics.get('dice'),metrics.get('iou'))
report=read(BASE/'comparison.json')
print('Delta PQ:',report.get('delta_pq'),'| epochs complete:',report.get('full_registered_epochs'))
print('Decision:',report.get('decision'),'| automatic next stage: False')
