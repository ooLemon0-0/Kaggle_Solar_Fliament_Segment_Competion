import json
import os
import sys
import time
from pathlib import Path
import psutil
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.routes_v3.resources import file_lock
BASE=ROOT/'output/benchmark/routes_v3'
BASE.mkdir(parents=True,exist_ok=True)
def relevant_processes():
    out=[]
    names={'run_route_experiments.py','train_route.py','evaluate_routes.py'}
    for p in psutil.process_iter(['pid','cmdline','create_time']):
        try:
            if any(os.path.basename(x) in names for x in (p.info['cmdline'] or [])):
                out.append(p.info)
        except psutil.Error:pass
    return out
if __name__=='__main__':
    if sys.argv[1]=='stop':
        temp=BASE/'stop_requested.tmp'
        temp.write_text(json.dumps({'time':time.time(),'requested_by':'linux/stop.sh'}));temp.replace(BASE/'stop_requested.json')
    elif sys.argv[1]=='start-check':
        with file_lock(BASE/'runner_process.lock',blocking=False),file_lock(BASE/'gpu_operation.lock',blocking=False):
            found=relevant_processes()
            if found: raise RuntimeError(f'Existing GPU jobs; duplicate launch rejected: {found}')
            (BASE/'stop_requested.json').unlink(missing_ok=True)
    else: raise RuntimeError('usage: control.py stop|start-check')
