import _bootstrap
import argparse
import traceback
import time
import os
import signal
from src.routes_v3.resources import file_lock, gpu_window
from src.utils.common import load_json
from src.routes_v3.engine import BASE, train, atomic_json

if __name__=='__main__':
    def request_stop(signum,frame):
        atomic_json(BASE/'stop_requested.json',dict(time=time.time(),signal=signum))
    signal.signal(signal.SIGTERM,request_stop)
    signal.signal(signal.SIGINT,request_stop)
    parser=argparse.ArgumentParser();parser.add_argument('--route',required=True);args=parser.parse_args()
    configs=load_json(BASE/'experiment_manifest.json')['runs'];cfg=next(c for c in configs if c['id']==args.route)
    if cfg['kind'] not in ['rf','flow']:cfg['size']=load_json(BASE/'rf/config.json')['size']
    try:
        with file_lock(BASE/'gpu_operation.lock',blocking=False), file_lock(BASE/cfg['id']/'route_process.lock',blocking=False):
            fraction=os.environ.get('KAGGLE_GPU_MEMORY_FRACTION')
            if fraction:
                import torch
                with gpu_window():torch.cuda.set_per_process_memory_fraction(float(fraction),0)
            train(cfg)
    except Exception as exc:
        atomic_json(BASE/cfg['id']/'failure.json',dict(time=time.time(),exception=type(exc).__name__,message=str(exc),traceback=traceback.format_exc()))
        raise
