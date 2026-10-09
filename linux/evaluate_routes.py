"""Full validation into separate artifacts; never replaces selected best weights."""
import argparse
import gc
import json
import os
import sys
import time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
os.environ['KAGGLE_COOPERATIVE_GPU']='0'
import torch
from src.routes_v3.engine import BASE,evaluate
from src.routes_v3.models import build
from src.routes_v3.resources import file_lock
parser=argparse.ArgumentParser()
parser.add_argument('--route',default='all',choices=['all','rf','flow','sam','image_projection','feature_control','feature_structure'])
parser.add_argument('--checkpoint',choices=['best','last'],default='best')
parser.add_argument('--sam-diagnostic',action='store_true')
args=parser.parse_args()
manifest=json.loads((BASE/'experiment_manifest.json').read_text())
with file_lock(BASE/'gpu_operation.lock',blocking=False):
    for cfg in manifest['runs']:
        if args.route!='all' and cfg['id']!=args.route:continue
        path=BASE/cfg['id']/f'{args.checkpoint}.pt'
        if not path.exists():
            if args.route!='all': raise FileNotFoundError(path)
            print('SKIP: no checkpoint',cfg['id']);continue
        saved=torch.load(path,map_location='cpu',weights_only=False)
        cfg=saved['config'];model=build(cfg);model.load_state_dict(saved['model'])
        epoch=saved['state'].get('best_epoch') if args.checkpoint=='best' else saved['state']['epoch']+1
        dest=BASE/'re_evaluation'/f'{cfg["id"]}_{args.checkpoint}_{time.time_ns()}'
        dest.mkdir(parents=True,exist_ok=True)
        del saved;gc.collect()
        metrics=evaluate(model,cfg,dest,epoch)
        print('FULL_VALIDATION_OUTPUT',dest,flush=True)
        if cfg['kind']=='sam' and args.sam_diagnostic:
            # This repairs the dedicated diagnostic only; uses GT prompts, never formal PQ.
            from src.routes_v3.diagnostics import sam_oracle_diagnostic
            sam_oracle_diagnostic(model)
            (BASE/'sam/diagnostic_failure.json').unlink(missing_ok=True)
        del model;gc.collect();torch.cuda.empty_cache()
