import argparse
import json
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'scripts'))
from src.routes_v3.engine import BASE
from run_route_experiments import report
parser=argparse.ArgumentParser();parser.add_argument('--final',action='store_true');args=parser.parse_args()
manifest=json.loads((BASE/'experiment_manifest.json').read_text())
status={'routes':{c['id']:'completed' if (BASE/c['id']/'completed.json').exists() else 'incomplete' for c in manifest['runs']}}
report(status)
checks=json.loads((BASE/'artifact_checks.json').read_text())
diagnostic_missing=not (BASE/'sam/oracle_diagnostics.json').is_file() or (BASE/'sam/diagnostic_failure.json').is_file()
if args.final and (not checks['passed'] or any(v!='completed' for v in status['routes'].values()) or diagnostic_missing):
    raise RuntimeError('Final report incomplete: inspect artifact_checks and SAM diagnostic; use evaluate_routes.py --route sam --sam-diagnostic if needed')
if args.final:
    for cfg in manifest['runs']:
        result=json.loads((BASE/cfg['id']/'completed.json').read_text())
        if result['config']!=cfg:raise RuntimeError('Completed config differs: '+cfg['id'])
        for which in ['train','val']:
            if result[f'{which}_split_sha256']!=manifest[f'{which}_split_sha256']:raise RuntimeError('Completed split differs')
        if result['completed_epochs']>cfg['epochs'] or not (result['completed_epochs']==cfg['epochs'] or result['training_seconds']>=cfg['hours']*3600):
            raise RuntimeError('Route has not met its declared stopping budget: '+cfg['id'])
print('FINAL' if args.final else 'PROVISIONAL',ROOT/'deliverables/routes_v3')
