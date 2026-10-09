"""Repair only the failed projection, using new source and original output/inputs.

The parent waiter owns the original runner/GPU locks throughout this process.
No original source file is replaced. Existing checkpoint budget/RNG are resumed.
"""
import _bootstrap
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
import traceback
import psutil

ROOT=Path(__file__).resolve().parents[1]


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def validate_inputs(active):
    base=active/'output/benchmark/routes_v3'
    manifest=read(base/'experiment_manifest.json')
    integrity=read(ROOT/'linux/integrity.json')['immutable_files']
    for name,expected in integrity.items():
        if hashlib.sha256((ROOT/name).read_bytes()).hexdigest()!=expected:
            raise RuntimeError('Immutable evaluation/split changed: '+name)
    for split in ['train','val']:
        if manifest[split+'_split_sha256']!=integrity['output/splits/'+split+'.json']:
            raise RuntimeError('Original manifest split mismatch: '+split)
    cfg=next(c.copy() for c in manifest['runs'] if c['id']=='image_projection')
    original_cfg=next(c for c in read(ROOT/'linux/integrity.json')['runs'] if c['id']=='image_projection')
    if cfg!=original_cfg:raise RuntimeError('Projection protocol differs; refusing to change the training budget')
    for c in manifest['runs']:
        if c['id']=='image_projection':continue
        if not (base/c['id']/'completed.json').is_file():raise RuntimeError('Other original route incomplete: '+c['id'])
    rf=base/'rf'
    for name in ['best.pt','best_predictions.json','config.json']:
        if not (rf/name).is_file():raise RuntimeError('Missing original RF asset: '+name)
    cfg['size']=read(rf/'config.json')['size']
    return base,cfg


def backup_attempt(base):
    folder=base/'image_projection'
    total=sum(p.stat().st_size for p in folder.rglob('*') if p.is_file()) if folder.exists() else 0
    if shutil.disk_usage(base).free<total+2*2**30:
        raise RuntimeError('Insufficient disk space to preserve failed projection attempt and checkpoints')
    archive=base/'repair_history'/('image_projection_'+str(time.time_ns()))
    archive.mkdir(parents=True)
    if folder.exists():shutil.copytree(folder,archive/'attempt')
    for name in ['pipeline_status.json','results.csv','comparison.md','artifact_checks.json']:
        if (base/name).exists():shutil.copy2(base/name,archive/name)
    return archive


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--active-project',type=Path,required=True)
    active=parser.parse_args().active_project.resolve()
    if active==ROOT:raise RuntimeError('Repair requires the separate Git source checkout')
    try:
        guard=psutil.Process(int(os.environ['SOLAR_FOLLOWUP_GUARD_PID']))
        valid=abs(guard.create_time()-float(os.environ['SOLAR_FOLLOWUP_GUARD_TIME']))<.01
        valid=valid and Path(guard.cwd()).resolve()==ROOT and any(Path(x).name=='queue_flow_followup.py' for x in guard.cmdline())
    except (KeyError,ValueError,psutil.Error):valid=False
    if not valid:raise RuntimeError('Use the guarded follow-up queue; direct repair launch is forbidden')
    sys.path.insert(0,str(ROOT/'linux'))
    from queue_flow_followup import running_jobs,readiness
    base,cfg=validate_inputs(active)
    required=[c['id'] for c in read(base/'experiment_manifest.json')['runs']]
    decision=readiness(read(base/'pipeline_status.json'),running_jobs(active),(base/'stop_requested.json').exists(),required,True)
    if decision!='repair_projection':raise RuntimeError('Projection repair is not eligible: '+decision)
    if (ROOT/'output/benchmark/flow_followup_v1/stop_requested.json').exists():
        raise RuntimeError('Follow-up cancellation requested before repair')
    os.environ['SOLAR_ROUTE_OUTPUT_ROOT']=str(base)
    os.environ['SOLAR_RF_WARMSTART']=str(base/'rf/best.pt')
    # Imports below initialize torch only once original training has ended.
    import torch
    from src.routes_v3.engine import train,probe,atomic_json
    from src.routes_v3.resources import file_lock
    from src.utils.common import digest,split_path
    status=read(base/'pipeline_status.json')
    def request_stop(signum,frame):
        atomic_json(base/'stop_requested.json',dict(time=time.time(),signal=signum,requested_by='projection_repair'))
    signal.signal(signal.SIGTERM,request_stop);signal.signal(signal.SIGINT,request_stop)
    folder=base/'image_projection';folder.mkdir(exist_ok=True)
    last=folder/'last.pt'
    if (folder/'completed.json').exists():raise RuntimeError('Projection already completed; refusing to overwrite')
    # Checkpoint exists => require a complete resumable checkpoint, never restart it.
    if last.exists():
        saved=torch.load(last,map_location='cpu',weights_only=False)
        for key in ['model','optimizer','state','config','rng']:
            if key not in saved:raise RuntimeError('Unsafe projection checkpoint: missing '+key)
        if saved['config']!=cfg:raise RuntimeError('Projection resume config mismatch')
        for which in ['train','val']:
            if saved['state'][which+'_split_sha256']!=digest(split_path(which)):
                raise RuntimeError('Projection checkpoint split mismatch')
        del saved
    elif any((folder/name).exists() for name in ['best.pt','best_metrics.json','training_log.csv']):
        raise RuntimeError('Training artifacts exist without last.pt; refusing a silent restart')
    archive=backup_attempt(base)
    commit=subprocess.check_output(['git','-C',str(ROOT),'rev-parse','HEAD'],text=True).strip()
    record=dict(source_commit=commit,backup=str(archive),resume=last.exists(),time=time.time(),
                policy='unchanged resolution/split/30 epochs or 4 training hours; preserve prior budget and RNG')
    atomic_json(folder/'repair_record.json',record)
    status.update(stage='image_projection_repair',active_routes={'image_projection':dict(pid=os.getpid(),create_time=psutil.Process().create_time())},updated=time.time())
    status['routes']['image_projection']='running';atomic_json(base/'pipeline_status.json',status)
    try:
        with file_lock(folder/'route_process.lock',blocking=False), (folder/'train_console.log').open('a',encoding='utf-8',buffering=1) as log:
            print('PROJECTION_REPAIR',json.dumps(record),flush=True)
            # Tee exceptions and training prints to the original stage log.
            original_out,original_err=sys.stdout,sys.stderr
            sys.stdout=sys.stderr=log
            try:
                print('PROJECTION_REPAIR',json.dumps(record),flush=True)
                fraction=os.environ.get('KAGGLE_GPU_MEMORY_FRACTION')
                if fraction:torch.cuda.set_per_process_memory_fraction(float(fraction),0)
                # Always revalidate all eight largest-target backward steps after this fix,
                # including when an old preflight existed. This does not mutate last.pt.
                probe(cfg,folder)
                result=train(cfg)
                if result.get('status')!='completed' or not (folder/'completed.json').exists():
                    raise RuntimeError('Projection paused; retain checkpoint and do not start Flow')
            except Exception:
                traceback.print_exc();raise
            finally:sys.stdout,sys.stderr=original_out,original_err
        status['routes']['image_projection']='completed'
        status.update(stage='completed',active_routes={},child_pid=None,updated=time.time(),repair_source_commit=commit)
        status.pop('report_error',None)
        atomic_json(base/'pipeline_status.json',status)
        # Rebuild the original six-route report with completed projection and retained RF/SAM/etc.
        from run_route_experiments import report
        try:report(status)
        except Exception as exc:
            status['report_error']=repr(exc);atomic_json(base/'pipeline_status.json',status)
            print('PROJECTION_REPORT_ERROR',repr(exc),flush=True)
        print('PROJECTION_REPAIR_OFFICIAL',json.dumps(read(folder/'best_metrics.json')),flush=True)
    except Exception as exc:
        status['routes']['image_projection']='paused' if (base/'stop_requested.json').exists() else 'failed'
        status.update(stage='paused' if (base/'stop_requested.json').exists() else 'needs_attention',active_routes={},updated=time.time())
        atomic_json(base/'pipeline_status.json',status)
        atomic_json(folder/'repair_failure.json',dict(time=time.time(),error=repr(exc),traceback=traceback.format_exc(),backup=str(archive)))
        raise


if __name__=='__main__':main()
