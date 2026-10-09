import _bootstrap
import argparse
import os
import psutil
import subprocess
import sys
import time
from src.utils.common import load_json
from src.routes_v3.engine import BASE,atomic_json
from src.routes_v3.resources import file_lock

def alive(lock):
    try:
        p=psutil.Process(lock['pid'])
        return abs(p.create_time()-lock['create_time'])<.01 and any(os.path.basename(x)=='run_route_experiments.py' for x in p.cmdline())
    except (psutil.Error,KeyError):return False

def report(status):
    import shutil,csv
    dest=load_json(BASE/'experiment_manifest.json')['deliverable_dir']
    from pathlib import Path
    dest=Path(dest);dest.mkdir(parents=True,exist_ok=True)
    rows=[]
    for c in load_json(BASE/'experiment_manifest.json')['runs']:
        folder=BASE/c['id'];metrics=load_json(folder/'best_metrics.json') if (folder/'best_metrics.json').exists() else {}
        complete=load_json(folder/'completed.json') if (folder/'completed.json').exists() else {}
        progress=load_json(folder/'progress.json') if (folder/'progress.json').exists() else {}
        epochs=complete.get('completed_epochs',progress.get('epoch',1)-1)
        train_seconds=complete.get('training_seconds',progress.get('training_seconds',0))
        inherited=load_json(BASE/'rf/completed.json').get('training_seconds',0)/3600 if c['dependency'] and (BASE/'rf/completed.json').exists() else 0
        row=dict(route=c['id'],status=status['routes'].get(c['id'],'queued'),pq=metrics.get('pq'),dice=metrics.get('dice'),iou=metrics.get('iou'),best_epoch=complete.get('best_epoch',metrics.get('training_epochs')),completed_epochs=epochs,partial_epoch_samples=complete.get('partial_epoch_samples',progress.get('sample',0)),training_hours=train_seconds/3600,updates=complete.get('optimizer_updates'),inherited_rf_selection_hours=inherited,gpu_window_wait_hours=complete.get('gpu_window_wait_seconds',0)/3600)
        rows.append(row)
    with (BASE/'results.csv').open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    lines=['# Modern route experiments','',f'Snapshot UTC Unix time: {time.time():.0f}. Running results are provisional.','', 'Official validation: 142 observations / 218 annotator groups, native 2048 masks. No test tuning.','',
           'Budget: each run up to 30 epochs or 4 hours of sample preparation plus model computation, whichever occurs first. Prior cooperative GPU-window waiting, validation and preflight excluded; this is not a hardware GPU-active-hours measurement. Warm starts inherit RF training; compare both incremental and total budget. Current scheduling is strictly serial: one training process on the GPU. Earlier cooperative scheduling and checkpoint recovery are recorded in serial_switch.json.','',
           '| Route | Status | PQ | Dice | IoU | Completed epochs | Train h |','|---|---|---:|---:|---:|---:|---:|']
    for r in rows:
        lines.append('| '+ ' | '.join(str(r[k]) if r[k] is not None else 'pending' for k in ['route','status','pq','dice','iou','completed_epochs','training_hours'])+' |')
    lines+=['','Feature_control vs feature_structure isolates structured supervision. Image projection uses frozen RF; feature routes train only the mask head and new detail branch. SAM uses RF-predicted boxes. Cellpose uses global stitched-flow reconstruction and cannot represent overlapping instances at one pixel.','',
            'These are route-screening experiments, not proof of the globally best architecture. Old baseline PQ: YOLO11s 0.371927, R50 U-Net 0.343608, frozen DINOv3 0.026902.']
    (BASE/'comparison.md').write_text('\n'.join(lines),encoding='utf-8')
    from src.routes_v3.diagnostics import compare_artifacts,visualize,detection_diagnostic
    compare_artifacts()
    if (BASE/'rf/best_predictions.json').exists():detection_diagnostic()
    visualize()
    for name in ['results.csv','comparison.md','experiment_manifest.json','pipeline_status.json','artifact_checks.json','same_scene_comparison.png']:
        if not (BASE/name).exists():continue
        shutil.copy2(BASE/name,dest/name)

def available_resources():
    try:
        result=subprocess.check_output(['nvidia-smi','--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True,timeout=5)
        return float(result.strip().splitlines()[0])>=4096 and psutil.virtual_memory().available>=6*2**30
    except (OSError,ValueError,subprocess.SubprocessError):return False

def main():
    BASE.mkdir(parents=True,exist_ok=True);lockpath=BASE/'runner_lock.json'
    if lockpath.exists() and alive(load_json(lockpath)):raise RuntimeError('Verified existing runner; duplicate launch rejected')
    # Also detect children after an abrupt parent exit, including venv launcher descendants.
    for p in psutil.process_iter(['pid','cmdline']):
        try:
            if any(os.path.basename(x)=='train_route.py' for x in (p.info['cmdline'] or [])):
                raise RuntimeError(f'Existing training process {p.pid}; refusing duplicate runner')
        except psutil.Error:pass
    me=psutil.Process();lock=dict(pid=os.getpid(),create_time=me.create_time())
    atomic_json(lockpath,lock)
    manifest=load_json(BASE/'experiment_manifest.json')
    status=dict(stage='starting',pid=os.getpid(),create_time=me.create_time(),routes={},active_routes={},updated=time.time(),scheduling='strict_serial_v1')
    pending=[];active={};deferred=set();retried=set()
    for cfg in manifest['runs']:
        folder=BASE/cfg['id'];folder.mkdir(parents=True,exist_ok=True)
        if (folder/'completed.json').exists():status['routes'][cfg['id']]='completed';continue
        status['routes'][cfg['id']]='queued';pending.append(cfg)
    def persist():
        status['active_routes']={key:entry['identity'] for key,entry in active.items()}
        status['stage']='parallel' if len(active)>1 else next(iter(active),'starting')
        first=next(iter(active.values()),None)
        status['child_pid']=first['identity']['pid'] if first else None
        status['child_create_time']=first['identity']['create_time'] if first else None
        status['updated']=time.time();atomic_json(BASE/'pipeline_status.json',status)
    while pending or active:
        completed_any=False
        for key,entry in list(active.items()):
            code=entry['process'].poll()
            if code is None:continue
            entry['log'].close();del active[key];folder=BASE/key
            if code==0 and (folder/'completed.json').exists():status['routes'][key]='completed'
            else:
                failure=load_json(folder/'failure.json') if (folder/'failure.json').exists() else {}
                oom='outofmemory' in failure.get('exception','').lower() or 'out of memory' in failure.get('message','').lower()
                if oom and entry['cohort'] and key not in retried:
                    atomic_json(folder/'cohort_failure.json',failure)
                    deferred.add(key);retried.add(key);pending.insert(0,entry['cfg']);status['routes'][key]='deferred_serial_retry'
                    print(f'COHORT OOM {key}; preserving checkpoint and deferring one serial recovery',flush=True)
                else:status['routes'][key]='failed'
            status['exit_code']=code;completed_any=True
        for cfg in list(pending):
            key=cfg['id']
            if cfg['dependency'] and not (BASE/'rf/completed.json').exists():
                if status['routes'].get('rf') in ['failed','blocked_missing_rf_checkpoint']:
                    status['routes'][key]='blocked_missing_rf_checkpoint';pending.remove(cfg)
                continue
            cohort=bool(active)
            if cohort:continue
            env=os.environ.copy();env['KAGGLE_COOPERATIVE_GPU']='0'
            fraction=.85
            if key in deferred:fraction=.85
            env['KAGGLE_GPU_MEMORY_FRACTION']=str(fraction)
            log=(BASE/key/'train_console.log').open('a',encoding='utf-8')
            try:
                child=subprocess.Popen([sys.executable,'-u',str(__file__).replace('run_route_experiments.py','train_route.py'),'--route',key],stdout=log,stderr=subprocess.STDOUT,cwd=str(BASE.parents[2]),env=env)
                identity=dict(pid=child.pid,create_time=psutil.Process(child.pid).create_time(),memory_fraction=fraction)
            except Exception:
                log.close();raise
            active[key]=dict(process=child,log=log,identity=identity,cfg=cfg,cohort=cohort)
            if cohort:
                for entry in active.values():entry['cohort']=True
            pending.remove(cfg);status['routes'][key]='running'
            print(f'START {key} pid={child.pid} active={list(active)}',flush=True)
        persist()
        if completed_any:
            try:report(status);status.pop('report_error',None)
            except Exception as exc:status['report_error']=repr(exc)
        if active:time.sleep(3)
        elif pending and all(c['dependency'] and not (BASE/'rf/completed.json').exists() for c in pending):
            for cfg in pending:status['routes'][cfg['id']]='blocked_missing_rf_checkpoint'
            pending.clear()
    status.update(stage='completed' if all(v=='completed' for v in status['routes'].values()) else 'needs_attention',updated=time.time(),child_pid=None,active_routes={})
    atomic_json(BASE/'pipeline_status.json',status);report(status)
    print(status,flush=True)

if __name__=='__main__':
    with file_lock(BASE/'runner_process.lock',blocking=False):main()
