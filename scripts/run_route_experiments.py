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
    lines+=['','Migration: prior preparation/computation seconds remain charged to the original budget. Windows RTX5070Ti -> Linux RTX4090D continuation preserves checkpoint states but is not bit-for-bit reproducible. Image projection operates after resizing to 1056, so this experiment does not test native ultra-high-resolution projection.','', 'Feature_control vs feature_structure isolates structured supervision. Image projection uses frozen RF; feature routes train only the mask head and new detail branch. SAM uses RF-predicted boxes. Cellpose uses global stitched-flow reconstruction and cannot represent overlapping instances at one pixel.','',
            'These are route-screening experiments, not proof of the globally best architecture. Old baseline PQ: YOLO11s 0.371927, R50 U-Net 0.343608, frozen DINOv3 0.026902.']
    (BASE/'comparison.md').write_text('\n'.join(lines),encoding='utf-8')
    from src.routes_v3.diagnostics import compare_artifacts,visualize,detection_diagnostic
    compare_artifacts()
    if (BASE/'rf/best_predictions.json').exists():detection_diagnostic()
    visualize()
    for name in ['results.csv','comparison.md','experiment_manifest.json','pipeline_status.json','artifact_checks.json','same_scene_comparison.png']:
        if not (BASE/name).exists():continue
        shutil.copy2(BASE/name,dest/name)
    for route,names in {'rf':['detection_diagnostics.json'],'sam':['oracle_diagnostics.json','diagnostic_failure.json']}.items():
        for name in names:
            if (BASE/route/name).exists():
                (dest/route).mkdir(exist_ok=True);shutil.copy2(BASE/route/name,dest/route/name)

def main():
    import signal
    BASE.mkdir(parents=True,exist_ok=True)
    for p in psutil.process_iter(['pid','cmdline']):
        try:
            if any(os.path.basename(x)=='train_route.py' for x in (p.info['cmdline'] or [])):
                raise RuntimeError(f'Existing training process {p.pid}; refusing duplicate runner')
        except psutil.Error:pass
    me=psutil.Process()
    identity=dict(pid=os.getpid(),create_time=me.create_time())
    atomic_json(BASE/'runner_lock.json',identity)
    status=dict(stage='starting',**identity,routes={},active_routes={},updated=time.time(),scheduling='linux_strict_serial')
    manifest=load_json(BASE/'experiment_manifest.json')
    for cfg in manifest['runs']:
        status['routes'][cfg['id']]='completed' if (BASE/cfg['id']/'completed.json').exists() else 'queued'
    def persist():
        status['updated']=time.time();atomic_json(BASE/'pipeline_status.json',status)
    def make_report():
        try:
            report(status);status.pop('report_error',None)
        except Exception as exc:
            status['report_error']=repr(exc);print('REPORT_ERROR',repr(exc),flush=True)
        persist()
    def stop(signum,frame):
        atomic_json(BASE/'stop_requested.json',dict(time=time.time(),signal=signum))
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    for cfg in manifest['runs']:
        key=cfg['id'];folder=BASE/key;folder.mkdir(parents=True,exist_ok=True)
        if status['routes'][key]=='completed':continue
        if (BASE/'stop_requested.json').exists():break
        if cfg['dependency'] and not all((BASE/'rf'/name).is_file() for name in ['completed.json','best.pt','best_predictions.json']):
            status['routes'][key]='blocked_missing_rf_checkpoint';persist();continue
        env=os.environ.copy();env['KAGGLE_COOPERATIVE_GPU']='0';env['KAGGLE_GPU_MEMORY_FRACTION']='0.85'
        with (folder/'train_console.log').open('a',encoding='utf-8') as log:
            child=subprocess.Popen([sys.executable,'-u',str(__file__).replace('run_route_experiments.py','train_route.py'),'--route',key],stdout=log,stderr=subprocess.STDOUT,cwd=str(BASE.parents[2]),env=env)
            child_id=dict(pid=child.pid,create_time=psutil.Process(child.pid).create_time())
            status.update(stage=key,child_pid=child.pid,child_create_time=child_id['create_time'],active_routes={key:child_id})
            status['routes'][key]='running';persist();print(f'START {key} pid={child.pid}',flush=True)
            code=child.wait()
        status.update(child_pid=None,active_routes={},exit_code=code)
        if code==0 and (folder/'completed.json').exists():status['routes'][key]='completed'
        elif code==0 and (BASE/'stop_requested.json').exists():status['routes'][key]='paused'
        else:status['routes'][key]='failed'
        print(f'END {key} status={status["routes"][key]} exit={code}',flush=True)
        persist();make_report()
    if (BASE/'stop_requested.json').exists():status['stage']='paused'
    else:status['stage']='completed' if all(s=='completed' for s in status['routes'].values()) else 'needs_attention'
    status.update(child_pid=None,active_routes={});persist();make_report()
    print(status,flush=True)

if __name__=='__main__':
    with file_lock(BASE/'runner_process.lock',blocking=False):main()
