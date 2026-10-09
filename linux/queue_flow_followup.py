"""Wait without occupying the GPU, then run a pinned separate Flow follow-up."""
import argparse
import contextlib
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path
import psutil

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.routes_v3.resources import file_lock


def running_jobs(active):
    jobs=[]
    for process in psutil.process_iter(['pid','cmdline','create_time']):
        try:
            command=process.info['cmdline'] or []
            if not any(Path(part).name in {'run_route_experiments.py','train_route.py','evaluate_routes.py'} for part in command):continue
            if process.cwd()==str(active) or any(str(active) in part for part in command):jobs.append(process.info)
        except (psutil.Error,OSError):pass
    return jobs


def readiness(status,jobs,stop_requested,required_routes,repair_projection=False):
    if jobs:return 'waiting'
    if stop_requested:return 'blocked_original_stop_request'
    routes=status.get('routes',{})
    if repair_projection and status.get('stage')=='needs_attention' and set(routes)==set(required_routes):
        failed={key for key,value in routes.items() if value!='completed'}
        if failed=={'image_projection'} and routes['image_projection']=='failed':return 'repair_projection'
    if status.get('stage')!='completed' or set(routes)!=set(required_routes) or not all(v=='completed' for v in routes.values()):
        return 'blocked_original_incomplete'
    return 'ready'


def link_readonly_assets(active):
    # Only link shared immutable input assets. Never link the original benchmark output.
    for relative in ['data','output/pretrained','output/splits']:
        target=active/relative;destination=ROOT/relative
        if not target.is_dir():raise RuntimeError('Missing existing input assets: '+str(target))
        if destination.is_symlink():
            if destination.resolve()!=target.resolve():raise RuntimeError('Input symlink points elsewhere: '+str(destination))
        elif destination.exists():
            raise RuntimeError('Separate checkout has its own input directory; refusing to replace: '+str(destination))
        else:
            destination.parent.mkdir(parents=True,exist_ok=True);destination.symlink_to(target,target_is_directory=True)


def prepare(active):
    active=Path(active).resolve()
    if active==ROOT.resolve():raise RuntimeError('Follow-up requires a separate source checkout')
    if not (active/'output/benchmark/routes_v3/experiment_manifest.json').is_file():raise RuntimeError('Active project does not contain the original experiment manifest')
    dirty=subprocess.check_output(['git','-C',str(ROOT),'status','--porcelain','--untracked-files=no'],text=True)
    if dirty.strip():raise RuntimeError('Commit source changes before arming the follow-up')
    commit=subprocess.check_output(['git','-C',str(ROOT),'rev-parse','HEAD'],text=True).strip()
    # Record hashes too: detecting uncommitted edits matters even when HEAD stays unchanged.
    files=subprocess.check_output(['git','-C',str(ROOT),'ls-files','-z']).decode().split('\0')
    fingerprints={name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in files if name}
    link_readonly_assets(active)
    return active,commit,fingerprints


def source_unchanged(commit,fingerprints):
    actual=subprocess.check_output(['git','-C',str(ROOT),'rev-parse','HEAD'],text=True).strip()
    return actual==commit and all(hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==expected for name,expected in fingerprints.items())


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--active-project',type=Path,required=True)
    parser.add_argument('--prepare-only',action='store_true')
    parser.add_argument('--repair-projection',action='store_true',help='Repair the sole failed image_projection after all other original routes finish, then run Flow')
    args=parser.parse_args()
    active,commit,fingerprints=prepare(args.active_project)
    required_routes=[c['id'] for c in json.loads((active/'output/benchmark/routes_v3/experiment_manifest.json').read_text())['runs']]
    base=ROOT/'output/benchmark/flow_followup_v1';base.mkdir(parents=True,exist_ok=True)
    if args.prepare_only:
        with file_lock(base/'queue_wrapper.lock',blocking=False):
            if (base/'stop_requested.json').exists():raise RuntimeError('Existing follow-up stop request; inspect before rearming')
            print('FOLLOWUP_INPUTS_READY',commit,'repair_projection='+str(args.repair_projection),flush=True)
        return
    stage_file=base/'queue_status.json'
    def state(stage,**extra):
        temporary=stage_file.with_suffix('.tmp')
        temporary.write_text(json.dumps(dict(stage=stage,time=time.time(),pid=os.getpid(),create_time=psutil.Process().create_time(),source_commit=commit,active_project=str(active),**extra),indent=2))
        temporary.replace(stage_file)
    # A dedicated wrapper lock rejects duplicate waiting processes too.
    with file_lock(base/'queue_wrapper.lock',blocking=False):
        state('waiting',repair_projection=args.repair_projection);last_stage=None
        while True:
            if (base/'stop_requested.json').exists():state('cancelled');print('FOLLOWUP_CANCELLED');return
            if not source_unchanged(commit,fingerprints):raise RuntimeError('Pinned source changed after arming; refusing mixed-version follow-up')
            original_base=active/'output/benchmark/routes_v3'
            status_path=original_base/'pipeline_status.json'
            status=json.loads(status_path.read_text()) if status_path.is_file() else {}
            jobs=running_jobs(active)
            stage=status.get('stage','unknown')
            if stage!=last_stage:print('WAIT_ORIGINAL',stage,'actual_jobs',len(jobs),flush=True);last_stage=stage
            decision=readiness(status,jobs,(original_base/'stop_requested.json').exists(),required_routes,args.repair_projection)
            if decision=='waiting':time.sleep(15);continue
            if decision=='blocked_original_incomplete':
                state(decision,original_stage=stage)
                raise RuntimeError('Original runner exited without completing all six experiments; inspect original logs and resume it before follow-up')
            if decision=='blocked_original_stop_request':
                state(decision);raise RuntimeError('Original queue was stopped; refusing automatic follow-up')
            try:
                with contextlib.ExitStack() as locks:
                    locks.enter_context(file_lock(original_base/'runner_process.lock',blocking=False))
                    locks.enter_context(file_lock(original_base/'gpu_operation.lock',blocking=False))
                    decision=readiness(json.loads(status_path.read_text()),running_jobs(active),(original_base/'stop_requested.json').exists(),required_routes,args.repair_projection)
                    if decision=='waiting':continue
                    if decision not in {'ready','repair_projection'}:state(decision);return
                    env=os.environ.copy()
                    env.update(CUDA_VISIBLE_DEVICES='0',KAGGLE_COOPERATIVE_GPU='0',KAGGLE_GPU_MEMORY_FRACTION='0.85',
                        HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',MPLCONFIGDIR=str(ROOT/'work/matplotlib'),
                        SOLAR_FOLLOWUP_GUARD_PID=str(os.getpid()),SOLAR_FOLLOWUP_GUARD_TIME=str(psutil.Process().create_time()))
                    if (base/'stop_requested.json').exists():state('cancelled');return
                    if not source_unchanged(commit,fingerprints):raise RuntimeError('Pinned source changed before launch')
                    if decision=='repair_projection':
                        state('repair_projection')
                        print('START_PROJECTION_REPAIR; original GPU and runner locks held',flush=True)
                        repaired=subprocess.run([sys.executable,'-u',str(ROOT/'scripts/run_projection_repair.py'),'--active-project',str(active)],cwd=ROOT,env=env)
                        if repaired.returncode:
                            state('blocked_projection_repair',exit_code=repaired.returncode)
                            raise RuntimeError('Projection repair failed or paused; Flow will not start')
                        status=json.loads(status_path.read_text())
                        if readiness(status,running_jobs(active),(original_base/'stop_requested.json').exists(),required_routes)!='ready':
                            state('blocked_after_projection_repair');return
                        print('PROJECTION_REPAIR_COMPLETED; next Flow v2',flush=True)
                    if (base/'stop_requested.json').exists():state('cancelled');return
                    state('running')
                    result=subprocess.run([sys.executable,'-u',str(ROOT/'scripts/run_flow_followup.py'),'--active-project',str(active)],cwd=ROOT,env=env)
                    stage='completed' if result.returncode==0 and (base/'completed.json').exists() else 'paused' if result.returncode==0 and (base/'stop_requested.json').exists() else 'needs_attention'
                    state(stage,exit_code=result.returncode)
                    if result.returncode:raise RuntimeError('Follow-up failed; inspect logs/flow_followup_console.log and follow-up failure.json')
                    return
            except BlockingIOError:
                time.sleep(15)


if __name__=='__main__':main()
