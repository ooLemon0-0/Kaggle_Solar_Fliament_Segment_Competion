"""Standalone CPU-only helper usable via git show BEFORE replacing pinned source.

Cancel only a waiting wrapper. Never signal a trainer, running Flow or the original
queue. Refuse upgrades after any Flow training/protocol has started.
"""
import argparse
import json
from pathlib import Path
import time
import psutil


def checkout_processes(root):
    found=[]
    names={'queue_flow_followup.py','run_flow_followup.py','train_route.py','run_projection_repair.py'}
    for p in psutil.process_iter(['pid','cmdline','create_time']):
        try:
            command=p.info['cmdline'] or []
            if any(Path(x).name in names for x in command) and (Path(p.cwd()).resolve()==root or any(str(root) in x for x in command)):
                found.append(p)
        except (psutil.Error,OSError):pass
    return found


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--checkout',type=Path,required=True)
    root=parser.parse_args().checkout.resolve();base=root/'output/benchmark/flow_followup_v1'
    if not (root/'linux/queue_flow_followup.py').exists():raise RuntimeError('Unexpected checkout')
    for relative in ['experiment_manifest.json','flow_v2/last.pt','flow_v2/completed.json','completed.json']:
        if (base/relative).exists():raise RuntimeError('Flow has already started; refusing mid-experiment source upgrade: '+relative)
    processes=checkout_processes(root)
    for p in processes:
        if not any(Path(x).name=='queue_flow_followup.py' for x in p.cmdline()) or p.children():
            raise RuntimeError('Follow-up has a running child; do not update its source')
    stop=base/'stop_requested.json'
    if stop.exists():
        old=json.loads(stop.read_text())
        if old.get('reason')!='waiter_rearm_only':raise RuntimeError('Existing stop request requires inspection; will not clear it')
    if processes:
        base.mkdir(parents=True,exist_ok=True)
        # Old pinned version already checks this file on each 15-second wait.
        temporary=stop.with_suffix('.tmp')
        temporary.write_text(json.dumps(dict(time=time.time(),reason='waiter_rearm_only')))
        temporary.replace(stop)
        deadline=time.monotonic()+35
        while checkout_processes(root):
            if time.monotonic()>=deadline:raise RuntimeError('Waiter did not exit; source must remain unchanged')
            time.sleep(.5)
    # Check actual processes and protocol again; a race into Flow blocks the merge.
    if checkout_processes(root) or (base/'experiment_manifest.json').exists():
        raise RuntimeError('Follow-up started during cancellation; source must remain unchanged')
    if stop.exists():stop.unlink()
    print('WAITING_FOLLOWUP_CANCELLED_SAFE_TO_MERGE; original training untouched',flush=True)


if __name__=='__main__':main()
