"""Run the complete baseline experiment sequentially on the single GPU."""
import _bootstrap
import json
import os
from pathlib import Path
import subprocess
import sys
import traceback
from datetime import datetime, timezone

from src.utils.common import ROOT, MODELS, load_json, split_path, digest


def now():
    return datetime.now(timezone.utc).isoformat()


def main():
    directory = ROOT/'output/benchmark/full'
    directory.mkdir(parents=True, exist_ok=True)
    status_path = directory/'pipeline_status.json'
    if status_path.exists() and load_json(status_path)['status'] == 'running':
        raise RuntimeError('Full baseline pipeline is already marked running; inspect its PID before restarting.')
    for model in MODELS:
        if (ROOT/f'output/{model}/checkpoints/last.pt').exists():
            raise RuntimeError(f'Existing full checkpoint for {model}; refusing to overwrite it.')
    data = load_json(split_path('train'))
    val = load_json(split_path('val'))
    state = {'status':'running','pid':os.getpid(),'started_at':now(),
             'epochs_per_model':30,'model_order':MODELS,'completed_steps':[],
             'train_observations':len({i['file_name'] for i in data['images']}),
             'train_annotation_groups':len(data['images']),
             'val_observations':len({i['file_name'] for i in val['images']}),
             'val_annotation_groups':len(val['images']),
             'train_split_sha256':digest(split_path('train')),'val_split_sha256':digest(split_path('val'))}

    def save():
        state['updated_at'] = now()
        temporary = status_path.with_suffix('.tmp')
        temporary.write_text(json.dumps(state, ensure_ascii=False, indent=2)+'\n',encoding='utf-8')
        temporary.replace(status_path)

    def run_step(label, arguments, log):
        state.update(current_step=label,log_path=str(log))
        log.parent.mkdir(parents=True,exist_ok=True)
        with log.open('w',encoding='utf-8') as stream:
            child = subprocess.Popen([sys.executable,'-u',*arguments],cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT)
            state['child_pid'] = child.pid
            save()
            result = child.wait()
        if result:
            raise RuntimeError(f'{label} failed with exit code {result}; inspect {log}')
        state['completed_steps'].append({'step':label,'finished_at':now()})
        save()

    save()
    try:
        for model in MODELS:
            run_step(f'train:{model}',['scripts/train.py','--model',model,'--epochs','30'],ROOT/f'output/{model}/train_console.log')
            run_step(f'submission:{model}',['scripts/make_submission.py','--model',model],ROOT/f'output/{model}/submission_console.log')
        run_step('compare',['scripts/compare_models.py'],directory/'comparison_console.log')
        run_step('check_artifacts',['scripts/check_artifacts.py'],directory/'artifact_console.log')
        state.update(status='completed',finished_at=now(),current_step=None,child_pid=None)
        save()
    except BaseException as error:
        state.update(status='failed',error=str(error),failed_at=now())
        save()
        traceback.print_exc()
        raise


if __name__ == '__main__':
    main()
