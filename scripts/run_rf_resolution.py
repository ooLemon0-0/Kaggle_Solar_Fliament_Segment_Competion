"""One RF 1536 experiment; isolate outputs, guard the GPU, retain safe resumes."""
import argparse
import contextlib
import csv
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time
import traceback

import psutil

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.routes_v3.resources import file_lock

BASE = ROOT / 'output/benchmark/rf_resolution_v1'
JOB_NAMES = {
    'run_route_experiments.py', 'train_route.py', 'evaluate_routes.py',
    'queue_flow_followup.py', 'run_flow_followup.py', 'run_projection_repair.py',
    'run_full_baselines.py', 'train.py', 'run_rf_resolution.py',
}


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding='utf-8')
    temporary.replace(path)


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def running_jobs():
    jobs = []
    for process in psutil.process_iter(['pid', 'cmdline', 'create_time']):
        try:
            if process.pid == os.getpid():
                continue
            if any(Path(part).name in JOB_NAMES for part in process.info['cmdline'] or []):
                jobs.append(process.info)
        except psutil.Error:
            pass
    return jobs


def configuration(original, expected):
    if original != expected or original['size'] != 1056:
        raise RuntimeError('Original RF recipe differs from the frozen 1056 baseline')
    return dict(original, id='rf_1536', size=1536)


def link_inputs(active):
    for name in ['data', 'output/pretrained', 'output/splits']:
        source, target = active / name, ROOT / name
        if not source.is_dir():
            raise RuntimeError('Missing original input directory: ' + str(source))
        if target.is_symlink():
            if target.resolve() != source.resolve():
                raise RuntimeError('Input link mismatch: ' + str(target))
        elif target.exists():
            raise RuntimeError('Refusing to replace separate checkout input directory: ' + str(target))
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            target.symlink_to(source, target_is_directory=True)


def inspect_inputs(active):
    if active == ROOT.resolve():
        raise RuntimeError('Use a separate Git source checkout, not the original training directory')
    original_base = active / 'output/benchmark/routes_v3'
    integrity = read(ROOT / 'linux/integrity.json')
    manifest = read(original_base / 'experiment_manifest.json')
    state = read(original_base / 'pipeline_status.json')
    required = {c['id'] for c in integrity['runs']}
    if (state.get('stage') != 'completed' or set(state.get('routes', {})) != required
            or any(v != 'completed' for v in state['routes'].values())):
        raise RuntimeError('Original experiments are not completed')
    if (original_base / 'stop_requested.json').exists():
        raise RuntimeError('Original stop request remains; inspect it before starting a new experiment')
    for name, expected in integrity['immutable_files'].items():
        owner = active if name.startswith('output/splits/') else ROOT
        if sha(owner / name) != expected:
            raise RuntimeError('Immutable split/evaluator mismatch: ' + name)
    for which in ['train', 'val']:
        expected = integrity['immutable_files']['output/splits/' + which + '.json']
        if manifest[which + '_split_sha256'] != expected:
            raise RuntimeError('Original manifest split mismatch: ' + which)
    expected_cfg = next(c for c in integrity['runs'] if c['id'] == 'rf')
    cfg = configuration(read(original_base / 'rf/config.json'), expected_cfg)
    done = read(original_base / 'rf/completed.json')
    if done.get('status') != 'completed' or done['config'] != expected_cfg:
        raise RuntimeError('Original RF completion/config mismatch')
    annotations = list((active / 'data/raw').rglob('MAGFiLO_1.0_Annotations_kaggle2026_train.json'))
    if len(annotations) != 1:
        raise RuntimeError('Expected one original training annotation file')
    expected_annotation = read(ROOT / 'linux/data_fingerprints.json')[
        'train/MAGFiLO_1.0_Annotations_kaggle2026_train.json']
    if sha(annotations[0]) != expected_annotation:
        raise RuntimeError('Original training annotations changed')
    assets = read(ROOT / 'expected_assets.json')['files']
    weight = 'output/pretrained/routes_v3/rf/rf-detr-seg-small.pt'
    if sha(active / weight) != assets[weight]['sha256']:
        raise RuntimeError('Original RF pretrained weight mismatch')
    return cfg, manifest, done, read(original_base / 'rf/best_metrics.json')


def source_record():
    dirty = subprocess.check_output(['git', '-C', str(ROOT), 'status', '--porcelain'], text=True)
    if dirty.strip():
        raise RuntimeError('Source must be committed and clean before launch')
    commit = subprocess.check_output(['git', '-C', str(ROOT), 'rev-parse', 'HEAD'], text=True).strip()
    names = subprocess.check_output(['git', '-C', str(ROOT), 'ls-files', '-z']).decode().split('\0')
    return dict(commit=commit, hashes={name: sha(ROOT / name) for name in names if name})


def validate_existing(folder, cfg, resume):
    if (folder / 'config.json').exists() and read(folder / 'config.json') != cfg:
        raise RuntimeError('Existing RF 1536 config mismatch')
    last = folder / 'last.pt'
    if last.exists() and not resume:
        raise RuntimeError('Checkpoint exists: use --resume to continue, never restart over it')
    if resume and not last.exists():
        raise RuntimeError('Safe resume requires last.pt with optimizer/RNG/offset')
    if not last.exists() and any((folder / name).exists() for name in [
        'formal_started.json', 'best.pt', 'best_metrics.json', 'best_predictions.json',
    ]):
        raise RuntimeError('Formal training evidence without last.pt; inspect before restart')
    if not last.exists() and (folder / 'training_log.csv').exists():
        with (folder / 'training_log.csv').open(encoding='utf-8-sig', newline='') as stream:
            if list(csv.DictReader(stream)):
                raise RuntimeError('Training rows without last.pt; cannot silently restart')


def decision(delta, complete_epochs, checks_passed):
    if not checks_passed:
        return 'blocked_artifact_checks'
    if complete_epochs < 30:
        return 'budget_truncated_not_a_matched_epoch_resolution_conclusion'
    if delta >= .005:
        return 'candidate_only_confirm_thin_target_recall_before_detail_1536'
    return 'hold_resolution_escalation_review_errors'


def recovered_wall(costs, now):
    # An unclosed session is conservatively charged through now, including downtime.
    # Never silently reset the wall cap after a crash/reboot.
    return costs.get('wall_seconds', 0.) + max(0., now - costs.get('active_since', now))


def report_completed(cfg, baseline_done, baseline_metrics):
    # Only cached masks/metrics are read. No model load, inference or retraining.
    from src.routes_v3.diagnostics import compare_artifacts
    compare_artifacts()
    checks = read(BASE / 'artifact_checks.json')
    folder = BASE / cfg['id']
    result, metrics = read(folder / 'completed.json'), read(folder / 'best_metrics.json')
    if result['config'] != cfg:
        raise RuntimeError('Completed result config mismatch')
    delta = metrics['pq'] - baseline_metrics['pq']
    comparison = dict(baseline=dict(pq=baseline_metrics['pq'], best_epoch=baseline_done['best_epoch'],
                                   completed_epochs=baseline_done['completed_epochs'],
                                   updates=baseline_done['optimizer_updates'], train_h=baseline_done['training_seconds']/3600),
                      rf_1536=dict(pq=metrics['pq'], dice=metrics['dice'], iou=metrics['iou'],
                                   best_epoch=result['best_epoch'], completed_epochs=result['completed_epochs'],
                                   updates=result['optimizer_updates'], train_h=result['training_seconds']/3600),
                      delta_pq=delta, checks_passed=checks['passed'],
                      decision=decision(delta, result['completed_epochs'], checks['passed']),
                      automatic_next_stage=False,
                      limitations=['Single seed; .005 is a spending gate, not statistical significance.',
                                   'Thin-target recall still needs paired size-stratified diagnosis.',
                                   'Original training crossed Windows/Linux hardware; not bitwise reproducible.'])
    write(BASE / 'comparison.json', comparison)
    print('RF_RESOLUTION_RESULT', json.dumps(comparison), flush=True)
    return checks['passed']


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--active-project', type=Path, required=True)
    parser.add_argument('--followup-project', type=Path,
                        default=Path('/root/autodl-tmp/solar_filament_source_clean'))
    parser.add_argument('--prepare-only', action='store_true', help='CPU checks only')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--preflight-only', action='store_true', help='Eight debug backward steps; no formal training')
    parser.add_argument('--report-only', action='store_true', help='Rebuild a completed report from caches; no model/inference')
    args = parser.parse_args()
    active, followup = args.active_project.resolve(), args.followup_project.resolve()
    original_base = active / 'output/benchmark/routes_v3'
    with contextlib.ExitStack() as locks:
        for path in [original_base / 'runner_process.lock', original_base / 'gpu_operation.lock',
                     followup / 'output/benchmark/flow_followup_v1/queue_wrapper.lock',
                     BASE / 'runner_process.lock']:
            locks.enter_context(file_lock(path, blocking=False))
        jobs = running_jobs()
        if jobs:
            raise RuntimeError('Existing experiment/waiter; duplicate launch refused: ' + json.dumps(jobs))
        cfg, original_manifest, baseline_done, baseline_metrics = inspect_inputs(active)
        source = source_record()
        folder = BASE / cfg['id']
        if (folder / 'completed.json').exists():
            if args.report_only:
                link_inputs(active)
                os.environ['SOLAR_ROUTE_OUTPUT_ROOT'] = str(BASE)
                passed = report_completed(cfg, baseline_done, baseline_metrics)
                state = read(BASE / 'pipeline_status.json')
                state.update(stage='completed' if passed else 'needs_attention', time=time.time())
                state.pop('error', None)
                write(BASE / 'pipeline_status.json', state)
                return
            print('ALREADY_COMPLETED: results preserved; read comparison.json', flush=True)
            return
        if args.report_only:
            raise RuntimeError('Report-only requires completed training; no training started')
        validate_existing(folder, cfg, args.resume)
        previous = read(BASE / 'experiment_manifest.json') if (BASE / 'experiment_manifest.json').exists() else None
        if previous and (previous['source'] != source or previous['runs'] != [cfg]):
            raise RuntimeError('Pinned source/config changed; resume the original commit in its checkout')
        costs = read(BASE / 'costs.json') if (BASE / 'costs.json').exists() else {'wall_seconds': 0.}
        costs['wall_seconds'] = recovered_wall(costs, time.time())
        costs.pop('active_since', None)
        remaining = 6 * 3600 - costs['wall_seconds']
        if remaining <= 0:
            raise RuntimeError('Six-hour cumulative wall budget exhausted; review before adding budget')
        if (BASE / 'stop_requested.json').exists() and not args.resume:
            raise RuntimeError('RF 1536 stop request exists; inspect before explicit safe resume')
        link_inputs(active)
        if args.prepare_only:
            print('RF_1536_INPUTS_READY commit=' + source['commit'] + '; no GPU/model loaded', flush=True)
            return
        # One namespace and one experiment. No warm start from the trained 1056 RF.
        os.environ.update(SOLAR_ROUTE_OUTPUT_ROOT=str(BASE), KAGGLE_COOPERATIVE_GPU='0',
                          HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1')
        os.environ.pop('SOLAR_RF_WARMSTART', None)
        os.environ['SOLAR_PRETRAINED_ROOT'] = str(active / 'output/pretrained/routes_v3')
        folder.mkdir(parents=True, exist_ok=True)
        manifest = dict(original_manifest, runs=[cfg], source=source, active_project=str(active),
                        deliverable_dir=str(ROOT / 'deliverables/rf_resolution_v1'),
                        hypothesis='Only change RF input 1056 -> 1536; same original pretrained initialization.',
                        baseline_pq=baseline_metrics['pq'], wall_budget_seconds=6 * 3600,
                        wall_policy='Cumulative soft cap: request checkpoint pause at accumulation boundary; in-flight preflight/evaluation may finish.',
                        automatic_next_stage=False)
        write(BASE / 'experiment_manifest.json', manifest)
        write(folder / 'config.json', cfg)
        identity = dict(pid=os.getpid(), create_time=psutil.Process().create_time(), source_commit=source['commit'])
        def status(stage, **extra):
            write(BASE / 'pipeline_status.json', dict(stage=stage, time=time.time(), **identity, **extra))
        deadline_done = threading.Event()
        stop_guard = threading.Lock()
        def stop(reason):
            with stop_guard:
                write(BASE / 'stop_requested.json', dict(time=time.time(), reason=reason, **identity))
            print('STOP_REQUESTED', reason, '; safe pause, no next-stage launch', flush=True)
        def deadline():
            if not deadline_done.wait(remaining):
                stop('six_hour_cumulative_wall_budget')
        signal.signal(signal.SIGTERM, lambda *_: stop('SIGTERM'))
        signal.signal(signal.SIGINT, lambda *_: stop('SIGINT'))
        if args.resume:
            (BASE / 'stop_requested.json').unlink(missing_ok=True)
        threading.Thread(target=deadline, daemon=True).start()
        started = time.monotonic()
        costs['active_since'] = time.time()
        write(BASE / 'costs.json', costs)
        try:
            import torch
            from src.routes_v3.engine import train, probe
            status('initializing')
            versions = {key: importlib.metadata.version(key) for key in ['torch', 'torchvision', 'rfdetr']}
            if versions['rfdetr'] != '1.11.2':
                raise RuntimeError('Keep RF-DETR 1.11.2 for the resolution comparison')
            weight = active / 'output/pretrained/routes_v3/rf/rf-detr-seg-small.pt'
            write(BASE / 'environment.json', dict(versions=versions, gpu=torch.cuda.get_device_name(0),
                                                original_versions=original_manifest.get('versions', {}),
                                                pretrained_sha256=sha(weight)))
            torch.cuda.set_per_process_memory_fraction(.85, 0)
            if args.resume:
                saved = torch.load(folder / 'last.pt', map_location='cpu', weights_only=False)
                for key in ['model', 'optimizer', 'state', 'config', 'rng']:
                    if key not in saved:
                        raise RuntimeError('Unsafe resume checkpoint: missing ' + key)
                if saved['config'] != cfg:
                    raise RuntimeError('Checkpoint recipe mismatch')
                for split in ['train', 'val']:
                    if saved['state'][split + '_split_sha256'] != original_manifest[split + '_split_sha256']:
                        raise RuntimeError('Checkpoint split mismatch')
                del saved
            if args.preflight_only or not (folder / 'preflight.json').exists():
                status('preflight_debug')
                evidence = probe(cfg, folder)
                print('PREFLIGHT_DEBUG', json.dumps(evidence), flush=True)
            if (BASE / 'stop_requested.json').exists():
                status('paused_before_formal_training')
                return
            if args.preflight_only:
                status('preflight_passed_no_formal_training')
                return
            status('training_and_validation')
            write(folder / 'formal_started.json', dict(time=time.time(), resume=args.resume, **identity))
            result = train(cfg)
            if result.get('status') != 'completed':
                status('paused')
                return
            status('reporting')
            passed = report_completed(cfg, baseline_done, baseline_metrics)
            status('completed' if passed else 'needs_attention')
        except Exception as exc:
            status('failed', error=repr(exc))
            write(BASE / 'failure.json', dict(time=time.time(), error=repr(exc), traceback=traceback.format_exc()))
            raise
        finally:
            deadline_done.set()
            costs['wall_seconds'] += time.monotonic() - started
            costs.pop('active_since', None)
            costs['updated'] = time.time()
            write(BASE / 'costs.json', costs)


if __name__ == '__main__':
    main()
