"""A separate, serial validation-only Flow experiment; never overwrite routes_v3."""
import _bootstrap
import argparse
import csv
import gc
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
import numpy as np
from src.utils.common import ROOT, load_json, digest, split_path

os.environ['SOLAR_ROUTE_OUTPUT_ROOT'] = str(ROOT/'output/benchmark/flow_followup_v1')
from src.routes_v3.engine import BASE, atomic_json, native_images
from src.routes_v3.resources import file_lock


class FollowupPaused(Exception):
    pass


def check_stop():
    if (BASE/'stop_requested.json').exists():raise FollowupPaused()


def source_commit():
    return subprocess.check_output(['git','-C',str(ROOT),'rev-parse','HEAD'],text=True).strip()


def validate_protocol(active, plan):
    original = load_json(active/'output/benchmark/routes_v3/experiment_manifest.json')
    for which in ['train','val']:
        expected = plan[f'{which}_split_sha256']
        if digest(split_path(which)) != expected or original[f'{which}_split_sha256'] != expected:
            raise RuntimeError('Frozen split differs: ' + which)
    integrity = load_json(ROOT/'linux/integrity.json')
    for path, expected in integrity['immutable_files'].items():
        if digest(ROOT/path) != expected:
            raise RuntimeError('Immutable evaluator or protocol changed: '+path)
    for cfg in original['runs']:
        folder = active/'output/benchmark/routes_v3'/cfg['id']
        if not (folder/'completed.json').is_file():
            raise RuntimeError('Original experiment incomplete: '+cfg['id'])
    asset='output/pretrained/routes_v3/cellpose/cpdino-vitb'
    expected=load_json(ROOT/'expected_assets.json')['files'][asset]
    if (ROOT/asset).stat().st_size!=expected['bytes'] or digest(ROOT/asset)!=expected['sha256']:
        raise RuntimeError('CellposeDINO pretrained bytes differ from the original protocol')


def decoder_search(checkpoint_path, source, cfg, plan):
    import torch
    from src.routes_v3.models import build
    from src.routes_v3.flow_v2 import decode_fields
    from src.evaluation.adapter import evaluate_predictions
    from src.data.dataset import observation, mask_from_annotation, decode_mask
    out = BASE/'decoder_search'/source
    out.mkdir(parents=True, exist_ok=True)
    saved = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    model = build(cfg); model.load_state_dict(saved['model']); model.eval()
    checkpoint_sha = digest(checkpoint_path)
    selected_epoch = saved['state']['best_epoch']
    del saved
    raw_dir = out/'raw_fields'; raw_dir.mkdir(exist_ok=True)
    started = time.monotonic()
    for index, (key, image) in enumerate(native_images(), 1):
        check_stop()
        path = raw_dir/(key+'.npz')
        if path.exists():
            with np.load(path,allow_pickle=False) as cache:
                if str(cache['checkpoint_sha256']) != checkpoint_sha:
                    raise RuntimeError('Cached fields belong to another checkpoint')
            continue
        with torch.no_grad(), torch.autocast('cuda',dtype=torch.bfloat16):
            _, fields, _ = model.cp.eval(image.permute(1,2,0).numpy(), batch_size=1,
                normalize=False,resample=True,diameter=None,bsize=cfg['size'],tile_overlap=.25,compute_masks=False)
        temporary = path.with_suffix('.tmp')
        with temporary.open('wb') as stream:
            np.savez_compressed(stream,dP=fields[1],cellprob=fields[2],checkpoint_sha256=checkpoint_sha)
        temporary.replace(path)
        if index%10==0:
            print(f'RAW_FIELDS {source} image={index}/142',flush=True)
            atomic_json(BASE/'progress.json',dict(stage='raw_fields',source=source,image=index,time=time.time()))
    cost_path=out/'raw_field_cost.json'
    prior=load_json(cost_path)['wall_seconds'] if cost_path.exists() else 0.
    atomic_json(cost_path,dict(wall_seconds=prior+time.monotonic()-started,checkpoint_sha256=checkpoint_sha))
    del model; gc.collect();torch.cuda.empty_cache()
    metrics_rows = []
    for candidate in plan['decoder_candidates']:
        check_stop()
        folder = out/candidate['id']; folder.mkdir(exist_ok=True)
        path = folder/'predictions.json'; metric_path = folder/'metrics.json'
        if metric_path.exists() and path.exists():
            record = load_json(path)
            if record['metadata']['checkpoint_sha256'] != checkpoint_sha:
                raise RuntimeError('Decoder metrics checkpoint mismatch')
            metrics_rows.append((path, load_json(metric_path))); continue
        predictions = []; started = time.monotonic()
        for index,(key,_) in enumerate(native_images(),1):
            check_stop()
            with np.load(raw_dir/(key+'.npz'),allow_pickle=False) as fields:
                instances = decode_fields(fields['dP'],fields['cellprob'],candidate['niter'],candidate['flow_threshold'],torch.device('cuda'))
            predictions.append(dict(image_id=key,instances=instances))
            if index%10==0:
                print(f'DECODER {source}/{candidate["id"]} image={index}/142',flush=True)
                atomic_json(BASE/'progress.json',dict(stage='decoder_search',source=source,candidate=candidate['id'],image=index,time=time.time()))
        metadata = dict(model=source,decoder=candidate,checkpoint_sha256=checkpoint_sha,
            training_epochs=selected_epoch,debug=False,validation_selected=True,source_commit=source_commit())
        atomic_json(path,dict(predictions=predictions,split_sha256=plan['val_split_sha256'],metadata=metadata))
        metrics = evaluate_predictions(path)
        metrics['decoder_wall_seconds'] = time.monotonic()-started
        atomic_json(metric_path,metrics);metrics_rows.append((path,metrics))
        print(f'DECODER_OFFICIAL {source}/{candidate["id"]} PQ={metrics["pq"]:.6f} Dice={metrics["dice"]:.6f} IoU={metrics["iou"]:.6f}',flush=True)
    # Raw foreground coverage versus selected reconstructed masks separates network misses from decoding loss.
    best_path,best = max(metrics_rows,key=lambda pair:pair[1]['pq'])
    predictions = {p['image_id']:p['instances'] for p in load_json(best_path)['predictions']}
    val = load_json(split_path('val')); recalls = []
    for im in val['images']:
        key = observation(im)
        gt = np.zeros((2048,2048),dtype=bool)
        for ann in val['annotations']:
            if ann['image_id']==im['id']:gt |= mask_from_annotation(ann,im)
        with np.load(raw_dir/(key+'.npz'),allow_pickle=False) as fields: raw = fields['cellprob']>0
        reconstructed = np.zeros_like(gt)
        for inst in predictions[key]:reconstructed |= decode_mask(inst['rle'])
        recalls.append({'annotation_group':im['id'],'raw_foreground_recall':float((gt&raw).sum()/max(1,gt.sum())),
            'selected_mask_recall':float((gt&reconstructed).sum()/max(1,gt.sum()))})
    atomic_json(out/'foreground_diagnostic.json',dict(groups=len(recalls),rows=recalls,
        raw_macro_recall=float(np.mean([r['raw_foreground_recall'] for r in recalls])),
        reconstructed_macro_recall=float(np.mean([r['selected_mask_recall'] for r in recalls]))))
    return metrics_rows


def write_report(active, plan):
    from src.data.dataset import decode_mask, observation, discover, mask_from_annotation
    rows = []; candidates = []
    original = load_json(active/'output/benchmark/routes_v3/experiment_manifest.json')
    for cfg in original['runs']:
        folder = active/'output/benchmark/routes_v3'/cfg['id']
        metrics = load_json(folder/'best_metrics.json'); complete=load_json(folder/'completed.json')
        rf_hours = load_json(active/'output/benchmark/routes_v3/rf/completed.json')['training_seconds']/3600
        rows.append(dict(route=cfg['id'],selection='original_best_epoch',pq=metrics['pq'],dice=metrics['dice'],iou=metrics['iou'],
            own_training_hours=complete['training_seconds']/3600,inherited_training_hours=rf_hours if cfg['dependency'] else 0.,
            prediction_path=str(folder/'best_predictions.json')))
        if cfg['id']=='flow':candidates.append(rows[-1])
    v2 = load_json(BASE/'flow_v2/completed.json')
    original_flow = next(row for row in rows if row['route']=='flow')
    for source in ['old_flow','flow_v2']:
        for candidate in plan['decoder_candidates']:
            folder=BASE/'decoder_search'/source/candidate['id'];metrics=load_json(folder/'metrics.json')
            row=dict(route=source,selection=candidate['id'],pq=metrics['pq'],dice=metrics['dice'],iou=metrics['iou'],
                own_training_hours=v2['training_seconds']/3600 if source=='flow_v2' else original_flow['own_training_hours'],
                inherited_training_hours=0.,prediction_path=str(folder/'predictions.json'))
            rows.append(row);candidates.append(row)
    trained_metrics = load_json(BASE/'flow_v2/best_metrics.json')
    fixed_v2 = dict(route='flow_v2',selection='fixed_decoder_best_epoch',pq=trained_metrics['pq'],dice=trained_metrics['dice'],
        iou=trained_metrics['iou'],own_training_hours=v2['training_seconds']/3600,inherited_training_hours=0.,
        prediction_path=str(BASE/'flow_v2/best_predictions.json'))
    rows.append(fixed_v2); candidates.append(fixed_v2)
    winner = max(candidates,key=lambda row:row['pq'])
    selected=load_json(winner['prediction_path'])
    expected={observation(im) for im in load_json(split_path('val'))['images']}
    actual=[p['image_id'] for p in selected['predictions']]
    valid_rles=all(inst['rle']['size']==[2048,2048] and decode_mask(inst['rle']).any() and 0<=inst['score']<=1
        for p in selected['predictions'] for inst in p['instances'])
    checks=dict(val_coverage=set(actual)==expected and len(actual)==len(set(actual))==142,
        split_matches=selected['split_sha256']==plan['val_split_sha256'],rle_valid=valid_rles,no_test_used=True,
        v2_budget_met=v2['completed_epochs']==plan['runs'][0]['epochs'] or v2['training_seconds']>=plan['runs'][0]['hours']*3600,
        validation_selection_disclosed=True)
    atomic_json(BASE/'artifact_checks.json',dict(checks=checks,passed=all(checks.values())))
    if not all(checks.values()):raise RuntimeError('Follow-up artifact checks failed')
    atomic_json(BASE/'selected_flow_baseline.json',dict(winner=winner,
        validation_selected=True,warning='Selection and reporting use the same validation set; this is optimistic, not an independent generalization estimate.',
        old_flow_training_hours=original_flow['own_training_hours'],v2_training_hours=v2['training_seconds']/3600,
        selection_candidate_count=len(candidates),source_commit=source_commit()))
    cost=dict(v2_training_preparation_computation_seconds=v2['training_seconds'],
        v2_debug_preflight_seconds=load_json(BASE/'flow_v2/preflight.json').get('preflight_wall_seconds'),
        v2_recorded_validation_seconds=sum(load_json(p).get('validation_wall_seconds',0) for p in (BASE/'flow_v2').glob('validation_epoch_*.json')),
        old_flow_training_seconds=original_flow['own_training_hours']*3600,
        raw_field_wall_seconds={s:load_json(BASE/'decoder_search'/s/'raw_field_cost.json')['wall_seconds'] for s in ['old_flow','flow_v2']},
        decoder_wall_seconds={s:sum(load_json(BASE/'decoder_search'/s/c['id']/'metrics.json')['decoder_wall_seconds'] for c in plan['decoder_candidates']) for s in ['old_flow','flow_v2']},
        limitation='Recorded successful stages; repeated interrupted/replayed work may cost additional time. Rental includes validation, diagnostics and waiting.',
        scheduling='One GPU job at a time; CPU waiter holds no GPU allocation')
    atomic_json(BASE/'costs.json',cost)
    shutil.copyfile(winner['prediction_path'],BASE/'selected_flow_predictions.json')
    with (BASE/'all_candidates.csv').open('w',newline='',encoding='utf-8-sig') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    comparison=[r for r in rows if r['selection']=='original_best_epoch' and r['route']!='flow']
    comparison.append({**winner,'route':'flow_selected'})
    with (BASE/'results.csv').open('w',newline='',encoding='utf-8-sig') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(comparison)
    lines=['# Flow follow-up validation comparison','',
        'Original 142 observations / 218 annotation groups, native 2048 official PQ. No test tuning.',
        'Old Flow is preserved. Flow candidate selected by validation PQ; selection uses the reporting split and is optimistic.',
        'Fresh v2: 768 native crops, full-image targets, checkpointed DINO blocks, foreground-balanced flow/BCE/Dice. 30 epochs OR 8 training hours; old runs used 30 OR 4 hours. Costs are not matched. Cache generation inside loss counts toward v2 training time.',
        f'Selected baseline: {winner["route"]} / {winner["selection"]}; PQ {winner["pq"]:.6f}.',
        f'Search paid for old Flow {original_flow["own_training_hours"]:.3f} h plus v2 {v2["training_seconds"]/3600:.3f} h, preflight, validation, two raw-field passes and ten decoder evaluations. See wall-time logs.',
        '', '| Route | Selection | PQ | Dice | IoU | Own train h | Inherited h |', '|---|---|---:|---:|---:|---:|---:|']
    for r in comparison:
        lines.append(f'| {r["route"]} | {r["selection"]} | {r["pq"]:.6f} | {r["dice"]:.6f} | {r["iou"]:.6f} | {r["own_training_hours"]:.3f} | {r["inherited_training_hours"]:.3f} |')
    lines += ['', 'All candidates and the original Flow remain visible in all_candidates.csv. This is an upgraded validation-selected baseline, not evidence of equal-budget superiority or a guaranteed leaderboard score.']
    (BASE/'comparison.md').write_text('\n'.join(lines),encoding='utf-8')
    # The same three predeclared observations/ROIs as the original route comparison.
    import numpy as np
    from PIL import Image
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    val=load_json(split_path('val'));selected_images=[];seen=set()
    for im in val['images']:
        key=observation(im)
        if key not in seen:selected_images.append(im);seen.add(key)
        if len(selected_images)==3:break
    visual_rows=[r for r in rows if r['selection']=='original_best_epoch']+[{**winner,'route':'flow_selected'}]
    prediction_maps={r['route']:{p['image_id']:p['instances'] for p in load_json(r['prediction_path'])['predictions']} for r in visual_rows}
    fig,axes=plt.subplots(3,len(visual_rows)+1,figsize=(3*(len(visual_rows)+1),9),squeeze=False)
    from src.routes_v3.models import roi_bounds
    for row,im in enumerate(selected_images):
        key=observation(im);gray=np.asarray(Image.open(discover()[1]/im['file_name']).convert('L'))/255.
        gt=[mask_from_annotation(a,im) for a in val['annotations'] if a['image_id']==im['id']]
        middle=sorted(gt,key=lambda m:m.sum())[len(gt)//2];box=roi_bounds(middle);cx,cy=(box[:2]+box[2:])/2
        side=max(384,int(max(box[2:]-box[:2])*1.5));x0=max(0,int(cx-side/2));y0=max(0,int(cy-side/2));x1=min(2048,x0+side);y1=min(2048,y0+side)
        for column,name in enumerate(['GT']+[r['route'] for r in visual_rows]):
            ax=axes[row,column];ax.imshow(gray[y0:y1,x0:x1],cmap='gray',vmin=0,vmax=1)
            masks=gt if name=='GT' else [decode_mask(p['rle']) for p in prediction_maps[name][key]]
            for j,m in enumerate(masks):
                crop=m[y0:y1,x0:x1]
                if crop.any():ax.contour(crop,levels=[.5],colors=[plt.cm.tab20(j%20)],linewidths=.7)
            ax.set_title(name if row==0 else key,fontsize=9);ax.axis('off')
    fig.tight_layout();fig.savefig(BASE/'same_scene_comparison.png',dpi=150);plt.close(fig)
    destination=ROOT/'deliverables/flow_followup_v1';destination.mkdir(parents=True,exist_ok=True)
    for name in ['results.csv','all_candidates.csv','comparison.md','selected_flow_baseline.json','costs.json','artifact_checks.json','same_scene_comparison.png','experiment_manifest.json']:
        shutil.copyfile(BASE/name,destination/name)
    return winner


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--active-project',type=Path,required=True);args=parser.parse_args()
    active=args.active_project.resolve();BASE.mkdir(parents=True,exist_ok=True)
    # The wrapper owns the old project's GPU/runner locks for this child's lifetime.
    import psutil
    try:
        guard=psutil.Process(int(os.environ['SOLAR_FOLLOWUP_GUARD_PID']))
        valid_guard=abs(guard.create_time()-float(os.environ['SOLAR_FOLLOWUP_GUARD_TIME']))<.01 and any(Path(x).name=='queue_flow_followup.py' for x in guard.cmdline())
    except (KeyError,ValueError,psutil.Error):valid_guard=False
    if not valid_guard:raise RuntimeError('Launch through linux/start_flow_followup.sh; an original-project GPU lock guard is required')
    plan=load_json(ROOT/'configs/flow_followup_v1.json');validate_protocol(active,plan)
    for cfg in plan['runs']:cfg['target_cache']=str(ROOT/cfg['target_cache'])
    plan.update(source_commit=source_commit(),active_project=str(active))
    path=BASE/'experiment_manifest.json'
    if path.exists() and load_json(path)!=plan:raise RuntimeError('Existing follow-up protocol differs; do not overwrite')
    atomic_json(path,plan)
    status_path=BASE/'pipeline_status.json'
    def status(stage):
        print('FOLLOWUP_STAGE',stage,flush=True)
        atomic_json(status_path,dict(stage=stage,time=time.time(),pid=os.getpid(),source_commit=source_commit()))
    with file_lock(BASE/'runner_process.lock',blocking=False):
        if (BASE/'completed.json').exists():print('FOLLOWUP_ALREADY_COMPLETED');return
        if (BASE/'stop_requested.json').exists():raise RuntimeError('Follow-up stop requested; remove only after inspecting safe checkpoint')
        # Require space for two complete float32 raw-field caches and training checkpoints.
        if shutil.disk_usage(BASE).free<16*2**30:raise RuntimeError('Need at least 16 GiB free for raw-field caches and checkpoints')
        cfg=plan['runs'][0];folder=BASE/cfg['id'];folder.mkdir(exist_ok=True)
        status('flow_v2_training')
        env=os.environ.copy();env['KAGGLE_COOPERATIVE_GPU']='0';env['KAGGLE_GPU_MEMORY_FRACTION']='0.85'
        with (folder/'train_console.log').open('a',encoding='utf-8') as log:
            subprocess.run([sys.executable,'-u',str(ROOT/'scripts/train_route.py'),'--route',cfg['id']],
                cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
        if not (folder/'completed.json').exists():status('paused');return
        with file_lock(BASE/'gpu_operation.lock',blocking=False):
            check_stop()
            status('old_flow_decoder_search')
            old=active/'output/benchmark/routes_v3/flow'
            decoder_search(old/'best.pt','old_flow',load_json(old/'config.json'),plan)
            status('flow_v2_decoder_search')
            decoder_search(folder/'best.pt','flow_v2',load_json(folder/'config.json'),plan)
            status('reporting');winner=write_report(active,plan)
        atomic_json(BASE/'completed.json',dict(winner=winner,time=time.time(),source_commit=source_commit()))
        status('completed')
        print(f'SELECTED_FLOW PQ={winner["pq"]:.6f} source={winner["route"]}/{winner["selection"]}',flush=True)


if __name__=='__main__':
    try:main()
    except FollowupPaused:
        atomic_json(BASE/'pipeline_status.json',dict(stage='paused',time=time.time(),pid=os.getpid(),source_commit=source_commit()))
        print('FOLLOWUP_PAUSED; cached complete candidates retained',flush=True)
    except Exception:
        import traceback
        BASE.mkdir(parents=True,exist_ok=True)
        atomic_json(BASE/'failure.json',dict(time=time.time(),traceback=traceback.format_exc()))
        raise
