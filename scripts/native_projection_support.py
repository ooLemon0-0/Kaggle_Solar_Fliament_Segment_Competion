"""Strict RF1536 provenance and cached reports for P0."""
from pathlib import Path
import json
import shutil


def paired_mask_diagnostics(baseline_path,prediction_path):
    """Per-GT best mask IoU; diagnostic, not the official all-pair PQ."""
    import numpy as np
    from pycocotools import mask as mu
    from src.utils.common import load_json,split_path
    from src.data.dataset import observation,annotation_rle
    val=load_json(split_path('val'))
    images={im['id']:im for im in val['images']}
    predictions={name:{p['image_id']:p['instances'] for p in load_json(path)['predictions']}
                 for name,path in [('RF1536',baseline_path),('P0',prediction_path)]}
    rows=[];cache={}
    for ann in val['annotations']:
        im=images[ann['image_id']];key=observation(im)
        gt=annotation_rle(ann,im)
        area=float(mu.area(gt));box=mu.toBbox(gt)
        proxy=area/max(1.,float(max(box[2:])))
        best={}
        for name,pred in predictions.items():
            if (name,key) not in cache:
                cache[name,key]=[p['rle'] for p in pred[key]]
            masks=cache[name,key]
            best[name]=float(mu.iou([gt],masks,[0]*len(masks)).max()) if masks else 0.
        rows.append(dict(annotation_group=ann['image_id'],annotation_id=ann['id'],
                         area=area,width_proxy=proxy,best_mask_iou=best))
    groups={
        'all':rows,
        'area_le_1024':[r for r in rows if r['area']<=1024],
        'area_1024_to_16384':[r for r in rows if 1024<r['area']<=16384],
        'area_gt_16384':[r for r in rows if r['area']>16384],
        'width_proxy_le_8':[r for r in rows if r['width_proxy']<=8],
        'width_proxy_gt_8':[r for r in rows if r['width_proxy']>8],
    }
    summary={}
    for label,subset in groups.items():
        stats={name:dict(best_mask_iou_mean=float(np.mean([r['best_mask_iou'][name] for r in subset])) if subset else None,
                         recalled=sum(r['best_mask_iou'][name]>.5 for r in subset),
                         mask_recall_iou_gt_05=sum(r['best_mask_iou'][name]>.5 for r in subset)/len(subset) if subset else None)
               for name in predictions}
        summary[label]=dict(gt_count=len(subset),models=stats,
                            gained=sum(r['best_mask_iou']['RF1536']<=.5<r['best_mask_iou']['P0'] for r in subset),
                            lost=sum(r['best_mask_iou']['P0']<=.5<r['best_mask_iou']['RF1536'] for r in subset))
    return dict(scope='full_validation_diagnostic_only',summary=summary,rows=rows,
                definitions='Each GT chooses its highest mask IoU; matches are not one-to-one. Width proxy = native mask area / longest axis-aligned bbox side, not measured physical width. Repeated annotator groups retained. Bins fixed in source before P0 results.')


def inspect_rf1536(project, expected_cfg, original_manifest, read, sha):
    base=project/'output/benchmark/rf_resolution_v1'
    folder=base/'rf_1536'
    manifest=read(base/'experiment_manifest.json')
    done=read(folder/'completed.json')
    metrics=read(folder/'best_metrics.json')
    if read(base/'pipeline_status.json').get('stage')!='completed':
        raise RuntimeError('RF1536 is not completed; do not start P0')
    if read(folder/'config.json')!=expected_cfg or done.get('config')!=expected_cfg:
        raise RuntimeError('RF1536 recipe mismatch')
    if done.get('status')!='completed' or done.get('completed_epochs')!=30 or done.get('optimizer_updates')!=7020:
        raise RuntimeError('Expected completed RF1536 30 epochs/7020 updates')
    if not read(base/'artifact_checks.json').get('passed') or not read(base/'comparison.json').get('checks_passed'):
        raise RuntimeError('RF1536 artifact checks must pass')
    for split in ['train','val']:
        key=split+'_split_sha256'
        if done[key]!=original_manifest[key] or manifest[key]!=original_manifest[key]:
            raise RuntimeError('RF1536 split mismatch')
    if metrics.get('debug') or abs(metrics['pq']-done['best_pq'])>1e-9:
        raise RuntimeError('RF1536 metrics are not the selected formal result')
    if metrics.get('training_epochs')!=done['best_epoch']:
        raise RuntimeError('RF1536 metrics epoch does not match best checkpoint')
    warm=folder/'best.pt'
    reference=dict(project=str(project),warmstart_path=str(warm),warmstart_sha256=sha(warm),
                   source=manifest['source'],completed=done,metrics=metrics,
                   predictions_path=str(folder/'best_predictions.json'),
                   predictions_sha256=sha(folder/'best_predictions.json'))
    return done,metrics,reference


def report(base,cfg,baseline_done,baseline_metrics,read,write):
    from src.routes_v3.diagnostics import compare_artifacts,visualize
    compare_artifacts()
    checks=read(base/'artifact_checks.json')
    manifest=read(base/'experiment_manifest.json')
    reference=manifest['rf1536_reference']
    if baseline_done!=reference['completed'] or baseline_metrics!=reference['metrics']:
        raise RuntimeError('RF1536 reference differs from the pinned experiment')
    # Report rebuilding must use the same inherited artifacts, even after remote restarts.
    from scripts.run_rf_resolution import sha
    if sha(reference['predictions_path'])!=reference['predictions_sha256']:
        raise RuntimeError('RF1536 cached predictions changed')
    folder=base/cfg['id']
    done,metrics=read(folder/'completed.json'),read(folder/'best_metrics.json')
    if done['config']!=cfg:raise RuntimeError('P0 completed config mismatch')
    full_budget=done['completed_epochs']==cfg['epochs'] and done['partial_epoch_samples']==0
    inherited=baseline_done['training_seconds']/3600
    own=done['training_seconds']/3600
    delta=metrics['pq']-baseline_metrics['pq']
    decision=('blocked_artifact_checks' if not checks['passed'] else
              'budget_truncated_review_actual_updates' if not full_budget else
              'candidate_for_D0_and_paired_error_review' if delta>=.005 else
              'review_masks_before_DINO_compute')
    comparison=dict(baseline=dict(pq=baseline_metrics['pq'],best_epoch=baseline_done['best_epoch'],
                                 inherited_training_hours=inherited),
                    native_projection=dict(pq=metrics['pq'],dice=metrics['dice'],iou=metrics['iou'],
                      best_epoch=done['best_epoch'],completed_epochs=done['completed_epochs'],
                      partial_epoch_samples=done['partial_epoch_samples'],updates=done['optimizer_updates'],
                      own_training_hours=own,total_training_hours=own+inherited),
                    delta_pq=delta,checks_passed=checks['passed'],full_registered_epochs=full_budget,
                    decision=decision,automatic_next_stage=False,
                    limitations=['Single seed and repeated use of the same validation set; .005 is a spending gate, not significance.',
                     'Gain over RF1536 includes extra mask-head adaptation and a larger mask grid; not a causal projection-only estimate.',
                     'Frozen RF queries/classes/boxes cannot recover absent proposals.',
                     'Native pixels enter a stride-4 512 grid; final masks are interpolated to 2048, not full-resolution logits.',
                     'M0 at 1056 with 30 extra epochs is not a matched control for this 12-epoch RF1536 experiment.'])
    write(base/'comparison.json',comparison)
    write(base/'paired_mask_diagnostics.json',paired_mask_diagnostics(
        Path(reference['predictions_path']),folder/'best_predictions.json'))
    visualize(references={'RF1536':Path(reference['predictions_path'])})
    lines=['# P0 native projection validation', '',
           f"RF1536 PQ {baseline_metrics['pq']:.6f}; P0 PQ {metrics['pq']:.6f}; delta {delta:+.6f}.",
           f"Dice {metrics['dice']:.6f}; IoU {metrics['iou']:.6f}; best epoch {done['best_epoch']}.",
           f"Completed epochs {done['completed_epochs']}; partial samples {done['partial_epoch_samples']}; updates {done['optimizer_updates']}.",
           f"Own training {own:.3f}h; inherited {inherited:.3f}h; validation/preflight excluded from training hours.",
           f"Artifact checks passed: {checks['passed']}; decision: {decision}; no next GPU job scheduled.", '',
           *['- '+s for s in comparison['limitations']]]
    (base/'comparison.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    deliverables=Path(manifest['deliverable_dir']);deliverables.mkdir(parents=True,exist_ok=True)
    for name in ['comparison.json','comparison.md','artifact_checks.json','paired_mask_diagnostics.json','same_scene_comparison.png']:
        shutil.copy2(base/name,deliverables/name)
    print('NATIVE_PROJECTION_RESULT',json.dumps(comparison),flush=True)
    return checks['passed']
