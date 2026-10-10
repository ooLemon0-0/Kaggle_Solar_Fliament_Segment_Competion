"""Read completed metrics/curves on CPU; no training, inference or threshold search."""
import argparse
import csv
import json
from pathlib import Path
import time


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig')) if path.is_file() else {}


def curve(path):
    if not path.is_file():return []
    with path.open(encoding='utf-8-sig',newline='') as stream:
        rows=list(csv.DictReader(stream))
    epochs={}
    for row in rows:
        if not row.get('epoch') or not row.get('pq'):continue
        epochs[int(row['epoch'])]=dict(epoch=int(row['epoch']),pq=float(row['pq']),
            training_hours=float(row.get('training_seconds',0))/3600)
    return [epochs[key] for key in sorted(epochs)]


def route_record(name,folder,inherited_hours=0):
    metrics=read(folder/'best_metrics.json');done=read(folder/'completed.json')
    config=read(folder/'config.json');history=curve(folder/'training_log.csv')
    groups=metrics.get('per_annotation_group',{})
    worst=sorted(groups.items(),key=lambda item:item[1].get('pq',1))[:5]
    return dict(route=name,completed=bool(done),pq=metrics.get('pq'),dice=metrics.get('dice'),iou=metrics.get('iou'),
        best_epoch=done.get('best_epoch'),completed_epochs=done.get('completed_epochs'),
        own_training_hours=done.get('training_seconds',0)/3600,inherited_training_hours=inherited_hours,
        input_size=config.get('size'),split_sha256=metrics.get('split_sha256'),
        metric_debug=metrics.get('debug'),metric_definitions=metrics.get('metric_definitions',{}),
        error_counts={k:metrics.get(k) for k in ['tp','fp','fn','fragmentation','over_merging']},
        recent_validation=history[-6:],validation_group_count=len(groups),
        worst_annotation_groups=[dict(group=key,**value) for key,value in worst])


def review(active,followup):
    base=active/'output/benchmark/routes_v3';extra=followup/'output/benchmark/flow_followup_v1'
    rf_hours=read(base/'rf/completed.json').get('training_seconds',0)/3600
    names=['rf','flow','sam','image_projection','feature_control','feature_structure']
    rows=[route_record(name,base/name,rf_hours if name in names[2:] else 0) for name in names]
    rows.append(route_record('flow_v2',extra/'flow_v2'))
    winner=read(extra/'completed.json').get('winner',{})
    candidates=[]
    candidate_path=extra/'all_candidates.csv'
    if candidate_path.is_file():
        with candidate_path.open(encoding='utf-8-sig',newline='') as stream:
            candidates=list(csv.DictReader(stream))
    best_new=max((r for r in candidates if r['route']=='flow_v2'),key=lambda r:float(r['pq']),default=None)
    return dict(time=time.time(),active_project=str(active),followup_project=str(followup),routes=rows,
        original_stage=read(base/'pipeline_status.json').get('stage'),
        followup_stage=read(extra/'pipeline_status.json').get('stage'),
        original_artifact_checks=read(base/'artifact_checks.json'),
        followup_artifact_checks=read(extra/'artifact_checks.json'),
        original_report_error=read(base/'pipeline_status.json').get('report_error'),
        sam_oracle=read(base/'sam/oracle_diagnostics.json'),sam_diagnostic_failure=read(base/'sam/diagnostic_failure.json'),
        selected_flow=winner,best_flow_v2_decoder=best_new,
        flow_foreground={s:read(extra/'decoder_search'/s/'foreground_diagnostic.json') for s in ['old_flow','flow_v2']},
        flow_costs=read(extra/'costs.json'),
        limitations=['Existing saved metrics only; no score recomputation.',
            'Worst groups are selected for diagnosis, not representative benchmark samples.',
            'Fragmentation/merging counts use any positive overlap, not proof of a segmentation error.',
            'Dice/IoU are foreground-union metrics, not instance matching quality.',
            'Best-validation and decoder selection share this validation split; scores have selection bias.'])


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--active-project',type=Path,default=Path('/root/autodl-tmp/solar_filament_linux'))
    parser.add_argument('--followup-project',type=Path,default=Path(__file__).resolve().parents[1])
    parser.add_argument('--output',type=Path)
    args=parser.parse_args();data=review(args.active_project.resolve(),args.followup_project.resolve())
    output=args.output or args.followup_project/'deliverables/stage1_review'/('review_'+str(time.time_ns())+'.json')
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding='utf-8')
    print('Saved CPU review:',output)
    print('Stages:',data['original_stage'],data['followup_stage'])
    for name in ['original_artifact_checks','followup_artifact_checks']:
        print(name,'passed=',data[name].get('passed','missing'))
    for row in data['routes']:
        print('\n'+row['route'], 'PQ=',row['pq'],'best_epoch=',row['best_epoch'],
              'train_h=',round(row['own_training_hours'],3),'inherited_h=',round(row['inherited_training_hours'],3))
        print('  official counts:',json.dumps(row['error_counts']))
        print('  last validations:',[(r['epoch'],round(r['pq'],6)) for r in row['recent_validation']])
    print('\nFlow winner:',json.dumps(data['selected_flow']))
    print('Best v2 decoder:',json.dumps(data['best_flow_v2_decoder']))
    for source,diag in data['flow_foreground'].items():
        print(source,'raw_recall=',diag.get('raw_macro_recall'),'reconstructed_recall=',diag.get('reconstructed_macro_recall'))
    oracle=data['sam_oracle']
    print('SAM diagnostic only:',{k:oracle.get(k) for k in ['gt_box_mean_iou','jitter_mean_iou','gt_box_fraction_iou_gt_05']})
    print('Report/diagnostic errors:',data['original_report_error'],data['sam_diagnostic_failure'])


if __name__=='__main__':main()
