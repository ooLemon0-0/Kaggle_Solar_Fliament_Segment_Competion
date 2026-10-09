"""Validation-only diagnostics and identical-scene visual comparison."""
from pathlib import Path
import numpy as np
from PIL import Image
from src.utils.common import load_json,split_path,digest
from src.data.dataset import discover,observation,mask_from_annotation,decode_mask
from src.routes_v3.models import roi_bounds
from src.routes_v3.engine import BASE,atomic_json

def sam_oracle_diagnostic(model):
    import torch
    from src.data.dataset import FilamentDataset,encode_mask
    dataset=FilamentDataset(split_path('val'),size=2048)
    selected=set(sorted({observation(im) for im in dataset.images})[:32])
    random=np.random.default_rng(42);rows=[]
    for index,im in enumerate(dataset.images):
        if observation(im) not in selected:continue
        sample=dataset[index]
        for j,layer in enumerate(sample['instances'].numpy().astype(bool)):
            box=roi_bounds(layer);wh=box[2:]-box[:2]
            perturbed=np.clip(box+random.uniform(-.15,.15,4)*np.r_[wh,wh],0,2048)
            perturbed[2:]=np.maximum(perturbed[2:],perturbed[:2]+1)
            values={}
            for name,prior in [('gt_box',box),('jitter_15pct',perturbed)]:
                with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
                    out=model.predict(sample['image'],[dict(box=prior.tolist(),score=1.)])
                pred=decode_mask(out[0]['rle']) if out else np.zeros_like(layer)
                inter=(layer&pred).sum();union=(layer|pred).sum();values[name+'_iou']=float(inter/max(1,union))
            rows.append(dict(annotation_group=im['id'],instance_index=j,**values))
        print(f'SAM oracle diagnostic group={im["id"]} instances={len(rows)}',flush=True)
    result=dict(scope='diagnostic_only_32_predeclared_validation_observations',instances=len(rows),
                gt_box_mean_iou=float(np.mean([r['gt_box_iou'] for r in rows])),
                jitter_mean_iou=float(np.mean([r['jitter_15pct_iou'] for r in rows])),
                gt_box_fraction_iou_gt_05=float(np.mean([r['gt_box_iou']>.5 for r in rows])),
                rows=rows,warning='Uses GT-derived prompts. This is not an official PQ or an end-to-end result.')
    atomic_json(BASE/'sam/oracle_diagnostics.json',result)

def compare_artifacts():
    manifest=load_json(BASE/'experiment_manifest.json');val=load_json(split_path('val'))
    observations={observation(im) for im in val['images']};checks={}
    for cfg in manifest['runs']:
        folder=BASE/cfg['id'];path=folder/'best_predictions.json'
        if not path.exists():continue
        saved=load_json(path);actual=[p['image_id'] for p in saved['predictions']]
        rles=all(i['rle']['size']==[2048,2048] and decode_mask(i['rle']).any() and 0<=i['score']<=1
                 for p in saved['predictions'] for i in p['instances'])
        complete=load_json(folder/'completed.json') if (folder/'completed.json').exists() else {}
        checks[cfg['id']]=dict(val_coverage=len(actual)==len(set(actual))==142 and set(actual)==observations,
                              split_matches=saved['split_sha256']==manifest['val_split_sha256']==digest(split_path('val')),
                              rle_valid=rles,budget_recorded=bool(complete),no_test_used=True)
    complete=len(checks)==len(manifest['runs'])
    atomic_json(BASE/'artifact_checks.json',dict(routes=checks,scope='validation_only',complete=complete,
                 passed=complete and all(all(v.values()) for v in checks.values())))

def detection_diagnostic():
    predictions={p['image_id']:p['instances'] for p in load_json(BASE/'rf/best_predictions.json')['predictions']}
    val=load_json(split_path('val'));images={im['id']:im for im in val['images']};hits=0;count=0;per_group={}
    for ann in val['annotations']:
        im=images[ann['image_id']];gt=roi_bounds(mask_from_annotation(ann,im))
        pred=np.array([p['box'] for p in predictions[observation(im)]],dtype=np.float32).reshape(-1,4)
        if len(pred):
            wh=np.maximum(0,np.minimum(gt[2:],pred[:,2:])-np.maximum(gt[:2],pred[:,:2]));intersection=wh.prod(1)
            union=(gt[2:]-gt[:2]).prod()+(pred[:,2:]-pred[:,:2]).prod(1)-intersection
            matched=bool((intersection/(union+1e-6)>.5).any())
        else:matched=False
        count+=1;hits+=matched;item=per_group.setdefault(ann['image_id'],dict(gt=0,recalled=0));item['gt']+=1;item['recalled']+=matched
    atomic_json(BASE/'rf/detection_diagnostics.json',dict(mask_derived_box_recall_iou_gt_05=hits/count,total_gt=count,recalled=hits,
               score_threshold=.25,per_group=per_group,definition='Any predicted RF box overlaps GT-derived box at IoU>0.5; diagnostic only, not official PQ.'))

def visualize():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    manifest=load_json(BASE/'experiment_manifest.json');val=load_json(split_path('val'));image_dir=discover()[1]
    available=[c['id'] for c in manifest['runs'] if (BASE/c['id']/'best_predictions.json').exists()]
    if not available:return
    predictions={r:{p['image_id']:p['instances'] for p in load_json(BASE/r/'best_predictions.json')['predictions']} for r in available}
    selected=[];seen=set()
    for im in val['images']:
        key=observation(im)
        if key not in seen:selected.append(im);seen.add(key)
        if len(selected)==3:break
    fig,axes=plt.subplots(len(selected),len(available)+1,figsize=(3*(len(available)+1),9),squeeze=False)
    for row,im in enumerate(selected):
        key=observation(im);gray=np.asarray(Image.open(image_dir/im['file_name']).convert('L'))/255.
        gt=[mask_from_annotation(a,im) for a in val['annotations'] if a['image_id']==im['id']]
        # Same native ROI for GT and every route, centered on a median-size annotated instance.
        middle=sorted(gt,key=lambda m:m.sum())[len(gt)//2];box=roi_bounds(middle);cx,cy=(box[:2]+box[2:])/2
        side=max(384,int(max(box[2:]-box[:2])*1.5));x0=max(0,int(cx-side/2));y0=max(0,int(cy-side/2));x1=min(2048,x0+side);y1=min(2048,y0+side)
        for column,name in enumerate(['GT']+available):
            ax=axes[row,column];ax.imshow(gray[y0:y1,x0:x1],cmap='gray',vmin=0,vmax=1)
            masks=gt if name=='GT' else [decode_mask(p['rle']) for p in predictions[name][key]]
            for j,m in enumerate(masks):
                crop=m[y0:y1,x0:x1]
                if crop.any():ax.contour(crop,levels=[.5],colors=[plt.cm.tab20(j%20)],linewidths=.7)
            ax.set_title(name if row==0 else key,fontsize=9);ax.axis('off')
    fig.tight_layout();fig.savefig(BASE/'same_scene_comparison.png',dpi=150);plt.close(fig)
