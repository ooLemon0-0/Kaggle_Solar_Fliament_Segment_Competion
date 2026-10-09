import _bootstrap
import argparse
import csv
import importlib.metadata
from pathlib import Path

import numpy as np
import torch
from pycocotools import mask as mu

from src.data.dataset import discover, FilamentDataset, observation
from src.pipeline import load_model
from src.utils.common import ROOT, MODELS, restored_config, split_path, load_json, save_json, digest


def main(debug):
    # A native-resolution compressed mask may legitimately exceed 128 KiB.
    csv.field_size_limit(2**31-1)
    expected={observation(i) for i in load_json(split_path('val',debug))['images']}
    test={p.stem for p in discover()[2].glob('*.jpeg')}
    results={}
    for name in MODELS:
        cfg=restored_config(name,debug);run=Path(cfg['run_dir'])
        for file in ['checkpoints/best.pt','checkpoints/last.pt','config.json','training_log.csv','metrics.json','submission.csv']:
            if not (run/file).is_file():raise AssertionError(f'Missing {run/file}')
        predictions=load_json(run/'predictions/val.json')
        assert {p['image_id'] for p in predictions['predictions']}==expected
        assert predictions['split_sha256']==digest(split_path('val',debug))
        trained=load_json(run/'config.json')
        assert trained['train_split_sha256']==digest(split_path('train',debug))
        test_predictions=load_json(run/'predictions/test.json')
        assert {p['image_id'] for p in test_predictions['predictions']}==test
        with (run/'submission.csv').open(newline='',encoding='utf-8') as f:
            reader=csv.DictReader(f);assert reader.fieldnames==['filament_id','segmentation_rle'];rows=list(reader)
        assert len({r['filament_id'] for r in rows})==len(rows)
        assert all(r['filament_id'].split('_',1)[0] in test for r in rows)
        assert len(rows)==sum(len(p['instances']) for p in test_predictions['predictions'])
        samples=rows[::max(1,len(rows)//10)][:10]
        for row in samples:
            mask=mu.decode({'size':[2048,2048],'counts':row['segmentation_rle']})
            assert mask.shape==(2048,2048) and mask.any()
            assert mu.encode(np.asfortranarray(mask))['counts'].decode('ascii')==row['segmentation_rle']
        results[name]={'status':'PASS','test_images':len(test),'submission_rows':len(rows),'sampled_rle_roundtrips':len(samples),
                       'val_observations':len(expected),'epochs':load_json(run/'training_summary.json')['epochs_completed']}
    cfg=restored_config('dino_transformer',debug);model=load_model(cfg)
    sample=FilamentDataset(split_path('train',debug),cfg['image_size'])[0]
    inputs=sample['image'][None].cuda()
    target={k:v.cuda() if isinstance(v,torch.Tensor) else v for k,v in sample.items() if k!='image'}
    with torch.autocast('cuda',dtype=torch.float16):
        outputs=model(inputs);loss=model.loss(inputs,[target])
    assert outputs['masks'].shape==(1,cfg['num_queries'],256,256)
    assert outputs['objectness'].shape==(1,cfg['num_queries'])
    loss.backward()
    assert all(p.grad is None for p in model.backbone.parameters())
    assert model.mask_embedding[-1].weight.grad is not None and model.mask_embedding[-1].weight.grad.abs().sum()>0
    assert torch.isfinite(loss)
    results['dino_query_head']={'status':'PASS','mask_shape':list(outputs['masks'].shape),'objectness_shape':list(outputs['objectness'].shape),
                               'frozen_backbone_gradient':False,'mask_head_backward':True,'loss':float(loss.detach())}
    packages={d.metadata['Name']:d.version for d in importlib.metadata.distributions() if d.metadata['Name']}
    save_json(ROOT/'output/environment_packages.json',packages)
    path=ROOT/'output/benchmark'/('debug' if debug else 'full')/'artifact_checks.json'
    save_json(path,results);print(results,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--debug',action='store_true');a=p.parse_args();main(a.debug)
