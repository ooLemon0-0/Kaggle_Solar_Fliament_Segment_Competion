"""Regression checks for observation leakage, COCO masks and official scoring."""
import unittest
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.data.dataset import discover, observation, encode_mask, decode_mask, FilamentDataset
from src.evaluation.official_pq import get_pq_score
from src.evaluation.adapter import evaluate_predictions, bounded_official_overlap
from src.evaluation.official_pq import get_overlap_df
from src.utils.common import load_json, split_path, digest, save_json


class PipelineTests(unittest.TestCase):
    def test_split_preserves_every_annotator_without_leakage(self):
        original=load_json(discover()[0]);train=load_json(split_path('train'));val=load_json(split_path('val'))
        train_obs={observation(i) for i in train['images']};val_obs={observation(i) for i in val['images']}
        self.assertFalse(train_obs&val_obs)
        self.assertEqual({i['id'] for i in original['images']},{i['id'] for i in train['images']+val['images']})
        self.assertEqual({a['id'] for a in original['annotations']},{a['id'] for a in train['annotations']+val['annotations']})
        for image in original['images']:
            target=train if observation(image) in train_obs else val
            self.assertIn(image['id'],{i['id'] for i in target['images']})

    def test_debug_subsets_keep_all_selected_annotations(self):
        for name in ['train','val']:
            parent=load_json(split_path(name));debug=load_json(split_path(name,True))
            obs={observation(i) for i in debug['images']}
            self.assertEqual({i['id'] for i in debug['images']},{i['id'] for i in parent['images'] if observation(i) in obs})

    def test_rle_orientation_roundtrip(self):
        mask=np.zeros((17,23),dtype=bool);mask[2:9,13:21]=True
        self.assertTrue(np.array_equal(mask,decode_mask(encode_mask(mask))))

    def test_official_strict_threshold_and_global_aggregation(self):
        table=pd.DataFrame([{'iou_matrix':torch.tensor([[.5]]),'n_gt':1,'n_pred':1}])
        self.assertEqual(get_pq_score(table),0.)
        table=pd.DataFrame([{'iou_matrix':torch.tensor([[1.]]),'n_gt':1,'n_pred':1},
                            {'iou_matrix':torch.zeros((1,0)),'n_gt':1,'n_pred':0}])
        self.assertAlmostEqual(get_pq_score(table),2/3)

    def test_chunked_overlap_matches_original_official_pipeline(self):
        a=np.zeros((2048,2048),dtype=bool);a[10:30,50:80]=True
        b=np.zeros_like(a);b[20:50,60:90]=True
        gt=pd.DataFrame({'filament_id':['01-obs_1','01-obs_2','02-obs_1'],'segmentation_rle':[encode_mask(a)['counts'],encode_mask(b)['counts'],encode_mask(a)['counts']]})
        pred=pd.DataFrame({'filament_id':['obs_1','obs_2'],'segmentation_rle':[encode_mask(a)['counts'],encode_mask(b)['counts']]})
        original=get_overlap_df(gt,pred);chunked=bounded_official_overlap(gt,pred,block_size=1)
        for r,s in zip(original.itertuples(),chunked.itertuples()):
            self.assertTrue(torch.equal(r.iou_matrix,s.iou_matrix))
            self.assertTrue(torch.equal(r.dice_matrix,s.dice_matrix))
        self.assertEqual(get_pq_score(original),get_pq_score(chunked))

    def test_adapter_rejects_missing_observations(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'pred.json'
            save_json(path,{'split_sha256':digest(split_path('val',True)),'predictions':[],'metadata':{}})
            with self.assertRaisesRegex(ValueError,'exactly cover'):
                evaluate_predictions(path,True)

    def test_loader_returns_instances_not_chirality_labels(self):
        item=FilamentDataset(split_path('train',True))[0]
        self.assertEqual(tuple(item['image'].shape),(3,1024,1024))
        self.assertEqual(tuple(item['instances'].shape[1:]),(1024,1024))
        self.assertTrue(torch.equal(item['instances'].bool().any(0),item['semantic'][0].bool()))


if __name__=='__main__':
    unittest.main()
