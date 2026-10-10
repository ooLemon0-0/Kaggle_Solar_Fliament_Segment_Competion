import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import types
import io
import contextlib

import torch
from torch import nn
from torch.nn import functional as F
from src.routes_v3.native_projection import NativeProjectionHead,NativeProjectionRoute,sample_native_targets
from scripts.native_projection_support import inspect_rf1536,paired_mask_diagnostics,report
from scripts.run_rf_resolution import read,write,sha


class Head(nn.Module):
    def __init__(self):
        super().__init__()
        self.scale=nn.Parameter(torch.tensor(1.))
        self.received=[]
    def forward(self,spatial,query,image_size,skip_blocks=False):
        self.received.append((tuple(spatial.shape[-2:]),image_size,skip_blocks))
        return [F.interpolate(spatial.mean(1,keepdim=True)*self.scale,
                              (image_size[0]//4,image_size[1]//4))]
    sparse_forward=forward


class NativeProjectionTests(unittest.TestCase):
    def test_paired_diagnostic_counts_gained_gt_without_changing_pq(self):
        import numpy as np
        from src.data.dataset import encode_mask
        gt=np.zeros((16,16),dtype=bool);gt[2:4,3:9]=True
        rle=encode_mask(gt)
        val={'images':[{'id':'g','file_name':'sun.png','height':16,'width':16}],
             'annotations':[{'id':1,'image_id':'g','segmentation':rle}]}
        old={'predictions':[{'image_id':'sun','instances':[]}]}
        new={'predictions':[{'image_id':'sun','instances':[{'rle':rle}]}]}
        def load(path):return old if str(path)=='old' else new if str(path)=='new' else val
        with patch('src.utils.common.load_json',side_effect=load):
            result=paired_mask_diagnostics('old','new')
        self.assertEqual(result['summary']['all']['gained'],1)
        self.assertEqual(result['summary']['all']['models']['P0']['mask_recall_iou_gt_05'],1)
        self.assertNotIn('pq',result)

    def test_report_keeps_inherited_cost_and_does_not_auto_promote_truncated_run(self):
        for epochs in [8,12]:
            with self.subTest(epochs=epochs),tempfile.TemporaryDirectory() as temporary:
                base=Path(temporary);folder=base/'native_projection'
                cfg=dict(id='native_projection',kind='native_projection',epochs=12)
                baseline=dict(training_seconds=14400,best_epoch=14)
                original_metrics={'pq':.428}
                reference_path=base/'rf_predictions.json';reference_path.write_text('{}')
                reference=dict(completed=baseline,metrics=original_metrics,predictions_path=str(reference_path),
                               predictions_sha256=sha(reference_path))
                write(base/'experiment_manifest.json',dict(rf1536_reference=reference,deliverable_dir=str(base/'deliverables')))
                write(folder/'completed.json',dict(config=cfg,best_epoch=6,completed_epochs=epochs,
                      partial_epoch_samples=0,optimizer_updates=epochs*234,training_seconds=3600))
                write(folder/'best_metrics.json',dict(pq=.44,dice=.67,iou=.52))
                fake=types.ModuleType('src.routes_v3.diagnostics')
                fake.compare_artifacts=lambda:write(base/'artifact_checks.json',{'passed':True})
                fake.visualize=lambda **kw:(base/'same_scene_comparison.png').write_bytes(b'fixture')
                with patch.dict('sys.modules',{'src.routes_v3.diagnostics':fake}),\
                     patch('scripts.native_projection_support.paired_mask_diagnostics',return_value={}),\
                     contextlib.redirect_stdout(io.StringIO()):
                    self.assertTrue(report(base,cfg,baseline,original_metrics,read,write))
                result=read(base/'comparison.json')
                self.assertEqual(result['native_projection']['total_training_hours'],5)
                self.assertFalse(result['automatic_next_stage'])
                if epochs==8:self.assertIn('truncated',result['decision'])
                else:self.assertIn('candidate',result['decision'])
                self.assertTrue((base/'deliverables/comparison.md').exists())
                self.assertFalse((folder/'last.pt').exists())

    def test_chunked_sampling_preserves_repeated_matches_border_and_tie_coordinates(self):
        from rfdetr.models.criterion import point_sample
        torch.manual_seed(42)
        targets=[{'masks':torch.rand(3,17,19)>.5},{'masks':torch.rand(2,17,19)>.5}]
        indices=[(torch.arange(19),torch.arange(19)%3),(torch.arange(9),torch.arange(9)%2)]
        coords=torch.rand(28,21,2)
        coords[:,0]=0;coords[:,1]=1;coords[:,2]=.5;coords[:,3]=-.1
        masks=torch.cat([t['masks'][ids] for t,(_,ids) in zip(targets,indices)])
        expected=point_sample(masks[:,None].float(),coords,align_corners=False,mode='nearest').squeeze(1)
        torch.testing.assert_close(sample_native_targets(targets,indices,coords,chunk_size=4),expected,rtol=0,atol=0)

    def test_mask_head_uses_native_grid_and_pixel_branch_learns(self):
        torch.manual_seed(42)
        original=Head();head=NativeProjectionHead(original,channels=8)
        image=torch.randn(1,3,32,32)
        context=torch.randn(1,8,3,3)
        head.set_image(image)
        expected=F.interpolate(context,(8,8),mode='bilinear',align_corners=False)
        torch.testing.assert_close(head.features(context),expected) # zero-initialized residual
        optimizer=torch.optim.AdamW(head.parameters(),lr=.01)
        before={n:p.detach().clone() for n,p in head.pixel.named_parameters()}
        for _ in range(3):
            head.set_image(image)
            output=head.sparse_forward(context,[torch.zeros(1,2,8)],(24,24))[0]
            self.assertEqual(output.shape[-2:],(8,8)) # never context /4 =6
            cached=head.cached
            head.sparse_forward(context,[torch.zeros(1,2,8)],(24,24),True)
            self.assertIs(cached,head.cached)
            optimizer.zero_grad();output.square().mean().backward();optimizer.step()
        self.assertTrue(any(not torch.equal(before[n],p) for n,p in head.pixel.named_parameters()))
        self.assertTrue(all(size==(32,32) for _,size,_ in original.received))

    def test_rf_input_downsampling_does_not_replace_native_branch_pixels(self):
        route=NativeProjectionRoute.__new__(NativeProjectionRoute);nn.Module.__init__(route)
        route.cfg=dict(native_size=32,size=24)
        route.register_buffer('mean',torch.zeros(1,3,1,1));route.register_buffer('std',torch.ones(1,3,1,1))
        route.core=nn.Module();route.core.segmentation_head=NativeProjectionHead(Head(),channels=8)
        image=torch.zeros(1,3,32,32);image[:,:,::2,::2]=1
        context=route.inputs(image)
        self.assertEqual(context.shape[-2:],(24,24))
        torch.testing.assert_close(route.core.segmentation_head.image,image,rtol=0,atol=0)
        with self.assertRaisesRegex(ValueError,'unresized'):route.inputs(context)

    def test_native_augmentation_moves_image_and_independent_masks_together(self):
        from src.routes_v3.engine import sample_at,training_dataset
        image=torch.zeros(3,8,8);image[:,1,2]=1
        mask=image[:1].clone()
        class Dataset:
            size=2048;augment_native=True
            def __getitem__(self,_):return dict(image=image.clone(),instances=mask.clone(),semantic=mask.clone())
        with patch('numpy.random.randint',return_value=1),patch('numpy.random.rand',return_value=0.),patch('numpy.random.uniform',return_value=1.):
            result=sample_at(Dataset(),0)
        torch.testing.assert_close(result['instances'],result['image'][:1])
        self.assertFalse(torch.equal(result['image'],image))
        with patch('src.routes_v3.engine.FilamentDataset',return_value=types.SimpleNamespace()) as factory:
            ds=training_dataset(dict(kind='native_projection',size=1536))
        self.assertEqual(factory.call_args.kwargs['size'],2048);self.assertTrue(ds.augment_native)

    def test_inherited_split_and_selected_epoch_cannot_be_mixed(self):
        with tempfile.TemporaryDirectory() as temporary:
            project=Path(temporary);base=project/'output/benchmark/rf_resolution_v1';folder=base/'rf_1536'
            cfg={'id':'rf_1536','size':1536};splits={'train_split_sha256':'a','val_split_sha256':'b'}
            done=dict(config=cfg,status='completed',completed_epochs=30,optimizer_updates=7020,
                      best_epoch=14,best_pq=.428,**splits)
            manifest=dict(source={'commit':'test'},**splits)
            for name,value in [('experiment_manifest.json',manifest),('pipeline_status.json',{'stage':'completed'}),
                ('artifact_checks.json',{'passed':True}),('comparison.json',{'checks_passed':True}),
                ('rf_1536/config.json',cfg),('rf_1536/completed.json',done),
                ('rf_1536/best_metrics.json',{'pq':.428,'training_epochs':14,'debug':False})]:write(base/name,value)
            (folder/'best.pt').write_bytes(b'fixture');(folder/'best_predictions.json').write_text('{}')
            _,_,reference=inspect_rf1536(project,cfg,splits,read,sha)
            self.assertEqual(reference['warmstart_sha256'],sha(folder/'best.pt'))
            write(folder/'best_metrics.json',{'pq':.428,'training_epochs':30})
            with self.assertRaisesRegex(RuntimeError,'epoch'):inspect_rf1536(project,cfg,splits,read,sha)
            write(folder/'best_metrics.json',{'pq':.428,'training_epochs':14})
            write(base/'experiment_manifest.json',dict(manifest,val_split_sha256='changed'))
            with self.assertRaisesRegex(RuntimeError,'split'):inspect_rf1536(project,cfg,splits,read,sha)


if __name__=='__main__':unittest.main()
