import copy
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

import torch
from torch import nn
from src.routes_v3.models import RFRoute, DetailHead

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('m0_runner',ROOT/'scripts/run_rf_resolution.py')
runner=importlib.util.module_from_spec(spec);spec.loader.exec_module(runner)


class TinyCore(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone=nn.Linear(4,4)
        self.decoder=nn.Linear(4,4)
        self.class_embed=nn.Linear(4,2)
        self.segmentation_head=nn.Linear(4,1)
    def forward(self,x):
        context=self.decoder(self.backbone(x)).relu()
        return self.segmentation_head(context)


class MaskControlTests(unittest.TestCase):
    def test_only_head_learns_and_original_weights_are_loaded(self):
        torch.manual_seed(42);original=TinyCore()
        def wrapper(**_):
            return types.SimpleNamespace(model=types.SimpleNamespace(model=copy.deepcopy(original)),
                                         model_config=types.SimpleNamespace(num_classes=1))
        rfdetr=types.ModuleType('rfdetr');rfdetr.RFDETRSegSmall=wrapper
        config=types.ModuleType('rfdetr.config');config.TrainConfig=lambda **kw:kw
        lwdetr=types.ModuleType('rfdetr.models.lwdetr')
        lwdetr.build_criterion_from_config=lambda *_:(nn.Identity(),None)
        with tempfile.TemporaryDirectory() as temporary:
            path=Path(temporary)/'original_rf_best.pt'
            torch.save({'model':{'core.'+name:value for name,value in original.state_dict().items()}},path)
            cfg=dict(kind='mask_control',size=1056,epochs=30,warmstart_path=str(path))
            with patch.dict(sys.modules,{'rfdetr':rfdetr,'rfdetr.config':config,'rfdetr.models.lwdetr':lwdetr}):
                route=RFRoute(cfg)
            self.assertIsInstance(route.projector,nn.Identity)
            self.assertNotIsInstance(route.core.segmentation_head,DetailHead)
            trainable=[name for name,p in route.named_parameters() if p.requires_grad]
            self.assertEqual(set(trainable),{'core.segmentation_head.weight','core.segmentation_head.bias'})
            sample=torch.randn(5,4)
            route.eval();original.eval()
            torch.testing.assert_close(route.core(sample),original(sample))
            before={name:p.detach().clone() for name,p in route.named_parameters()}
            route.training_mode()
            self.assertFalse(route.core.backbone.training)
            optimizer=torch.optim.AdamW([p for p in route.parameters() if p.requires_grad],lr=.01)
            loss=route.core(sample).square().mean();loss.backward()
            self.assertTrue(all(p.grad is None for p in route.parameters() if not p.requires_grad))
            self.assertGreater(sum(p.grad.abs().sum().item() for p in route.parameters() if p.grad is not None),0)
            optimizer.step()
            for name,p in route.named_parameters():
                if not p.requires_grad:torch.testing.assert_close(p,before[name],rtol=0,atol=0)
            self.assertTrue(any(not torch.equal(p,before[name]) for name,p in route.named_parameters() if p.requires_grad))

    def test_same_budget_and_size_as_detail_control(self):
        integrity=json.loads((ROOT/'linux/integrity.json').read_text())
        rf=next(c for c in integrity['runs'] if c['id']=='rf')
        detail=next(c for c in integrity['runs'] if c['id']=='feature_control')
        actual=runner.configuration(rf.copy(),rf,'mask_control')
        self.assertEqual({k for k in detail if detail[k]!=actual[k]},{'id','kind'})
        self.assertEqual(actual['dependency'],'rf')
        self.assertEqual(actual['size'],1056)

    def test_report_compares_detail_and_preserves_inherited_cost(self):
        for epochs,updates,expected in [(30,7020,'detail_branch_candidate'),(22,5148,'training_amount_not_matched')]:
            with self.subTest(epochs=epochs),tempfile.TemporaryDirectory() as temporary:
                base=Path(temporary);active=base/'original';folder=base/'mask_control';folder.mkdir()
                cfg={'id':'mask_control','kind':'mask_control','size':1056}
                done=dict(config=cfg,best_epoch=18,completed_epochs=epochs,
                          optimizer_updates=updates,training_seconds=7000.)
                baseline=dict(done,best_epoch=20,completed_epochs=30,optimizer_updates=7020,training_seconds=9270.)
                (folder/'completed.json').write_text(json.dumps(done))
                (folder/'best_metrics.json').write_text(json.dumps(dict(pq=.415,dice=.65,iou=.50)))
                detail=active/'output/benchmark/routes_v3/feature_control';detail.mkdir(parents=True)
                (detail/'completed.json').write_text(json.dumps(dict(completed_epochs=30,optimizer_updates=7020,training_seconds=8107.2)))
                (base/'experiment_manifest.json').write_text(json.dumps(dict(active_project=str(active),detail_reference={'pq':.422718})))
                fake=types.ModuleType('src.routes_v3.diagnostics')
                fake.compare_artifacts=lambda:(base/'artifact_checks.json').write_text('{"passed":true}')
                with patch.object(runner,'BASE',base),patch.dict(sys.modules,{'src.routes_v3.diagnostics':fake}):
                    self.assertTrue(runner.report_completed(cfg,baseline,{'pq':.411695}))
                report=json.loads((base/'comparison.json').read_text())
                self.assertIn(expected,report['decision'])
                self.assertAlmostEqual(report['delta_detail_minus_mask_control'],.007718)
                self.assertAlmostEqual(report['inherited_rf_training_hours'],2.575)
                self.assertFalse(report['automatic_next_stage'])
                self.assertIn('mask_control',report);self.assertNotIn('rf_1536',report)

    def test_new_runner_is_detected_as_real_job(self):
        class Process:
            pid=551
            info={'pid':551,'cmdline':['python','/other/scripts/run_mask_control.py'],'create_time':1.}
        with patch.object(runner.psutil,'process_iter',return_value=[Process()]):
            self.assertEqual(runner.running_jobs()[0]['pid'],551)


if __name__=='__main__':unittest.main()
