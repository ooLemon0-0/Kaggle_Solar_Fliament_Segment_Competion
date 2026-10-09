import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import torch
from torch import nn
from src.routes_v3.models import RFRoute,ImageProjector

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('projection_repair',ROOT/'scripts/run_projection_repair.py')
# _bootstrap is only needed by the script entry point.
import sys
sys.path.insert(0,str(ROOT/'scripts'))
repair=importlib.util.module_from_spec(spec);spec.loader.exec_module(repair)


class ProjectionRepairTests(unittest.TestCase):
    def route(self,kind):
        route=RFRoute.__new__(RFRoute);nn.Module.__init__(route)
        route.cfg={'kind':kind}
        route.projector=ImageProjector() if kind=='image_projection' else nn.Identity()
        route.core=nn.Module();route.core.segmentation_head=nn.Identity()
        route.register_buffer('mean',torch.tensor([.485,.456,.406])[None,:,None,None])
        route.register_buffer('std',torch.tensor([.229,.224,.225])[None,:,None,None])
        return route

    def test_projection_graph_through_frozen_consumer_and_pack_equivalence(self):
        from rfdetr.utilities.tensors import NestedTensor
        torch.manual_seed(42)
        route=self.route('image_projection');image=torch.rand(1,3,32,32)
        expected=(image-route.mean)/route.std
        packed=route.inputs(image)
        self.assertIsInstance(packed,NestedTensor)
        torch.testing.assert_close(packed.tensors,expected)
        self.assertFalse(packed.mask.any());self.assertEqual(tuple(packed.mask.shape),(1,32,32))
        consumer=nn.Conv2d(3,1,3,padding=1);consumer.requires_grad_(False)
        loss=consumer(packed.tensors).square().mean();loss.backward()
        self.assertTrue(all(p.grad is None for p in consumer.parameters()))
        gradient=route.projector.net[-1].weight.grad
        self.assertIsNotNone(gradient);self.assertTrue(torch.isfinite(gradient).all());self.assertGreater(gradient.abs().sum().item(),0)
        before=packed.tensors.detach().clone()
        optimizer=torch.optim.SGD(route.projector.parameters(),lr=.01);optimizer.step()
        self.assertFalse(torch.equal(before,route.inputs(image).tensors))

    def test_projection_inference_and_other_routes_unchanged(self):
        with torch.no_grad():
            packed=self.route('image_projection').inputs(torch.rand(2,3,16,16))
        self.assertFalse(packed.tensors.requires_grad)
        for kind in ['rf','feature_control','feature_structure']:
            self.assertIsInstance(self.route(kind).inputs(torch.rand(1,3,16,16)),torch.Tensor)

    def test_backup_preserves_original_checkpoint_and_failed_logs(self):
        with tempfile.TemporaryDirectory() as temporary:
            base=Path(temporary);folder=base/'image_projection';folder.mkdir()
            for name,data in [('last.pt',b'checkpoint'),('train_console.log',b'failed trace')]:
                (folder/name).write_bytes(data)
            with patch.object(repair.shutil,'disk_usage') as usage:
                usage.return_value.free=4*2**30
                archive=repair.backup_attempt(base)
            for name in ['last.pt','train_console.log']:
                self.assertEqual((folder/name).read_bytes(),(archive/'attempt'/name).read_bytes())


if __name__=='__main__':unittest.main()
