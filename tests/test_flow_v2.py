import tempfile
import unittest
from pathlib import Path
import numpy as np
import torch
from src.routes_v3.flow_v2 import transform_patch, crop_origin, flow_foreground_loss, make_labels


class FlowV2Tests(unittest.TestCase):
    def test_vector_rotation_and_flip(self):
        image=torch.zeros(3,4,4);flow=np.zeros((2,4,4),np.float32);flow[1]=1
        foreground=np.zeros((4,4),bool);foreground[1,2]=True
        valid=np.ones((4,4),bool)
        _,rotated,fg,_=transform_patch(image,flow,foreground,valid,1,False)
        self.assertTrue(np.all(rotated[0]==-1));self.assertTrue(np.all(rotated[1]==0))
        np.testing.assert_array_equal(fg,np.rot90(foreground))
        _,flipped,_,_=transform_patch(image,flow,foreground,valid,0,True)
        self.assertTrue(np.all(flipped[1]==-1))

    def test_crop_keeps_small_complete_instance(self):
        mask=np.zeros((64,64),bool);mask[3:9,50:57]=True
        for _ in range(40):
            y,x=crop_origin(mask,24,np.random.RandomState(_))
            self.assertEqual(mask[y:y+24,x:x+24].sum(),mask.sum())

    def test_background_cannot_dilute_foreground_flow_error(self):
        def loss(size):
            prediction=torch.zeros(3,size,size,requires_grad=True)
            target=torch.zeros(2,size,size);target[:,1,1]=5
            foreground=torch.zeros(size,size);foreground[1,1]=1
            keep=torch.ones(size,size,dtype=torch.bool)
            value=flow_foreground_loss(prediction,target,foreground,keep)
            value.backward()
            self.assertTrue(torch.isfinite(prediction.grad).all())
            return prediction.grad[:2,1,1]
        torch.testing.assert_close(loss(8),loss(32))

    def test_empty_crop_and_overlap_remain_finite(self):
        prediction=torch.zeros(3,8,8,requires_grad=True)
        value=flow_foreground_loss(prediction,torch.zeros(2,8,8),torch.zeros(8,8),torch.ones(8,8,dtype=torch.bool))
        value.backward();self.assertTrue(torch.isfinite(value));self.assertTrue(torch.isfinite(prediction.grad).all())
        ignored=flow_foreground_loss(prediction,torch.zeros(2,8,8),torch.ones(8,8),torch.zeros(8,8,dtype=torch.bool))
        self.assertEqual(float(ignored.detach()),0.)
        masks=np.zeros((3,8,8),bool);masks[1,2:4,2:4]=True;masks[2,3:5,3:5]=True
        labels=make_labels(masks)
        self.assertEqual(set(np.unique(labels)),{0,1,2})
        self.assertEqual(labels[3,3],1)


if __name__=='__main__':unittest.main()
