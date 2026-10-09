import cv2
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


class SemanticUNet(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        import segmentation_models_pytorch as smp
        self.network=smp.Unet(encoder_name='resnet50',encoder_weights=None,in_channels=3,classes=1)
        if pretrained:
            from torchvision.models import ResNet50_Weights
            weights=ResNet50_Weights.IMAGENET1K_V1.get_state_dict(progress=True,check_hash=True)
            self.network.encoder.load_state_dict(weights)
        self.register_buffer('mean',torch.tensor([.485,.456,.406]).view(1,3,1,1))
        self.register_buffer('std',torch.tensor([.229,.224,.225]).view(1,3,1,1))

    def forward(self, images):
        return self.network((images-self.mean)/self.std)

    def loss(self, images, targets):
        logits=self(images)
        truth=torch.stack([t['semantic'] for t in targets])
        bce=F.binary_cross_entropy_with_logits(logits,truth)
        p=logits.sigmoid().float();truth=truth.float()
        dice=(2*(p*truth).sum((1,2,3))+1)/(p.sum((1,2,3))+truth.sum((1,2,3))+1)
        return bce+1-dice.mean()

    def predict_instances(self, images, cfg):
        probability=self(images).sigmoid()[0,0].float().cpu().numpy()
        native=cv2.resize(probability,(2048,2048),interpolation=cv2.INTER_LINEAR)
        binary=(native>cfg['mask_threshold']).astype(np.uint8)
        if cfg.get('morphology_kernel',0):
            k=cfg['morphology_kernel']
            binary=cv2.morphologyEx(binary,cv2.MORPH_CLOSE,np.ones((k,k),np.uint8))
        count,labels,stats,_=cv2.connectedComponentsWithStats(binary,connectivity=8)
        sums=np.bincount(labels.ravel(),weights=native.ravel(),minlength=count)
        candidates=[(i,float(sums[i]/stats[i,cv2.CC_STAT_AREA])) for i in range(1,count) if stats[i,cv2.CC_STAT_AREA]>=cfg['min_area']]
        self.raw_instance_count=len(candidates)
        candidates=sorted(candidates,key=lambda pair:pair[1],reverse=True)[:cfg['max_instances']]
        return [{'mask':labels==i,'score':score} for i,score in candidates]


def build_model(cfg, pretrained=True):
    return SemanticUNet(pretrained)
