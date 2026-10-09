from pathlib import Path
import numpy as np
import cv2
import torch
from torch import nn
from torch.nn import functional as F
from scipy.optimize import linear_sum_assignment


class DINOQueryMasks(nn.Module):
    def __init__(self,cfg):
        super().__init__()
        from transformers import AutoModel
        path=Path(cfg['backbone_weights'])
        if not path.is_dir():raise FileNotFoundError(f'Local DINOv3 weights missing: {path}')
        self.backbone=AutoModel.from_pretrained(str(path),local_files_only=True,attn_implementation='sdpa')
        self.backbone.requires_grad_(False);self.backbone.eval()
        self.special=1+self.backbone.config.num_register_tokens
        self.patch=self.backbone.config.patch_size
        dim=cfg['hidden_dim'];self.dim=dim
        self.projection=nn.Linear(self.backbone.config.hidden_size,dim)
        self.queries=nn.Embedding(cfg['num_queries'],dim)
        nn.init.normal_(self.queries.weight,std=.02)
        layer=nn.TransformerDecoderLayer(dim,4,dim*4,dropout=0.,batch_first=True,norm_first=True)
        self.decoder=nn.TransformerDecoder(layer,cfg['decoder_layers'],norm=nn.LayerNorm(dim))
        self.objectness=nn.Linear(dim,1)
        self.mask_embedding=nn.Sequential(nn.Linear(dim,dim),nn.GELU(),nn.Linear(dim,dim))
        self.pixel_head=nn.Sequential(nn.Conv2d(dim,dim,3,padding=1),nn.GELU(),nn.Conv2d(dim,dim,3,padding=1))
        self.register_buffer('mean',torch.tensor([.485,.456,.406]).view(1,3,1,1))
        self.register_buffer('std',torch.tensor([.229,.224,.225]).view(1,3,1,1))

    def train(self,mode=True):
        super().train(mode);self.backbone.eval();return self

    def positional(self,h,w,device):
        y,x=torch.meshgrid(torch.linspace(0,1,h,device=device),torch.linspace(0,1,w,device=device),indexing='ij')
        frequency=torch.exp(torch.arange(self.dim//4,device=device)*(-np.log(10000)/(self.dim//4)))
        x=x.flatten()[:,None]*2*np.pi/frequency;y=y.flatten()[:,None]*2*np.pi/frequency
        return torch.cat([x.sin(),x.cos(),y.sin(),y.cos()],dim=1)[None]

    def forward(self,images):
        with torch.no_grad():
            tokens=self.backbone(pixel_values=(images-self.mean)/self.std).last_hidden_state[:,self.special:]
        h,w=images.shape[-2]//self.patch,images.shape[-1]//self.patch
        features=self.projection(tokens)
        memory=features+self.positional(h,w,images.device).to(features.dtype)
        queries=self.queries.weight[None].expand(len(images),-1,-1)
        decoded=self.decoder(queries,memory)
        pixels=self.pixel_head(features.transpose(1,2).reshape(len(images),self.dim,h,w))
        pixels=F.interpolate(pixels,size=(256,256),mode='bilinear',align_corners=False)
        masks=torch.einsum('bqd,bdhw->bqhw',self.mask_embedding(decoded),pixels)/np.sqrt(self.dim)
        return {'objectness':self.objectness(decoded)[...,0],'masks':masks}

    def loss(self,images,targets):
        outputs=self(images);loss=images.new_tensor(0.)
        for b,target in enumerate(targets):
            logits=outputs['masks'][b].float();objects=outputs['objectness'][b].float()
            truth=target['instances'].float()
            labels=torch.zeros_like(objects);weights=torch.full_like(objects,.1)
            if len(truth)>len(objects):raise ValueError('More GT instances than queries; increase num_queries')
            if len(truth):
                # Dense supervised masks are area-resized, retaining fractional thin structures.
                coarse_logits=F.adaptive_avg_pool2d(logits,(32,32)).flatten(1)
                coarse_truth=F.adaptive_avg_pool2d(truth,(32,32)).flatten(1)
                p=coarse_logits.sigmoid()
                bce=F.softplus(coarse_logits).mean(1)[:,None]-coarse_logits@coarse_truth.T/coarse_logits.shape[1]
                dice=1-(2*p@coarse_truth.T+1)/(p.sum(1)[:,None]+coarse_truth.sum(1)[None]+1)
                cost=2*bce+5*dice-objects.sigmoid()[:,None]
                source,destination=linear_sum_assignment(cost.detach().cpu().numpy())
                source=torch.as_tensor(source,device=images.device);destination=torch.as_tensor(destination,device=images.device)
                labels[source]=1;weights[source]=1
                matched=logits[source]
                expected=F.interpolate(truth[destination,None],size=matched.shape[-2:],mode='area')[:,0]
                bce_loss=F.binary_cross_entropy_with_logits(matched,expected)
                p=matched.sigmoid()
                dice_loss=1-((2*(p*expected).sum((1,2))+1)/(p.sum((1,2))+expected.sum((1,2))+1)).mean()
                loss=loss+2*bce_loss+5*dice_loss
            loss=loss+(F.binary_cross_entropy_with_logits(objects,labels,reduction='none')*weights).mean()
        return loss/len(targets)

    def predict_instances(self,images,cfg):
        output=self(images);scores=output['objectness'][0].sigmoid().float().cpu().numpy()
        candidates=np.flatnonzero(scores>=cfg['confidence'])
        probabilities=output['masks'][0].sigmoid().float().cpu().numpy()
        instances=[]
        for index in candidates:
            native=cv2.resize(probabilities[index],(2048,2048),interpolation=cv2.INTER_LINEAR)>cfg['mask_threshold']
            if native.sum()>=cfg['min_area']:
                instances.append({'mask':native,'score':float(scores[index])})
        self.raw_instance_count=len(instances)
        return instances


def build_model(cfg,pretrained=True):
    return DINOQueryMasks(cfg)
