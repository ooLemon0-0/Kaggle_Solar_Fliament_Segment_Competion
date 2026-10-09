import os
import math
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from src.utils.common import ROOT
from src.data.dataset import encode_mask, decode_mask

ASSETS=ROOT/'output/pretrained/routes_v3'
os.environ['RF_HOME']=str(ASSETS/'rf')
os.environ['CELLPOSE_LOCAL_MODELS_PATH']=str(ASSETS/'cellpose')

def dice_loss(logits, target):
    p=logits.sigmoid();return 1-(2*(p*target).sum()+1)/(p.sum()+target.sum()+1)

def conv(cin,cout,stride=1):
    return nn.Sequential(nn.Conv2d(cin,cout,3,stride,padding=1),nn.GroupNorm(4,cout),nn.GELU())

class ImageProjector(nn.Module):
    def __init__(self):
        super().__init__();self.net=nn.Sequential(conv(3,24),conv(24,24),nn.Conv2d(24,3,1))
        nn.init.zeros_(self.net[-1].weight);nn.init.zeros_(self.net[-1].bias)
    def forward(self,x):return x+.3*torch.tanh(self.net(x))

class DetailHead(nn.Module):
    def __init__(self,original):
        super().__init__();self.original=original
        self.detail=nn.Sequential(conv(3,24,2),conv(24,32,2),conv(32,32))
        self.context=nn.Conv2d(256,32,1)
        self.mix=conv(32,32)
        self.fuse=nn.Conv2d(32,256,1);self.foreground=nn.Conv2d(32,1,1)
        nn.init.zeros_(self.fuse.weight);nn.init.zeros_(self.fuse.bias)
        self.image=None;self.z=None;self.fg=None
    def features(self,spatial,image_size,skip):
        detail=self.detail(self.image)
        spatial=F.interpolate(spatial,detail.shape[-2:],mode='bilinear',align_corners=False)
        z=self.mix(detail+self.context(spatial))
        if not skip:self.z=z;self.fg=self.foreground(z)
        spatial=F.interpolate(spatial,z.shape[-2:],mode='bilinear',align_corners=False)
        return spatial+self.fuse(z)
    def sparse_forward(self,spatial,query,image_size,skip_blocks=False):
        return self.original.sparse_forward(self.features(spatial,image_size,skip_blocks),query,image_size,skip_blocks)
    def forward(self,spatial,query,image_size,skip_blocks=False):
        return self.original(self.features(spatial,image_size,skip_blocks),query,image_size,skip_blocks)

class RFRoute(nn.Module):
    def __init__(self,cfg):
        super().__init__()
        from rfdetr import RFDETRSegSmall
        from rfdetr.config import TrainConfig
        from rfdetr.models.lwdetr import build_criterion_from_config
        wrapper=RFDETRSegSmall(device='cuda',resolution=cfg['size'],num_classes=1,gradient_checkpointing=True)
        self.core=wrapper.model.model
        wrapper.model_config.num_classes=1
        if self.core.class_embed.out_features!=2:self.core.reinitialize_detection_head(2)
        self.cfg=cfg;self.projector=ImageProjector() if cfg['kind']=='image_projection' else nn.Identity()
        if cfg['kind']!='rf':
            warm=torch.load(cfg.get('warmstart_path',ROOT/'output/benchmark/routes_v3/rf/best.pt'),map_location='cpu',weights_only=False)
            self.core.load_state_dict({k.removeprefix('core.'):v for k,v in warm['model'].items() if k.startswith('core.')})
            self.core.requires_grad_(False)
        if cfg['kind'].startswith('feature'):
            self.core.segmentation_head=DetailHead(self.core.segmentation_head)
            self.core.segmentation_head.requires_grad_(True)
        tc=TrainConfig(dataset_dir=str(ROOT),output_dir=str(ROOT/'work'),batch_size=1,epochs=cfg['epochs'],use_ema=False)
        self.criterion,_=build_criterion_from_config(wrapper.model_config,tc)
        self.register_buffer('mean',torch.tensor([.485,.456,.406])[None,:,None,None])
        self.register_buffer('std',torch.tensor([.229,.224,.225])[None,:,None,None])
    def training_mode(self):
        self.train()
        if self.cfg['kind']!='rf':
            self.core.backbone.eval()
            if self.cfg['kind']=='image_projection':
                # Sparse mask losses require core training mode; backbone checkpoint still propagates input gradients.
                for name,m in self.core.named_modules():
                    if isinstance(m,nn.Dropout):m.eval()
    def inputs(self,x):
        x=(x-self.mean)/self.std;x=self.projector(x)
        if isinstance(self.core.segmentation_head,DetailHead):self.core.segmentation_head.image=x
        return x
    def target(self,masks):
        from torchvision.ops import masks_to_boxes
        masks=masks.cuda();n=len(masks)
        if n:
            boxes=masks_to_boxes(masks.bool());xy=(boxes[:,:2]+boxes[:,2:])/2
            wh=(boxes[:,2:]-boxes[:,:2]).clamp_min(1)
            boxes=torch.cat((xy,wh),1)/masks.shape[-1]
        else:boxes=torch.empty(0,4,device='cuda')
        return {'labels':torch.zeros(n,dtype=torch.int64,device='cuda'),'boxes':boxes,'masks':masks}
    def loss(self,sample):
        x=sample['image'].cuda()[None];target=self.target(sample['instances'])
        out=self.core(self.inputs(x)); losses=self.criterion(out,[target])
        loss=sum(v*self.criterion.weight_dict[k] for k,v in losses.items() if k in self.criterion.weight_dict)
        if self.cfg['kind']=='feature_structure':
            head=self.core.segmentation_head
            masks=F.interpolate(target['masks'][None],head.z.shape[-2:],mode='nearest')[0]
            union=masks.amax(0,keepdim=True)[None] if len(masks) else torch.zeros_like(head.fg)
            auxiliary=F.binary_cross_entropy_with_logits(head.fg,union)+dice_loss(head.fg,union)
            unique=masks.sum(0)==1;means=[];pull=head.z.sum()*0
            for mask in masks:
                keep=(mask>0)&unique
                if keep.sum()>2:
                    pixels=head.z[0,:,keep].T.float();center=pixels.mean(0)
                    pull=pull+(pixels-center).square().mean();means.append(center)
            if means:pull=pull/len(means)
            push=head.z.sum()*0
            if len(means)>1:
                distances=torch.pdist(torch.stack(means))/math.sqrt(32)
                push=F.relu(1-distances).square().mean()
            loss=loss+.5*auxiliary+.1*(pull+push)
        return loss
    @torch.no_grad()
    def predict(self,image):
        self.eval();x=F.interpolate(image[None].cuda(),(self.cfg['size'],)*2,mode='bilinear',align_corners=False)
        out=self.core(self.inputs(x));scores=out['pred_logits'][0,:,0].sigmoid()
        selected=(scores>=self.cfg.get('debug_prediction_confidence',.25)).nonzero().flatten();selected=selected[scores[selected].argsort(descending=True)][:64]
        masks=out['pred_masks'][0];boxes=out['pred_boxes'][0]
        instances=[]
        for q in selected:
            mask=F.interpolate(masks[q][None,None].float(),(2048,2048),mode='bilinear',align_corners=False)[0,0]>0
            if mask.any():
                box=boxes[q].float().cpu().numpy();xy=box[:2];wh=box[2:]
                instances.append({'rle':encode_mask(mask.cpu().numpy()),'score':float(scores[q]),
                                  'box':np.clip(np.r_[xy-wh/2,xy+wh/2]*2048,0,2048).tolist()})
        return instances

def roi_bounds(mask):
    ys,xs=np.where(mask);x0,x1=xs.min(),xs.max()+1;y0,y1=ys.min(),ys.max()+1
    return np.array([x0,y0,x1,y1],dtype=np.float32)

def expanded(box):
    x0,y0,x1,y1=box;side=max(x1-x0,y1-y0,64)*1.5
    cx=(x0+x1)/2;cy=(y0+y1)/2
    return np.array([max(0,int(cx-side/2)),max(0,int(cy-side/2)),min(2048,int(cx+side/2)),min(2048,int(cy+side/2))])

class SAMRoute(nn.Module):
    def __init__(self,cfg):
        super().__init__()
        from transformers import Sam2Model
        self.sam=Sam2Model.from_pretrained(ASSETS/'sam2_tiny');self.sam.vision_encoder.requires_grad_(False)
        # Adapt the last visual stage and neck to H-alpha imagery while preserving the low-level pretrained encoder.
        for block in self.sam.vision_encoder.backbone.blocks[-2:]:block.requires_grad_(True)
        self.sam.vision_encoder.neck.requires_grad_(True)
        self.refine=nn.Sequential(conv(4,24),conv(24,24),nn.Conv2d(24,1,1))
        nn.init.zeros_(self.refine[-1].weight);nn.init.zeros_(self.refine[-1].bias)
        self.register_buffer('mean',torch.tensor([.485,.456,.406])[None,:,None,None]);self.register_buffer('std',torch.tensor([.229,.224,.225])[None,:,None,None])
        self.cfg=cfg
    def training_mode(self):self.train();self.sam.vision_encoder.eval()
    def forward_roi(self,image,box):
        # Stretch each square clipped ROI consistently in training and inference; map output back to exact native bounds.
        rgb=F.interpolate(image[None],(1024,1024),mode='bilinear',align_corners=False)
        scaled=torch.tensor(box,device='cuda',dtype=torch.float32)
        scaled=scaled*torch.tensor([1024/image.shape[2],1024/image.shape[1]]*2,device='cuda')
        out=self.sam(pixel_values=(rgb-self.mean)/self.std,input_boxes=scaled[None,None],multimask_output=False)
        logits=out.pred_masks.reshape(1,1,*out.pred_masks.shape[-2:]).float()
        logits=F.interpolate(logits,(512,512),mode='bilinear',align_corners=False)
        rgbsmall=F.interpolate(rgb,logits.shape[-2:],mode='bilinear',align_corners=False)
        logits=logits+self.refine(torch.cat([rgbsmall,logits],1))
        return logits,out.iou_scores.reshape(-1)[0]
    def loss(self,sample):
        masks=sample['instances'].numpy().astype(bool)
        if np.random.rand()<.1:
            for _ in range(8):
                y0,x0=np.random.randint(0,2048-192,2)
                if not masks[:,y0:y0+192,x0:x0+192].any():
                    logits,quality=self.forward_roi(sample['image'][:,y0:y0+192,x0:x0+192].cuda(),[24,24,168,168])
                    return F.binary_cross_entropy_with_logits(logits,torch.zeros_like(logits))+dice_loss(logits,torch.zeros_like(logits))+.1*quality.square()
        if len(masks)==0:raise ValueError('SAM training requires an annotated instance')
        index=np.random.randint(len(masks));mask=masks[index]
        gtbox=roi_bounds(mask);bounds=expanded(gtbox);x0,y0,x1,y1=bounds
        prior=gtbox.copy();wh=prior[2:]-prior[:2]
        prior+=np.random.uniform(-.15,.15,4)*np.r_[wh,wh];prior[2:]=np.maximum(prior[2:],prior[:2]+2)
        prior-=np.array([x0,y0,x0,y0]);prior=np.clip(prior,0,np.array([x1-x0,y1-y0]*2))
        image=sample['image'][:,y0:y1,x0:x1].cuda();logits,quality=self.forward_roi(image,prior)
        target=torch.tensor(mask[y0:y1,x0:x1],device='cuda',dtype=torch.float32)[None,None]
        target=F.interpolate(target,logits.shape[-2:],mode='nearest')
        binary=logits.detach()>0;inter=(binary*target).sum();iou=inter/((binary+target).bool().sum()+1e-6)
        return F.binary_cross_entropy_with_logits(logits,target)+dice_loss(logits,target)+.1*(quality-iou).square()
    @torch.no_grad()
    def predict(self,image,proposals):
        self.eval();instances=[]
        for inst in proposals:
            box=np.array(inst['box'] if 'box' in inst else roi_bounds(decode_mask(inst['rle'])),dtype=np.float32)
            bounds=expanded(box);x0,y0,x1,y1=bounds
            if x1<=x0 or y1<=y0:continue
            local=box-np.array([x0,y0,x0,y0]);logits,quality=self.forward_roi(image[:,y0:y1,x0:x1].cuda(),local)
            roi=F.interpolate(logits,(y1-y0,x1-x0),mode='bilinear',align_corners=False)[0,0]>0
            mask=np.zeros((2048,2048),dtype=bool);mask[y0:y1,x0:x1]=roi.cpu().numpy()
            if mask.any():instances.append({'rle':encode_mask(mask),'score':float(inst['score'])})
        return instances

class FlowRoute(nn.Module):
    def __init__(self,cfg):
        super().__init__()
        from cellpose.models import CellposeModel
        self.cp=CellposeModel(gpu=True,pretrained_model='cpdino-vitb',device=torch.device('cuda'),use_bfloat16=False)
        self.net=self.cp.net;self.cfg=cfg
    def training_mode(self):self.train()
    def loss(self,sample):
        from cellpose.dynamics import masks_to_flows_gpu
        masks=sample['instances'].numpy().astype(bool);side=self.cfg['size']
        if np.random.rand()<.8 and masks.any():
            positions=np.argwhere(masks.any(0));y,x=positions[np.random.randint(len(positions))]
        else:y,x=np.random.randint(0,2048,2)
        x0=int(np.clip(x-side//2,0,2048-side));y0=int(np.clip(y-side//2,0,2048-side))
        layers=masks[:,y0:y0+side,x0:x0+side];image=sample['image'][:,y0:y0+side,x0:x0+side]
        turns=np.random.randint(4);layers=np.rot90(layers,turns,axes=(1,2)).copy();image=torch.rot90(image,turns,(1,2))
        labels=np.zeros((side,side),dtype=np.int32)
        for i,m in enumerate(layers,1):labels[(labels==0)&m]=i
        import fastremap
        labels,_=fastremap.renumber(labels,in_place=True)
        flow,_=masks_to_flows_gpu(labels,device=torch.device('cpu')) if labels.any() else (np.zeros((2,side,side)),None)
        target=torch.as_tensor(flow,device='cuda',dtype=torch.float32)*5
        union=torch.as_tensor(layers.any(0),device='cuda',dtype=torch.float32)
        ignore=torch.as_tensor(layers.sum(0)>1,device='cuda')
        y,_=self.net(image[None].cuda())
        flow_error=(y[0,:2].float()-target).square().mean(0)
        return .5*flow_error[~ignore].mean()+F.binary_cross_entropy_with_logits(y[0,2].float(),union)
    @torch.no_grad()
    def predict(self,image):
        self.eval();x=image.permute(1,2,0).numpy()
        labels,flows,_=self.cp.eval(x,batch_size=1,normalize=False,resample=True,diameter=None,bsize=self.cfg['size'],
                                  tile_overlap=.25,flow_threshold=.4,cellprob_threshold=0.,min_size=16,max_size_fraction=.4)
        prob=1/(1+np.exp(-np.clip(flows[2],-30,30)));instances=[]
        for i in np.unique(labels):
            if i==0:continue
            mask=labels==i;score=float(prob[mask].mean())
            if score>=.25:instances.append({'rle':encode_mask(mask),'score':score})
        return sorted(instances,key=lambda i:i['score'],reverse=True)[:64]

def build(cfg):
    if cfg['kind']=='sam':return SAMRoute(cfg).cuda()
    if cfg['kind']=='flow':return FlowRoute(cfg).cuda()
    return RFRoute(cfg).cuda()
