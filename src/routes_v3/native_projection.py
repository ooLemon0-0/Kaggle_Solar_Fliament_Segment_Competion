"""Native pixels supply mask features; RF supplies frozen global instance queries."""
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from src.routes_v3.models import RFRoute, conv
from src.data.dataset import encode_mask
import numpy as np


def sample_native_targets(targets, indices, point_coords, chunk_size=8):
    """Exact RF nearest sampling, with bounded duplicate native-mask allocation."""
    from rfdetr.models.criterion import point_sample
    sampled=[];offset=0
    for target,(_,matched) in zip(targets,indices):
        for start in range(0,len(matched),chunk_size):
            ids=matched[start:start+chunk_size]
            masks=target['masks'][ids].unsqueeze(1).float()
            coords=point_coords[offset+start:offset+start+len(ids)]
            sampled.append(point_sample(masks,coords,align_corners=False,mode='nearest').squeeze(1))
        offset+=len(matched)
    if offset!=len(point_coords):raise ValueError('Target sampling order/count mismatch')
    return torch.cat(sampled,0) if sampled else point_coords.new_empty((0,point_coords.shape[1]))


class NativeProjectionHead(nn.Module):
    def __init__(self, original, channels=256):
        super().__init__()
        self.original = original
        self.pixel = nn.Sequential(conv(3,24,2), conv(24,32,2), conv(32,32))
        self.context = nn.Conv2d(channels,32,1)
        self.mix = conv(32,32)
        self.fuse = nn.Conv2d(32,channels,1)
        nn.init.zeros_(self.fuse.weight)
        nn.init.zeros_(self.fuse.bias)
        self.image = None
        self.cached = None
        self.last_grid = None

    def set_image(self, image):
        self.image = image
        self.cached = None

    def project(self, spatial, image):
        pixels = self.pixel(image)
        context = F.interpolate(spatial, pixels.shape[-2:], mode='bilinear', align_corners=False)
        return context + self.fuse(self.mix(pixels + self.context(context)))

    def features(self, spatial):
        if self.image is None:
            raise RuntimeError('Set native pixels before RF forward')
        if self.cached is None:
            self.cached = (checkpoint(self.project, spatial, self.image, use_reentrant=False)
                           if self.training and torch.is_grad_enabled()
                           else self.project(spatial, self.image))
            self.last_grid = list(self.cached.shape[-2:])
        return self.cached

    def sparse_forward(self, spatial, query, image_size, skip_blocks=False):
        # The RF input is 1536, but the mask head must retain the native /4 grid.
        return self.original.sparse_forward(self.features(spatial), query,
                                            tuple(self.image.shape[-2:]), skip_blocks)

    def forward(self, spatial, query, image_size, skip_blocks=False):
        return self.original(self.features(spatial), query,
                             tuple(self.image.shape[-2:]), skip_blocks)


class NativeProjectionRoute(RFRoute):
    def __init__(self, cfg):
        # Load the unmodified trained RF head before wrapping it.
        super().__init__(dict(cfg, kind='mask_control'))
        self.cfg = cfg
        self.core.segmentation_head = NativeProjectionHead(self.core.segmentation_head)
        self.core.segmentation_head.requires_grad_(True)
        # Process-local training optimization, not an edit to the installed package/PQ.
        import rfdetr.models.criterion as criterion_module
        criterion_module._sample_target_masks_at_points=sample_native_targets

    def target(self, masks):
        return super().target(masks.bool())

    def training_mode(self):
        super().training_mode()
        # Keep frozen localization deterministic; core train mode retains sparse masks/groups.
        for module in self.core.modules():
            if isinstance(module, nn.Dropout):
                module.eval()

    def inputs(self, image):
        native = self.cfg.get('native_size',2048)
        if image.shape[-2:] != (native,native):
            raise ValueError('Native projection requires unresized native pixels and masks')
        normalized = (image-self.mean)/self.std
        self.core.segmentation_head.set_image(normalized)
        return F.interpolate(normalized, (self.cfg['size'],)*2, mode='bilinear', align_corners=False)

    @torch.no_grad()
    def predict(self, image):
        self.eval()
        out = self.core(self.inputs(image.cuda()[None]))
        scores = out['pred_logits'][0,:,0].sigmoid()
        selected = (scores >= self.cfg.get('debug_prediction_confidence',.25)).nonzero().flatten()
        selected = selected[scores[selected].argsort(descending=True)][:64]
        masks, boxes = out['pred_masks'][0], out['pred_boxes'][0]
        instances = []
        for q in selected:
            mask = F.interpolate(masks[q][None,None].float(), (2048,2048),
                                 mode='bilinear',align_corners=False)[0,0]>0
            if mask.any():
                box = boxes[q].float().cpu().numpy()
                instances.append(dict(rle=encode_mask(mask.cpu().numpy()),score=float(scores[q]),
                                      box=np.clip(np.r_[box[:2]-box[2:]/2,box[:2]+box[2:]/2]*2048,0,2048).tolist()))
        return instances
