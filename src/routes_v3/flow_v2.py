"""Native-context CellposeDINO with globally consistent targets; old Flow stays intact."""
import os
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint
from src.routes_v3.models import FlowRoute, dice_loss
from src.utils.common import ROOT
from src.data.dataset import encode_mask


def make_labels(masks):
    labels = np.zeros(masks.shape[-2:], dtype=np.int32)
    for index, mask in enumerate(masks, 1):
        labels[(labels == 0) & mask] = index
    # Dense IDs are required by Cellpose's diffusion implementation.
    import fastremap
    return fastremap.renumber(labels, in_place=True)[0]


def transform_patch(image, flow, foreground, valid, turns, flip):
    image = torch.rot90(image, turns, (-2, -1))
    flow = np.rot90(flow, turns, axes=(-2, -1)).copy()
    foreground = np.rot90(foreground, turns).copy()
    valid = np.rot90(valid, turns).copy()
    # Cellpose channels are dy, dx; spatial rotation also rotates the vectors.
    for _ in range(turns):
        flow = np.stack((-flow[1], flow[0]))
    if flip:
        image = image.flip(-1)
        flow = flow[..., ::-1].copy(); flow[1] *= -1
        foreground = foreground[:, ::-1].copy(); valid = valid[:, ::-1].copy()
    return image, flow, foreground, valid


def crop_origin(mask, side, rng=np.random):
    height, width = mask.shape
    ys, xs = np.where(mask)
    if len(ys):
        # Prefer a full selected instance in view, rather than a random foreground pixel.
        def origin(lo, hi, extent):
            left, right = max(0, int(hi) + 1 - side), min(int(lo), extent - side)
            if left <= right:
                return int(rng.randint(left, right + 1))
            return int(np.clip((int(lo) + int(hi)) // 2 - side // 2, 0, extent - side))
        return origin(ys.min(), ys.max(), height), origin(xs.min(), xs.max(), width)
    return int(rng.randint(height - side + 1)), int(rng.randint(width - side + 1))


def flow_foreground_loss(prediction, target, foreground, valid):
    prediction = prediction.float()
    if not valid.any():return prediction.sum() * 0
    error = (prediction[:2] - target).square().mean(0)
    positive = (foreground > .5) & valid
    negative = (foreground <= .5) & valid
    fg_flow = error[positive].mean() if positive.any() else error.sum() * 0
    bg_flow = error[negative].mean() if negative.any() else error.sum() * 0
    logits = prediction[2]
    weight = (negative.sum() / positive.sum().clamp_min(1)).float().clamp(1, 20)
    bce = F.binary_cross_entropy_with_logits(logits[valid], foreground[valid], pos_weight=weight)
    dice = dice_loss(logits[valid], foreground[valid]) if positive.any() else logits.sum() * 0
    return .5 * fg_flow + .05 * bg_flow + bce + dice


def decode_fields(dP, cellprob, niter, flow_threshold, device):
    from cellpose.dynamics import resize_and_compute_masks
    labels = resize_and_compute_masks(dP, cellprob, niter=niter,
        cellprob_threshold=0., flow_threshold=flow_threshold, min_size=16,
        max_size_fraction=.4, device=device)
    probability = 1 / (1 + np.exp(-np.clip(cellprob, -30, 30)))
    instances = []
    for index in np.unique(labels):
        if index == 0: continue
        mask = labels == index
        score = float(probability[mask].mean())
        if score >= .25:
            instances.append({'rle': encode_mask(mask), 'score': score})
    return sorted(instances, key=lambda item: item['score'], reverse=True)[:64]


class FlowV2Route(FlowRoute):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.cache = Path(cfg['target_cache']) / cfg['train_split_sha256']
        self.cache.mkdir(parents=True, exist_ok=True)
        # Preserve all state_dict names and stochastic block behavior.
        for block in self.net.encoder.blocks:
            original = block.forward
            def forward(*args, _original=original, **kwargs):
                if torch.is_grad_enabled():
                    return checkpoint(_original, *args, use_reentrant=False, **kwargs)
                return _original(*args, **kwargs)
            block.forward = forward

    def targets(self, sample, masks):
        from cellpose.dynamics import masks_to_flows_gpu
        name = str(sample['annotation_group'])
        if Path(name).name != name or name in {'.', '..'}:
            raise ValueError('Unsafe annotation group ID')
        path = self.cache / (name + '.npz')
        if path.exists():
            with np.load(path, allow_pickle=False) as data:
                if int(data['version']) != 2:
                    raise ValueError('Target cache version mismatch')
                return data['flow'].astype(np.float32), data['foreground'].astype(bool), data['valid'].astype(bool)
        labels = make_labels(masks)
        # Crucially, generate on the complete native annotation group BEFORE cropping.
        flow = masks_to_flows_gpu(labels, device=torch.device('cpu'))[0] if labels.any() else np.zeros((2, *labels.shape), np.float32)
        foreground = masks.any(0)
        valid = masks.sum(0) <= 1
        temporary = path.with_suffix('.tmp')
        with temporary.open('wb') as stream:
            np.savez_compressed(stream, flow=flow.astype(np.float16), foreground=foreground,
                                valid=valid, version=np.array(2))
            stream.flush(); os.fsync(stream.fileno())
        temporary.replace(path)
        # Match subsequent cache reads exactly, including float16 target quantization.
        return flow.astype(np.float16).astype(np.float32), foreground, valid

    def loss(self, sample):
        masks = sample['instances'].numpy().astype(bool)
        flow, foreground, valid = self.targets(sample, masks)
        side = self.cfg['size']
        present = np.flatnonzero(masks.reshape(len(masks), -1).any(1)) if len(masks) else []
        selected = masks[np.random.choice(present)] if len(present) and np.random.rand() < .8 else np.zeros_like(foreground)
        y0, x0 = crop_origin(selected, side)
        image = sample['image'][:, y0:y0+side, x0:x0+side]
        patch = (slice(y0, y0+side), slice(x0, x0+side))
        image, target, fg, keep = transform_patch(image, flow[:, patch[0], patch[1]],
            foreground[patch], valid[patch], np.random.randint(4), np.random.rand() < .5)
        image = image.pow(float(np.random.uniform(.85, 1.15)))
        prediction, _ = self.net(image[None].cuda())
        return flow_foreground_loss(prediction[0], torch.as_tensor(target, device='cuda') * 5,
            torch.as_tensor(fg, dtype=torch.float32, device='cuda'), torch.as_tensor(keep, device='cuda'))

    @torch.no_grad()
    def predict(self, image):
        self.eval()
        _, fields, _ = self.cp.eval(image.permute(1, 2, 0).numpy(), batch_size=1,
            normalize=False, resample=True, diameter=None, bsize=self.cfg['size'],
            tile_overlap=.25, compute_masks=False)
        return decode_fields(fields[1], fields[2], self.cfg['niter'], .4, torch.device('cuda'))
