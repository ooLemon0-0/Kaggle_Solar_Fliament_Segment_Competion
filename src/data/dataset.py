from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from pycocotools import mask as mask_util
from torch.utils.data import Dataset

from src.utils.common import ROOT, load_json


def discover():
    found = list((ROOT / 'data/raw').rglob('MAGFiLO_1.0_Annotations_kaggle2026_train.json'))
    if len(found) != 1:
        raise FileNotFoundError(f'Expected exactly one training JSON, found {len(found)}')
    annotation = found[0]
    base = annotation.parent.parent
    return annotation, base / 'train/train_images', base / 'test/test_images'


def observation(image):
    return Path(image['file_name']).stem


def annotation_rle(annotation, image):
    seg = annotation['segmentation']
    h, w = image['height'], image['width']
    if isinstance(seg, list):
        return mask_util.merge(mask_util.frPyObjects(seg, h, w))
    if isinstance(seg['counts'], list):
        return mask_util.frPyObjects(seg, h, w)
    return seg


def mask_from_annotation(annotation, image):
    return mask_util.decode(annotation_rle(annotation, image)).astype(bool)


def encode_mask(mask):
    rle = mask_util.encode(np.asfortranarray(mask, dtype=np.uint8))
    return {'size': list(rle['size']), 'counts': rle['counts'].decode('ascii')}


def decode_mask(rle):
    return mask_util.decode(rle).astype(bool)


class FilamentDataset(Dataset):
    """One independent annotator-image record per training sample, one class."""
    def __init__(self, annotation_path, size=1024):
        self.data = load_json(annotation_path)
        self.images = self.data['images']
        self.size = size
        self.image_dir = discover()[1]
        self.annotations = defaultdict(list)
        for a in self.data['annotations']:
            self.annotations[a['image_id']].append(a)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image = self.images[index]
        with Image.open(self.image_dir / image['file_name']) as original:
            gray = np.asarray(original.convert('L').resize((self.size, self.size), Image.Resampling.BILINEAR), dtype=np.float32) / 255.
        masks = []
        for a in self.annotations[image['id']]:
            native = mask_from_annotation(a, image)
            resized = np.asarray(Image.fromarray(native).resize((self.size, self.size), Image.Resampling.NEAREST), dtype=bool)
            if resized.any():
                masks.append(resized)
        instances = np.stack(masks) if masks else np.zeros((0, self.size, self.size), dtype=bool)
        return {'image': torch.from_numpy(np.repeat(gray[None], 3, axis=0)),
                'instances': torch.from_numpy(instances.astype(np.float32)),
                'semantic': torch.from_numpy(instances.any(axis=0).astype(np.float32))[None],
                'image_id': observation(image), 'annotation_group': image['id']}


def collate(items):
    return items
