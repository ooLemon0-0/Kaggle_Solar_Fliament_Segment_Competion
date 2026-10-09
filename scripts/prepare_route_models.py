"""Cache public pretrained assets for the isolated route comparison."""
import _bootstrap
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
from src.utils.common import ROOT, save_json

CACHE = ROOT/'output/pretrained/routes_v3'
os.environ['RF_HOME'] = str(CACHE/'rf')
os.environ['HF_HOME'] = str(CACHE/'hf')
os.environ['CELLPOSE_LOCAL_MODELS_PATH'] = str(CACHE/'cellpose')
os.environ['TORCH_HOME'] = str(CACHE/'torch')

def main():
    from rfdetr import RFDETRSegSmall
    from transformers import Sam2Model
    from cellpose.models import cache_model_path
    CACHE.mkdir(parents=True, exist_ok=True)
    manifest = {'versions':{n:importlib.metadata.version(n) for n in ['torch','torchvision','transformers','rfdetr','cellpose']}}
    print('Caching RF-DETR-Seg Small', flush=True)
    rf = RFDETRSegSmall(device='cpu')
    manifest['rf'] = rf.model_config.model_dump(mode='json')
    del rf
    print('Caching SAM2.1 Hiera Tiny', flush=True)
    sam = Sam2Model.from_pretrained('facebook/sam2.1-hiera-tiny', cache_dir=CACHE/'hf')
    manifest['sam2_revision'] = getattr(sam.config,'_commit_hash',None)
    sam.save_pretrained(CACHE/'sam2_tiny')
    del sam
    print('Caching CellposeDINO ViT-B', flush=True)
    manifest['cellpose_path'] = cache_model_path('cpdino-vitb')
    manifest['sha256'] = {}
    for path in [*list((CACHE/'rf').rglob('*.pt')),Path(manifest['cellpose_path']),CACHE/'sam2_tiny/model.safetensors']:
        if path.is_file():
            manifest['sha256'][str(path)] = hashlib.file_digest(path.open('rb'),'sha256').hexdigest()
    save_json(CACHE/'asset_manifest.json',manifest)
    print(json.dumps({'assets_ready':True,'manifest':str(CACHE/'asset_manifest.json')}),flush=True)

if __name__ == '__main__':
    main()
