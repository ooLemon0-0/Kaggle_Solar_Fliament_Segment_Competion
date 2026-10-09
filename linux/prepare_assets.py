"""Prefer exact migrated assets. Optional immutable public downloads run on CPU."""
import argparse
import hashlib
import json
import os
import sys
import urllib.request
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
parser=argparse.ArgumentParser()
parser.add_argument('--download',action='store_true',help='Download public originals if migrated assets are absent')
args=parser.parse_args()
expected=json.loads((ROOT/'expected_assets.json').read_text())
def digest(path):
    with path.open('rb') as f: return hashlib.file_digest(f,'sha256').hexdigest()
def download(url,path):
    path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix('.download')
    with urllib.request.urlopen(url,timeout=120) as response,temp.open('wb') as stream:
        import shutil
        shutil.copyfileobj(response,stream)
    temp.replace(path)
for rel,item in expected['files'].items():
    path=ROOT/rel
    if path.exists():
        if path.stat().st_size!=item['bytes'] or digest(path)!=item['sha256']:
            raise RuntimeError(f'Asset mismatch: {rel}; preserving file, use exact resume-assets bundle')
        print('VERIFIED',rel);continue
    if not args.download: raise RuntimeError(f'Missing {rel}; unpack resume-assets bundle or use --download')
    if '/rf/' in rel:
        # Use installed registry; verify immutable original MD5 and packaged SHA256.
        from rfdetr import RFDETRSegSmall
        os.environ['RF_HOME']=str(path.parent)
        wrapper=RFDETRSegSmall(device='cpu');del wrapper
    elif '/cellpose/' in rel:
        from huggingface_hub import hf_hub_download
        import shutil
        source=hf_hub_download('mouseland/cellpose-sam','cpdino-vitb',revision=expected['cellpose_hf_revision'])
        path.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(source,path)
    else:
        from transformers import Sam2Model
        model=Sam2Model.from_pretrained('facebook/sam2.1-hiera-tiny',revision='de431c4043854a71d8101e17995dfe596bf101a5')
        model.save_pretrained(path.parent);del model
    if not path.is_file() or path.stat().st_size!=item['bytes'] or digest(path)!=item['sha256']:
        raise RuntimeError(f'{rel}: exported/downloaded bytes differ; use exact migrated assets, do not silently switch pretrained weights')
if not (ROOT/'output/pretrained/routes_v3/sam2_tiny/config.json').is_file():
    raise RuntimeError('SAM config.json absent; unpack complete resume-assets bundle')
print('ASSETS_READY')
