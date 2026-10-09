import _bootstrap
import argparse
import concurrent.futures
import hashlib
import math
import time
from pathlib import Path
import requests
from src.utils.common import ROOT


def download(url, path, sha_prefix=None):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists():
        if sha_prefix:
            with path.open('rb') as f:
                assert hashlib.file_digest(f,'sha256').hexdigest().startswith(sha_prefix)
        return
    metadata=requests.head(url,allow_redirects=True,timeout=30)
    metadata.raise_for_status();size=int(metadata.headers['Content-Length'])
    block=4*1024*1024;parts=ROOT/'work/weight_parts'/path.name;parts.mkdir(parents=True,exist_ok=True)
    count=math.ceil(size/block)
    # Reuse the prefix already received by torch.hub, then verify full SHA256.
    old=list(path.parent.glob(path.name+'.*.partial'))
    if old:
        prefix=max(old,key=lambda p:p.stat().st_size)
        with prefix.open('rb') as f:
            for i in range(min(count,prefix.stat().st_size//block)):
                target=parts/f'{i:03d}'
                payload=f.read(block)
                if not target.exists():target.write_bytes(payload)
    def transfer(i):
        start=i*block;end=min(size,start+block)-1;target=parts/f'{i:03d}'
        if target.exists() and target.stat().st_size==end-start+1:return target
        for attempt in range(5):
            try:
                r=requests.get(url,headers={'Range':f'bytes={start}-{end}'},timeout=(20,45))
                r.raise_for_status()
                if r.status_code!=206 or r.headers.get('Content-Range')!=f'bytes {start}-{end}/{size}':raise RuntimeError('Weight server did not honor HTTP Range')
                if len(r.content)!=end-start+1:raise RuntimeError('Incomplete weight range')
                target.write_bytes(r.content);return target
            except requests.RequestException:
                time.sleep(2)
        raise RuntimeError('Weight range download failed')
    with concurrent.futures.ThreadPoolExecutor(max_workers=12) as pool:
        paths=list(pool.map(transfer,range(count)))
    complete=path.with_suffix(path.suffix+'.complete')
    with complete.open('wb') as f:
        for p in paths:f.write(p.read_bytes())
    with complete.open('rb') as f:sha=hashlib.file_digest(f,'sha256').hexdigest()
    if sha_prefix and not sha.startswith(sha_prefix):raise RuntimeError('Official pretrained weight SHA256 mismatch')
    complete.replace(path);print(path,'SHA256',sha,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--model',choices=['all','yolo_seg','semantic_seg'],default='all');a=p.parse_args()
    if a.model in ['all','semantic_seg']:
        download('https://download.pytorch.org/models/resnet50-0676ba61.pth',ROOT/'output/pretrained/torch/hub/checkpoints/resnet50-0676ba61.pth','0676ba61')
    if a.model in ['all','yolo_seg']:
        download('https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo11s-seg.pt',ROOT/'output/pretrained/yolo11s-seg.pt')
