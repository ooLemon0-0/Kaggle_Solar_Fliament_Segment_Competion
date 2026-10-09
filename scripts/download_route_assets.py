"""Resumable range download; assembled files are published only after verification."""
import _bootstrap
import concurrent.futures as cf
import hashlib
import time
from pathlib import Path
import requests
from src.utils.common import ROOT

def fetch(url, destination, expected_md5=None, workers=12,expected_sha256=None):
    destination=Path(destination); destination.parent.mkdir(parents=True,exist_ok=True)
    if destination.exists() and expected_md5 and hashlib.file_digest(destination.open('rb'),'md5').hexdigest()==expected_md5:
        return
    response=requests.get(url,headers={'Range':'bytes=0-99'},stream=True,timeout=90)
    response.raise_for_status()
    if response.status_code!=206: raise RuntimeError('Server does not support checked range downloads')
    total=int(response.headers['Content-Range'].split('/')[-1]); response.close()
    chunks=destination.parent/(destination.name+'.parts'); chunks.mkdir(exist_ok=True)
    width=1024*1024
    def part(index):
        start=index*width; end=min(total,start+width)-1; path=chunks/str(index)
        if path.exists() and path.stat().st_size==end-start+1:return
        for attempt in range(8):
            try:
                with requests.get(url,headers={'Range':f'bytes={start}-{end}'},stream=True,timeout=(90,90)) as r:
                    r.raise_for_status()
                    if r.status_code!=206 or r.headers.get('Content-Range')!=f'bytes {start}-{end}/{total}':
                        raise ValueError('Unverified Content-Range')
                    tmp=path.with_suffix('.tmp')
                    with tmp.open('wb') as f:
                        for block in r.iter_content(65536):f.write(block)
                    if tmp.stat().st_size!=end-start+1:raise ValueError('Short download')
                    tmp.replace(path)
                return
            except Exception as exc:
                print(f'retry part={index} attempt={attempt} {type(exc).__name__}',flush=True); time.sleep(2)
        raise RuntimeError(f'Failed chunk {index}')
    with cf.ThreadPoolExecutor(max_workers=workers) as pool:
        futures=[pool.submit(part,i) for i in range((total+width-1)//width)]
        for n,f in enumerate(cf.as_completed(futures),1):
            f.result();print(f'{destination.name}: {n}/{len(futures)} MiB chunks',flush=True)
    tmp=destination.with_suffix('.assembled')
    with tmp.open('wb') as out:
        for i in range(len(futures)):out.write((chunks/str(i)).read_bytes())
    if tmp.stat().st_size!=total:raise ValueError('Assembled size mismatch')
    actual=hashlib.file_digest(tmp.open('rb'),'md5').hexdigest()
    if expected_md5 and actual!=expected_md5:raise ValueError(f'MD5 mismatch {actual}')
    if expected_sha256 and hashlib.file_digest(tmp.open('rb'),'sha256').hexdigest()!=expected_sha256:raise ValueError('SHA256 mismatch')
    if destination.exists():destination.replace(destination.with_suffix('.old-partial'))
    tmp.replace(destination)
    print(f'VERIFIED {destination.name} bytes={total} md5={actual}',flush=True)

if __name__=='__main__':
    from rfdetr.assets.model_weights import ModelWeights
    a=ModelWeights.from_filename('rf-detr-seg-small.pt')
    fetch(a.url,ROOT/'output/pretrained/routes_v3/rf'/a.filename,a.md5_hash)
