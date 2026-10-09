"""Readiness audit. Never launches formal training or downloads implicitly."""
import argparse
import gc
import hashlib
import importlib.metadata as md
import json
import platform
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['host', 'gpu', 'ready'], default='ready')
    parser.add_argument('--preflight', choices=['rf','flow','sam','image_projection','feature_control','feature_structure'])
    args = parser.parse_args()
    report = {'mode': args.mode, 'python': sys.version, 'platform': platform.platform(), 'checks': {}, 'errors': []}
    def check(name, fn):
        try:
            report['checks'][name] = fn()
        except Exception as exc:
            report['errors'].append(f'{name}: {type(exc).__name__}: {exc}')
    def host():
        if platform.system() != 'Linux':
            raise RuntimeError('Deployment audit requires Linux; local Windows syntax checks do not prove server readiness')
        if sys.version_info[:2] != (3,11): raise RuntimeError('Use isolated Python 3.11 environment')
        return {'disk_free_gib': shutil.disk_usage(ROOT).free / 2**30,
                'conda': shutil.which('conda'), 'nvidia_smi': shutil.which('nvidia-smi')}
    check('host', host)
    if args.mode != 'host':
        def versions():
            wanted = json.loads((ROOT/'local_reference_versions.json').read_text())
            actual = {name: md.version(name) for name in wanted}
            mismatch = {k: (wanted[k], actual[k]) for k in wanted if wanted[k] != actual[k]}
            if mismatch: raise RuntimeError(f'Package versions differ: {mismatch}')
            subprocess.run([sys.executable,'-m','pip','check'],check=True)
            return actual
        check('packages', versions)
        def cuda():
            import torch
            import torchvision
            if not torch.cuda.is_available(): raise RuntimeError('CUDA unavailable; check host driver and cu128 wheel')
            p = torch.cuda.get_device_properties(0)
            x = torch.randn(64,64,device='cuda',requires_grad=True)
            with torch.autocast('cuda',dtype=torch.bfloat16): loss = (x@x).square().mean()
            loss.backward()
            if not torch.isfinite(loss) or not torch.isfinite(x.grad).all(): raise RuntimeError('CUDA/BF16 gradients nonfinite')
            torchvision.ops.nms(torch.tensor([[0.,0.,2.,2.]],device='cuda'),torch.tensor([1.],device='cuda'),.5)
            torch.cuda.synchronize()
            if p.total_memory < 22*2**30: raise RuntimeError('Expected 4090D 24GB; less than 22GiB detected')
            return {'name': p.name, 'vram_gib': p.total_memory/2**30, 'torch_cuda': torch.version.cuda,
                    'cuda_matmul_backward': True, 'torchvision_cuda_nms': True}
        check('cuda', cuda)
    if args.mode == 'ready':
        def protocol():
            expected=json.loads((ROOT/'linux/integrity.json').read_text())
            for rel, value in expected['immutable_files'].items():
                if sha(ROOT/rel) != value: raise RuntimeError(f'Immutable protocol file changed: {rel}')
            m=json.loads((ROOT/'output/benchmark/routes_v3/experiment_manifest.json').read_text())
            original=expected['runs']
            if m['runs'] != original: raise RuntimeError('Route configs differ from migration protocol')
            return {'splits_and_official_evaluator_unchanged': True}
        check('protocol',protocol)
        def data():
            from src.data.dataset import discover, observation
            annotation, train_dir, test_dir = discover()
            fingerprints=json.loads((ROOT/'linux/data_fingerprints.json').read_text())
            original_root=annotation.parent.parent
            for rel,value in fingerprints.items():
                if sha(original_root/rel)!=value:raise RuntimeError('Original dataset bytes differ: '+rel)
            result={}
            for name, groups, observations in [('train',936,565),('val',218,142)]:
                d=json.loads((ROOT/f'output/splits/{name}.json').read_text())
                if len(d['images'])!=groups or len({observation(i) for i in d['images']})!=observations:
                    raise RuntimeError(f'{name} annotation groups/observations changed')
                absent=[i['file_name'] for i in d['images'] if not (train_dir/i['file_name']).is_file()]
                if absent: raise RuntimeError(f'Missing images: {absent[:3]}')
                result[name]={'observations': observations, 'annotation_groups':groups}
            def image_files(folder):return [p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in {'.png','.jpg','.jpeg','.tif','.tiff'}]
            train_images=image_files(train_dir); test_images=image_files(test_dir)
            if len(train_images)!=707 or len(test_images)!=180:
                raise RuntimeError(f'Expected 707 train / 180 test files, got {len(train_images)} / {len(test_images)}')
            result['test_files']=180
            result['original_file_sha256_verified']=len(fingerprints)
            return result
        check('data',data)
        def assets():
            expected=json.loads((ROOT/'expected_assets.json').read_text())['files']
            for rel,value in expected.items():
                path=ROOT/rel
                if not path.is_file() or path.stat().st_size!=value['bytes'] or sha(path)!=value['sha256']:
                    raise RuntimeError(f'Missing or differing pretrained asset: {rel}; unpack resume-assets ZIP')
            if not (ROOT/'output/pretrained/routes_v3/sam2_tiny/config.json').is_file(): raise RuntimeError('Missing SAM config')
            return {'verified_files':len(expected)}
        check('assets',assets)
        def resumes():
            import torch
            m=json.loads((ROOT/'output/benchmark/routes_v3/experiment_manifest.json').read_text())
            out={}
            for c in m['runs']:
                folder=ROOT/'output/benchmark/routes_v3'/c['id']
                if (folder/'completed.json').is_file():
                    if not (folder/'best.pt').is_file(): raise RuntimeError(f'{c["id"]} completed but best checkpoint missing')
                    out[c['id']]={'completed':True}; continue
                if not (folder/'last.pt').is_file():
                    out[c['id']]={'fresh_route':True}; continue
                saved=torch.load(folder/'last.pt',map_location='cpu',weights_only=False)
                if saved['config']!=c: raise RuntimeError(f'{c["id"]}: checkpoint config differs')
                if saved['config']!=json.loads((folder/'config.json').read_text()): raise RuntimeError('Resume config.json differs')
                for k in ['model','optimizer','state','rng']:
                    if k not in saved: raise RuntimeError(f'Missing {k}')
                s=saved['state']
                for which in ['train','val']:
                    if s[f'{which}_split_sha256']!=m[f'{which}_split_sha256']: raise RuntimeError('Resume split differs')
                order=s['order']
                if order and sorted(order)!=list(range(936)): raise RuntimeError('Epoch permutation invalid')
                if not 0<=s['offset']<=936 or (s['offset'] and not order): raise RuntimeError('Resume offset invalid')
                if s['offset']%c['accumulation'] and s['offset']!=936: raise RuntimeError('Checkpoint contains incomplete accumulation')
                if not all(k in saved['rng'] for k in ['python','numpy','torch','cuda']): raise RuntimeError('RNG incomplete')
                out[c['id']]={k:s[k] for k in ['epoch','offset','updates','train_seconds','validation_pending']}
                del saved; gc.collect()
            return out
        check('checkpoints',resumes)
    report['passed']=not report['errors']
    (ROOT/'work').mkdir(exist_ok=True)
    (ROOT/'work/check_environment.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
    print(json.dumps(report,ensure_ascii=False,indent=2))
    if report['errors']: return 1
    if args.preflight:
        if args.mode!='ready': raise RuntimeError('Preflight requires --mode ready')
        from src.routes_v3.engine import BASE,probe
        from src.routes_v3.resources import file_lock
        m=json.loads((BASE/'experiment_manifest.json').read_text())
        cfg=next(c for c in m['runs'] if c['id']==args.preflight)
        folder=BASE/'linux_preflight'/cfg['id'];folder.mkdir(parents=True,exist_ok=True)
        with file_lock(BASE/'gpu_operation.lock',blocking=False): probe(cfg,folder)
        print('DEBUG_PREFLIGHT_OK: eight steps, excluded from formal scores/budget.')
    return 0

if __name__=='__main__': sys.exit(main())
