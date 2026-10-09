"""Download with the official CLI; validate ZIP members before extraction."""
import argparse
import hashlib
import json
import subprocess
import sys
import zipfile
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
parser=argparse.ArgumentParser()
parser.add_argument('--kaggle',default=str(ROOT/'.envs/kaggle-tools/bin/kaggle'))
parser.add_argument('--official-notebook',action='store_true',help='Fetch separate reference copy, never replace evaluator')
args=parser.parse_args()
raw=ROOT/'data/raw';raw.mkdir(parents=True,exist_ok=True)
existing=list(raw.rglob('MAGFiLO_1.0_Annotations_kaggle2026_train.json'))
if not existing:
    subprocess.run([args.kaggle,'competitions','files','-c','filament-segmentation-2026'],check=True)
    subprocess.run([args.kaggle,'competitions','download','-c','filament-segmentation-2026','-p',str(raw)],check=True)
    archives=list(raw.glob('*.zip'))
    if not archives:raise RuntimeError('CLI did not produce a ZIP')
    for path in archives:
        with zipfile.ZipFile(path) as archive:
            if archive.testzip() is not None:raise RuntimeError('Corrupt competition archive')
            for member in archive.infolist():
                name=member.filename.replace('\\','/')
                target=(raw/name).resolve()
                if not target.is_relative_to(raw.resolve()) or (member.external_attr>>16)&0o170000==0o120000:
                    raise RuntimeError('Unsafe ZIP member')
            archive.extractall(raw)
existing=list(raw.rglob('MAGFiLO_1.0_Annotations_kaggle2026_train.json'))
if len(existing)!=1:raise RuntimeError(f'Expected exactly one annotation JSON; found {len(existing)}')
subprocess.run([sys.executable,str(ROOT/'scripts/restore_frozen_splits.py'),'--annotation',str(existing[0])],check=True)

data_root=existing[0].parent.parent
for split in ['train','val']:
    path=ROOT/f'output/splits/{split}.json'
    manifest=json.loads((ROOT/'output/benchmark/routes_v3/experiment_manifest.json').read_text())
    assert hashlib.sha256(path.read_bytes()).hexdigest()==manifest[f'{split}_split_sha256']
    data=json.loads(path.read_text())
    missing=[im['file_name'] for im in data['images'] if not (data_root/'train/train_images'/im['file_name']).exists()]
    if missing:raise RuntimeError(f'Missing {len(missing)} {split} files: {missing[:3]}')
print('DATA_READY: original train/val images and frozen split verified.')
if args.official_notebook:
    dest=ROOT/'work/downloaded_official_notebook';dest.mkdir(parents=True,exist_ok=True)
    subprocess.run([args.kaggle,'kernels','pull','azimahmadzadeh/self-evaluation-notebook','-p',str(dest),'-m'],check=True)
    for path in dest.glob('*.ipynb'):
        print('Downloaded reference notebook SHA256:',hashlib.sha256(path.read_bytes()).hexdigest())
    print('Bundled official evaluator and frozen notebook were not overwritten.')
