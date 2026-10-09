import hashlib
import json
import random
from pathlib import Path

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[2]
MODELS = ['yolo_seg', 'semantic_seg', 'dino_transformer']


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding='utf-8')


def load_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def config_for(model, debug=False, config_path=None):
    cfg = yaml.safe_load(Path(config_path or ROOT / f'configs/{model}.yaml').read_text(encoding='utf-8'))
    cfg.update(model=model, debug=debug)
    cfg.setdefault('max_instances',64)
    if debug:
        cfg['epochs'] = cfg['debug_epochs']
    cfg['run_dir'] = str(ROOT / 'output' / model / ('debug' if debug else ''))
    return cfg


def split_path(which, debug=False):
    return ROOT / 'output/splits' / (f'debug_{which}.json' if debug else f'{which}.json')


def restored_config(model, debug=False):
    cfg=config_for(model,debug)
    snapshot=Path(cfg['run_dir'])/'config.json'
    if snapshot.exists():
        run_dir=cfg['run_dir']
        cfg.update(load_json(snapshot))
        cfg['run_dir']=run_dir
    return cfg


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()
