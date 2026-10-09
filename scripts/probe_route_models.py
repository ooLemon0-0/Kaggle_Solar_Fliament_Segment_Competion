import _bootstrap
import argparse
from src.utils.common import load_json
from src.routes_v3.engine import BASE,probe
parser=argparse.ArgumentParser();parser.add_argument('--routes',nargs='+');args=parser.parse_args()
for cfg in load_json(BASE/'experiment_manifest.json')['runs']:
    if cfg['id'] not in args.routes:continue
    if cfg['kind'] not in ['rf','flow','sam']:cfg['warmstart_path']=str(BASE/'rf/debug_probe.pt')
    probe(cfg,BASE/'debug_preflight'/cfg['id'])
