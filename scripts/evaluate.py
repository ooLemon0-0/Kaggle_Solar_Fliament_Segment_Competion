import _bootstrap
import argparse
from src.pipeline import evaluate
from src.utils.common import restored_config as config_for, MODELS
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--model',choices=MODELS+['all'],default='all');p.add_argument('--debug',action='store_true');a=p.parse_args()
    for model in MODELS if a.model=='all' else [a.model]:
        r=evaluate(config_for(model,a.debug));print(model,r['pq'],r['dice'],r['iou'])
