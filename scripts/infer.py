import _bootstrap
import argparse
from src.pipeline import infer
from src.utils.common import restored_config as config_for, MODELS
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--model',required=True,choices=MODELS);p.add_argument('--debug',action='store_true');p.add_argument('--checkpoint');p.add_argument('--split',choices=['val','test'],default='val')
    a=p.parse_args();print(infer(config_for(a.model,a.debug),a.split,a.checkpoint))
