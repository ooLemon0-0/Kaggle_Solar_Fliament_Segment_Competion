import _bootstrap
import argparse
from src.pipeline import submission
from src.utils.common import restored_config as config_for, MODELS
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--model',required=True,choices=MODELS);p.add_argument('--debug',action='store_true');p.add_argument('--checkpoint');a=p.parse_args()
    submission(config_for(a.model,a.debug),a.checkpoint)
