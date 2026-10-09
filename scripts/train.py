import _bootstrap
import argparse
from src.pipeline import train
from src.utils.common import config_for, MODELS
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--model',required=True,choices=MODELS);p.add_argument('--debug',action='store_true');p.add_argument('--config');p.add_argument('--epochs',type=int)
    a=p.parse_args();cfg=config_for(a.model,a.debug,a.config)
    if a.epochs is not None:cfg['epochs']=a.epochs
    train(cfg)
