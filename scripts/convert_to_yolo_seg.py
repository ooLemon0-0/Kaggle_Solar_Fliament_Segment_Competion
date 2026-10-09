import _bootstrap
import argparse
from src.models.yolo_seg.model import convert
if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--debug',action='store_true');args=parser.parse_args()
    print(convert(args.debug))
