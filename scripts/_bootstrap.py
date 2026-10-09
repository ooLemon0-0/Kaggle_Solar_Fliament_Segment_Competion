import sys
import os
from pathlib import Path
root=Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root))
for variable,relative in [('YOLO_CONFIG_DIR','work/ultralytics'),('TORCH_HOME','output/pretrained/torch'),('MPLCONFIGDIR','work/matplotlib')]:
    target=root/relative
    target.mkdir(parents=True,exist_ok=True)
    os.environ.setdefault(variable,str(target))
(root/'work/ultralytics/Ultralytics').mkdir(parents=True,exist_ok=True)
