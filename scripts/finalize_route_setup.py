import _bootstrap
import hashlib
import importlib.metadata
import shutil
from pathlib import Path
import torch
from src.utils.common import ROOT,load_json
from src.routes_v3.engine import BASE,atomic_json
from src.routes_v3.models import DetailHead
from rfdetr.models.heads.segmentation import SegmentationHead

# Check the contextual projection change on CPU so the active GPU run stays exclusive.
torch.set_num_threads(2)
head=DetailHead(SegmentationHead(256,4));head.image=torch.rand(1,3,96,96)
spatial=torch.rand(1,256,8,8);queries=[torch.rand(1,5,256) for _ in range(4)]
optimizer=torch.optim.AdamW(head.parameters(),lr=1e-4)
for _ in range(2):
    output=head(spatial,queries,(96,96));loss=sum(x.square().mean() for x in output)+head.fg.square().mean()
    optimizer.zero_grad();loss.backward()
    assert all(torch.isfinite(p.grad).all() for p in head.parameters() if p.grad is not None)
    optimizer.step()
assert head.context.weight.grad is not None and head.context.weight.grad.abs().sum()>0
assert all(x.shape==(1,5,24,24) for x in output)
atomic_json(BASE/'debug_preflight/context_projection_check.json',dict(debug=True,device='cpu',finite_gradients=True,
            context_receives_gradient=True,mask_shape=[1,5,24,24]))
assets=ROOT/'output/pretrained/routes_v3';paths=[assets/'rf/rf-detr-seg-small.pt',assets/'cellpose/cpdino-vitb',assets/'sam2_tiny/model.safetensors']
asset_manifest=dict(files={p.relative_to(ROOT).as_posix():dict(bytes=p.stat().st_size,sha256=hashlib.file_digest(p.open('rb'),'sha256').hexdigest()) for p in paths},
    rf_registry_md5='0a2a3006381d0c42853907e700eadd08',cellpose_hf_revision='7c61431b5fbb078f3296754bd15d9f51b320f837',
    cellpose_official_sha256='3ed4c06a3963ab13ff377d4e2957174aaaf637434eb031ae9931fcbfaf9a217f')
assert asset_manifest['files']['output/pretrained/routes_v3/cellpose/cpdino-vitb']['sha256']==asset_manifest['cellpose_official_sha256']
atomic_json(assets/'asset_manifest.json',asset_manifest)
dest=Path(load_json(BASE/'experiment_manifest.json')['deliverable_dir'])
shutil.copy2(assets/'asset_manifest.json',dest/'asset_manifest.json')
evidence={}
for route in ['rf','sam','flow','image_projection','feature_control','feature_structure']:
    p=BASE/route/'preflight.json' if route=='rf' else BASE/'debug_preflight'/route/'preflight.json'
    evidence[route]=load_json(p)
evidence['context_projection_cpu']=load_json(BASE/'debug_preflight/context_projection_check.json')
atomic_json(dest/'preflight_checks.json',evidence)
from run_route_experiments import report
report(load_json(BASE/'pipeline_status.json'))
print('Setup and initial deliverables verified; training continues independently.',flush=True)
