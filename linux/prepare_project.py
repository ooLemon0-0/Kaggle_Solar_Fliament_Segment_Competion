"""Relocate report paths without changing run configs or split fingerprints."""
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
path=ROOT/'output/benchmark/routes_v3/experiment_manifest.json'
manifest=json.loads(path.read_text(encoding='utf-8'))
manifest['deliverable_dir']=str(ROOT/'deliverables/routes_v3')
manifest['migration_target']='single RTX 4090D 24GB, strict serial, original protocol'
path.write_text(json.dumps(manifest,indent=2,ensure_ascii=False),encoding='utf-8')
for name in ['logs','work','deliverables/routes_v3']:(ROOT/name).mkdir(parents=True,exist_ok=True)
print('PROJECT_READY',ROOT)
