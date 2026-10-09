import json
import sys
import time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'linux'))
from control import relevant_processes
BASE=ROOT/'output/benchmark/routes_v3'
def read(path):return json.loads(path.read_text()) if path.is_file() else {}
print('Actual processes:',json.dumps(relevant_processes(),ensure_ascii=False))
print('Runner stage:',read(BASE/'pipeline_status.json').get('stage','not started'))
for c in read(BASE/'experiment_manifest.json').get('runs',[]):
    folder=BASE/c['id'];p=read(folder/'progress.json');m=read(folder/'best_metrics.json');done=read(folder/'completed.json')
    last=folder/'last.pt';log=folder/'train_console.log'
    age=min([time.time()-f.stat().st_mtime for f in [last,log,folder/'progress.json'] if f.exists()],default=None)
    print(c['id'], 'completed' if done else p.get('stage','queued'),
          'epoch',p.get('epoch'),'sample/image',p.get('sample',p.get('image')),
          'best PQ',m.get('pq'),'last activity seconds',round(age) if age is not None else None)
