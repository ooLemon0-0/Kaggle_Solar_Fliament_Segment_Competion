import csv
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.data.dataset import FilamentDataset, collate
from src.pipeline import create_model, infer, evaluate
from src.utils.common import split_path, save_json, sync


def train_torch(cfg):
    model=create_model(cfg).cuda()
    loader=DataLoader(FilamentDataset(split_path('train',cfg['debug']),cfg['image_size']),
                      batch_size=cfg['batch_size'],shuffle=True,num_workers=0,collate_fn=collate)
    optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=cfg['lr'],weight_decay=1e-4)
    scaler=torch.amp.GradScaler('cuda')
    run=Path(cfg['run_dir']);(run/'checkpoints').mkdir(exist_ok=True)
    best=-1.;best_epoch=None;peak=0.
    with (run/'training_log.csv').open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=['epoch','loss','official_pq','dice','iou','training_gpu_peak_mib'])
        writer.writeheader()
        for epoch in range(cfg['epochs']):
            model.train();torch.cuda.reset_peak_memory_stats();losses=[];started=time.monotonic()
            for step,items in enumerate(loader,1):
                images=torch.stack([i['image'] for i in items]).cuda()
                targets=[{k:v.cuda() if isinstance(v,torch.Tensor) else v for k,v in i.items() if k!='image'} for i in items]
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast('cuda',dtype=torch.float16):
                    loss=model.loss(images,targets)
                if not torch.isfinite(loss):raise FloatingPointError('Nonfinite loss')
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
                scaler.step(optimizer);scaler.update();losses.append(float(loss.detach()))
                if step%25==0 or step==len(loader):
                    print(f"{cfg['model']} epoch {epoch+1}/{cfg['epochs']} step {step}/{len(loader)} loss={np.mean(losses):.6f} elapsed={time.monotonic()-started:.1f}s",flush=True)
            sync();peak=max(peak,torch.cuda.max_memory_allocated()/1048576)
            state={'model':model.state_dict(),'optimizer':optimizer.state_dict(),'scaler':scaler.state_dict(),
                   'epoch':epoch+1,'config':cfg}
            torch.save(state,run/'checkpoints/last.pt')
            model.eval();infer(cfg,model=model);metrics=evaluate(cfg)
            if metrics['pq']>best:
                best=metrics['pq'];best_epoch=epoch+1;torch.save(state,run/'checkpoints/best.pt')
            row={'epoch':epoch+1,'loss':float(np.mean(losses)),'official_pq':metrics['pq'],
                 'dice':metrics['dice'],'iou':metrics['iou'],'training_gpu_peak_mib':peak}
            writer.writerow(row);f.flush();print(row,flush=True)
    save_json(run/'training_summary.json',{'training_gpu_peak_mib':peak,'best_epoch':best_epoch,
              'checkpoint_selection':'official global validation PQ','epochs_completed':cfg['epochs']})
    del model,optimizer;torch.cuda.empty_cache()
    infer(cfg);return evaluate(cfg)
