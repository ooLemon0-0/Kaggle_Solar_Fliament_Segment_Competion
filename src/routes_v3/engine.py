import csv
import gc
import json
import os
import random
import time
from pathlib import Path
import numpy as np
import psutil
import torch
from PIL import Image
from src.utils.common import ROOT, digest, load_json, split_path, seed_everything
from src.data.dataset import FilamentDataset, discover, observation
from src.evaluation.adapter import evaluate_predictions
from src.routes_v3.models import build
from src.routes_v3.resources import gpu_window, gpu_wait_seconds, exclusive_gpu

BASE=Path(os.environ.get('SOLAR_ROUTE_OUTPUT_ROOT', ROOT/'output/benchmark/routes_v3'))

def atomic_json(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix('.tmp');temp.write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf-8');temp.replace(path)

def atomic_torch(path,value):
    path=Path(path);temp=path.with_suffix('.tmp');torch.save(value,temp);temp.replace(path)

def rng_state():
    return dict(python=random.getstate(),numpy=np.random.get_state(),torch=torch.get_rng_state(),cuda=torch.cuda.get_rng_state_all())

def restore_rng(s):
    random.setstate(s['python']);np.random.set_state(s['numpy']);torch.set_rng_state(s['torch']);torch.cuda.set_rng_state_all(s['cuda'])

def native_images():
    data=load_json(split_path('val'));seen=set();directory=discover()[1]
    for im in data['images']:
        key=observation(im)
        if key in seen:continue
        seen.add(key)
        with Image.open(directory/im['file_name']) as original:
            gray=np.asarray(original.convert('L'),dtype=np.float32)/255
        yield key,torch.from_numpy(np.repeat(gray[None],3,axis=0))

def evaluate(model,cfg,directory,epoch):
    model.eval();predictions=[];started=time.monotonic()
    proposals={}
    if cfg['kind']=='sam':
        proposals={p['image_id']:p['instances'] for p in load_json(BASE/'rf/best_predictions.json')['predictions']}
    for i,(key,image) in enumerate(native_images(),1):
        with gpu_window():
            with torch.autocast('cuda',dtype=torch.bfloat16):
                instances=model.predict(image,proposals[key]) if cfg['kind']=='sam' else model.predict(image)
        predictions.append({'image_id':key,'instances':instances})
        if i%10==0:
            print(f'validation epoch={epoch} image={i}/142 elapsed={time.monotonic()-started:.1f}s',flush=True)
            atomic_json(directory/'progress.json',dict(stage='validation',epoch=epoch,image=i,time=time.time(),pid=os.getpid()))
    metadata=dict(model=cfg['id'],training_epochs=epoch,debug=False,parameter_count=sum(p.numel() for p in model.parameters()),
                  gpu_memory_mib=torch.cuda.max_memory_allocated()/2**20,inference_time=(time.monotonic()-started)/142)
    prediction_path=directory/'validation_predictions.json'
    atomic_json(prediction_path,dict(predictions=predictions,split_sha256=digest(split_path('val')),metadata=metadata))
    metrics=evaluate_predictions(prediction_path)
    if cfg.get('flow_version')==2:metrics['validation_wall_seconds']=time.monotonic()-started
    atomic_json(directory/f'validation_epoch_{epoch}.json',metrics)
    print(f'OFFICIAL epoch={epoch} PQ={metrics["pq"]:.6f} Dice={metrics["dice"]:.6f} IoU={metrics["iou"]:.6f}',flush=True)
    return metrics

def optimizer_for(model,cfg):
    encoder=[];head=[]
    for name,p in model.named_parameters():
        if not p.requires_grad:continue
        (encoder if 'backbone' in name or 'vision_encoder' in name or cfg['kind']=='flow' else head).append(p)
    groups=[]
    if encoder:groups.append(dict(params=encoder,lr=cfg['encoder_lr']))
    if head:groups.append(dict(params=head,lr=cfg['lr']))
    return torch.optim.AdamW(groups,weight_decay=1e-4)

def sample_at(dataset,index,augment=True):
    item=dataset[index]
    if augment and dataset.size!=2048:
        turns=np.random.randint(4)
        for key in ['image','instances','semantic']:item[key]=torch.rot90(item[key],turns,(-2,-1))
        if np.random.rand()<.5:
            for key in ['image','instances','semantic']:item[key]=item[key].flip(-1)
        gamma=np.random.uniform(.85,1.15);item['image']=item['image'].pow(gamma)
    return item

@exclusive_gpu
def probe(cfg,folder):
    probe_started=time.monotonic()
    cfg=cfg.copy();cfg['debug_prediction_confidence']=0.
    seed_everything(42)
    with gpu_window():model=build(cfg);optimizer=optimizer_for(model,cfg)
    dataset=FilamentDataset(split_path('train'),size=2048 if cfg['kind'] in ['sam','flow'] else cfg['size'])
    counts={}
    for a in dataset.data['annotations']:counts[a['image_id']]=counts.get(a['image_id'],0)+1
    index=max(range(len(dataset)),key=lambda i:counts.get(dataset.images[i]['id'],0))
    losses=[];projection_grad_norms=[];torch.cuda.reset_peak_memory_stats();model.training_mode()
    # Worst instance-count image plus a repeated-image optimization check. Smoke artifacts are never reported as formal results.
    for step in range(8):
        item=sample_at(dataset,index,augment=False)
        with torch.autocast('cuda',dtype=torch.bfloat16):loss=model.loss(item)
        if not torch.isfinite(loss):raise FloatingPointError('Nonfinite preflight loss')
        optimizer.zero_grad(set_to_none=True);loss.backward()
        if cfg['kind']=='image_projection':
            gradients=[p.grad for p in model.projector.parameters() if p.grad is not None]
            magnitude=sum(float(g.detach().float().square().sum()) for g in gradients)**.5
            if not gradients or not np.isfinite(magnitude) or magnitude<=0:
                raise FloatingPointError('Projection preflight must backpropagate finite nonzero gradients through frozen RF')
            if any(p.grad is not None for p in model.core.parameters()):
                raise RuntimeError('Projection protocol requires frozen RF weights')
            projection_grad_norms.append(magnitude)
        norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
        if not torch.isfinite(norm):raise FloatingPointError('Nonfinite preflight gradients')
        optimizer.step();losses.append(float(loss.detach()));print(f'PREFLIGHT {cfg["id"]} step={step+1} loss={float(loss.detach()):.4f}',flush=True)
    evidence=dict(losses=losses,largest_annotation_count=counts[dataset.images[index]['id']],
                  peak_memory_mib=torch.cuda.max_memory_allocated()/2**20,finite=True,
                  optimization_trend='decreasing' if np.mean(losses[-3:])<np.mean(losses[:3]) else 'needs_longer_overfit',debug=True)
    if projection_grad_norms:evidence['projection_grad_norms']=projection_grad_norms
    with torch.autocast('cuda',dtype=torch.bfloat16):
        if cfg['kind']!='sam':
            key,native=next(native_images());pred=model.predict(native)
            for instance in pred:
                from src.data.dataset import decode_mask
                assert instance['rle']['size']==[2048,2048] and decode_mask(instance['rle']).any()
            evidence['native_prediction_count']=len(pred)
    if cfg['kind']=='rf':atomic_torch(folder/'debug_probe.pt',dict(model=model.state_dict(),debug=True))
    if cfg.get('flow_version')==2:evidence['preflight_wall_seconds']=time.monotonic()-probe_started
    atomic_json(folder/'preflight.json',evidence)
    del model,optimizer,dataset;gc.collect();torch.cuda.empty_cache()
    return evidence

def train(cfg):
    directory=BASE/cfg['id'];directory.mkdir(parents=True,exist_ok=True)
    completed=directory/'completed.json'
    if completed.exists():return load_json(completed)
    last=directory/'last.pt'
    if (BASE/'stop_requested.json').exists():
        return {'status':'paused','route':cfg['id']}
    if not last.exists() and not (directory/'preflight.json').exists():
        if cfg['kind']=='rf':
            for size in [cfg['size']]:
                cfg['size']=size
                try:probe(cfg,directory);break
                except torch.OutOfMemoryError:
                    print(f'PREFLIGHT OOM size={size}; retry next predefined size',flush=True)
                    gc.collect();torch.cuda.empty_cache()
            else:raise RuntimeError('RF preflight OOM at fixed protocol resolution; do not silently reduce input size')
        else:probe(cfg,directory)
        atomic_json(directory/'config.json',cfg)
    elif last.exists():cfg=load_json(directory/'config.json')
    else:atomic_json(directory/'config.json',cfg)
    seed_everything(42)
    with gpu_window():model=build(cfg);optimizer=optimizer_for(model,cfg)
    dataset=FilamentDataset(split_path('train'),size=2048 if cfg['kind'] in ['sam','flow'] else cfg['size'])
    state=dict(epoch=0,offset=0,order=[],train_seconds=0.,best_pq=-1.,best_epoch=None,updates=0,
               train_split_sha256=digest(split_path('train')),val_split_sha256=digest(split_path('val')),validation_pending=False)
    if last.exists():
        saved=torch.load(last,map_location='cpu',weights_only=False)
        if saved['config']!=cfg:raise ValueError('Resume config mismatch')
        with gpu_window():
            model.load_state_dict(saved['model']);optimizer.load_state_dict(saved['optimizer'])
        state=saved['state'];restore_rng(saved['rng'])
        if state['train_split_sha256']!=digest(split_path('train')) or state['val_split_sha256']!=digest(split_path('val')):
            raise ValueError('Resume split mismatch')
        with gpu_window():
            for s in optimizer.state.values():
                for k,v in s.items():
                    if isinstance(v,torch.Tensor):s[k]=v.cuda()
        print(f'RESUMED epoch={state["epoch"]} offset={state["offset"]}',flush=True)
    prior_wait=state.get('gpu_window_wait_seconds',0.)
    initial_wait=gpu_wait_seconds()
    def checkpoint(path=last):
        state['gpu_window_wait_seconds']=prior_wait+gpu_wait_seconds()-initial_wait
        with gpu_window():
            atomic_torch(path,dict(model=model.state_dict(),optimizer=optimizer.state_dict(),state=state.copy(),config=cfg,rng=rng_state()))
    def finish_validation(writer,stream):
        epoch_number=state['epoch']+1
        metrics=evaluate(model,cfg,directory,epoch_number)
        row=dict(epoch=epoch_number,samples=state['offset'],updates=state['updates'],training_seconds=state['train_seconds'],pq=metrics['pq'],dice=metrics['dice'],iou=metrics['iou'])
        writer.writerow(row);stream.flush()
        if metrics['pq']>state['best_pq']:
            state['best_pq']=metrics['pq'];state['best_epoch']=epoch_number
            checkpoint(directory/'best.pt')
            atomic_json(directory/'best_metrics.json',metrics)
            atomic_json(directory/'best_predictions.json',load_json(directory/'validation_predictions.json'))
        state['validation_pending']=False
        if state['offset']==len(dataset):state['epoch']+=1;state['offset']=0;state['order']=[]
        checkpoint()
    def pause_if_requested():
        if not (BASE/'stop_requested.json').exists(): return False
        # Revalidate on resume if the saved boundary ended an evaluation epoch/budget.
        due=(state['epoch']+1)%cfg['eval_every']==0 or state['epoch']+1==cfg['epochs'] or state['train_seconds']>=cfg['hours']*3600
        if (state['offset']==len(dataset) and due) or state['train_seconds']>=cfg['hours']*3600:
            state['validation_pending']=True
        checkpoint()
        atomic_json(directory/'paused.json',dict(status='paused',time=time.time(),epoch=state['epoch'],offset=state['offset'],updates=state['updates']))
        print(f'PAUSED {cfg["id"]} safe checkpoint epoch={state["epoch"]} offset={state["offset"]}',flush=True)
        return True
    logpath=directory/'training_log.csv'
    with logpath.open('a',newline='',encoding='utf-8') as stream:
        writer=csv.DictWriter(stream,fieldnames=['epoch','samples','updates','training_seconds','pq','dice','iou'])
        if stream.tell()==0:writer.writeheader();stream.flush()
        if state.get('validation_pending'):finish_validation(writer,stream)
        if pause_if_requested():return {'status':'paused'}
        while state['epoch']<cfg['epochs'] and state['train_seconds']<cfg['hours']*3600:
            if not state['order']:
                state['order']=np.random.permutation(len(dataset)).tolist();state['offset']=0
            model.training_mode();optimizer.zero_grad(set_to_none=True);window=0;recent=[]
            for position in range(state['offset'],len(state['order'])):
                started=time.monotonic();item=sample_at(dataset,state['order'][position])
                preparation_seconds=time.monotonic()-started
                with gpu_window():
                    computation_started=time.monotonic()
                    with torch.autocast('cuda',dtype=torch.bfloat16):loss=model.loss(item)/cfg['accumulation']
                    if not torch.isfinite(loss):raise FloatingPointError('Nonfinite formal training loss')
                    loss.backward();window+=1;recent.append(float(loss)*cfg['accumulation'])
                    end=position+1==len(state['order'])
                    if window==cfg['accumulation'] or end:
                        if window<cfg['accumulation']:
                            for p in model.parameters():
                                if p.grad is not None:p.grad.mul_(cfg['accumulation']/window)
                        norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
                        if not torch.isfinite(norm):raise FloatingPointError('Nonfinite formal gradients')
                        progress=min(1.,(state['epoch']+(position+1)/len(dataset))/cfg['epochs'])
                        for group in optimizer.param_groups:
                            base_lr=group.setdefault('initial_lr',group['lr']);group['lr']=base_lr*(.1+.9*(1+np.cos(np.pi*progress))/2)
                        optimizer.step();optimizer.zero_grad(set_to_none=True);state['updates']+=1;window=0
                    torch.cuda.synchronize()
                    computation_seconds=time.monotonic()-computation_started
                state['train_seconds']+=preparation_seconds+computation_seconds;state['offset']=position+1
                if (position+1)%20==0 or end:
                    print(f'TRAIN {cfg["id"]} epoch={state["epoch"]+1}/{cfg["epochs"]} sample={position+1}/{len(dataset)} loss={np.mean(recent):.5f} train_h={state["train_seconds"]/3600:.3f}',flush=True)
                    recent=[]
                    atomic_json(directory/'progress.json',dict(stage='training',epoch=state['epoch']+1,sample=position+1,total=len(dataset),training_seconds=state['train_seconds'],time=time.time(),pid=os.getpid()))
                if window==0 and (position+1)%100==0:checkpoint()
                if window==0 and pause_if_requested():return {'status':'paused'}
                if window==0 and state['train_seconds']>=cfg['hours']*3600:break
            epoch_number=state['epoch']+1
            final=state['train_seconds']>=cfg['hours']*3600 or epoch_number==cfg['epochs']
            if epoch_number%cfg['eval_every']==0 or final:
                state['validation_pending']=True;checkpoint();finish_validation(writer,stream)
            elif state['offset']==len(dataset):state['epoch']+=1;state['offset']=0;state['order']=[]
            checkpoint()
            if pause_if_requested():return {'status':'paused'}
            if final:break
    result=dict(config=cfg,best_pq=state['best_pq'],best_epoch=state['best_epoch'],completed_epochs=state['epoch'],
                partial_epoch_samples=state['offset'],optimizer_updates=state['updates'],training_seconds=state['train_seconds'],
                train_split_sha256=state['train_split_sha256'],val_split_sha256=state['val_split_sha256'],status='completed',
                gpu_window_wait_seconds=prior_wait+gpu_wait_seconds()-initial_wait,
                scheduling=os.environ.get('KAGGLE_COOPERATIVE_GPU','0'),
                timing_definition='sample preparation plus exclusive computation; GPU-window waiting and validation excluded')
    atomic_json(completed,result)
    if cfg['kind']=='sam':
        saved=torch.load(directory/'best.pt',map_location='cpu',weights_only=False);model.load_state_dict(saved['model']);del saved
        from src.routes_v3.diagnostics import sam_oracle_diagnostic
        try:
            with gpu_window():sam_oracle_diagnostic(model)
        except Exception as exc:atomic_json(directory/'diagnostic_failure.json',dict(error=repr(exc),stage='oracle_diagnostic_only'))
    return result
