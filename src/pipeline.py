import csv
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from src.data.dataset import discover, observation
from src.evaluation.adapter import serialize_prediction, deserialize_prediction, evaluate_predictions
from src.utils.common import ROOT, config_for, load_json, save_json, split_path, digest, seed_everything, sync
from src.utils.visualization import panel, join_panels


def create_model(cfg, pretrained=True):
    import importlib
    module = importlib.import_module('src.models.' + cfg['model'] + '.model')
    return module.build_model(cfg, pretrained)


def load_model(cfg, checkpoint=None):
    checkpoint=Path(checkpoint or Path(cfg['run_dir'])/'checkpoints/best.pt')
    if cfg['model']=='yolo_seg':
        from ultralytics import YOLO
        model=YOLO(str(checkpoint));model.model.cuda()
        return model
    model=create_model(cfg,pretrained=False).cuda()
    state=torch.load(checkpoint,map_location='cpu',weights_only=False)
    model.load_state_dict(state['model'])
    return model.eval()


def infer(cfg, split='val', checkpoint=None, model=None):
    model=model or load_model(cfg,checkpoint)
    if split=='test':
        files=sorted(discover()[2].glob('*.jpeg'));fingerprint=None
    else:
        data=load_json(split_path(split,cfg['debug']))
        files=[discover()[1]/name for name in sorted({i['file_name'] for i in data['images']})]
        fingerprint=digest(split_path(split,cfg['debug']))
    parameters=sum(p.numel() for p in (model.model if cfg['model']=='yolo_seg' else model).parameters())
    # One GPU warmup per model; all measured runs use FP16/AMP consistently.
    if files:
        if cfg['model']=='yolo_seg':
            model.predict(str(files[0]),imgsz=cfg['inference_size'],conf=cfg['confidence'],max_det=cfg['max_instances'],retina_masks=True,half=True,device=0,verbose=False)
        else:
            with Image.open(files[0]) as source:
                gray=np.asarray(source.convert('L').resize((cfg['inference_size'],cfg['inference_size']),Image.Resampling.BILINEAR),dtype=np.float32)/255.
            warmup=torch.from_numpy(np.repeat(gray[None],3,axis=0)).unsqueeze(0).cuda()
            with torch.inference_mode(),torch.autocast('cuda',dtype=torch.float16):
                model(warmup)
            del warmup
        sync()
    torch.cuda.reset_peak_memory_stats()
    predictions=[];times=[];raw_counts={}
    for path in files:
        sync();started=time.perf_counter()
        if cfg['model']=='yolo_seg':
            result=model.predict(str(path),imgsz=cfg['inference_size'],conf=cfg['confidence'],max_det=cfg['max_instances'],retina_masks=True,half=True,device=0,verbose=False)[0]
            masks=[] if result.masks is None else result.masks.data.cpu().numpy()>.5
            scores=[] if result.masks is None else result.boxes.conf.cpu().numpy()
            instances=[{'mask':m,'score':float(s)} for m,s in zip(masks,scores) if m.sum()>=cfg['min_area']]
        else:
            with Image.open(path) as source:
                array=np.asarray(source.convert('L').resize((cfg['inference_size'],cfg['inference_size']),Image.Resampling.BILINEAR),dtype=np.float32)/255.
            tensor=torch.from_numpy(np.repeat(array[None],3,axis=0)).unsqueeze(0).cuda()
            with torch.inference_mode(), torch.autocast('cuda',dtype=torch.float16):
                instances=model.predict_instances(tensor,cfg)
        raw_counts[path.stem]=getattr(model,'raw_instance_count',len(instances))
        instances=sorted(instances,key=lambda i:i['score'],reverse=True)[:cfg['max_instances']]
        sync();times.append(time.perf_counter()-started)
        prediction={'image_id':path.stem,'instances':instances}
        predictions.append(serialize_prediction(prediction))
        if split=='test' and len(predictions)%20==0:
            print(f"{cfg['model']} test inference {len(predictions)}/{len(files)}",flush=True)
        if split=='val':
            with Image.open(path) as source:
                join_panels([panel(source,[],path.stem),panel(source,[i['mask'] for i in instances],f"{cfg['model']} | Pred {len(instances)}")],Path(cfg['run_dir'])/f'visualizations/{path.stem}.png')
    saved={'predictions':predictions,'split_sha256':fingerprint,'metadata':{'model':cfg['model'],
           'parameter_count':parameters,'gpu_memory_mib':torch.cuda.max_memory_allocated()/1048576,
           'inference_time':float(np.mean(times)),'inference_seconds_per_image':dict(zip([p.stem for p in files],times)),
           'train_size':cfg['image_size'],'inference_size':cfg['inference_size'],'number_images':len(files),
           'timing_definition':'image read/preprocess + GPU forward + instance postprocess; synchronized; excludes loading model, plotting and RLE serialization; one GPU warmup',
           'training_epochs':cfg['epochs'],'debug':cfg['debug']}}
    saved['metadata'].update(inference_precision='FP16 / AMP',warmup_images=1)
    saved['metadata'].update(max_instances=cfg['max_instances'],min_area=cfg['min_area'],raw_instance_counts_before_cap=raw_counts)
    output=Path(cfg['run_dir'])/f'predictions/{split}.json';save_json(output,saved)
    return output


def evaluate(cfg):
    metrics=evaluate_predictions(Path(cfg['run_dir'])/'predictions/val.json',cfg['debug'])
    training=Path(cfg['run_dir'])/'training_summary.json'
    if training.exists():
        metrics.update(load_json(training))
    save_json(Path(cfg['run_dir'])/'metrics.json',metrics)
    return metrics


def train(cfg):
    seed_everything(cfg['seed'])
    run=Path(cfg['run_dir']);run.mkdir(parents=True,exist_ok=True)
    cfg['train_split_sha256']=digest(split_path('train',cfg['debug']))
    cfg['val_split_sha256']=digest(split_path('val',cfg['debug']))
    save_json(run/'config.json',cfg)
    if cfg['model']=='yolo_seg':
        from src.models.yolo_seg.model import train_yolo
        train_yolo(cfg)
        prediction=infer(cfg)
        metrics=evaluate(cfg)
    else:
        from src.training import train_torch
        metrics=train_torch(cfg)
    print({key:metrics[key] for key in ['model','pq','dice','iou','tp','fp','fn']},flush=True)
    return metrics


def submission(cfg, checkpoint=None):
    prediction_path=infer(cfg,'test',checkpoint)
    saved=load_json(prediction_path)
    path=Path(cfg['run_dir'])/'submission.csv'
    with path.open('w',newline='',encoding='utf-8') as f:
        writer=csv.writer(f);writer.writerow(['filament_id','segmentation_rle'])
        for p in saved['predictions']:
            for n,instance in enumerate(p['instances'],1):
                writer.writerow([p['image_id']+f'_{n}',instance['rle']['counts']])
    save_json(Path(cfg['run_dir'])/'submission_summary.json',{'test_images_covered':len(saved['predictions']),
               'zero_prediction_images':[p['image_id'] for p in saved['predictions'] if not p['instances']],
               'instance_rows':sum(len(p['instances']) for p in saved['predictions']),
               'note':'Images with zero detections contribute no instance rows; no fabricated empty filament.'})
    print(path,flush=True)
    return path
