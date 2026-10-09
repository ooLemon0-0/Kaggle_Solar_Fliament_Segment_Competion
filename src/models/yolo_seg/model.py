import shutil
from pathlib import Path

import yaml

from src.data.dataset import discover
from src.utils.common import ROOT, load_json, split_path, save_json


def convert(debug=False):
    base = ROOT / 'output/data_analysis/yolo' / ('debug' if debug else 'full')
    image_dir = discover()[1]
    for split in ['train','val']:
        data = load_json(split_path(split,debug))
        by_image = {i['id']:[] for i in data['images']}
        dimensions = {i['id']:(i['width'],i['height']) for i in data['images']}
        for annotation in data['annotations']:
            w,h = dimensions[annotation['image_id']]
            # The official audit verified one polygon for every V0 instance.
            if len(annotation['segmentation'])!=1:
                raise ValueError('Multi-polygon instances require an explicit conversion policy')
            polygon=annotation['segmentation'][0]
            normalized=[max(0.,min(1.,v/(w if j%2==0 else h))) for j,v in enumerate(polygon)]
            by_image[annotation['image_id']].append('0 '+' '.join(f'{v:.8f}' for v in normalized))
        images=base/f'images/{split}';labels=base/f'labels/{split}'
        images.mkdir(parents=True,exist_ok=True);labels.mkdir(parents=True,exist_ok=True)
        for im in data['images']:
            target=images/(im['id']+'.jpeg')
            if not target.exists():
                shutil.copy2(image_dir/im['file_name'],target)
            (labels/(im['id']+'.txt')).write_text('\n'.join(by_image[im['id']])+'\n',encoding='utf-8')
    path=base/'dataset.yaml'
    path.write_text(yaml.safe_dump({'path':str(base),'train':'images/train','val':'images/val','names':{0:'filament'}}),encoding='utf-8')
    return path


def train_yolo(cfg):
    import torch
    from ultralytics import YOLO
    run=Path(cfg['run_dir']);run.mkdir(parents=True,exist_ok=True)
    data=convert(cfg['debug'])
    model=YOLO(cfg['weights'])
    selection={'best_pq':-1.,'best_epoch':None,'training_gpu_peak_mib':0.,'last_evaluated_epoch':-1}
    official_log=[]
    (run/'checkpoints').mkdir(exist_ok=True)
    def epoch_start(trainer):
        torch.cuda.reset_peak_memory_stats()
    def epoch_end(trainer):
        selection['training_gpu_peak_mib']=max(selection['training_gpu_peak_mib'],torch.cuda.max_memory_allocated()/1048576)
    def official_validation(trainer):
        if selection['last_evaluated_epoch']==trainer.epoch:
            return
        from src.pipeline import infer, evaluate
        infer(cfg,checkpoint=trainer.last)
        metrics=evaluate(cfg)
        official_log.append({'epoch':trainer.epoch+1,'official_pq':metrics['pq']})
        if metrics['pq']>selection['best_pq']:
            selection['best_pq']=metrics['pq'];selection['best_epoch']=trainer.epoch+1
            shutil.copy2(trainer.last,run/'checkpoints/best.pt')
        selection['last_evaluated_epoch']=trainer.epoch
        save_json(run/'official_validation_log.json',official_log)
        print('Official validation:',official_log[-1],flush=True)
    model.add_callback('on_train_epoch_start',epoch_start)
    model.add_callback('on_train_epoch_end',epoch_end)
    model.add_callback('on_fit_epoch_end',official_validation)
    model.train(data=str(data),epochs=cfg['epochs'],imgsz=cfg['image_size'],batch=cfg['batch_size'],
                device=0,workers=0,seed=cfg['seed'],deterministic=True,amp=True,pretrained=True,
                project=str(run),name='training',exist_ok=True,save=True,plots=False,
                mosaic=0,mixup=0,copy_paste=0,fliplr=0,flipud=0,hsv_h=0,hsv_s=0,hsv_v=0,
                translate=0,scale=0,auto_augment=None,erasing=0,overlap_mask=False,mask_ratio=1,
                val=True,optimizer='AdamW',lr0=cfg['lr'],close_mosaic=0)
    shutil.copy2(run/'training/weights/last.pt',run/'checkpoints/last.pt')
    save_json(run/'training_summary.json',{'training_gpu_peak_mib':selection['training_gpu_peak_mib'],
              'best_epoch':selection['best_epoch'],'checkpoint_selection':'official global validation PQ',
              'epochs_completed':len(official_log)})
    shutil.copy2(run/'training/results.csv',run/'training_log.csv')
    return model
