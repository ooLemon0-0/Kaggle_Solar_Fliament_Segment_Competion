import _bootstrap
import argparse
import csv
from pathlib import Path
import numpy as np
from PIL import Image

from src.data.dataset import FilamentDataset, observation, mask_from_annotation
from src.evaluation.adapter import deserialize_prediction
from src.utils.common import ROOT, MODELS, config_for, load_json, split_path, save_json
from src.utils.visualization import panel, join_panels


def compare(debug=False):
    metrics={name:load_json(Path(config_for(name,debug)['run_dir'])/'metrics.json') for name in MODELS}
    assert len({m['split_sha256'] for m in metrics.values()})==1,'Different validation splits'
    assert len({(m['train_size'],m['inference_size'],m['number_images'],m['training_epochs']) for m in metrics.values()})==1,'Different experimental budgets'
    target=ROOT/'output/benchmark'/('debug' if debug else 'full');target.mkdir(parents=True,exist_ok=True)
    fields=['model','pq','dice','iou','tp','fp','fn','fragmentation','over_merging','inference_time','gpu_memory_mib','training_gpu_peak_mib','parameter_count','training_epochs','debug']
    rows=[{key:metrics[name][key] for key in fields} for name in MODELS]
    for dest in [target/'results.csv',ROOT/'output/benchmark/results.csv']:
        with dest.open('w',newline='',encoding='utf-8') as f:
            writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader();writer.writerows(rows)
    dataset=FilamentDataset(split_path('val',debug))
    # Keep RLEs compressed until the displayed observation; full-val dense masks
    # across all three models can otherwise consume tens of GiB of CPU memory.
    predicted={name:{p['image_id']:p for p in load_json(Path(config_for(name,debug)['run_dir'])/'predictions/val.json')['predictions']} for name in MODELS}
    selected={}
    for image in dataset.images:
        selected.setdefault(observation(image),image)
    for obs,image in selected.items():
        with Image.open(dataset.image_dir/image['file_name']) as src:gray=src.convert('L')
        anns=dataset.annotations[image['id']];gt=[mask_from_annotation(a,image) for a in anns]
        panels=[panel(gray,[],f'Original | {obs}'),panel(gray,gt,f"GT {len(gt)} | {image['id']}",anns)]
        zoom_source=[([], 'Original'),(gt,f'GT {len(gt)}')]
        for name in MODELS:
            pred=deserialize_prediction(predicted[name][obs])['instances'];masks=[i['mask'] for i in pred];score=metrics[name]['per_annotation_group'][image['id']]
            text=f"{name} | Pred {len(masks)}\nPQ {score['pq']:.4f} | Dice {score['dice']:.4f}"
            panels.append(panel(gray,masks,text));zoom_source.append((masks,text))
        full=Image.new('RGB',(2560,1120),'white')
        for index,p in enumerate(panels):full.paste(p,(index*512,0))
        largest=max(anns,key=lambda a:a['area']);x,y,w,h=largest['bbox']
        x0=max(0,int(x)-40);y0=max(0,int(y)-40);x1=min(2048,int(x+w)+40);y1=min(2048,int(y+h)+40)
        crop=gray.crop((x0,y0,x1,y1))
        for index,(masks,text) in enumerate(zoom_source):
            full.paste(panel(crop,[m[y0:y1,x0:x1] for m in masks],text+' | visual zoom'),(index*512,560))
        path=target/f'visualizations/{obs}.png';path.parent.mkdir(parents=True,exist_ok=True);full.save(path)
        canonical=ROOT/'output/benchmark/visualizations';canonical.mkdir(parents=True,exist_ok=True)
        full.save(canonical/f"{'debug' if debug else 'full'}_{obs}.png")
    lines=['# Three-model baseline comparison', '',
           '**Debug smoke run, not a trained-model ranking.**' if debug else '**Full training run.**','',
           'All models use the same saved observation split, 1024 training/inference input, native 2048 evaluation masks, FP16/AMP inference and one GPU warmup. Predictions are capped at 64 instances, retaining score order; all models filter masks smaller than 32 native pixels. No tiling or chirality supervision.', '',
           f"Validation: {len(selected)} original observations, {len(dataset)} independent annotator groups. Training epochs: {rows[0]['training_epochs']}.", '',
           '| model | PQ | Dice | IoU | TP/FP/FN | fragmentation | over-merging | train peak MiB | infer peak MiB | ms/image | parameters |',
           '|---|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|']
    for r in rows:
        lines.append(f"| {r['model']} | {r['pq']:.6f} | {r['dice']:.6f} | {r['iou']:.6f} | {r['tp']}/{r['fp']}/{r['fn']} | {r['fragmentation']} | {r['over_merging']} | {r['training_gpu_peak_mib']:.1f} | {r['gpu_memory_mib']:.1f} | {1000*r['inference_time']:.1f} | {r['parameter_count']:,} |")
    lines+=['','PQ is the unchanged official global score at IoU > 0.5, comparing each observation prediction separately with every annotator group. Dice/IoU are mean union-foreground pixel scores per annotator group; official positive pair score means are additionally saved in metrics.json. Fragmentation counts GT with >1 overlapping predictions, over-merging counts predictions overlapping >1 GT, at the official relationship threshold IoU >0.', '',
            f'Inference latency includes image reading, preprocessing, forward and instance postprocessing; excludes model loading, plotting, CSV/RLE serialization, and official evaluation. CUDA timings are synchronized and measured on all {len(selected)} validation observations. Training peak and inference peak are allocated PyTorch GPU memory, not total nvidia-smi process memory. Parameter counts include frozen backbone parameters (YOLO counted before inference fusion).', '',
            'YOLO-Seg: mature direct instance baseline with pretrained COCO weights. Masks are native-size segmentation results. Thin-filament recall and false positives should be inspected on validation images. Best weights are selected by official PQ.', '',
            'U-Net: mature ImageNet ResNet50 encoder, BCEWithLogits + Dice. Connected components are simple and reproducible, but can fragment one filament or merge neighboring filaments. Raw component counts before the 64-instance cap are retained in metrics.json. Morphology is disabled by default.', '',
            'DINO + Transformer: frozen local DINOv3 ViT-S/16 with a 3-layer, 128-dimensional query decoder, 64 queries, objectness and dense 256x256 mask logits. Hungarian matching is used only for training; evaluation uses official matching. The new decoder needs substantially more training, and coarse masks can lose thin branches.']
    for name in MODELS:
        summary=load_json(Path(config_for(name,debug)['run_dir'])/'submission_summary.json')
        lines+=['',f"{name} submission: {summary['test_images_covered']} test observations processed; {summary['instance_rows']} instance rows; {len(summary['zero_prediction_images'])} images with zero detections."]
    lines+=['','Images with zero detections contribute no instance rows; no fake filament rows are added. Files were generated and sampled RLE roundtrips verified, but were not uploaded to Kaggle.', '',
            'The figure top row shows the whole disk, and bottom row enlarges the largest GT filament region for visual inspection. This crop is visualization only, not model tiling. Each figure uses one displayed annotator GT, identified in its title; global metrics use all annotators.', '',
            ('Next experiment: train all three baselines on the complete saved train split, using the provided 30-epoch configs. Avoid drawing an architectural ranking from the one-epoch debug numbers.' if debug else 'This is a single-seed baseline comparison, not a general architectural ranking. Pretraining, trainable parameters and learning-rate schedules differ across models. Next investigate instance false positives, fragmentation and overlapping queries on val.'),
            'Tune thresholds only on val; retain the same val observations and original-resolution official scoring. Do not tune on test predictions.']
    text='\n'.join(lines)+'\n'
    (target/'comparison.md').write_text(text,encoding='utf-8');(ROOT/'output/benchmark/comparison.md').write_text(text,encoding='utf-8')
    save_json(target/'manifest.json',{'debug':debug,'split_sha256':rows and metrics[MODELS[0]]['split_sha256'],'validation_observations':sorted(selected),'models':MODELS,'epochs':rows[0]['training_epochs']})
    print(target/'comparison.md',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--debug',action='store_true');a=p.parse_args();compare(a.debug)
