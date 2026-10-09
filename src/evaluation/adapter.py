"""Validation/serialization adapter around the unchanged official evaluator."""
import numpy as np
import pandas as pd
import torch
from pycocotools import mask as mu

from src.data.dataset import annotation_rle, observation, decode_mask, encode_mask
from src.evaluation import official_pq as official
from src.utils.common import load_json, save_json, split_path, digest


def bounded_official_overlap(gt, pred, block_size=16):
    """Same official pairwise functions, chunked to bound 2048x2048 memory.

    Binary float32 sums are exact at this image size. No matching/scoring
    logic changes; return CPU matrices for the unchanged official PQ function.
    """
    device='cuda' if torch.cuda.is_available() else 'cpu'
    gt_groups=gt['filament_id'].str.split('_',n=1).str[0]
    pred_images=pred['filament_id'].str.split('_',n=1).str[0]
    rows=[]
    for group in gt_groups.unique():
        obs=group.split('-',1)[1]
        gt_rles=gt.loc[gt_groups==group,'segmentation_rle'].tolist()
        pred_rles=pred.loc[pred_images==obs,'segmentation_rle'].tolist()
        gt_layers=torch.from_numpy(official.rles_to_layers(gt_rles)).to(device)
        ious=[];dices=[]
        for start in range(0,max(1,len(pred_rles)),block_size):
            pred_layers=torch.from_numpy(official.rles_to_layers(pred_rles[start:start+block_size])).to(device)
            with torch.autocast(device_type=device,enabled=False):
                iou,dice=official.get_overlap_matrices(gt_layers,pred_layers)
            ious.append(iou.cpu());dices.append(dice.cpu())
            del pred_layers
        rows.append({'annotator_image':group,'iou_matrix':torch.cat(ious,1),'dice_matrix':torch.cat(dices,1),
                     'n_gt':len(gt_rles),'n_pred':len(pred_rles)})
        del gt_layers
    return pd.DataFrame(rows)


def serialize_prediction(prediction):
    return {'image_id': prediction['image_id'], 'instances': [{'rle':encode_mask(i['mask']), 'score':float(i['score'])} for i in prediction['instances']]}


def deserialize_prediction(prediction):
    return {'image_id':prediction['image_id'], 'instances':[{'mask':decode_mask(i['rle']), 'score':float(i['score'])} for i in prediction['instances']]}


def evaluate_predictions(prediction_path, debug=False):
    data = load_json(split_path('val', debug))
    saved = load_json(prediction_path)
    if saved['split_sha256'] != digest(split_path('val', debug)):
        raise ValueError('Prediction split fingerprint mismatch')
    predictions = saved['predictions']
    expected = {observation(i) for i in data['images']}
    actual = [p['image_id'] for p in predictions]
    if len(set(actual)) != len(actual) or set(actual) != expected:
        raise ValueError('Prediction observations must exactly cover the validation split')
    images = {im['id']:im for im in data['images']}
    gt_rows = []
    for a in data['annotations']:
        rle = annotation_rle(a, images[a['image_id']])
        gt_rows.append({'filament_id':a['image_id']+'_'+a['id'], 'segmentation_rle':rle['counts'].decode('ascii')})
    pred_rows = []
    pred_by_id = {p['image_id']:p for p in predictions}
    for p in predictions:
        for index, inst in enumerate(p['instances'], 1):
            if inst['rle']['size'] != [2048,2048] or not np.isfinite(inst['score']) or not 0 <= inst['score'] <= 1:
                raise ValueError('Invalid prediction mask size or score')
            if not decode_mask(inst['rle']).any():
                raise ValueError('Empty instance mask')
            pred_rows.append({'filament_id':p['image_id']+f'_{index}', 'segmentation_rle':inst['rle']['counts']})
    gt = pd.DataFrame(gt_rows, columns=['filament_id','segmentation_rle'])
    pred = pd.DataFrame(pred_rows, columns=['filament_id','segmentation_rle'])
    overlap = bounded_official_overlap(gt, pred)
    total = {'tp':0,'fp':0,'fn':0,'fragmentation':0,'over_merging':0}
    per_group = {}
    positive_dice, positive_iou = [], []
    for row in overlap.itertuples(index=False):
        hits = row.iou_matrix > .5
        tp = int(hits.sum())
        fp = row.n_pred if row.n_gt == 0 else official.fp_count_hit(hits)
        fn = row.n_gt if row.n_pred == 0 else official.fn_count_hit(hits)
        links = row.iou_matrix > 0
        fragmentation = int((links.sum(1)>1).sum())
        merging = int((links.sum(0)>1).sum())
        obs = row.annotator_image.split('-',1)[1]
        gt_rles = [{'size':[2048,2048],'counts':r['segmentation_rle']} for r in gt_rows if r['filament_id'].split('_',1)[0]==row.annotator_image]
        pred_rles = [i['rle'] for i in pred_by_id[obs]['instances']]
        gt_union = mu.decode(mu.merge(gt_rles)).astype(bool)
        pred_union = mu.decode(mu.merge(pred_rles)).astype(bool) if pred_rles else np.zeros_like(gt_union)
        intersection = np.logical_and(gt_union,pred_union).sum()
        a,b = gt_union.sum(),pred_union.sum()
        dice = float(2*intersection/(a+b)) if a+b else 1.
        iou = float(intersection/(a+b-intersection)) if a+b-intersection else 1.
        values = {'tp':tp,'fp':fp,'fn':fn,'fragmentation':fragmentation,'over_merging':merging}
        for key,value in values.items():
            total[key]+=value
        per_group[row.annotator_image] = {**values, 'pq':official.get_pq_score(pd.DataFrame([row._asdict()])), 'dice':dice,'iou':iou,'gt_count':row.n_gt,'pred_count':row.n_pred}
        positive_dice.extend(row.dice_matrix[row.iou_matrix>0].tolist())
        positive_iou.extend(row.iou_matrix[row.iou_matrix>0].tolist())
    result = {**saved['metadata'], **total, 'pq':official.get_pq_score(overlap),
              'dice':float(np.mean([v['dice'] for v in per_group.values()])),
              'iou':float(np.mean([v['iou'] for v in per_group.values()])),
              'official_positive_pair_dice_mean':float(np.mean(positive_dice)) if positive_dice else 0.,
              'official_positive_pair_iou_mean':float(np.mean(positive_iou)) if positive_iou else 0.,
              'split_sha256':saved['split_sha256'], 'debug':debug,
              'metric_definitions':{'pq':'unchanged official global PQ; IoU > 0.5; all annotator groups',
                                    'dice_iou':'mean foreground union pixel Dice/IoU per annotator group',
                                    'fragmentation':'count GT instances overlapping >1 predictions at IoU >0',
                                    'over_merging':'count predictions overlapping >1 GT instances at IoU >0'},
              'per_annotation_group':per_group}
    return result
