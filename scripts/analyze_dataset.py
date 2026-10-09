import _bootstrap
import csv
import random
from collections import defaultdict, Counter

import numpy as np

from audit_official_resources import main as audit
from src.data.dataset import discover, observation, annotation_rle
from src.utils.common import ROOT, load_json, save_json
from pycocotools import mask as mu


def make_splits():
    original = load_json(discover()[0])
    groups = defaultdict(list)
    for image in original['images']:
        groups[observation(image)].append(image)
    identities = sorted(groups)
    random.Random(42).shuffle(identities)
    nval = int(np.ceil(.2 * len(identities)))
    sets = {'train': set(identities[nval:]), 'val': set(identities[:nval])}
    assert not sets['train'] & sets['val']
    for name, chosen in list(sets.items()):
        # Debug subsets retain every annotator of each chosen observation.
        sets['debug_' + name] = set(sorted(chosen)[:8 if name == 'train' else 4])
    for name, chosen in sets.items():
        images = [im for im in original['images'] if observation(im) in chosen]
        ids = {im['id'] for im in images}
        value = {**original, 'images': images, 'annotations': [a for a in original['annotations'] if a['image_id'] in ids],
                 'split_metadata': {'seed': 42, 'observation_ids': sorted(chosen), 'group_key': 'file_name stem', 'split': name}}
        path = ROOT / f'output/splits/{name}.json'
        if path.exists() and load_json(path) != value:
            raise RuntimeError(f'Refusing to replace an existing different split: {path}')
        save_json(path, value)
    with (ROOT / 'output/data_analysis/instance_statistics.csv').open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['annotation_id','annotation_group','observation','station','split','mask_area','bbox_width','bbox_height','aspect_ratio','spine_length'])
        writer.writeheader()
        image_by_id = {im['id']: im for im in original['images']}
        for a in original['annotations']:
            im = image_by_id[a['image_id']]
            obs = observation(im)
            spine = np.asarray(a['spine']).reshape(-1, 2)
            w, h = a['bbox'][2:]
            writer.writerow({'annotation_id':a['id'],'annotation_group':im['id'],'observation':obs,'station':obs[-2:],
                             'split':'train' if obs in sets['train'] else 'val','mask_area':int(mu.area(annotation_rle(a,im))),
                             'bbox_width':w,'bbox_height':h,'aspect_ratio':w/h,'spine_length':np.linalg.norm(np.diff(spine,axis=0),axis=1).sum()})
    counts = {name: {'observations':len(ids),'annotation_groups':sum(observation(i) in ids for i in original['images'])} for name,ids in sets.items()}
    save_json(ROOT/'output/data_analysis/split_summary.json', counts)
    summary_path=ROOT/'output/data_analysis/dataset_summary.json'
    summary=load_json(summary_path)
    summary.update(splits=counts,split_rule='Saved seed 42 observation-group 80/20 split; debug subsets preserve all annotators.')
    save_json(summary_path,summary)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    with (ROOT/'output/data_analysis/instance_statistics.csv').open(encoding='utf-8') as f:
        statistics=list(csv.DictReader(f))
    fig,axes=plt.subplots(2,3,figsize=(13,7))
    for ax,key in zip(axes.flat,['mask_area','bbox_width','bbox_height','aspect_ratio','spine_length','instance_count']):
        values=np.asarray([float(row[key]) for row in statistics]) if key!='instance_count' else np.asarray(list(Counter(a['image_id'] for a in original['annotations']).values()))
        bins=np.geomspace(values.min(),values.max(),32) if key=='mask_area' else 30
        ax.hist(values,bins=bins,color='#315b80',alpha=.85)
        if key=='mask_area':ax.set_xscale('log')
        ax.set_title(key);ax.set_ylabel('Count');ax.grid(alpha=.2)
    fig.suptitle('MAGFiLO official train annotations | native 2048 pixels | all annotators')
    fig.tight_layout();fig.savefig(ROOT/'output/data_analysis/distributions.png',dpi=150);plt.close(fig)
    print(counts)


if __name__ == '__main__':
    audit()
    make_splits()
