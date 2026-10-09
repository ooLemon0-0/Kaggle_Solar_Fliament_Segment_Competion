"""Read-only audit of the downloaded competition files; write audit reports only."""
import argparse
import collections
import concurrent.futures
import json
from pathlib import Path

import numpy as np
from PIL import Image
from pycocotools import mask as mask_util


def distribution(values):
    a = np.asarray(values, dtype=float)
    return dict(zip(['min', 'p25', 'median', 'p75', 'p95', 'max'], np.quantile(a, [0, .25, .5, .75, .95, 1]).tolist()))


def inspect_image(path):
    with Image.open(path) as image:
        result = {'file': str(path), 'size': list(image.size), 'mode': image.mode, 'format': image.format}
        image.verify()
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--project', type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    project = args.project.resolve()
    candidates = list((project / 'data/raw').rglob('MAGFiLO_1.0_Annotations_kaggle2026_train.json'))
    if len(candidates) != 1:
        raise RuntimeError(f'Expected one official annotation JSON, found {len(candidates)}')
    annotation_path = candidates[0]
    data_root = annotation_path.parent.parent
    train_dir = data_root / 'train/train_images'
    test_dir = data_root / 'test/test_images'
    train_files = sorted(train_dir.glob('*.jpeg'))
    test_files = sorted(test_dir.glob('*.jpeg'))
    d = json.loads(annotation_path.read_bytes())
    errors = []
    if not all(k in d for k in ('images', 'annotations', 'categories')):
        raise RuntimeError('Missing required COCO collections')
    images = {image['id']: image for image in d['images']}
    if len(images) != len(d['images']):
        errors.append('Duplicate annotation-group image IDs')
    annotation_ids = [a['id'] for a in d['annotations']]
    if len(set(annotation_ids)) != len(annotation_ids):
        errors.append('Duplicate annotation IDs')
    groups_by_observation = collections.defaultdict(list)
    for image in d['images']:
        observation = Path(image['file_name']).stem
        groups_by_observation[observation].append(image['id'])
        if image['id'].split('-', 1)[-1] != observation:
            errors.append(f'Observation identity mismatch: {image["id"]}')
        if not (train_dir / image['file_name']).is_file():
            errors.append(f'Missing training file: {image["file_name"]}')
        if (image['width'], image['height']) != (2048, 2048):
            errors.append(f'Unexpected JSON image dimensions: {image["id"]}')
    per_group = collections.Counter(a['image_id'] for a in d['annotations'])
    required = ['id', 'image_id', 'category_id', 'segmentation', 'area', 'spine', 'bbox', 'iscrowd']
    missing_fields = collections.Counter()
    categories = {c['id'] for c in d['categories']}
    area_values, widths, heights, aspects, spine_lengths = [], [], [], [], []
    segmentation_types, polygon_counts, crowd = collections.Counter(), collections.Counter(), collections.Counter()
    raster_areas, declared_area_deltas = [], []
    for annotation in d['annotations']:
        for name in required:
            if name not in annotation:
                missing_fields[name] += 1
        if any(name not in annotation for name in required):
            continue
        if annotation['image_id'] not in images:
            errors.append(f'Orphan annotation: {annotation["id"]}')
            continue
        image = images[annotation['image_id']]
        if annotation['category_id'] not in categories:
            errors.append(f'Unknown category: {annotation["id"]}')
        segmentation = annotation['segmentation']
        segmentation_types[type(segmentation).__name__] += 1
        crowd[annotation['iscrowd']] += 1
        if not isinstance(segmentation, list) or not segmentation:
            errors.append(f'Unsupported polygon segmentation: {annotation["id"]}')
            continue
        polygon_counts[len(segmentation)] += 1
        for polygon in segmentation:
            points = np.asarray(polygon, dtype=float)
            if len(points) < 6 or len(points) % 2 or not np.isfinite(points).all():
                errors.append(f'Invalid polygon: {annotation["id"]}')
        rle = mask_util.merge(mask_util.frPyObjects(segmentation, image['height'], image['width']))
        actual_area = int(mask_util.area(rle))
        if actual_area == 0:
            errors.append(f'Empty rasterized mask: {annotation["id"]}')
        raster_areas.append(actual_area)
        area_values.append(float(annotation['area']))
        declared_area_deltas.append(actual_area - float(annotation['area']))
        x, y, width, height = annotation['bbox']
        if width <= 0 or height <= 0:
            errors.append(f'Nonpositive bbox: {annotation["id"]}')
        widths.append(width)
        heights.append(height)
        aspects.append(width / height if height else 0)
        spine = np.asarray(annotation['spine'], dtype=float)
        if spine.size % 2:
            errors.append(f'Odd spine coordinate count: {annotation["id"]}')
        elif spine.size:
            spine = spine.reshape(-1, 2)
            spine_lengths.append(float(np.linalg.norm(np.diff(spine, axis=0), axis=1).sum()))
    if missing_fields:
        errors.append('Required annotation fields are missing')
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        headers = list(pool.map(inspect_image, train_files + test_files))
    modes = collections.Counter(h['mode'] for h in headers)
    sizes = collections.Counter(f'{h["size"][0]}x{h["size"][1]}' for h in headers)
    if any(h['size'] != [2048, 2048] for h in headers):
        errors.append('JPEG dimensions are not uniformly 2048x2048')
    train_ids = {p.stem for p in train_files}
    test_ids = {p.stem for p in test_files}
    overlap = sorted(train_ids & test_ids)
    if overlap:
        errors.append('Train/test observation filename overlap')
    if train_ids != set(groups_by_observation):
        errors.append('Training files and annotated observations differ')
    counts = [per_group.get(image['id'], 0) for image in d['images']]
    summary = {
        'dataset_directory': str(data_root), 'train_images_directory': str(train_dir),
        'test_images_directory': str(test_dir), 'annotation_json': str(annotation_path),
        'train_image_files': len(train_files), 'test_image_files': len(test_files),
        'annotation_group_records': len(d['images']), 'unique_train_observations': len(groups_by_observation),
        'annotations': len(d['annotations']), 'categories': d['categories'],
        'target': 'single-class filament instance segmentation; chirality is not a prediction target',
        'observation_group_count_distribution': dict(collections.Counter(len(v) for v in groups_by_observation.values())),
        'observations_with_multiple_annotation_groups': sum(len(v) > 1 for v in groups_by_observation.values()),
        'station_distribution_unique_observations': dict(collections.Counter(k[-2:] for k in groups_by_observation)),
        'station_distribution_annotation_groups': dict(collections.Counter(Path(i['file_name']).stem[-2:] for i in d['images'])),
        'instance_count_per_annotation_group': distribution(counts),
        'annotation_groups_with_no_instances': sum(v == 0 for v in counts),
        'declared_area_pixels': distribution(area_values), 'rasterized_mask_area_pixels': distribution(raster_areas),
        'declared_area_vs_raster_area_nonzero_deltas': sum(v != 0 for v in declared_area_deltas),
        'bbox_width_pixels': distribution(widths), 'bbox_height_pixels': distribution(heights),
        'bbox_width_height_ratio': distribution(aspects), 'spine_length_pixels': distribution(spine_lengths),
        'segmentation_types': dict(segmentation_types), 'polygons_per_instance': dict(polygon_counts),
        'iscrowd': dict(crowd), 'required_annotation_fields': required,
        'missing_field_counts': dict(missing_fields), 'train_test_observation_overlap': overlap,
        'jpeg_modes': dict(modes), 'jpeg_sizes': dict(sizes), 'jpeg_header_verification': 'PASS',
        'sample_submission_files': [str(p) for p in (project / 'data/raw').rglob('*submission*') if p.is_file()],
        'split_rule': 'Group by file_name stem (original observation), keeping all annotator image records together; seed 42, 80/20; split not created in this resource-audit stage.',
        'errors': errors, 'status': 'PASS' if not errors else 'FAIL',
    }
    output = project / 'output/data_analysis'
    output.mkdir(parents=True, exist_ok=True)
    (output / 'dataset_summary.json').write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding='utf-8')
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    if errors:
        raise RuntimeError(f'Data audit found {len(errors)} errors')


if __name__ == '__main__':
    main()
