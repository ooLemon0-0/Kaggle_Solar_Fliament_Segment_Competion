"""Restore frozen splits from separately downloaded Kaggle annotations (stdlib only)."""
import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def build_split_bytes(original, spec):
    chosen = set(spec['metadata']['observation_ids'])
    images = [im for im in original['images'] if Path(im['file_name']).stem in chosen]
    found = {Path(im['file_name']).stem for im in images}
    if found != chosen:
        raise RuntimeError('Downloaded data does not contain all frozen observations')
    ids = {im['id'] for im in images}
    annotations = [a for a in original['annotations'] if a['image_id'] in ids]
    if len(images) != spec['annotation_groups'] or len(annotations) != spec['instances']:
        raise RuntimeError('Frozen split counts differ from downloaded annotations')
    value = {**original, 'images': images, 'annotations': annotations,
             'split_metadata': spec['metadata']}
    # Original Windows splits used CRLF; preserve those bytes on Linux too.
    data = json.dumps(value, indent=2, ensure_ascii=False).replace('\n', spec.get('newline', '\n')).encode('utf-8')
    if sha256(data) != spec['sha256']:
        raise RuntimeError('Reconstructed split SHA256 differs; refusing to change the protocol')
    return data


def restore(annotation_path, root=ROOT):
    root = Path(root)
    manifest = json.loads((root / 'configs/frozen_split_manifest.json').read_text(encoding='utf-8'))
    source = Path(annotation_path).read_bytes()
    if sha256(source) != manifest['source_annotation_sha256']:
        raise RuntimeError('Original Kaggle annotation SHA256 differs from the frozen dataset')
    original = json.loads(source)
    train = set(manifest['splits']['train']['metadata']['observation_ids'])
    val = set(manifest['splits']['val']['metadata']['observation_ids'])
    if train & val:
        raise RuntimeError('Train and validation observations overlap')
    pending = []
    # Validate every generated/existing file before writing any file.
    for name, spec in manifest['splits'].items():
        data = build_split_bytes(original, spec)
        target = root / 'output/splits' / (name + '.json')
        if target.exists():
            if sha256(target.read_bytes()) != spec['sha256']:
                raise RuntimeError(f'Refusing to overwrite a different existing split: {target}')
        else:
            pending.append((target, data))
    for target, data in pending:
        target.parent.mkdir(parents=True, exist_ok=True)
        fd, temporary = tempfile.mkstemp(prefix=target.name + '.', suffix='.tmp', dir=target.parent)
        try:
            with os.fdopen(fd, 'wb') as stream:
                stream.write(data)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, target)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)
    return {name: spec['sha256'] for name, spec in manifest['splits'].items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--annotation', type=Path, help='Path to original Kaggle train annotation JSON')
    args = parser.parse_args()
    annotation = args.annotation
    if annotation is None:
        matches = list((ROOT / 'data/raw').rglob('MAGFiLO_1.0_Annotations_kaggle2026_train.json'))
        if len(matches) != 1:
            raise RuntimeError(f'Expected one original annotation JSON, found {len(matches)}; download data first')
        annotation = matches[0]
    for name, digest in restore(annotation).items():
        print(f'FROZEN_SPLIT_OK {name} SHA256={digest}')


if __name__ == '__main__':
    main()
