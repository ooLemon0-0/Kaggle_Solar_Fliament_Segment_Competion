import hashlib
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('restore_frozen_splits', ROOT / 'scripts/restore_frozen_splits.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class FrozenSplitTests(unittest.TestCase):
    def test_restore_preserves_all_annotators_and_refuses_overwrite(self):
        original = {'images': [{'id': 1, 'file_name': 'a.jpeg'}, {'id': 2, 'file_name': 'b.jpeg'},
                               {'id': 3, 'file_name': 'a.jpeg'}],
                    'annotations': [{'id': 4, 'image_id': 1}, {'id': 5, 'image_id': 2}, {'id': 6, 'image_id': 3}],
                    'categories': [{'id': 1, 'name': 'example'}]}
        splits = {}
        for name, chosen in [('train', ['a']), ('val', ['b'])]:
            metadata = {'seed': 42, 'observation_ids': chosen, 'group_key': 'file_name stem', 'split': name}
            images = [im for im in original['images'] if Path(im['file_name']).stem in chosen]
            ids = {im['id'] for im in images}
            annotations = [a for a in original['annotations'] if a['image_id'] in ids]
            data = json.dumps({**original, 'images': images, 'annotations': annotations,
                               'split_metadata': metadata}, indent=2, ensure_ascii=False).encode()
            splits[name] = {'metadata': metadata, 'annotation_groups': len(images),
                            'instances': len(annotations), 'sha256': hashlib.sha256(data).hexdigest()}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'configs').mkdir()
            source = root / 'annotations.json'
            source.write_text(json.dumps(original), encoding='utf-8')
            manifest = {'source_annotation_sha256': hashlib.sha256(source.read_bytes()).hexdigest(), 'splits': splits}
            (root / 'configs/frozen_split_manifest.json').write_text(json.dumps(manifest), encoding='utf-8')
            module.restore(source, root)
            train_path = root / 'output/splits/train.json'
            train = json.loads(train_path.read_bytes())
            self.assertEqual([im['id'] for im in train['images']], [1, 3])
            self.assertEqual([a['id'] for a in train['annotations']], [4, 6])
            before = train_path.stat().st_mtime_ns
            module.restore(source, root)
            self.assertEqual(train_path.stat().st_mtime_ns, before)
            train_path.write_bytes(b'other split')
            val_path = root / 'output/splits/val.json'
            val_path.unlink()
            with self.assertRaisesRegex(RuntimeError, 'Refusing to overwrite'):
                module.restore(source, root)
            self.assertEqual(train_path.read_bytes(), b'other split')
            self.assertFalse(val_path.exists())
            source.write_bytes(b'{}')
            with self.assertRaisesRegex(RuntimeError, 'annotation SHA256'):
                module.restore(source, root)


if __name__ == '__main__':
    unittest.main()
