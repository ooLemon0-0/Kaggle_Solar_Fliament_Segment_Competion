import importlib.util
import json
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('rf_resolution', ROOT / 'scripts/run_rf_resolution.py')
experiment = importlib.util.module_from_spec(spec)
spec.loader.exec_module(experiment)


class ResolutionTests(unittest.TestCase):
    def test_resolution_only_changes_input_and_output_id(self):
        frozen = json.loads((ROOT / 'linux/integrity.json').read_text())
        original = next(c for c in frozen['runs'] if c['id'] == 'rf')
        actual = experiment.configuration(original.copy(), original)
        self.assertEqual({k for k in original if original[k] != actual[k]}, {'size', 'id'})
        self.assertEqual(actual['size'], 1536)
        self.assertIsNone(actual['dependency'])
        wrong = dict(original, lr=.003)
        with self.assertRaises(RuntimeError):
            experiment.configuration(wrong, original)

    def test_truncated_training_cannot_pass_resolution_gate(self):
        self.assertIn('truncated', experiment.decision(.03, 18, True))
        self.assertIn('confirm_thin', experiment.decision(.0051, 30, True))
        self.assertIn('hold', experiment.decision(.003, 30, True))
        self.assertEqual(experiment.decision(.03, 30, False), 'blocked_artifact_checks')

    def test_crashed_session_cannot_reset_wall_budget(self):
        self.assertEqual(experiment.recovered_wall({'wall_seconds':100}, 1000), 100)
        self.assertEqual(experiment.recovered_wall({'wall_seconds':100, 'active_since':900}, 1000), 200)
        self.assertEqual(experiment.recovered_wall({'wall_seconds':100, 'active_since':1100}, 1000), 100)

    def fixture(self, base):
        source, active = base / 'source', base / 'active'
        integrity = json.loads((ROOT / 'linux/integrity.json').read_text())
        source.mkdir(); active.mkdir()
        frozen = next(c for c in integrity['runs'] if c['id'] == 'rf')
        for name in integrity['immutable_files']:
            owner = active if name.startswith('output/splits/') else source
            target = owner / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(b'frozen fixture')
            integrity['immutable_files'][name] = experiment.sha(target)
        def save(root, name, value):
            target = root / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(json.dumps(value))
        save(source, 'linux/integrity.json', integrity)
        save(active, 'output/benchmark/routes_v3/experiment_manifest.json', {
            k+'_split_sha256':integrity['immutable_files']['output/splits/'+k+'.json'] for k in ['train','val']})
        save(active, 'output/benchmark/routes_v3/pipeline_status.json', {
            'stage':'completed', 'routes':{c['id']:'completed' for c in integrity['runs']}})
        save(active, 'output/benchmark/routes_v3/rf/config.json', frozen)
        save(active, 'output/benchmark/routes_v3/rf/completed.json', {'status':'completed','config':frozen})
        save(active, 'output/benchmark/routes_v3/rf/best_metrics.json', {'pq':.41})
        annotation = active / 'data/raw/train/MAGFiLO_1.0_Annotations_kaggle2026_train.json'
        annotation.parent.mkdir(parents=True); annotation.write_bytes(b'frozen annotations')
        save(source, 'linux/data_fingerprints.json', {
            'train/MAGFiLO_1.0_Annotations_kaggle2026_train.json':experiment.sha(annotation)})
        weight = active / 'output/pretrained/routes_v3/rf/rf-detr-seg-small.pt'
        weight.parent.mkdir(parents=True); weight.write_bytes(b'frozen pretrained')
        save(source, 'expected_assets.json', {'files':{
            'output/pretrained/routes_v3/rf/rf-detr-seg-small.pt':{'sha256':experiment.sha(weight)}}})
        return source, active

    def test_immutable_inputs_and_completion_are_checked_before_gpu(self):
        with tempfile.TemporaryDirectory() as directory:
            source, active = self.fixture(Path(directory))
            with patch.object(experiment, 'ROOT', source):
                self.assertEqual(experiment.inspect_inputs(active)[0]['size'],1536)
                split = active / 'output/splits/val.json'
                split.write_bytes(b'changed')
                with self.assertRaisesRegex(RuntimeError, 'Immutable split'):
                    experiment.inspect_inputs(active)

    def test_changed_pretrained_and_stopped_original_block(self):
        with tempfile.TemporaryDirectory() as directory:
            source, active = self.fixture(Path(directory))
            with patch.object(experiment, 'ROOT', source):
                weight = active / 'output/pretrained/routes_v3/rf/rf-detr-seg-small.pt'
                weight.write_bytes(b'other pretrained')
                with self.assertRaisesRegex(RuntimeError, 'pretrained weight mismatch'):
                    experiment.inspect_inputs(active)
                (active / 'output/benchmark/routes_v3/stop_requested.json').write_text('{}')
                with self.assertRaisesRegex(RuntimeError, 'stop request'):
                    experiment.inspect_inputs(active)

    def test_missing_checkpoint_cannot_overwrite_formal_training(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            experiment.validate_existing(folder, {}, False)
            (folder / 'formal_started.json').write_text('{}')
            with self.assertRaisesRegex(RuntimeError, 'Formal training evidence'):
                experiment.validate_existing(folder, {}, False)
            (folder / 'last.pt').write_bytes(b'untouched checkpoint')
            with self.assertRaisesRegex(RuntimeError, '--resume'):
                experiment.validate_existing(folder, {}, False)
            experiment.validate_existing(folder, {}, True)
            self.assertEqual((folder / 'last.pt').read_bytes(), b'untouched checkpoint')

    def test_duplicate_waiters_and_old_trainers_are_real_jobs(self):
        class Process:
            def __init__(self, pid, name):
                self.pid = pid
                self.info = dict(pid=pid, cmdline=['python', '/project/' + name], create_time=100.)
        processes = [Process(71, 'queue_flow_followup.py'), Process(72, 'train_route.py'),
                     Process(73, 'status.py'), Process(experiment.os.getpid(), 'run_rf_resolution.py')]
        with patch.object(experiment.psutil, 'process_iter', return_value=processes):
            self.assertEqual([p['pid'] for p in experiment.running_jobs()], [71, 72])

    def test_resume_requires_checkpoint_and_matching_config(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            with self.assertRaisesRegex(RuntimeError, 'Safe resume'):
                experiment.validate_existing(folder, {}, True)
            (folder / 'config.json').write_text('{"size":1056}')
            with self.assertRaisesRegex(RuntimeError, 'config mismatch'):
                experiment.validate_existing(folder, {'size':1536}, False)

    def test_cached_report_does_not_load_checkpoint_or_restart_training(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory); folder = base / 'rf_1536'; folder.mkdir()
            cfg = {'id':'rf_1536','size':1536}
            done = dict(config=cfg, best_epoch=20, completed_epochs=30,
                        optimizer_updates=7020, training_seconds=13000)
            baseline = dict(done, training_seconds=9000)
            (folder/'completed.json').write_text(json.dumps(done))
            (folder/'best_metrics.json').write_text(json.dumps(dict(pq=.42,dice=.66,iou=.5)))
            def check():
                (base/'artifact_checks.json').write_text('{"passed":true}')
            fake = types.ModuleType('src.routes_v3.diagnostics')
            fake.compare_artifacts = check
            with patch.object(experiment, 'BASE', base), patch.dict(
                    experiment.sys.modules, {'src.routes_v3.diagnostics':fake}):
                self.assertTrue(experiment.report_completed(cfg, baseline, {'pq':.411}))
            report = json.loads((base/'comparison.json').read_text())
            self.assertAlmostEqual(report['delta_pq'],.009)
            self.assertFalse(report['automatic_next_stage'])
            self.assertFalse((folder/'last.pt').exists())


if __name__ == '__main__':
    unittest.main()
