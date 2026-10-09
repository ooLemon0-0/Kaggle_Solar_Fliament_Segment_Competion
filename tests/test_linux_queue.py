"""No CUDA training: exercise queue advancement, stop and RF dependency blocking."""
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'scripts'))
import run_route_experiments as runner

class QueueTests(unittest.TestCase):
    def execute(self, fail_rf=False, stop_after_flow=False):
        (ROOT/'work').mkdir(exist_ok=True)
        temp=tempfile.TemporaryDirectory(dir=ROOT/'work')
        assert Path(temp.name).resolve().is_relative_to((ROOT/'work').resolve())
        self.addCleanup(temp.cleanup)
        base=Path(temp.name);runs=[{'id':k,'dependency':'rf' if k=='sam' else None} for k in ['rf','flow','sam']]
        (base/'experiment_manifest.json').write_text(json.dumps({'runs':runs}))
        starts=[]
        class Child:
            pid=43210
            def __init__(self,cmd,**kwargs):self.route=cmd[-1];starts.append(self.route)
            def wait(self):
                if self.route=='rf' and fail_rf:return 1
                d=base/self.route
                (d/'completed.json').write_text('{}')
                (d/'best.pt').write_bytes(b'checkpoint')
                (d/'best_predictions.json').write_text('{}')
                if self.route=='flow' and stop_after_flow:(base/'stop_requested.json').write_text('{}')
                return 0
        class Identity:
            def create_time(self):return 1.
        with patch.object(runner,'BASE',base),patch.object(runner,'report'),patch.object(runner.psutil,'process_iter',return_value=[]),patch.object(runner.psutil,'Process',return_value=Identity()),patch.object(runner.subprocess,'Popen',Child):
            runner.main()
        return starts,json.loads((base/'pipeline_status.json').read_text())
    def test_serial_success(self):
        starts,s=self.execute();self.assertEqual(starts,['rf','flow','sam']);self.assertEqual(s['stage'],'completed')
    def test_independent_continues_dependency_blocked(self):
        starts,s=self.execute(fail_rf=True);self.assertEqual(starts,['rf','flow']);self.assertEqual(s['routes']['sam'],'blocked_missing_rf_checkpoint')
    def test_pause_does_not_launch_next(self):
        starts,s=self.execute(stop_after_flow=True);self.assertEqual(starts,['rf','flow']);self.assertEqual(s['stage'],'paused');self.assertEqual(s['routes']['sam'],'queued')
if __name__=='__main__':unittest.main()
