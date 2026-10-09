import importlib.util
import unittest
import json
import sys
import tempfile
from unittest.mock import patch
from pathlib import Path

spec=importlib.util.spec_from_file_location('queue_flow_followup',Path(__file__).resolve().parents[1]/'linux/queue_flow_followup.py')
queue=importlib.util.module_from_spec(spec);spec.loader.exec_module(queue)


class FollowupQueueTests(unittest.TestCase):
    def test_completed_status_cannot_override_live_process(self):
        status={'stage':'completed','routes':{'a':'completed','b':'completed'}}
        self.assertEqual(queue.readiness(status,[{'pid':123}],False,['a','b']),'waiting')

    def test_dead_runner_with_missing_or_failed_route_is_blocked(self):
        self.assertEqual(queue.readiness({'stage':'completed','routes':{'a':'completed'}},[],False,['a','b']),'blocked_original_incomplete')
        self.assertEqual(queue.readiness({'stage':'completed','routes':{'a':'completed','b':'failed'}},[],False,['a','b']),'blocked_original_incomplete')

    def test_stop_request_blocks_even_after_completion(self):
        status={'stage':'completed','routes':{'a':'completed','b':'completed'}}
        self.assertEqual(queue.readiness(status,[],True,['a','b']),'blocked_original_stop_request')
        self.assertEqual(queue.readiness(status,[],False,['a','b']),'ready')

    def test_projection_repair_requires_opt_in_and_other_routes_completed(self):
        status={'stage':'needs_attention','routes':{'rf':'completed','image_projection':'failed','feature_structure':'completed'}}
        required=list(status['routes'])
        self.assertEqual(queue.readiness(status,[],False,required),'blocked_original_incomplete')
        self.assertEqual(queue.readiness(status,[],False,required,True),'repair_projection')
        self.assertEqual(queue.readiness(status,[{'pid':123}],False,required,True),'waiting')
        self.assertEqual(queue.readiness(status,[],True,required,True),'blocked_original_stop_request')
        status['routes']['feature_structure']='failed'
        self.assertEqual(queue.readiness(status,[],False,required,True),'blocked_original_incomplete')

    def test_projection_repair_cannot_skip_queued_or_paused_routes(self):
        for value in ['queued','running','paused','blocked_missing_rf_checkpoint']:
            status={'stage':'needs_attention','routes':{'rf':'completed','image_projection':value}}
            self.assertEqual(queue.readiness(status,[],False,list(status['routes']),True),'blocked_original_incomplete')

    def test_repaired_queue_becomes_ready_without_bypassing_completion(self):
        status={'stage':'completed','routes':{'rf':'completed','image_projection':'completed'}}
        self.assertEqual(queue.readiness(status,[],False,list(status['routes']),True),'ready')

    def exercise_wrapper(self,fail_repair):
        from src.routes_v3.resources import file_lock
        import subprocess
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary)/'source';active=Path(temporary)/'original'
            oldbase=active/'output/benchmark/routes_v3';oldbase.mkdir(parents=True)
            route_states={'rf':'completed','image_projection':'failed','feature_structure':'completed'}
            (oldbase/'experiment_manifest.json').write_text(json.dumps({'runs':[{'id':key} for key in route_states]}))
            status={'stage':'needs_attention','routes':route_states}
            path=oldbase/'pipeline_status.json';path.write_text(json.dumps(status))
            calls=[]
            def run(command,**kwargs):
                # Both original-project locks must remain held while either child runs.
                for name in ['runner_process.lock','gpu_operation.lock']:
                    with self.assertRaises(BlockingIOError):
                        with file_lock(oldbase/name,blocking=False):pass
                script=Path(command[2]).name;calls.append(script)
                if script=='run_projection_repair.py':
                    if fail_repair:return subprocess.CompletedProcess(command,1)
                    status['stage']='completed';status['routes']['image_projection']='completed'
                    path.write_text(json.dumps(status))
                else:(root/'output/benchmark/flow_followup_v1/completed.json').write_text('{}')
                return subprocess.CompletedProcess(command,0)
            with patch.object(queue,'ROOT',root),patch.object(queue,'prepare',return_value=(active,'commit',{})),patch.object(queue,'source_unchanged',return_value=True),patch.object(queue,'running_jobs',return_value=[]),patch.object(queue.subprocess,'run',side_effect=run),patch.object(sys,'argv',['queue_flow_followup.py','--active-project',str(active),'--repair-projection']):
                if fail_repair:
                    with self.assertRaisesRegex(RuntimeError,'Projection repair failed'):queue.main()
                else:queue.main()
            saved=json.loads((root/'output/benchmark/flow_followup_v1/queue_status.json').read_text())
            self.assertEqual(calls,['run_projection_repair.py'] if fail_repair else ['run_projection_repair.py','run_flow_followup.py'])
            self.assertEqual(saved['stage'],'blocked_projection_repair' if fail_repair else 'completed')

    def test_wrapper_serial_repair_then_flow_with_locks_held(self):self.exercise_wrapper(False)

    def test_wrapper_repair_failure_blocks_flow(self):self.exercise_wrapper(True)


if __name__=='__main__':unittest.main()
