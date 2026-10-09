import importlib.util
import unittest
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


if __name__=='__main__':unittest.main()
