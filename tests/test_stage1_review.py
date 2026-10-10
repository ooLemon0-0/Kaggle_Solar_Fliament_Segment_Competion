import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

spec=importlib.util.spec_from_file_location('stage1_review',Path(__file__).resolve().parents[1]/'linux/review_completed_routes.py')
review=importlib.util.module_from_spec(spec);spec.loader.exec_module(review)


class ReviewTests(unittest.TestCase):
    def test_curve_replayed_epoch_uses_latest_record(self):
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/'training_log.csv'
            path.write_text('epoch,pq,training_seconds\n2,.3,100\n4,.4,200\n4,.41,210\n')
            data=review.curve(path)
            self.assertEqual([row['epoch'] for row in data],[2,4]);self.assertEqual(data[-1]['pq'],.41)

    def test_read_only_review_and_best_v2_decoder(self):
        with tempfile.TemporaryDirectory() as temp:
            active=Path(temp)/'original';followup=Path(temp)/'followup'
            base=active/'output/benchmark/routes_v3';rf=base/'rf';rf.mkdir(parents=True)
            done=rf/'completed.json';done.write_text(json.dumps({'training_seconds':3600,'best_epoch':10,'completed_epochs':30}))
            metrics=rf/'best_metrics.json';metrics.write_text(json.dumps({'pq':.4,'dice':.6,'iou':.5,'tp':10}))
            extra=followup/'output/benchmark/flow_followup_v1';extra.mkdir(parents=True)
            (extra/'all_candidates.csv').write_text('route,selection,pq\nold_flow,n2000,.3999\nflow_v2,n2000,.3876\nflow_v2,n1000,.39\n')
            before=metrics.read_bytes();data=review.review(active,followup)
            self.assertEqual(metrics.read_bytes(),before)
            self.assertEqual(data['routes'][0]['pq'],.4);self.assertFalse(data['routes'][1]['completed'])
            self.assertEqual(data['routes'][2]['inherited_training_hours'],1.)
            self.assertEqual(data['best_flow_v2_decoder']['selection'],'n1000')


if __name__=='__main__':unittest.main()
