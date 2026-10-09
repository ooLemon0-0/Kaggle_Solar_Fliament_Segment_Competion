import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import Mock,patch

spec=importlib.util.spec_from_file_location('cancel_waiter',Path(__file__).resolve().parents[1]/'linux/cancel_waiting_followup.py')
cancel=importlib.util.module_from_spec(spec);spec.loader.exec_module(cancel)


class CancelWaiterTests(unittest.TestCase):
    def root(self,directory):
        root=Path(directory);(root/'linux').mkdir();(root/'linux/queue_flow_followup.py').write_text('')
        return root

    def test_waiter_cancellation_does_not_signal_any_process(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=self.root(temporary)
            process=Mock();process.cmdline.return_value=['python','queue_flow_followup.py'];process.children.return_value=[]
            with patch.object(cancel,'checkout_processes',side_effect=[[process],[],[]]),patch.object(sys,'argv',['cancel.py','--checkout',str(root)]):
                cancel.main()
            process.terminate.assert_not_called();process.kill.assert_not_called()
            self.assertFalse((root/'output/benchmark/flow_followup_v1/stop_requested.json').exists())

    def test_running_child_or_prior_flow_checkpoint_refuses_upgrade(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=self.root(temporary);process=Mock();process.cmdline.return_value=['python','queue_flow_followup.py'];process.children.return_value=[Mock()]
            with patch.object(cancel,'checkout_processes',return_value=[process]),patch.object(sys,'argv',['cancel.py','--checkout',str(root)]):
                with self.assertRaisesRegex(RuntimeError,'running child'):cancel.main()
            base=root/'output/benchmark/flow_followup_v1/flow_v2';base.mkdir(parents=True)
            (base/'last.pt').write_bytes(b'checkpoint')
            with patch.object(sys,'argv',['cancel.py','--checkout',str(root)]):
                with self.assertRaisesRegex(RuntimeError,'already started'):cancel.main()
            self.assertEqual((base/'last.pt').read_bytes(),b'checkpoint')


if __name__=='__main__':unittest.main()
