"""A known scheduler-only upgrade must preserve finished execution identities."""
from pathlib import Path
import json
import sys
import tempfile
import unittest
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'src'))
from initskill.methods.benchmark_init_v1 import common
from scripts.runtime_parallel import OLD_RESOURCE_SHA,migrate_resource_protocol


class ResourceUpgradeTest(unittest.TestCase):
    def test_completed_results_preserved_pending_checkpoint_audited(self):
        with tempfile.TemporaryDirectory() as temp:
            out=Path(temp);old=common.protocol(out);old['source_sha256']['resources.py']=OLD_RESOURCE_SHA
            common.dump(out/'protocol.json',old);before=(out/'protocol.json').read_bytes();old_sha=common.sha(out/'protocol.json')
            done=out/'full_evaluation/test_id300/uniform/completed';done.mkdir(parents=True)
            common.dump(done/'identity.json',dict(protocol_sha256=old_sha,method='uniform'))
            (done/'result.json').write_text('{"historical_execution": true}')
            original=(done/'result.json').read_bytes()
            original_identity=(done/'identity.json').read_bytes()
            pending=done.parent/'pending';pending.mkdir()
            identity=dict(protocol_sha256=old_sha,method='uniform',subset='test_id300',max_steps=500)
            common.dump(pending/'identity.json',identity)
            state=pending/'oc_checkpoint_slots/0/state.npz';state.parent.mkdir(parents=True);np.savez(state,x=np.array([.5]))
            common.dump(pending/'checkpoint.json',dict(step=10,state_file='oc_checkpoint_slots/0/state.npz',state_sha256=common.sha(state)))
            migrate_resource_protocol(out,common)
            self.assertEqual((done/'result.json').read_bytes(),original)
            self.assertEqual((done/'identity.json').read_bytes(),original_identity)
            self.assertEqual(json.loads((pending/'identity.json').read_text())['protocol_sha256'],common.sha(out/'protocol.json'))
            audit=next((out/'audit').iterdir())
            self.assertEqual((audit/'protocol.before.json').read_bytes(),before)
            saved=audit/'pending_before/test_id300/uniform/pending/oc_checkpoint_slots/0/state.npz'
            self.assertEqual(common.sha(saved),common.sha(state))
            self.assertEqual(common.protocol(out)['oc_max_steps'],500)
            # Simulate interruption after pending identities but before the
            # global protocol pointer was committed; a retry must recover.
            (out/'protocol.json').write_bytes(before)
            migrate_resource_protocol(out,common)
            self.assertEqual(json.loads((pending/'identity.json').read_text())['protocol_sha256'],common.sha(out/'protocol.json'))

    def test_unrecognized_source_change_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            out=Path(temp);record=common.protocol(out);record['source_sha256']['common.py']='unrecognized'
            common.dump(out/'protocol.json',record)
            with self.assertRaisesRegex(ValueError,'Unrecognized source change'):migrate_resource_protocol(out,common)


if __name__=='__main__':unittest.main()
