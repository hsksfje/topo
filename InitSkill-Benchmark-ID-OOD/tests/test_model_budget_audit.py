from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from initskill.methods.benchmark_init_v1 import common
from scripts import analyze

class ModelBudgetAuditTest(unittest.TestCase):
    def test_short_supervised_model_cannot_pass_full_audit(self):
        for method in ['ulu','wang','cnn']:
            with tempfile.TemporaryDirectory() as temp:
                out=Path(temp);folder=out/'models'/method;folder.mkdir(parents=True)
                (folder/'best.pt').write_bytes(b'engineering placeholder, deliberately not a formal model')
                cfg=dict(train_keys=[r['task_key'] for r in common.rows('train1000')],valid_keys=[r['task_key'] for r in common.rows('valid100')],
                    epochs=500 if method=='cnn' else 1000,batch_size=16,rank=80 if method=='ulu' else 200)
                common.dump(folder/'config.json',cfg)
                common.dump(folder/'frozen.json',dict(model_sha256=common.sha(folder/'best.pt'),config_sha256=common.sha(folder/'config.json')))
                key='valid_MAE' if method=='cnn' else 'validation_MSE'
                common.dump(folder/'history.json',[dict(epoch=1,**{key:1.})])
                with patch.object(analyze,'METHODS',(method,)):
                    with self.assertRaisesRegex(ValueError,'stopping rule'):analyze.model_audit(out)
