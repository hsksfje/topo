"""Exercise actual trainer/save/reload paths on tiny engineering data only."""
from pathlib import Path
import json
import sys
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from initskill.methods.benchmark_init_v1 import common

class TrainerIOTest(unittest.TestCase):
    def test_pca_train_save_and_predict_both_optimizers(self):
        from initskill.methods.benchmark_init_v1.ulu_pca import core as ulu
        from initskill.methods.benchmark_init_v1.wang_pca import core as wang
        for module,name,n,d in [(ulu,'ulu',85,85),(wang,'wang',205,205)]:
            rng=np.random.default_rng(42)
            train=(common.rows('train1000')[:n],rng.normal(size=(n,3)),rng.uniform(.1,.9,size=(n,d)))
            valid=(common.rows('valid100')[:4],rng.normal(size=(4,3)),rng.uniform(.1,.9,size=(4,d)))
            def arrays(out,subset):return train if subset=='train1000' else valid
            with tempfile.TemporaryDirectory() as temp,patch.object(module,'arrays',side_effect=arrays):
                module.train(temp,epochs=1,seed=42)
                with patch.object(module,'features',return_value=train[1][0]):prediction,meta=module.predict(temp,train[0][0])
                self.assertEqual(prediction.shape,(d,));self.assertTrue(np.isfinite(prediction).all())
                report=json.loads((Path(temp)/'models'/name/'training_health.json').read_text())
                self.assertIn('mean_training_topology_coefficient_MSE',report)
                cfg=json.loads((Path(temp)/'models'/name/'config.json').read_text())
                self.assertNotEqual(len(cfg['train_keys']),1000,'Engineering model must not satisfy formal model audit')

    def test_cnn_train_freeze_health_and_real64_inference(self):
        from initskill.methods.benchmark_init_v1.cnn_encoder_decoder import core
        torch.manual_seed(42)
        def tensors(out,subset):
            n=4 if subset=='train1000' else 2
            records=common.rows(subset)[:n]
            return records,torch.randn(n,4,9,9),torch.full((n,),.3),torch.rand(n,1,8,8)
        with tempfile.TemporaryDirectory() as temp,patch.object(core,'tensors',side_effect=tensors):
            core.train_cnn(temp,epochs=1,batch_size=2,seed=42)
            model,frozen=core.load_cnn(temp)
            prediction,meta=core.predict_cnn(temp,common.rows('train1000')[0])
            self.assertEqual(prediction.shape,(4096,));self.assertTrue(np.isfinite(prediction).all())
            health=json.loads((Path(temp)/'models/cnn/training_health.json').read_text())
            self.assertEqual(health['selected_epoch'],1);self.assertIn('warnings',health['selected'])
            self.assertEqual(common.sha(Path(temp)/'models/cnn/training_health.json'),frozen['health_report_sha256'])
            self.assertEqual(len(json.loads((Path(temp)/'models/cnn/config.json').read_text())['train_keys']),4)
