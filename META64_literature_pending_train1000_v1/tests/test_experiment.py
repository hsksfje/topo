import os
for _key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[_key] = '1'
import json
from pathlib import Path
import sys
import tempfile
import unittest
import zipfile

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'code'))
import numpy as np
from experiment import run_one, converged, read
from initializers import generate, METHODS
from physics import SolverConfig
from solver import SIMP


class ExperimentTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.out=Path(self.temp.name)
        self.records=read(ROOT/'data_identity.json')['records']
        self.r=self.records[0]
        self.cfg=read(ROOT/'protocol.json')
        with zipfile.ZipFile(ROOT/'physical_inputs.zip') as z:
            for info in z.infolist():
                if info.filename.startswith(self.r['path']+'/'):
                    z.extract(info,self.out)
        self.model=SIMP(self.out/self.r['path'],SolverConfig(**self.cfg['solver']))

    def tearDown(self):
        self.temp.cleanup()

    def test_initializers_volume_seed_and_geometry(self):
        for target in (.100001,.21393619775098704,.499999):
            self.model.target=target
            fields={name:generate(self.model,name)[0] for name in METHODS}
            for name,x in fields.items():
                self.assertLess(abs(self.model.physical(x).mean()-target),1e-8)
                self.assertTrue(np.isfinite(x).all())
                self.assertGreaterEqual(x.min(),self.model.config.xmin)
                self.assertLessEqual(x.max(),1.)
                np.testing.assert_array_equal(x,generate(self.model,name)[0])
            self.assertFalse(np.array_equal(fields['RAND_IID_S01'],fields['RAND_IID_S02']))
            hole=fields['HC1_B020'].reshape(64,64,order='F')
            np.testing.assert_allclose(hole,hole[::-1,:],atol=1e-12)
            np.testing.assert_allclose(hole,hole[:,::-1],atol=1e-12)
            self.assertLess(hole[32,32],hole[0,0])

    def test_reference_uniform_C0_C50(self):
        result=run_one(self.out,self.r,'U',self.cfg,'50')
        self.assertEqual(result['status'],'complete50')
        self.assertAlmostEqual(result['C0']/1792.8443495857668,1.,places=8)
        self.assertAlmostEqual(result['C50']/80.17467347023177,1.,places=8)

    def test_resume_equals_uninterrupted_and_counts_recovery(self):
        cfg=dict(self.cfg,max_step=70,earliest_certifiable_step=999)
        run_one(self.out,self.r,'HC1_B020',cfg,'50')
        resumed=run_one(self.out,self.r,'HC1_B020',cfg,'converge')
        self.assertEqual(resumed['status'],'max_steps')
        self.assertEqual(resumed['cost']['recovery_FE'],1)
        dest=self.out/'tasks'/self.r['task_id']/'HC1_B020'
        with np.load(dest/'state_final.npz') as a:
            resume_x=a['x'].copy()
        other=self.out/'independent';other.mkdir()
        import shutil
        shutil.copytree(self.out/'inputs',other/'inputs')
        direct=run_one(other,self.r,'HC1_B020',cfg,'all')
        with np.load(other/'tasks'/self.r['task_id']/'HC1_B020/state_final.npz') as a:
            np.testing.assert_array_equal(resume_x,a['x'])
        self.assertAlmostEqual(resumed['Cend'],direct['Cend'],places=10)
        self.assertEqual(resumed['cost']['attempted_FE'],direct['cost']['attempted_FE']+1)

    def test_pause_and_step50_artifact_recovery(self):
        (self.out/'PAUSE_REQUESTED').touch()
        self.assertEqual(run_one(self.out,self.r,'U',self.cfg,'50')['status'],'paused')
        self.assertFalse((self.out/'tasks'/self.r['task_id']/'U/state_0000.npz').exists())
        (self.out/'PAUSE_REQUESTED').unlink()
        run_one(self.out,self.r,'U',self.cfg,'50')
        dest=self.out/'tasks'/self.r['task_id']/'U'
        (dest/'result_50.json').unlink();(dest/'state_0050.npz').unlink()
        result=run_one(self.out,self.r,'U',self.cfg,'50')
        self.assertEqual(result['status'],'complete50')
        self.assertTrue((dest/'state_0050.npz').exists())

    def test_convergence_requires_entire_window_and_minimum_step(self):
        cfg=dict(self.cfg,_target=.3)
        def row(step):
            return dict(step=step,compliance=100.,volume=.3,residual=1e-10,
                        max_design_change=.001,relative_compliance_change=1e-6)
        h=[row(i) for i in range(61)]
        self.assertTrue(converged(h,cfg))
        self.assertFalse(converged(h[:51],cfg))
        h[-5]['max_design_change']=.02
        self.assertFalse(converged(h,cfg))
        h[-5]['max_design_change']=.001;h[-2]['volume']=.301
        self.assertFalse(converged(h,cfg))


if __name__=='__main__':
    unittest.main()
