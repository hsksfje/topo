"""Numerical regressions for switching, exact resume, and return integrity."""
import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'code'))
from experiment import is_converged, model_for, read, repair_volume, run_one
from physics import mesh
from storage import digest,dump,save_npz,sha
from run import select_records
from export_results import seal_output,verify_return,export_zip


class NumericalProtocolTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory()
        self.out=Path(self.tmp.name)
        folder=self.out/'inputs/small';folder.mkdir(parents=True)
        nodes,cells,centers=mesh(8,6)
        force=np.zeros(2*len(nodes))
        force[2*np.flatnonzero((nodes[:,0]==8)&(nodes[:,1]==3))[0]+1]=-1.
        fixed=np.array([2*i+a for i in np.flatnonzero(nodes[:,0]==0) for a in (0,1)])
        save_npz(folder/'domain.npz',nodes=nodes,cells=cells,centers=centers)
        save_npz(folder/'boundary_conditions.npz',force=force,fixed_dofs=fixed)
        dump(folder/'task.json',dict(schema='initskill-isotropic-task-v1',elements_xy=[8,6],volume_fraction=.35))
        self.record=dict(task_id='small',group_id='test',path='inputs/small',assets={n:sha(folder/n) for n in ('domain.npz','boundary_conditions.npz','task.json')})
        self.cfg=read(ROOT/'protocol.json')
        self.cfg['max_step']=70
        self.cfg['protocol_sha256']=digest(self.cfg)
        self.cfg['package_sha256']='offline-test'

    def tearDown(self):self.tmp.cleanup()

    def profile(self,name,p=2.,r=1.5,m=.1):
        return dict(id=name,penal=p,rmin=r,move=m)

    def test_default_is_exactly_original_solver(self):
        p=self.profile('default',3.,2.5,.2)
        result=run_one(self.out,self.record,p,self.cfg)
        self.assertNotEqual(result['status'],'failed',result)
        model=model_for(self.out,self.record,self.cfg,p,common=True)
        x=model.initial();ev=model.evaluate(x)
        for _ in range(result['step']):
            x=model.update(x,ev);ev=model.evaluate(x)
        with np.load(self.out/'tasks/small/default/state_final.npz') as a:
            np.testing.assert_array_equal(a['x'],x)
        self.assertEqual(result['Cend'],ev['compliance'])
        self.assertFalse(result['transition']['volume_repair']['applied'])
        self.assertEqual(result['cost']['attempted_FE'],result['step']+1)

    def test_resume_exact_before_at_and_after_switch(self):
        direct=run_one(self.out,self.record,self.profile('direct'),self.cfg)
        self.assertNotEqual(direct['status'],'failed',direct)
        with np.load(self.out/'tasks/small/direct/state_final.npz') as a:expected=a['x'].copy()
        for boundary in (49,50,55):
            p=self.profile('resume'+str(boundary))
            paused=run_one(self.out,self.record,p,self.cfg,pause_after=boundary)
            self.assertEqual(paused['status'],'paused')
            finished=run_one(self.out,self.record,p,self.cfg)
            self.assertEqual(finished['status'],direct['status'])
            self.assertEqual(finished['Cend'],direct['Cend'])
            self.assertEqual(finished['cost'],direct['cost'])
            with np.load(self.out/'tasks/small'/p['id']/'state_final.npz') as a:
                np.testing.assert_array_equal(a['x'],expected)
        self.assertEqual(direct['cost']['switch_FE'],1)
        self.assertEqual(direct['cost']['attempted_FE'],direct['step']+2)

    def test_filter_switch_volume_repair(self):
        p=self.profile('repair')
        model=model_for(self.out,self.record,self.cfg,p,common=True)
        # Deliberately infeasible and nonuniform: exercises repair, clipping, orientation.
        x=np.linspace(.2,.95,len(model.cells))
        repaired,metadata=repair_volume(model,x,self.cfg)
        self.assertTrue(metadata['applied'])
        self.assertLess(abs(model.physical(repaired).mean()-model.target),1e-11)
        self.assertTrue(np.all(repaired>=model.config.xmin))
        self.assertTrue(np.all(repaired<=1))
        same,metadata=repair_volume(model,repaired,self.cfg)
        np.testing.assert_array_equal(same,repaired)
        self.assertFalse(metadata['applied'])

    def test_common_window_and_iteration_cap(self):
        p=self.profile('cap')
        cfg=dict(self.cfg,max_step=60)
        # Force no convergence certification to distinguish cap from convergence.
        cfg['convergence']=dict(cfg['convergence'],max_design_change=0.)
        cfg['protocol_sha256']=digest(cfg)
        result=run_one(self.out,self.record,p,cfg)
        self.assertEqual(result['status'],'max_steps')
        self.assertEqual(result['step'],60)
        history=read(self.out/'tasks/small/cap/history.json')
        self.assertFalse(is_converged(history[:60],cfg))
        self.assertEqual(history[50]['phase'],'common_boundary')
        self.assertIsNone(history[50]['relative_compliance_change'])
        self.assertTrue(all(r['phase']=='common' for r in history[51:]))

    def test_identity_changes_are_rejected(self):
        p=self.profile('identity')
        run_one(self.out,self.record,p,self.cfg,pause_after=2)
        with self.assertRaises(ValueError):
            run_one(self.out,self.record,dict(p,move=.3),self.cfg)

    def test_seal_detects_tampering_and_export_roundtrip(self):
        p=self.profile('seal')
        run_one(self.out,self.record,p,self.cfg)
        cfg=dict(self.cfg,records=[self.record],profiles=[p])
        dump(self.out/'config.json',cfg)
        dump(self.out/'progress.json',dict(status='complete',settled=1))
        dump(self.out/'COMPLETION.json',dict(settled=1))
        seal_output(self.out)
        self.assertEqual(verify_return(self.out)['trajectory_count'],1)
        archive=export_zip(self.out,self.out.parent/(self.out.name+'_return.zip'))
        import zipfile
        with tempfile.TemporaryDirectory() as returned:
            with zipfile.ZipFile(archive) as z:z.extractall(returned)
            self.assertEqual(verify_return(Path(returned)/self.out.name)['trajectory_count'],1)
        archive.unlink();archive.with_name(archive.name+'.sha256').unlink()
        (self.out/'tasks/small/seal/history.json').write_text('[]')
        with self.assertRaises(AssertionError):verify_return(self.out)


class FrozenDataTests(unittest.TestCase):
    def test_exact20_fields_and_reference_matrix(self):
        protocol=read(ROOT/'protocol.json');records=read(ROOT/'data/tasks.json')
        self.assertEqual(len(protocol['profiles']),6)
        self.assertEqual(len(records),20)
        self.assertEqual([sum(r['vf_bin']==b for r in records) for b in range(10)],[2]*10)
        self.assertEqual(len(select_records(records,10)),10)
        refs=read(ROOT/'data/references.json')
        self.assertEqual(len(refs),120)
        self.assertEqual(len({(r['task_id'],r['initializer'],r['strategy']) for r in refs}),120)
        self.assertTrue(all(r['status']=='converged' for r in refs))
        for r in records:
            path=ROOT/r['starts_file']
            self.assertEqual(sha(path),r['starts_file_sha256'])
            with np.load(path,allow_pickle=False) as a:
                self.assertEqual(set(a.files),{'PG1','L1_PG'})
                for name in a.files:
                    from experiment import array_digest
                    self.assertEqual(array_digest(a[name]),r['starts'][name]['array_sha256'])
        new={(r['task_id'],p['initializer'],p['strategy']) for r in records for p in protocol['profiles']}
        old={(r['task_id'],r['initializer'],r['strategy']) for r in refs}
        self.assertFalse(new&old)
        self.assertEqual(len(new|old),240)


if __name__=='__main__':unittest.main()
