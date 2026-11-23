import json,sys,tempfile,unittest
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'code'))
from experiment import array_digest,model_for,repair_volume,run_one,read
from storage import digest,dump,save_npz,sha
from physics import mesh
from analyze import report


class FrozenNonuniformTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.out=Path(self.tmp.name);folder=self.out/'inputs/small';folder.mkdir(parents=True)
        nodes,cells,centers=mesh(8,6);force=np.zeros(2*len(nodes));force[-1]=-1.
        fixed=np.array([2*i+a for i in np.flatnonzero(nodes[:,0]==0) for a in (0,1)])
        save_npz(folder/'domain.npz',nodes=nodes,cells=cells,centers=centers)
        save_npz(folder/'boundary_conditions.npz',force=force,fixed_dofs=fixed)
        dump(folder/'task.json',dict(schema='initskill-isotropic-task-v1',elements_xy=[8,6],volume_fraction=.35))
        self.cfg=read(ROOT/'protocol.json');self.cfg['max_step']=70;self.cfg['protocol_sha256']=digest(self.cfg);self.cfg['package_sha256']='test'
        self.record=dict(task_id='small',group_id='small',path='inputs/small')
        p=dict(id='dummy',penal=3.,rmin=2.5,move=.2)
        m=model_for(self.out,self.record,self.cfg,p)
        x=.15+.6*(centers[:,0]/8)**2;x,_=repair_volume(m,x,self.cfg)
        self.original=x.copy();save_npz(folder/'starts.npz',PG1=x,L1_PG=x)
        self.record['starts']={name:dict(array_sha256=array_digest(x)) for name in ('PG1','L1_PG')}
        self.record['assets']={n:sha(folder/n) for n in ('domain.npz','boundary_conditions.npz','task.json','starts.npz')}

    def tearDown(self):self.tmp.cleanup()

    def profile(self,name):return dict(id=name,initializer='PG1',strategy='P39',penal=3.,rmin=3.,move=.2)

    def test_original_field_preserved_and_launch_filter_repaired(self):
        p=self.profile('initial');result=run_one(self.out,self.record,p,self.cfg,pause_after=0)
        self.assertEqual(result['status'],'paused')
        folder=self.out/'tasks/small/initial'
        with np.load(folder/'state_source.npz') as a:np.testing.assert_array_equal(a['x'],self.original)
        info=read(folder/'initialization.json');self.assertTrue(info['volume_repair']['applied'])
        self.assertEqual(info['source_array_sha256'],array_digest(self.original))
        self.assertEqual(read(folder/'cost.json')['attempted_FE'],1)
        self.assertEqual(info['new_LLM_calls'],0)
        model=model_for(self.out,self.record,self.cfg,p)
        with np.load(folder/'state_0000.npz') as a:self.assertLess(abs(model.physical(a['x']).mean()-model.target),1e-8)
        with np.load(self.out/self.record['path']/'starts.npz') as a:np.testing.assert_array_equal(a['PG1'],self.original)

    def test_failed_field_identity_keeps_comparison_labels(self):
        import copy
        record=copy.deepcopy(self.record)
        record['starts']['PG1']['array_sha256']='0'*64
        value=run_one(self.out,record,self.profile('bad_field'),self.cfg)
        self.assertEqual(value['status'],'failed')
        self.assertEqual(value['initializer'],'PG1')
        self.assertEqual(value['strategy'],'P39')
        self.assertEqual(value['cost']['attempted_FE'],0)

    def test_nonuniform_resume_across_both_repairs(self):
        direct=run_one(self.out,self.record,self.profile('direct'),self.cfg)
        self.assertNotEqual(direct['status'],'failed',direct)
        with np.load(self.out/'tasks/small/direct/state_final.npz') as a:expected=a['x'].copy()
        for stop in (0,49,50,55):
            p=self.profile('resume'+str(stop));paused=run_one(self.out,self.record,p,self.cfg,pause_after=stop)
            self.assertEqual(paused['status'],'paused')
            value=run_one(self.out,self.record,p,self.cfg)
            self.assertEqual(value['Cend'],direct['Cend']);self.assertEqual(value['cost'],direct['cost'])
            with np.load(self.out/'tasks/small'/p['id']/'state_final.npz') as a:np.testing.assert_array_equal(a['x'],expected)


class SameStrategyReportTests(unittest.TestCase):
    def test_uses_matching_uniform_and_same_task_default_interaction(self):
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp);cfg=read(ROOT/'protocol.json');cfg['records']=[dict(task_id='t')]
            refs=[]
            for strategy,c in [('P34',100),('P39',200),('P38',300),('P20',400)]:
                refs.append(dict(task_id='t',initializer='U',strategy=strategy,status='converged',N=100,Cend=c,C50_common=c*1.1))
            for initializer,c in [('PG1',90),('L1_PG',80)]:
                refs.append(dict(task_id='t',initializer=initializer,strategy='P34',status='converged',N=90,Cend=c,C50_common=c*1.1))
            cfg['references']=refs
            for p in cfg['profiles']:
                u=next(r for r in refs if r['initializer']=='U' and r['strategy']==p['strategy'])
                dump(out/'tasks/t'/p['id']/'result.json',dict(initializer=p['initializer'],strategy=p['strategy'],status='converged',step=80,Cend=.9*u['Cend'],C50_common=u['Cend'],cost=dict(attempted_FE=82)))
            progress=report(out,cfg,'complete');self.assertEqual(progress['settled'],6);self.assertEqual(progress['comparison_planned'],12)
            import csv
            with (out/'reports/same_strategy_summary.csv').open(encoding='utf-8-sig') as f:rows=list(csv.DictReader(f))
            row=next(r for r in rows if r['initializer']=='PG1' and r['strategy']=='P39')
            self.assertAlmostEqual(float(row['mean_I']),10.) # vs U(P39)=200, never U(P34)=100.
            with (out/'reports/initialization_strategy_interaction.csv').open(encoding='utf-8-sig') as f:rows=list(csv.DictReader(f))
            row=next(r for r in rows if r['initializer']=='L1_PG' and r['strategy']=='P39')
            self.assertAlmostEqual(float(row['mean_gain_change_pp']),-10.)



class SourceSnapshotTests(unittest.TestCase):
    def test_required_source_is_complete_and_immutable(self):
        from export_results import verify_source_snapshot
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp);snapshot=out/'source_snapshot';(snapshot/'code').mkdir(parents=True)
            for name in ('code/engine.py','run.py','protocol.json'):
                (snapshot/name).write_text('{}\n',encoding='utf-8')
            source=dict(files={name:sha(snapshot/name) for name in ('code/engine.py','run.py','protocol.json')})
            dump(snapshot/'package_MANIFEST.json',source)
            value=sha(snapshot/'package_MANIFEST.json');(snapshot/'package_MANIFEST.sha256').write_text(value+'  MANIFEST.json\n')
            cfg=dict(source_snapshot_required=True,package_sha256=value)
            verify_source_snapshot(out,cfg)
            (snapshot/'code/engine.py').write_text('changed\n')
            with self.assertRaises(AssertionError):verify_source_snapshot(out,cfg)

if __name__=='__main__':unittest.main()
