import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import splu
from initskill.methods.benchmark_init_v1 import common
from initskill.methods.benchmark_init_v1.precision_fem import refined_evaluation
from initskill.methods.benchmark_init_v1.convergence import run_to_convergence
from scripts.knn import descriptor,distances,build_bank,prediction,CONFIG,resize_design
from scripts.experiment import allocation

ROOT=common.ROOT

class KNNTest(unittest.TestCase):
    def test_scope_and_separation(self):
        groups={s:{r['group_id'] for r in common.rows(s)} for s in ['train1000','valid100','test_id300','test_ood300','test_cr300']}
        for a in ['train1000','valid100']:
            for b in groups:
                if b!=a:self.assertFalse(groups[a]&groups[b],(a,b))
        self.assertEqual(sum(len(groups[s]) for s in ['test_id300','test_ood300','test_cr300']),900)

    def test_sign_invariance_and_self_retrieval(self):
        row=common.rows('train1000')[0];f,s,v=descriptor(row)
        bank=dict(load=np.stack([f,-f,np.roll(f,40)]),support=np.stack([s,s,np.roll(s,30)]),vf=np.array([v,v,v+.1]),vf_span=np.array(.3))
        d1,_=distances(bank,f,s,v);d2,_=distances(bank,-f,s,v)
        np.testing.assert_array_equal(d1,d2);self.assertLess(d1[0],1e-24);self.assertLess(d1[1],1e-24)
        self.assertGreater(d1[2],d1[0])

    def test_softening_volume_and_initialization(self):
        row=common.rows('train1000')[0]
        with tempfile.TemporaryDirectory() as temp:
            m=common.solver(temp,'train1000',row);rng=np.random.default_rng(42)
            x=common.calibrate(rng.uniform(size=4096),m);soft=.9*x+.1*m.target
            self.assertLess(abs(m.physical(soft).mean()-m.target),1e-10)
            self.assertGreaterEqual(soft.min(),.1*m.target)

    def test_decimal_healthy_solution_gradient_and_work(self):
        row=common.rows('train1000')[0]
        with tempfile.TemporaryDirectory() as temp:
            m=common.solver(temp,'train1000',row);x=m.initial();ev=m.evaluate(x);rho=m.physical(x);c=m.config
            w=c.emin+(c.emax-c.emin)*rho**c.penal
            K=sparse.coo_matrix(((w[:,None,None]*m.ke).ravel(),(m.rows,m.cols)),shape=(len(m.force),len(m.force))).tocsc();factor=splu(K[m.free][:,m.free])
            dec=refined_evaluation(m,x,rho,factor,ev['u'],ev['residual'])
            self.assertLess(abs(dec['compliance']/ev['compliance']-1),1e-8)
            self.assertLess(np.linalg.norm(dec['gradient']-ev['gradient'])/np.linalg.norm(ev['gradient']),1e-7)
            self.assertLess(dec['energy_balance_relative'],1e-8)
            # A small equilibrium residual alone must not accept corrupted
            # cancellation-prone strain energy or its resulting gradients.
            from unittest.mock import patch
            original=np.einsum
            def wrong_energy(expression,*args,**kwargs):
                answer=original(expression,*args,**kwargs)
                return answer*2 if expression=='ei,ij,ej->e' else answer
            with patch.object(common.np,'einsum',side_effect=wrong_energy):
                guarded=m.evaluate(x)
            self.assertEqual(guarded['precision'],'decimal40-Q4-iterative-refinement')
            self.assertLess(abs(guarded['compliance']/ev['compliance']-1),1e-8)


    def test_eight_windows_failed_initial_states(self):
        with np.load(ROOT/'tests/fixtures/windows_initial_failures.npz') as z:fields={k:z[k].copy() for k in z.files}
        rows={r['task_key']:r for s in ['test_id300','test_ood300'] for r in common.rows(s)};report=[]
        with tempfile.TemporaryDirectory() as temp:
            for key,subset,x in zip(fields['keys'],fields['subsets'],fields['x']):
                m=common.solver(temp,str(subset),rows[str(key)]);e=m.evaluate(x)
                self.assertLess(e['residual'],1e-6);self.assertGreater(e['compliance'],0)
                self.assertIn(e.get('precision'),[None,'decimal40-Q4-iterative-refinement'])
                report.append(dict(task=str(key),residual=e['residual'],compliance=e['compliance'],energy_balance=e['energy_balance_relative'],refinements=e['refinement_steps']))
        common.dump(ROOT/'verification/windows_failure_regression.json',dict(passed=True,FE_attempts=8,decimal_digits=40,cases=report))

    def test_coherent_oc_resume(self):
        row=common.rows('train1000')[0]
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);a=common.solver(root/'a','train1000',row);b=common.solver(root/'b','train1000',row)
            full=run_to_convergence(a,a.initial(),12,root/'continuous',dict(check='resume'))
            original=b.evaluate;counter=[0]
            def interrupted(x):
                counter[0]+=1
                if counter[0]==12:raise RuntimeError('test interruption')
                return original(x)
            b.evaluate=interrupted
            with self.assertRaises(RuntimeError):run_to_convergence(b,b.initial(),12,root/'split',dict(check='resume'))
            (root/'split/failure.json').unlink();b.evaluate=original
            resumed=run_to_convergence(b,b.initial(),12,root/'split',dict(check='resume'))
            self.assertEqual(full['history'],resumed['history'])
            with np.load(root/'continuous/state_final.npz') as x,np.load(root/'split/state_final.npz') as y:np.testing.assert_array_equal(x['x'],y['x'])

    def test_cpu_budget(self):
        a=allocation(28,64);self.assertLessEqual(a['total_worker_limit'],28)
        self.assertEqual(a['threads_per_worker'],1)

    def test_cross_resolution_half_pixel_mapping(self):
        y,x=np.meshgrid(np.arange(64.),np.arange(64.),indexing='ij')
        raw=(2*x+3*y).ravel(order='F')
        actual=resize_design(raw,256,256).reshape(256,256,order='F')
        v=np.clip((np.arange(256)+.5)/4-.5,0,63)
        yy,xx=np.meshgrid(v,v,indexing='ij')
        np.testing.assert_allclose(actual,2*xx+3*yy,atol=1e-12,rtol=0)

    def test_full_train_bank_and_engineering_trajectories(self):
        row=common.rows('train1000')[0]
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);build_bank(root)
            with np.load(root/'models/knn/bank.npz') as z:bank={k:z[k].copy() for k in z.files}
            self.assertEqual(bank['x'].shape,(1000,4096))
            stats=[]
            for method in ['uniform','knn_legacy','knn_v2']:
                raw,meta=prediction(bank,row,method)
                m=common.solver(root/method,'train1000',row)
                if raw is None:raw=m.initial()
                else:
                    self.assertEqual(meta['neighbors'][0],row['task_key']);self.assertLess(meta['distances'][0],1e-20)
                x=common.calibrate(raw,m)
                if method=='knn_v2':x=.9*x+.1*m.target
                result=run_to_convergence(m,x,30,root/'runs'/method,dict(engineering_train_only=True))
                self.assertLessEqual(result['final_compliance'],result['history'][0]['compliance']*1.01)
                stats.append(dict(method=method,FE=result['FE'],C0=result['history'][0]['compliance'],C30=result['final_compliance']))
            # Actual spawned worker exercises the same per-task3-method path
            # with an explicitly labelled engineering-only12-step budget.
            import multiprocessing
            from concurrent.futures import ProcessPoolExecutor
            from scripts.experiment import worker
            common.dump(root/'protocol.json',dict(engineering_train_only=True))
            with ProcessPoolExecutor(max_workers=1,mp_context=multiprocessing.get_context('spawn')) as pool:
                answer=pool.submit(worker,(str(root),'train1000',row,12)).result(timeout=180)
            self.assertTrue(all(r['status']=='max_steps' for r in answer['methods']),answer)
            common.dump(ROOT/'verification/train_bank_engineering.json',dict(passed=True,training_count=1000,task=row['task_key'],formal_test_tasks_used=0,trajectories=stats))

    def test_256_training_case_initial_fields(self):
        row=common.rows('train1000')[0];f,b,v=common.load(row)
        highf=np.zeros((257,257,2));highb=np.zeros_like(highf)
        highf[::4,::4]=f;highb[::4,::4]=b;high=dict(row,nx=256,ny=256)
        with np.load(ROOT/'data/labels/train1000'/(row['source_task_id']+'.npz')) as z:low=z['x'].copy()
        records=[]
        with tempfile.TemporaryDirectory() as temp:
            m=common.solver(temp,'engineering_train256',high,(highf,highb,v));raw=resize_design(low,256,256)
            calibrated=common.calibrate(raw,m)
            for name,x in [('uniform',m.initial()),('knn_legacy',calibrated),('knn_v2',.9*calibrated+.1*v)]:
                e=m.evaluate(x);self.assertLess(e['residual'],1e-6);self.assertLess(abs(e['volume']-v),1e-8)
                records.append(dict(method=name,compliance=e['compliance'],residual=e['residual'],precision=e.get('precision')))
        common.dump(ROOT/'verification/train256_engineering.json',dict(passed=True,training_task=row['task_key'],FE_attempts=3,formal_test_tasks_used=0,cases=records))
