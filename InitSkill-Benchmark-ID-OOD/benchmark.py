#!/usr/bin/env python3
"""Remaining methods: full Train1000/Valid100/ID300/OOD300; CR/cGAN opt-in."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'src'))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=['plan','prepare','verify','start','status','stop','analyze','export','configure','ulu','wang','cnn','meta','early','train','test64','pairs','testcr','cgan'])
    parser.add_argument('--enable-cross-resolution',action='store_true',help='Explicit future opt-in: also generate256 teachers, train cGAN and test CR300. Uses a separate results directory.')
    parser.add_argument('--cpus',type=int);parser.add_argument('--ram-gb',type=float)
    parser.add_argument('--allow-partial',action='store_true')
    args=parser.parse_args()
    os.environ['INITSKILL_INCLUDE_CR']='1' if args.enable_cross_resolution else '0'
    from initskill.methods.benchmark_init_v1.scope import make_scope
    spec=make_scope(args.enable_cross_resolution)
    out=ROOT/('results_cross' if args.enable_cross_resolution else 'results_id_ood')
    os.environ['INITSKILL_RESULT_DIR']=str(out)
    if args.action=='plan':print(json.dumps(dict(**spec,output=str(out)),ensure_ascii=False,indent=2));return
    if args.action in {'pairs','testcr','cgan'} and not args.enable_cross_resolution:
        raise SystemExit('This action is deferred. Cross-resolution is disabled by default.')
    if args.action=='prepare':
        from scripts.prepare_data import prepare
        prepare();return
    profile_path=ROOT/'.runtime/profile.json'
    if not profile_path.exists():raise SystemExit('Run: python scripts/setup_env.py --profile cpu --cpus 28 --ram-gb 64')
    profile=json.loads(profile_path.read_text(encoding='utf-8'))
    if Path(sys.prefix).resolve()!=Path(profile['main_prefix']).resolve():
        os.execve(profile['main_python'],[profile['main_python'],str(Path(__file__).resolve()),*sys.argv[1:]],dict(os.environ))
    from initskill import locking
    from scripts.runtime_parallel import settings
    if args.action=='configure' or args.cpus is not None or args.ram_gb is not None:
        old=profile.get('hardware',dict(cpus=28,ram_gb=64))
        profile['hardware']=dict(cpus=args.cpus if args.cpus is not None else old['cpus'],ram_gb=args.ram_gb if args.ram_gb is not None else old['ram_gb'])
        profile_path.write_text(json.dumps(profile,indent=2)+'\n',encoding='utf-8')
    parallel=settings(profile)
    if args.action=='configure':print(json.dumps(parallel,indent=2));return
    os.environ.update(PYTHONPATH=str(ROOT/'src'),INITSKILL_FE_SOLVER=profile['fe_solver'],
        INITSKILL_TRAIN_DEVICE=profile['torch_device'],CUBLAS_WORKSPACE_CONFIG=':4096:8',
        JAX_PLATFORMS=profile['jax_platforms'],XLA_PYTHON_CLIENT_PREALLOCATE='false',
        OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',
        MPLCONFIGDIR=str(ROOT/'.runtime/cache/matplotlib'))
    if parallel['large_slots']:os.environ['INITSKILL_HIGH_RES_SLOTS']=str(parallel['large_slots'])
    else:os.environ.pop('INITSKILL_HIGH_RES_SLOTS',None)
    from initskill.methods.benchmark_init_v1 import common,campaign
    control=out/'campaign';control.mkdir(parents=True,exist_ok=True);(control/'logs').mkdir(exist_ok=True)
    if args.action=='stop':
        from scripts.stop_run import stop
        stop(out);return
    if args.action=='status':
        print(json.dumps(dict(output=str(out),scope=spec,concurrency=parallel,
            complete=(control/'COMPLETE.json').exists(),completed=sorted(p.name for p in control.glob('*.complete.json')),
            failures=[json.loads(p.read_text()) for p in control.glob('ACTOR_FAILURE*.json')],
            progress=json.loads((control/'progress.json').read_text()) if (control/'progress.json').exists() else None),indent=2));return
    with (control/'bootstrap.lock').open('a') as lease:
        locking.flock(lease,locking.LOCK_EX)
        from scripts.prepare_data import prepare
        if args.action in {'start','verify','analyze','export'} or not (control/'data_ready.json').exists():
            prepare()
            import shutil
            for subset in ['train1000','valid100']:
                source=ROOT/'data/labels'/subset;dest=out/'labels'/subset;dest.mkdir(parents=True,exist_ok=True)
                for row in common.rows(subset):
                    for suffix in ['.json','.npz']:
                        name=row['source_task_id']+suffix
                        if not (dest/name).exists():shutil.copy2(source/name,dest/name)
                        if common.sha(dest/name)!=common.sha(source/name):raise ValueError('Changed local teacher: '+name)
            common.protocol(out)
            common.dump(control/'data_ready.json',dict(scope=spec,teacher_labels_copied=True,test_results_imported=False))
        plan_path=out/'RUN_PLAN.json'
        if plan_path.exists() and json.loads(plan_path.read_text())!=spec:raise ValueError('Execution scope changed; use the matching separate results directory')
        if not plan_path.exists():common.dump(plan_path,spec)
    if args.action=='verify':
        from scripts.preflight import run_checks
        run_checks()
        import unittest
        result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.discover(str(ROOT/'tests')))
        if not result.wasSuccessful():raise RuntimeError('Independent package regression failed')
        return
    if args.action=='analyze':
        command=[profile['main_python'],str(ROOT/'scripts/analyze.py'),'--out',str(out)]
        if args.allow_partial:command.append('--allow-partial')
        subprocess.run(command,check=True);return
    if args.action=='export':
        from scripts.export_results import export
        export(out,args.allow_partial);return
    common.protocol(out)
    executable=profile['main_python'];jax=profile['jax_python']
    command_root=[executable,str(Path(__file__).resolve())]
    def child(action):return command_root+[action]+(['--enable-cross-resolution'] if args.enable_cross_resolution else [])
    def model_folder(method):
        if method=='meta':return out/'models/meta_native'
        if method=='early':return out/'early_trajectories/models/meta_native'
        return out/'models'/method
    def ready(method,wait=False):
        if method=='uniform':return True
        folder=model_folder(method);f=folder/'frozen.json'
        while not f.exists():
            if not wait:return False
            time.sleep(3)
        frozen=json.loads(f.read_text());model=folder/('learned.eqx' if method in {'meta','early'} else 'best.pt')
        if common.sha(model)!=frozen['model_sha256']:raise ValueError('Changed frozen model: '+method)
        if method in {'ulu','wang'} and common.sha(folder/'pca.npz')!=frozen['pca_sha256']:raise ValueError('Changed PCA basis: '+method)
        if 'health_report_sha256' in frozen and common.sha(folder/'training_health.json')!=frozen['health_report_sha256']:raise ValueError('Changed training health report: '+method)
        cfg=json.loads((folder/'config.json').read_text())
        if common.sha(folder/'config.json')!=frozen['config_sha256']:raise ValueError('Changed frozen config')
        if cfg['train_keys']!=[r['task_key'] for r in common.rows('train1000')] or cfg['valid_keys']!=[r['task_key'] for r in common.rows('valid100')]:raise ValueError('Model split drift')
        if frozen.get('smoke_only') or cfg.get('smoke_only'):raise ValueError('Engineering model cannot enter formal tests')
        if method in {'meta','early'} and (cfg['outer_steps']!=6000 or cfg['batch_size']!=5 or cfg['inner_steps']!=(10 if method=='meta' else 5)):raise ValueError('Native training budget mismatch')
        if method=='cnn' and (cfg['epochs']!=500 or cfg['batch_size']!=16):raise ValueError('CNN declared budget mismatch')
        if method in {'ulu','wang'} and (cfg['epochs']!=1000 or cfg['rank']!=(80 if method=='ulu' else 200)):raise ValueError('PCA declared budget mismatch')
        return True
    def stage(name,command,training=False):
        if training:
            with (control/'training.lock').open('a') as lease:
                locking.flock(lease,locking.LOCK_EX);campaign.stage(out,name,list(map(str,command)))
        else:campaign.stage(out,name,list(map(str,command)))
    def train_one(method):
        run='initskill.methods.benchmark_init_v1.run'
        commands=dict(ulu=child('ulu'),wang=child('wang'),cnn=[executable,'-m',run,'train-cnn','--out',out,'--epochs',500,'--batch-size',16],
            meta=[jax,'-m',run,'train-meta','--out',out,'--outer-steps',6000,'--inner-steps',10],
            early=[jax,'-m','initskill.methods.benchmark_init_v1.early_trajectories.run','--out',out])
        if method in {'ulu','wang'}:
            if not ready(method):
                if method=='ulu':from initskill.methods.benchmark_init_v1.ulu_pca.core import train
                else:from initskill.methods.benchmark_init_v1.wang_pca.core import train
                train(out,epochs=1000,seed=42)
            ready(method);return
        if not ready(method):stage(method+'_train',commands[method],training=True)
        elif not (control/(method+'_train.complete.json')).exists():
            common.dump(control/(method+'_train.complete.json'),dict(command=list(map(str,commands[method])),adopted_verified_frozen_model=True,model_sha256=common.sha(model_folder(method)/('learned.eqx' if method in {'meta','early'} else 'best.pt'))))
        ready(method)
        if method in {'meta','early'}:
            stage(method+'_export',[jax,'-m','initskill.methods.benchmark_init_v1.meta_neural.export','--method',method,'--out',out,'--subsets',*spec['test_subsets']])
    with (control/(args.action+'.lock')).open('a') as lease:
        try:locking.flock(lease,locking.LOCK_EX|locking.LOCK_NB)
        except BlockingIOError:raise RuntimeError('This action is already running in this package/scope')
        if args.action=='start':
            import psutil,shutil
            from scripts.stop_run import stop_descendants
            previous=control/'execution.json'
            if previous.exists():
                history=control/'execution_history';history.mkdir(exist_ok=True);shutil.copy2(previous,history/(str(time.time_ns())+'.json'))
            old_failures=list(control.glob('ACTOR_FAILURE*.json'))
            if old_failures:
                archive=control/'failures';archive.mkdir(exist_ok=True)
                for item in old_failures:item.rename(archive/(str(time.time_ns())+'_'+item.name))
            actors={};streams=[]
            for actor in spec['actors']:
                stream=(control/'logs'/(actor+'.actor.log')).open('a');streams.append(stream)
                actors[actor]=subprocess.Popen(child(actor),stdout=stream,stderr=subprocess.STDOUT)
            common.dump(control/'execution.json',dict(pid=os.getpid(),process_created=psutil.Process().create_time(),root=str(ROOT.resolve()),scope=spec,started_unix=time.time(),concurrency=parallel,actors={k:v.pid for k,v in actors.items()},script_sha256=common.sha(Path(__file__))))
            try:
                while True:
                    failed=[(name,p.returncode) for name,p in actors.items() if p.poll() not in (None,0)]
                    if failed:
                        for name,code in failed:common.dump(control/('ACTOR_FAILURE_'+name+'.json'),dict(actor=name,returncode=code,unix=time.time()))
                        raise RuntimeError('Actor failed: '+','.join(name for name,_ in failed))
                    if all(p.poll() is not None for p in actors.values()):break
                    time.sleep(3)
            except BaseException as exc:
                common.dump(control/'LAST_STOP.json',dict(error=str(exc),unix=time.time()));stop_descendants(psutil.Process());raise
            finally:
                for stream in streams:stream.close()
            subprocess.run([executable,str(ROOT/'scripts/analyze.py'),'--out',str(out)],check=True)
            common.dump(control/'COMPLETE.json',dict(scope=spec,finished_unix=time.time(),strict_report_verified=True));return
        if args.action in {'ulu','wang','cnn','meta','early'}:train_one(args.action);return
        if args.action=='train':
            for method in ['ulu','wang']:
                stage(method+'_train',child(method),training=True);ready(method)
            for method in ['cnn','meta','early']:train_one(method)
        elif args.action in {'test64','testcr'}:
            import concurrent.futures
            subsets=['test_id300','test_ood300'] if args.action=='test64' else ['test_cr300']
            for method in spec['methods']:
                ready(method,wait=True)
                if method in {'meta','early'}:
                    while not (control/(method+'_export.complete.json')).exists():time.sleep(3)
                for subset in subsets:
                    jobs,settled=campaign.evaluation_plan(out,method,subset,500)
                    workers=parallel['small_workers'] if args.action=='test64' else parallel['cr_workers']
                    with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as pool:
                        futures=[pool.submit(campaign.evaluation_job,job) for job in jobs]
                        for i,future in enumerate(concurrent.futures.as_completed(futures),1):
                            record=future.result();print(method,subset,i,record,flush=True)
                            common.dump(control/'progress.json',dict(method=method,subset=subset,newly_settled=i,previously_settled=len(settled),planned=300,last=record,unix=time.time()))
        elif args.action=='pairs':
            for name,kind in [('paired_train','pairs-train'),('paired_valid','pairs-valid')]:
                stage(name,[executable,'-m','initskill.methods.benchmark_init_v1.campaign','--out',out,'--stage',kind,'--workers',parallel['teacher_workers']])
        elif args.action=='cgan':
            for name in ['paired_train','paired_valid','cnn_train']:
                while not (control/(name+'.complete.json')).exists():time.sleep(3)
            if not ready('cgan'):stage('cgan_train',[executable,'-m','initskill.methods.benchmark_init_v1.run','train-cgan','--out',out,'--epochs',20,'--batch-size',16],training=True)
            ready('cgan')
        common.dump(control/(args.action+'.actor.complete.json'),dict(scope=spec,unix=time.time()))

if __name__=='__main__':
    if not sys.flags.utf8_mode:os.execve(sys.executable,[sys.executable,str(Path(__file__).resolve()),*sys.argv[1:]],dict(os.environ,PYTHONUTF8='1'))
    import multiprocessing
    multiprocessing.freeze_support();multiprocessing.set_start_method('spawn',force=True)
    main()
