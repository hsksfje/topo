#!/usr/bin/env python3
"""Current authorized scope: Train1000, uniformValid100, three uniformTest300.

All numerical methods and budgets stay unchanged. Older Valid300 selected
models are retired. Evaluation waits for a verified import manifest; every
method sees exactly the same sampled tasks. Training/teachers run independently.
"""
import argparse
import concurrent.futures
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'results'
OLD=ROOT/'.runtime'
CUDA=CPU=JAX=None
sys.path.insert(0,str(ROOT/'src'))
from initskill import locking as fcntl
METHODS=['uniform','knn','ulu','wang','cnn','meta','early','cgan']
ACTORS=['pca','neural','pairs','test64','testcr','cgan']


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=['prepare','verify','analyze','configure','start','stop','status','ulu','wang']+ACTORS)
    parser.add_argument('--cpus',type=int)
    parser.add_argument('--ram-gb',type=float)
    args=parser.parse_args()
    if args.action=='stop':
        from scripts.stop_run import stop
        stop();return
    if args.action=='prepare':
        from scripts.prepare_data import prepare
        prepare();return
    profile_path=OLD/'profile.json'
    if not profile_path.exists():raise SystemExit('Run: python scripts/setup_env.py --profile cpu --use-current-main')
    profile=json.loads(profile_path.read_text())
    global CUDA,CPU,JAX
    CUDA=Path(profile['main_python']);CPU=Path(profile['pca_python']);JAX=Path(profile['jax_python'])
    selected=CPU if args.action in {'pca','ulu','wang'} else CUDA
    selected_prefix=Path(profile['pca_prefix' if args.action in {'pca','ulu','wang'} else 'main_prefix'])
    if Path(sys.prefix).resolve()!=selected_prefix.resolve():
        os.execve(str(selected),[str(selected),str(Path(__file__).resolve()),*sys.argv[1:]],dict(os.environ))
    from scripts.runtime_parallel import cpu_allocation,settings,migrate_resource_protocol
    if args.action=='configure' or args.cpus is not None or args.ram_gb is not None:
        if profile['fe_solver']!='superlu':raise SystemExit('The CPU preset requires the CPU/SuperLU profile')
        if args.cpus is None or args.ram_gb is None:import psutil
        cpus=args.cpus or psutil.cpu_count()
        ram=args.ram_gb or psutil.virtual_memory().total/1024**3
        profile['concurrency']=cpu_allocation(cpus,ram)
        profile_path.write_text(json.dumps(profile,indent=2)+'\n',encoding='utf-8')
        if args.action=='configure':print(json.dumps(profile['concurrency'],indent=2));return
    parallel=settings(profile)
    if parallel['large_slots'] is not None:os.environ['INITSKILL_HIGH_RES_SLOTS']=str(parallel['large_slots'])
    else:os.environ.pop('INITSKILL_HIGH_RES_SLOTS',None)
    os.environ.update(PYTHONPATH=str(ROOT/'src'),INITSKILL_FE_SOLVER=profile['fe_solver'],
        INITSKILL_TRAIN_DEVICE=profile['torch_device'],CUBLAS_WORKSPACE_CONFIG=':4096:8',
        JAX_PLATFORMS=profile['jax_platforms'],XLA_PYTHON_CLIENT_PREALLOCATE='false',
        OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',MPLCONFIGDIR=str(OLD/'cache/matplotlib'))
    sys.path.insert(0,str(ROOT/'src'))
    from initskill.methods.benchmark_init_v1 import common,campaign
    control=OUT/'campaign';control.mkdir(parents=True,exist_ok=True)
    (control/'logs').mkdir(exist_ok=True)
    if args.action=='status':
        print(json.dumps(dict(output=str(OUT),concurrency=parallel,scope={'train1000':1000,'valid100':100,'test_id300':300,'test_ood300':300,'test_cr300':300},
            completed=sorted(p.name for p in control.glob('*.complete.json')),
            failures=[json.loads(p.read_text()) for p in control.glob('ACTOR_FAILURE*.json')]),indent=2));return
    with (control/'bootstrap.lock').open('a') as bootstrap:
        fcntl.flock(bootstrap,fcntl.LOCK_EX)
        migrate_resource_protocol(OUT,common)
        if args.action in {'start','verify','analyze'} or not (control/'data_ready.json').exists():
            from scripts.prepare_data import prepare
            prepare()
            import shutil
            for subset in ['train1000','valid100']:
                src=ROOT/'data/labels'/subset;dest=OUT/'labels'/subset
                dest.mkdir(parents=True,exist_ok=True)
                for row in common.rows(subset):
                    for suffix in ['.json','.npz']:
                        name=row['source_task_id']+suffix
                        if not (dest/name).exists():shutil.copy2(src/name,dest/name)
            common.protocol(OUT)
            from initskill.methods.benchmark_init_v1.knn.core import train_knn
            if not (OUT/'models/knn/frozen.json').exists():train_knn(OUT)
            common.dump(control/'data_ready.json',dict(passed=True,counts=common.protocol(OUT)['task_counts'],existing_test_outcomes_imported=False))
    if args.action=='verify':
        from scripts.preflight import run_checks
        run_checks();return
    if args.action=='analyze':
        subprocess.run([str(CPU),str(ROOT/'scripts/analyze.py'),'--out',str(OUT),'--allow-partial'],check=True);return
    common.protocol(OUT)
    leases=control/'actor_locks';leases.mkdir(exist_ok=True)
    with (leases/(args.action+'.lock')).open('a') as lease:
        fcntl.flock(lease,fcntl.LOCK_EX)
        if args.action=='start':
            previous=control/'failures';previous.mkdir(exist_ok=True)
            for path in control.glob('ACTOR_FAILURE*.json'):
                path.rename(previous/(str(int(time.time()))+'_'+path.name))
            execution_path=control/'execution.json'
            if execution_path.exists():
                import shutil
                history=control/'execution_history';history.mkdir(exist_ok=True)
                shutil.copy2(execution_path,history/(str(time.time_ns())+'.json'))
            processes={};logs=[]
            for actor in ACTORS:
                stream=(control/'logs'/(actor+'.actor.log')).open('a');logs.append(stream)
                command=[str(CPU if actor=='pca' else CUDA),str(Path(__file__).resolve()),actor]
                process=subprocess.Popen(command,stdout=stream,stderr=subprocess.STDOUT,env=os.environ.copy())
                processes[actor]=process
            common.dump(control/'execution.json',dict(started_unix=time.time(),pid=os.getpid(),
                script_sha256=common.sha(Path(__file__)),process_created=__import__('psutil').Process().create_time(),
                root=str(ROOT.resolve()),scope=common.protocol(OUT)['task_counts'],
                actors={k:v.pid for k,v in processes.items()},concurrency=parallel,external_handoff=True,
                note='Fresh models under uniformValid100; no imported old test outcomes; no train-set performance comparison'))
            recorded=set()
            try:
                while True:
                    for actor,process in processes.items():
                        code=process.poll()
                        if code not in (None,0) and actor not in recorded:
                            common.dump(control/('ACTOR_FAILURE_'+actor+'.json'),dict(actor=actor,returncode=code,observed_unix=time.time()))
                            recorded.add(actor)
                    if recorded:
                        from scripts.stop_run import stop_descendants
                        stop_descendants(__import__('psutil').Process());break
                    if all(p.poll() is not None for p in processes.values()):break
                    time.sleep(5)
            except KeyboardInterrupt:
                from scripts.stop_run import stop_descendants
                stop_descendants(__import__('psutil').Process())
                common.dump(control/'INTERRUPTED.json',dict(unix=time.time(),scope=common.protocol(OUT)['task_counts'],resume='Run the same start command; coherent checkpoints and attempted FE are retained.'))
                raise SystemExit(130)
            for stream in logs:stream.close()
            if recorded:raise RuntimeError('Independent actor failed: '+','.join(sorted(recorded)))
            subprocess.run([str(CPU),str(ROOT/'scripts/analyze.py'),
                '--out',str(OUT)],check=True,env=os.environ.copy())
            common.dump(control/'COMPLETE.json',dict(methods=METHODS,scope=common.protocol(OUT)['task_counts'],
                finished_unix=time.time(),strict_report_verified=True));return
        marker=control/(args.action+'.actor.complete.json')
        if marker.exists():return

        def wait(path):
            while not path.exists():time.sleep(5)

        def ready(method):
            if method=='uniform':return
            folder=OUT/'models'/method
            if method in {'meta','early'}:
                base=OUT if method=='meta' else OUT/'early_trajectories'
                folder=base/'models/meta_native'
                wait(control/(method+'_export.complete.json'))
            wait(folder/'frozen.json')
            frozen=json.loads((folder/'frozen.json').read_text())
            model=folder/('model.npz' if method=='knn' else 'learned.eqx' if method in {'meta','early'} else 'best.pt')
            assert common.sha(model)==frozen['model_sha256']
            if method!='knn':
                cfg=json.loads((folder/'config.json').read_text())
                assert common.sha(folder/'config.json')==frozen['config_sha256']
                assert cfg['train_keys']==[r['task_key'] for r in common.rows('train1000')]
                assert cfg['valid_keys']==[r['task_key'] for r in common.rows('valid100')]

        def stage(name,command):campaign.stage(OUT,name,[str(v) for v in command])
        run='initskill.methods.benchmark_init_v1.run'
        if args.action in {'ulu','wang'}:
            if not (OUT/'models'/args.action/'frozen.json').exists():
                if args.action=='ulu':from initskill.methods.benchmark_init_v1.ulu_pca.core import train
                else:from initskill.methods.benchmark_init_v1.wang_pca.core import train
                train(OUT,epochs=1000,seed=42)
            ready(args.action)
        elif args.action=='pca':
            for method in ['ulu','wang']:
                stage(method+'_train',[CPU,Path(__file__).resolve(),method])
        elif args.action=='neural':
            stage('cnn_train',[CUDA,'-m',run,'train-cnn','--out',OUT,'--epochs',500,'--batch-size',16])
            stage('meta_train',[JAX,'-m',run,'train-meta','--out',OUT,'--outer-steps',6000,'--inner-steps',10])
            stage('meta_export',[JAX,'-m','initskill.methods.benchmark_init_v1.meta_neural.export','--method','meta','--out',OUT])
            stage('early_train',[JAX,'-m','initskill.methods.benchmark_init_v1.early_trajectories.run','--out',OUT])
            stage('early_export',[JAX,'-m','initskill.methods.benchmark_init_v1.meta_neural.export','--method','early','--out',OUT])
        elif args.action=='pairs':
            for name,kind in [('paired_train','pairs-train'),('paired_valid','pairs-valid')]:
                stage(name,[CUDA,'-m','initskill.methods.benchmark_init_v1.campaign','--out',OUT,'--stage',kind,'--workers',parallel['teacher_workers']])
        elif args.action=='cgan':
            for name in ['paired_train','paired_valid','cnn_train']:wait(control/(name+'.complete.json'))
            stage('cgan_train',[CUDA,'-m',run,'train-cgan','--out',OUT,'--epochs',20,'--batch-size',16])
        elif args.action in {'test64','testcr'}:
            wait(control/'data_ready.json')
            subsets=['test_id300','test_ood300'] if args.action=='test64' else ['test_cr300']
            for method in METHODS:
                ready(method)
                for subset in subsets:
                    jobs,settled=campaign.evaluation_plan(OUT,method,subset,common.MAX_OC_STEPS)
                    assert len(jobs)+len(settled)==300
                    print('sampled-evaluation',method,subset,'settled',len(settled),'pending',len(jobs),flush=True)
                    with concurrent.futures.ProcessPoolExecutor(max_workers=parallel['small_workers'] if args.action=='test64' else parallel['cr_workers']) as pool:
                        futures=[pool.submit(campaign.evaluation_job,job) for job in jobs]
                        for i,future in enumerate(concurrent.futures.as_completed(futures),1):
                            print(method,subset,i,future.result(),flush=True)
                            if i%25==0:campaign.report(OUT)
                    campaign.report(OUT)
        common.dump(marker,dict(finished_unix=time.time(),actor=args.action,scope=common.protocol(OUT)['task_counts']))


if __name__=='__main__':
    if not sys.flags.utf8_mode:
        os.execve(sys.executable,[sys.executable,str(Path(__file__).resolve()),*sys.argv[1:]],dict(os.environ,PYTHONUTF8='1'))
    import multiprocessing
    multiprocessing.freeze_support()
    multiprocessing.set_start_method('spawn',force=True)
    main()
