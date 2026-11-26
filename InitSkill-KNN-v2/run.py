"""Independent KNN benchmark: verify, start, status, stop, report, export."""
from pathlib import Path
import argparse
import json
import multiprocessing
import os
import sys
ROOT=Path(__file__).resolve().parent

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=['verify','start','status','stop','report','export'])
    parser.add_argument('--cpus',type=int,default=28);parser.add_argument('--ram-gb',type=float,default=64)
    parser.add_argument('--allow-partial',action='store_true')
    args=parser.parse_args()
    path=ROOT/'.runtime/profile.json'
    if not path.exists():raise SystemExit('First run: python setup.py')
    profile=json.loads(path.read_text())
    if Path(sys.prefix).resolve()!=Path(profile['prefix']).resolve():os.execve(profile['python'],[profile['python'],str(Path(__file__).resolve()),*sys.argv[1:]],dict(os.environ))
    sys.path.insert(0,str(ROOT/'src'))
    from scripts.experiment import OUT,allocation,start,stop,summarize
    if args.action=='start':start(args.cpus,args.ram_gb)
    elif args.action=='verify':
        from scripts.verify import verify
        verify()
    elif args.action=='stop':stop()
    elif args.action=='status':
        execution=OUT/'campaign/execution.json'
        print(json.dumps(dict(execution=json.loads(execution.read_text()) if execution.exists() else None,
            requested_concurrency=allocation(args.cpus,args.ram_gb),complete=(OUT/'campaign/COMPLETE.json').exists(),
            last_progress=json.loads((OUT/'campaign/progress.json').read_text()) if (OUT/'campaign/progress.json').exists() else None,
            progress=json.loads((OUT/'reports/coverage.json').read_text()) if (OUT/'reports/coverage.json').exists() else {}),indent=2))
    elif args.action=='report':summarize(strict=not args.allow_partial)
    elif args.action=='export':
        from scripts.export import export
        export(args.allow_partial)

if __name__=='__main__':
    if not sys.flags.utf8_mode:os.execve(sys.executable,[sys.executable,str(Path(__file__).resolve()),*sys.argv[1:]],dict(os.environ,PYTHONUTF8='1'))
    # Set before NumPy/SciPy imports, including spawn workers.
    for name in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']:os.environ[name]='1'
    os.environ['INITSKILL_FE_SOLVER']='superlu'
    multiprocessing.freeze_support();multiprocessing.set_start_method('spawn',force=True)
    main()
