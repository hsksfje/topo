#!/usr/bin/env python3
"""Helper-device runner: converge ONLY the fixed (non-LLM) subset transferred from device A.

Usage (from this directory or anywhere):
    python run_fixed_convergence.py --data-dir <extracted package dir> --output <results dir> [--workers N]

The package dir must contain: config.json, corpus/, extra/, inputs/.
"""
import os, sys, json, argparse
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED

for k in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[k] = '1'

TOPO = Path(__file__).resolve().parents[1]
CODE = TOPO / 'META64_train1000_distributed_v1' / 'code'
sys.path.insert(0, str(CODE))
sys.path.insert(0, str(CODE / 'vendor'))

import continuation as cont


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data-dir', type=Path, required=True,
                    help='extracted package dir (contains config.json, corpus/, extra/, inputs/)')
    ap.add_argument('--output', type=Path, required=True,
                    help='where to write convergence results')
    ap.add_argument('--workers', type=int, default=None,
                    help='number of worker processes (default: usable CPUs - 1)')
    args = ap.parse_args()

    data = args.data_dir.resolve()
    cfg = json.loads((data / 'config.json').read_text(encoding='utf-8'))
    cfg['fresh_output'] = str(data)  # so source_folder() and new_model() resolve to the transferred data

    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    cont.OUT = out

    jobs = [(r, m) for r in cfg['records'] for m in cfg['methods']
            if cfg['trajectories'].get(r['task_id'] + '/' + m, {}).get('source_status') == 'complete'
            and not (out / 'tasks' / r['task_id'] / m / 'result.json').exists()]

    cpus = cont.available_cpus()
    workers = args.workers if args.workers is not None else max(1, len(cpus) - 1)
    print('fixed convergence jobs:', len(jobs), 'workers:', workers, flush=True)

    try:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            pending = {pool.submit(cont.run_one, r, m, cfg, str(out)) for r, m in jobs}
            while pending:
                done, pending = wait(pending, timeout=15, return_when=FIRST_COMPLETED)
                for f in done:
                    v = f.result()
                    print('FIXED_CONV', v['task_id'], v['method'], v['status'], v.get('step'), flush=True)
                if (out / 'PAUSE_REQUESTED').exists():
                    for f in pending:
                        f.cancel()
                    raise SystemExit('paused by flag')
        print('FIXED_CONV_DONE', flush=True)
    except BaseException as e:
        print('FIXED_CONV_STOP', type(e).__name__, e, flush=True)
        raise


if __name__ == '__main__':
    main()
