"""Install isolated main/JAX environments, including native Windows311 wheels."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import venv

ROOT=Path(__file__).resolve().parents[1]


def executable(folder):return folder/('Scripts/python.exe' if os.name=='nt' else 'bin/python')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--profile',choices=['cpu','cuda'],default='cpu')
    parser.add_argument('--use-current-main',action='store_true',help='Use current meta64-fresh environment for NumPy/SciPy/Torch')
    args=parser.parse_args()
    if sys.version_info[:2]!=(3,11):raise SystemExit('Please run setup with Python3.11 (your meta64-fresh environment).')
    if args.profile=='cuda' and os.name=='nt':raise SystemExit('Native Windows: use --profile cpu. Linux/WSL2 supports --profile cuda.')
    runtime=ROOT/'.runtime';runtime.mkdir(exist_ok=True)
    main_folder=runtime/('cuda_environment' if args.profile=='cuda' else 'environment')
    if args.use_current_main:main_python=Path(sys.executable)
    else:
        if not executable(main_folder).exists():venv.EnvBuilder(with_pip=True).create(main_folder)
        main_python=executable(main_folder)
    jax_folder=runtime/'jax_environment'
    if not executable(jax_folder).exists():venv.EnvBuilder(with_pip=True).create(jax_folder)
    jax_python=executable(jax_folder)
    def pip(python,*arguments):subprocess.run([str(python),'-m','pip',*map(str,arguments)],check=True)
    pip(main_python,'install','-r',ROOT/'requirements/main.txt')
    index='https://download.pytorch.org/whl/'+('cu124' if args.profile=='cuda' else 'cpu')
    pip(main_python,'install','torch==2.5.1','--index-url',index)
    pip(jax_python,'install','-r',ROOT/'requirements/jax.txt')
    if args.profile=='cuda':
        pip(jax_python,'install','jax[cuda12]==0.6.2')
        pip(main_python,'install','--target',runtime/'solver_gpu_packages','--no-deps','-r',ROOT/'requirements/cudss.txt',
            '--report',runtime/'solver_gpu_install_report.json')
        if args.use_current_main:
            # cuDSS discovers actual wheel DLLs via the registered interpreter.
            pass
    config=dict(profile=args.profile,main_python=str(main_python.absolute()),pca_python=str(main_python.absolute()),
        jax_python=str(jax_python.absolute()),fe_solver='cudss' if args.profile=='cuda' else 'superlu',
        torch_device='cuda' if args.profile=='cuda' else 'cpu',jax_platforms='cuda,cpu' if args.profile=='cuda' else 'cpu',
        main_prefix=str(Path(sys.prefix).absolute() if args.use_current_main else main_folder.absolute()),
        pca_prefix=str(Path(sys.prefix).absolute() if args.use_current_main else main_folder.absolute()),
        jax_prefix=str(jax_folder.absolute()),native_windows=os.name=='nt')
    (runtime/'profile.json').write_text(json.dumps(config,indent=2)+'\n')
    for name,python in [('main',main_python),('jax',jax_python)]:
        packages=json.loads(subprocess.check_output([str(python),'-m','pip','list','--format=json'],text=True))
        (runtime/(name+'_environment.json')).write_text(json.dumps(packages,indent=2)+'\n')
    subprocess.run([str(jax_python),'-c','import jax, cvxopt.cholmod; print("JAX",jax.__version__,jax.devices(),"CVXOPT CHOLMOD OK")'],check=True,
                   env=dict(os.environ,JAX_PLATFORMS=config['jax_platforms'],XLA_PYTHON_CLIENT_PREALLOCATE='false'))
    print('Environment ready. Next: python benchmark.py verify')


if __name__=='__main__':main()
