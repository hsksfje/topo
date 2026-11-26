"""Create a clean Python3.11 environment; no Torch/JAX/Conda MKL imports."""
from pathlib import Path
import json
import os
import subprocess
import sys
import venv
ROOT=Path(__file__).resolve().parent
if __name__=='__main__':
    if sys.version_info[:2]!=(3,11):raise SystemExit('Run setup with Python3.11')
    folder=ROOT/'.runtime/environment';python=folder/('Scripts/python.exe' if os.name=='nt' else 'bin/python')
    if not python.exists():venv.EnvBuilder(with_pip=True).create(folder)
    subprocess.run([str(python),'-m','pip','install','-r',str(ROOT/'requirements.txt')],check=True)
    profile=dict(python=str(python.absolute()),prefix=str(folder.absolute()))
    (ROOT/'.runtime/profile.json').write_text(json.dumps(profile,indent=2)+'\n',encoding='utf-8')
    print('Environment ready. Next: python run.py verify')
