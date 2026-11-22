"""One-command entry: python -u run.py. Run from any working directory."""
import runpy
from pathlib import Path

if __name__ == '__main__':
    runpy.run_path(str(Path(__file__).resolve().parent/'code/run.py'),run_name='__main__')
