"""Portable entry point; no API or external dataset dependency."""
import runpy
from pathlib import Path

if __name__ == '__main__':
    runpy.run_path(str(Path(__file__).resolve().parent / 'code/run.py'), run_name='__main__')
