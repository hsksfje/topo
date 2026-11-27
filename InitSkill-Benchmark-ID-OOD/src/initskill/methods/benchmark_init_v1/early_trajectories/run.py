import argparse
from pathlib import Path
from .core import train
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args();train(a.out)
