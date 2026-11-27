import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parent


def verify_package():
    manifest=ROOT/'PACKAGE_MANIFEST.json'
    if not manifest.exists():
        raise ValueError('PACKAGE_MANIFEST.json missing')
    for name,expected in json.loads(manifest.read_text()).items():
        digest=hashlib.sha256((ROOT/name).read_bytes()).hexdigest()
        if digest!=expected:raise ValueError('Package changed: '+name)


if __name__=='__main__':
    verify_package()
    print('Package hashes verified. No API/FE.')
