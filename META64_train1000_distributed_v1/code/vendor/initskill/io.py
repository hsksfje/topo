import hashlib
import json
import os
from pathlib import Path
import tempfile
import time


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def _replace(src, dst):
    last = None
    for attempt in range(20):
        try:
            os.replace(src, dst)
            return
        except OSError as exc:
            if getattr(exc, 'winerror', None) not in (5, 32) and getattr(exc, 'errno', None) not in (13,):
                raise
            last = exc
            time.sleep(0.05 * (attempt + 1))
    raise last


def dump(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(dir=path.parent, prefix='.' + path.name)
    try:
        with os.fdopen(fd, 'w', encoding='utf-8') as f:
            json.dump(value, f, indent=2, ensure_ascii=False, allow_nan=False)
            f.write('\n')
        _replace(name, path)
    finally:
        if os.path.exists(name):
            os.unlink(name)


def save_npz(path, **arrays):
    import numpy as np
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(dir=path.parent, suffix='.npz')
    os.close(fd)
    try:
        np.savez_compressed(name, **arrays)
        _replace(name, path)
    finally:
        if os.path.exists(name):
            os.unlink(name)
