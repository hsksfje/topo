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


def dump(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(dir=path.parent, prefix='.' + path.name)
    try:
        with os.fdopen(fd, 'w', encoding='utf-8') as f:
            json.dump(value, f, indent=2, ensure_ascii=False, allow_nan=False)
            f.write('\n')
        _retry_file_operation(os.replace, name, path)
    finally:
        if os.path.exists(name):
            _retry_file_operation(os.unlink, name)


def save_npz(path, **arrays):
    import numpy as np
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(dir=path.parent, suffix='.npz')
    os.close(fd)
    try:
        np.savez_compressed(name, **arrays)
        _retry_file_operation(os.replace, name, path)
    finally:
        if os.path.exists(name):
            _retry_file_operation(os.unlink, name)


def _retry_file_operation(fn, *args):
    for attempt in range(9):
        try:
            return fn(*args)
        except PermissionError:
            if attempt == 8:
                raise
            time.sleep(min(2., .05 * 2**attempt))
