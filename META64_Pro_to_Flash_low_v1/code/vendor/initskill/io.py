import hashlib
import json
import os
from pathlib import Path
import tempfile
import time

def replace_with_retry(source, target):
    """Allow Windows antivirus/indexer transient file handles to close."""
    for attempt in range(20):
        try:
            os.replace(source, target)
            return
        except PermissionError:
            if attempt == 19:
                raise
            time.sleep(min(.05 * (attempt + 1), .5))


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
        replace_with_retry(name, path)
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
        replace_with_retry(name, path)
    finally:
        if os.path.exists(name):
            os.unlink(name)
