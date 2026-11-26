"""Platform-only reader: no JAX/NumPy2 runtime enters the common OC evaluator."""
import json
from pathlib import Path
import numpy as np
from ..common import sha


def predict(out,row,method='meta'):
    out=Path(out);folder=out/'native_predictions'/method/row['task_key'];p=folder/'density.npz'
    metadata=json.loads((folder/'metadata.json').read_text())
    model_out=out if method=='meta' else out/'early_trajectories'
    frozen=json.loads((model_out/'models/meta_native/frozen.json').read_text())
    if (metadata['prediction_sha256']!=sha(p) or metadata['task_key']!=row['task_key'] or
        metadata['group_id']!=row['group_id'] or metadata['model_sha256']!=frozen['model_sha256'] or
        metadata['physical_inputs']!={k:row['assets'][k]['sha256'] for k in ('forces','normals','target_vf')}):
        raise ValueError('Native-to-platform handoff identity mismatch')
    with np.load(p,allow_pickle=False) as a:raw=a['raw'].copy()
    return raw,metadata
