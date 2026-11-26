"""Explicit training device selection; common OC and inference remain CPU."""
import os
import torch


def training_device():
    name=os.environ.get('INITSKILL_TRAIN_DEVICE','cpu')
    if name not in {'cpu','cuda'}:
        raise ValueError('INITSKILL_TRAIN_DEVICE must be cpu or cuda')
    if name=='cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA training explicitly requested but unavailable')
    if name=='cuda':
        if os.environ.get('CUBLAS_WORKSPACE_CONFIG') not in {':4096:8',':16:8'}:
            raise RuntimeError('Set CUBLAS_WORKSPACE_CONFIG before deterministic CUDA training')
        torch.backends.cuda.matmul.allow_tf32=False
        torch.backends.cudnn.allow_tf32=False
        torch.backends.cudnn.benchmark=False
        torch.backends.cudnn.deterministic=True
    return torch.device(name)


def device_metadata(device):
    return dict(device=str(device),torch_version=str(torch.__version__),
                cuda_runtime=torch.version.cuda,
                gpu_name=torch.cuda.get_device_name(device) if device.type=='cuda' else None,
                TF32=False,deterministic_algorithms=torch.are_deterministic_algorithms_enabled())
