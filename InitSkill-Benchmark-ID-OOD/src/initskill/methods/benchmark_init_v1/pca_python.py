"""CLI compatibility router; each PCA method owns its complete training code."""
def train(out,method,epochs=1000,seed=42):
    if method=='ulu':from .ulu_pca.core import train as run
    elif method=='wang':from .wang_pca.core import train as run
    else:raise ValueError(method)
    return run(out,epochs,seed)


def predict(out,row,method):
    if method=='ulu':from .ulu_pca.core import predict as run
    elif method=='wang':from .wang_pca.core import predict as run
    else:raise ValueError(method)
    return run(out,row)
