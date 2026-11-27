"""Reptile short-trajectory ablation: independent checkpoint, k=5 inner steps."""
from pathlib import Path
from ..meta_neural import core as meta


def train(out,outer_steps=6000,seed=42):
    return meta.train(Path(out)/'early_trajectories',outer_steps=outer_steps,inner_steps=5,seed=seed)


def predict(out,row):
    raw,metadata=meta.predict(Path(out)/'early_trajectories',row)
    metadata['ablation']='same meta-neural mechanism; inner steps 5 vs 10, equal outer steps with cost disclosed'
    return raw,metadata
