"""Shared dense layer primitive; method training/optimizers live in independent folders."""
import torch
from torch import nn

class Dense(nn.Module):
    def __init__(self,input_size,output_size,hidden):
        super().__init__();layers=[];a=input_size
        for b in hidden:layers.extend([nn.Linear(a,b),nn.Tanh()]);a=b
        layers.append(nn.Linear(a,output_size));self.layers=nn.Sequential(*layers)
        for layer in self.layers:
            if isinstance(layer,nn.Linear):nn.init.xavier_uniform_(layer.weight);nn.init.zeros_(layer.bias)
    def forward(self,z):return self.layers(z)
