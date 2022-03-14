import torch
import torch.nn.functional as F
from torch import nn, einsum

from einops import rearrange
from einops.layers.torch import Rearrange

# helper functions

def exists(val):
    return val is not None

# classes

def FeedForward(dim, mult = 4, dropout = 0.):
    return nn.Sequential(
        nn.Linear(dim, dim * mult),
        nn.Sigmoid(),
        nn.Dropout(dropout),
        nn.Linear(dim * mult, dim),
        nn.Dropout(dropout)
    )

def InputEmbedding(time_features, model_dim, kernel = 3, dropout = 0.):
    return nn.Sequential(
        Rearrange('b n d -> b d n'),
        nn.Conv1d(time_features, model_dim, kernel = kernel, padding = kernel // 2),
        nn.Dropout(dropout),
        Rearrange('b d n -> b n d'),
    )

# main class

class ETSFormer(nn.Module):
    def __init__(self, time_features, model_dim):
        super().__init__()

    def forward(self, x):
        return x
