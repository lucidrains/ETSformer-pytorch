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

class FrequencyAttention(nn.Module):
    def __init__(
        self,
        *,
        K = 4
    ):
        super().__init__()
        self.K = K

    def forward(self, x):
        freqs = torch.fft.rfft(x, dim = 1)

        # get amplitudes

        amp = freqs.abs()

        # topk amplitudes - for seasonality, branded as attention

        _, topk_amp_indices = amp.topk(k = self.K, dim = 1)

        # scatter back

        topk_freqs = torch.zeros_like(freqs).scatter(1, topk_amp_indices, freqs)

        # inverse fft

        return torch.fft.irfft(topk_freqs, dim = 1)

# main class

class ETSFormer(nn.Module):
    def __init__(
        self,
        *,
        time_features,
        model_dim,
        heads = 8,
        K = 4
    ):
        super().__init__()

    def forward(self, x):
        return x
