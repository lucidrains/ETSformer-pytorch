from math import pi
import torch
import torch.nn.functional as F
from torch import nn, einsum

from scipy.fftpack import next_fast_len
from einops import rearrange, repeat
from einops.layers.torch import Rearrange

# helper functions

def exists(val):
    return val is not None

# fourier helpers

def fourier_extrapolate(signal, start, end):
    device = signal.device
    fhat = torch.fft.fft(signal)
    fhat_len = fhat.shape[-1]
    time = torch.linspace(start, end - 1, end - start, device = device, dtype = torch.complex64)
    freqs = torch.linspace(0, fhat_len - 1, fhat_len, device = device, dtype = torch.complex64)
    res = fhat[..., None, :] * (1.j * 2 * pi * freqs[..., None, :] * time[..., :, None] / fhat_len).exp() / fhat_len
    return res.sum(dim = -1).real

# classes

def InputEmbedding(time_features, model_dim, kernel_size = 3, dropout = 0.):
    return nn.Sequential(
        Rearrange('b n d -> b d n'),
        nn.Conv1d(time_features, model_dim, kernel_size = kernel_size, padding = kernel_size // 2),
        nn.Dropout(dropout),
        Rearrange('b d n -> b n d'),
    )

def FeedForward(dim, mult = 4, dropout = 0.):
    return nn.Sequential(
        nn.Linear(dim, dim * mult),
        nn.Sigmoid(),
        nn.Dropout(dropout),
        nn.Linear(dim * mult, dim),
        nn.Dropout(dropout)
    )

class FeedForwardBlock(nn.Module):
    def __init__(
        self,
        *,
        dim,
        **kwargs
    ):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.ff = FeedForward(dim, **kwargs)
        self.post_norm = nn.LayerNorm(dim)

    def forward(self, x):
        x = self.norm(x)
        return self.post_norm(x + self.ff(x))

# encoder related classes

## multi-head exponential smoothing attention

def conv1d_fft(x, weights, dim = -2, weight_dim = -1):
    # Algorithm 3 in paper

    N = x.shape[dim]
    M = weights.shape[weight_dim]

    fast_len = next_fast_len(N + M - 1)

    f_x = torch.fft.rfft(x, n = fast_len, dim = dim)
    f_weight = torch.fft.rfft(weights, n = fast_len, dim = weight_dim)

    f_v_weight = f_x * rearrange(f_weight.conj(), '... -> ... 1')
    out = torch.fft.irfft(f_v_weight, fast_len, dim = dim)
    out = out.roll(-1, dims = (dim,))

    indices = torch.arange(start = fast_len - N, end = fast_len, dtype = torch.long, device = x.device)
    out = out.index_select(dim, indices)
    return out

class MHESA(nn.Module):
    def __init__(
        self,
        *,
        dim,
        heads = 8,
        dropout = 0.
    ):
        super().__init__()
        self.heads = heads
        self.initial_state = nn.Parameter(torch.randn(heads, dim // heads))

        self.dropout = nn.Dropout(dropout)
        self.alpha = nn.Parameter(torch.randn(heads))

        self.project_in = nn.Linear(dim, dim)
        self.project_out = nn.Linear(dim, dim)

    def naive_Aes(self, x, weights):
        n, h = x.shape[-2], self.heads

        # in appendix A.1 - Algorithm 2

        arange = torch.arange(n, device = x.device)

        weights = repeat(weights, '... l -> ... t l', t = n)
        indices = repeat(arange, 'l -> h t l', h = h, t = n)

        indices = (indices - rearrange(arange + 1, 't -> 1 t 1')) % n

        weights = weights.gather(-1, indices)
        weights = self.dropout(weights)

        # causal

        weights = weights.tril()

        # multiply

        output = einsum('b h n d, h m n -> b h m d', x, weights)
        return output

    def forward(self, x, naive = False):
        b, n, d, h, device = *x.shape, self.heads, x.device

        # linear project in

        x = self.project_in(x)

        # split out heads

        x = rearrange(x, 'b n (h d) -> b h n d', h = h)

        # temporal difference

        x = torch.cat((
            repeat(self.initial_state, 'h d -> b h 1 d', b = b),
            x
        ), dim = -2)

        x = x[:, :, 1:] - x[:, :, :-1]

        # prepare exponential alpha

        alpha = self.alpha.sigmoid()
        alpha = rearrange(alpha, 'h -> h 1')

        # arange == powers

        arange = torch.arange(n, device = device)
        weights = alpha * (1 - alpha) ** torch.flip(arange, dims = (0,))

        if naive:
            output = self.naive_Aes(x, weights)
        else:
            output = conv1d_fft(x, weights)

        # get initial state contribution

        init_weight = (1 - alpha) ** (arange + 1)
        init_output = rearrange(init_weight, 'h n -> h n 1') * rearrange(self.initial_state, 'h d -> h 1 d')

        output = output + init_output

        # merge heads

        output = rearrange(output, 'b h n d -> b n (h d)')
        return self.project_out(output)

## frequency attention

class FrequencyAttention(nn.Module):
    def __init__(
        self,
        *,
        K = 4,
        dropout = 0.
    ):
        super().__init__()
        self.K = K
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        freqs = torch.fft.rfft(x, dim = 1)

        # get amplitudes

        amp = freqs.abs()
        amp = self.dropout(amp)

        # topk amplitudes - for seasonality, branded as attention

        topk_amp, _ = amp.topk(k = self.K, dim = 1, sorted = True)

        # mask out all freqs with lower amplitudes than the lowest value of the topk above

        topk_freqs = freqs.masked_fill(amp < topk_amp[:, -1:], 0.+0.j)

        # inverse fft

        return torch.fft.irfft(topk_freqs, dim = 1)

## level module

class Level(nn.Module):
    def __init__(self, time_features, model_dim):
        super().__init__()
        self.alpha = nn.Parameter(torch.Tensor([0.]))
        self.to_growth = nn.Linear(model_dim, time_features)
        self.to_seasonal = nn.Linear(model_dim, time_features)

    def forward(self, x, latent_growth, latent_seasonal):
        # following equation in appendix A.2

        n, device = x.shape[1], x.device

        alpha = self.alpha.sigmoid()

        arange = torch.arange(n, device = device)
        powers = torch.flip(arange, dims = (0,))

        # Aes for raw time series signal with seasonal terms (from frequency attention) subtracted out

        seasonal =self.to_seasonal(latent_seasonal)
        Aes_weights = alpha * (1 - alpha) ** powers
        seasonal_normalized_term = conv1d_fft(x - seasonal, Aes_weights)

        # auxiliary term

        growth = self.to_growth(latent_growth)
        growth_smoothing_weights = (1 - alpha) ** powers
        growth_term = conv1d_fft(growth, growth_smoothing_weights)

        return seasonal_normalized_term + growth_term

# decoder classes

class LevelStack(nn.Module):
    def forward(self, x, num_steps_forecast):
        return repeat(x[:, -1], 'b d -> b n d', n = num_steps_forecast)

class GrowthDampening(nn.Module):
    def __init__(
        self,
        dim,
        heads = 8
    ):
        super().__init__()
        self.heads = heads
        self.dampen_factor = nn.Parameter(torch.randn(heads))

    def forward(self, growth, *, num_steps_forecast):
        device, h = growth.device, self.heads

        dampen_factor = self.dampen_factor.sigmoid()

        # like level stack, it takes the last growth for forecasting

        last_growth = growth[:, -1]
        last_growth = rearrange(last_growth, 'b l (h d) -> b l 1 h d', h = h)

        # prepare dampening factors per head and the powers

        dampen_factor = rearrange(dampen_factor, 'h -> 1 1 1 h 1')
        powers = (torch.arange(num_steps_forecast, device = device) + 1)
        powers = rearrange(powers, 'n -> 1 1 n 1 1')

        # following Eq(2) in the paper

        dampened_growth = last_growth * (dampen_factor ** powers).cumsum(dim = 2)
        return rearrange(dampened_growth, 'b l n h d -> b l n (h d)')

# main class

class ETSFormer(nn.Module):
    def __init__(
        self,
        *,
        time_features,
        model_dim,
        embed_kernel_size = 3,
        layers = 2,
        heads = 8,
        K = 4,
        dropout = 0.
    ):
        super().__init__()
        assert (model_dim % heads) == 0, 'model dimension must be divisible by number of heads'
        self.embed = InputEmbedding(time_features, model_dim, kernel_size = embed_kernel_size, dropout = dropout)

        self.encoder_layers = nn.ModuleList([])

        for ind in range(layers):
            is_last_layer = (ind - 1) == layers

            self.encoder_layers.append(nn.ModuleList([
                FrequencyAttention(K = K, dropout = dropout),
                MHESA(dim = model_dim, heads = heads, dropout = dropout),
                FeedForwardBlock(dim = model_dim) if not is_last_layer else None,
                Level(time_features = time_features, model_dim = model_dim)
            ]))

        self.growth_dampening_module = GrowthDampening(dim = model_dim, heads = heads)

        self.latents_to_time_features = nn.Linear(model_dim, time_features)
        self.level_stack = LevelStack()

    def forward(self, x, *, num_steps_forecast):
        z = self.embed(x)

        latent_growths = []
        latent_seasonals = []

        for freq_attn, mhes_attn, ff_block, level in self.encoder_layers:
            latent_seasonal = freq_attn(z)
            z = z - latent_seasonal

            latent_growth = mhes_attn(z)
            z = z - latent_growth

            if exists(ff_block):
                z = ff_block(z)

            x = level(x, latent_growth, latent_seasonal)

            latent_growths.append(latent_growth)
            latent_seasonals.append(latent_seasonal)

        latent_growths = torch.stack(latent_growths, dim = -2)
        latent_seasonals = torch.stack(latent_seasonals, dim = -2)

        latent_seasonals = rearrange(latent_seasonals, 'b n l d -> b l d n')
        extrapolated_seasonals = fourier_extrapolate(latent_seasonals, x.shape[1], x.shape[1] + num_steps_forecast)
        extrapolated_seasonals = rearrange(extrapolated_seasonals, 'b l d n -> b l n d')

        dampened_growths = self.growth_dampening_module(latent_growths, num_steps_forecast = num_steps_forecast)
        level = self.level_stack(x, num_steps_forecast = num_steps_forecast)

        summed_latents = dampened_growths.sum(dim = 1) + extrapolated_seasonals.sum(dim = 1)
        return level + self.latents_to_time_features(summed_latents)
