<img src="./etsformer.png" width="400px"></img>

## ETSformer - Pytorch

Implementation of <a href="https://arxiv.org/abs/2202.01381">ETSformer</a>, state of the art time-series Transformer, in Pytorch

## Install

```bash
$ pip install etsformer-pytorch
```

## Usage

```python
import torch
from etsformer_pytorch import ETSFormer

model = ETSFormer(
    time_features = 4,
    model_dim = 512,                # in paper they use 512
    embed_kernel_size = 3,          # kernel size for 1d conv for input embedding
    layers = 2,                     # number of encoder and corresponding decoder layers
    heads = 8,                      # number of exponential smoothing attention heads
    K = 4,                          # num frequencies with highest amplitude to keep (attend to)
    dropout = 0.2                   # dropout (in paper they did 0.2)
)

timeseries = torch.randn(1, 1024, 4)

pred = model(timeseries, num_steps_forecast = 32) # (1, 32, 4) - (batch, num steps forecast, num time features)
```

For using ETSFormer for classification, using cross attention pooling on all latents and level output

```python
import torch
from etsformer_pytorch import ETSFormer, ClassificationWrapper

etsformer = ETSFormer(
    time_features = 1,
    model_dim = 512,
    embed_kernel_size = 3,
    layers = 2,
    heads = 8,
    K = 4,
    dropout = 0.2
)

adapter = ClassificationWrapper(
    etsformer = etsformer,
    dim_head = 32,
    heads = 16,
    dropout = 0.2,
    level_kernel_size = 5,
    num_classes = 10
)

timeseries = torch.randn(1, 1024)

logits = adapter(timeseries) # (1, 10)
```

## Citation

```bibtex
@misc{woo2022etsformer,
    title   = {ETSformer: Exponential Smoothing Transformers for Time-series Forecasting}, 
    author  = {Gerald Woo and Chenghao Liu and Doyen Sahoo and Akshat Kumar and Steven Hoi},
    year    = {2022},
    eprint  = {2202.01381},
    archivePrefix = {arXiv},
    primaryClass = {cs.LG}
}
```
