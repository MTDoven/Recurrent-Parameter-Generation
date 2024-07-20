import torch
import torch.nn as nn
from torch.nn import functional as F
import math


class TimestepEmbedder(nn.Module):
    def __init__(self, hidden_dim, device, frequency_embedding_size=256, max_period=10000):
        super().__init__()
        assert frequency_embedding_size % 2 == 0
        self.frequency_embedding_size = frequency_embedding_size
        self.mlp = nn.Sequential(
                nn.Linear(frequency_embedding_size, hidden_dim, bias=True), nn.SiLU(),
                nn.Linear(hidden_dim, hidden_dim, bias=True))
        half = frequency_embedding_size // 2
        freqs = torch.exp(-math.log(max_period) * torch.arange(start=0, end=half, device=device) / half)
        self.register_buffer("freqs", freqs)

    def forward(self, t):
        args = t[:, None].float() * self.freqs
        t_freq = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        t_emb = self.mlp(t_freq)
        return t_emb


class AdaptiveLayer(nn.Module):
    def __init__(self, input_dim, output_dim=None, condition_dim=None, activation=None):
        super().__init__()
        output_dim = input_dim if output_dim is None else output_dim
        condition_dim = input_dim if condition_dim is None else condition_dim
        self.norm = nn.LayerNorm(input_dim, elementwise_affine=False, eps=1e-6)
        self.adaptive_layer_norm_modulation = nn.Sequential(
                nn.ELU(), nn.Linear(condition_dim, 2 * input_dim, bias=True))
        self.linear = nn.Sequential(
                nn.Linear(input_dim, output_dim, bias=True),
                activation if activation is not None else nn.Identity(),)

    def forward(self, x, c):
        shift, scale = self.adaptive_layer_norm_modulation(c).chunk(2, dim=-1)
        x = self.norm(x) * (1 + scale) + shift
        x = self.linear(x)
        return x


class ConditionalMLP(nn.Module):
    def __init__(self, layer_dims: list, condition_dim: int,
                 device, activation=nn.ELU()):
        super().__init__()
        self.time_embedder = TimestepEmbedder(hidden_dim=condition_dim, device=device)
        self.module_list = nn.ModuleList()
        for i in range(len(layer_dims) - 1):
            self.module_list.append(AdaptiveLayer(
                    input_dim=layer_dims[i],
                    output_dim=layer_dims[i+1],
                    condition_dim=condition_dim,
                    activation=activation if (i+2 != len(layer_dims)) else None,
            ),)  # all AdaptiveLayerNorm + linear

    def forward(self, x, t, c):
        t = self.time_embedder(t)
        for module in self.module_list:
            x = module(x, c + t)
        return x


class ConditionalUNet(nn.Module):
    def __init__(self, layer_channels: list, condition_dim: int, kernel_size: int, device: torch.device):
        super().__init__()
        self.time_embedder = TimestepEmbedder(hidden_dim=condition_dim, device=device)
        self.module_list = nn.ModuleList([])
        for i in range(len(layer_channels)-1):
            self.module_list.append(
                nn.ModuleList([
                    nn.Conv1d(layer_channels[i], layer_channels[i+1], kernel_size, 1, kernel_size // 2),
                    nn.ELU() if i+1 != len(layer_channels)-1 else nn.Identity(),
                ])
            )

    def forward(self, x, t, c):
        c = (c + self.time_embedder(t))[:, None, :]
        x = x[:, None, :]
        for module, activation in self.module_list:
            x = module(x + c)
            x = activation(x)
        return x[:, 0, :]