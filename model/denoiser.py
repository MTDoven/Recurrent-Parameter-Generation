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


class ConditionalMLP(nn.Module):
    def __init__(self, layer_dims: list, condition_dim: int, device: torch.device):
        super().__init__()
        self.time_embedder = TimestepEmbedder(hidden_dim=condition_dim, device=device)
        self.encoder_list = nn.ModuleList([])
        for i in range(len(layer_dims) // 2 + 1):
            self.encoder_list.append(nn.ModuleList([
                nn.Linear(layer_dims[i], layer_dims[i+1]),
                nn.Sequential(nn.LayerNorm(layer_dims[i+1]), nn.LeakyReLU()),
            ]))
        self.decoder_list = nn.ModuleList([])
        for i in range(len(layer_dims) // 2 + 1, len(layer_dims) - 1):
            self.decoder_list.append(nn.ModuleList([
                nn.Linear(layer_dims[i], layer_dims[i+1]),
                nn.Sequential(nn.LayerNorm(layer_dims[i+1]), nn.LeakyReLU())
                    if layer_channels[i+1] != 1 else nn.Identity(),
            ]))
        self._init_weight()

    def _init_weight(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, x, t, c):
        c = (c + self.time_embedder(t))[:, None, :]
        x = x[:, None, :]
        x_list = []
        for i, (module, activation) in enumerate(self.encoder_list):
            x = module(x + c)
            x = activation(x)
            if i < len(self.encoder_list) - 2:
                x_list.append(x)
        for i, (module, activation) in enumerate(self.decoder_list):
            x = x + x_list[-i-1]
            x = module(x + c)
            x = activation(x)
        return x[:, 0, :]


class ConditionalUNet(nn.Module):
    def __init__(self, layer_channels: list, condition_dim: int, kernel_size: int, device: torch.device):
        super().__init__()
        self.time_embedder = TimestepEmbedder(hidden_dim=condition_dim, device=device)
        self.encoder_list = nn.ModuleList([])
        for i in range(len(layer_channels) // 2 + 1):
            self.encoder_list.append(nn.ModuleList([
                nn.Conv1d(layer_channels[i], layer_channels[i+1], kernel_size, 1, kernel_size // 2),
                nn.Sequential(nn.BatchNorm1d(layer_channels[i+1]), nn.ELU(),),
            ]))
        self.decoder_list = nn.ModuleList([])
        for i in range(len(layer_channels) // 2 + 1, len(layer_channels) - 1):
            self.decoder_list.append(nn.ModuleList([
                nn.Conv1d(layer_channels[i], layer_channels[i+1], kernel_size, 1, kernel_size // 2),
                nn.Sequential(nn.BatchNorm1d(layer_channels[i+1]), nn.ELU())
                    if layer_channels[i+1] != 1 else nn.Identity(),
            ]))
        # self.output_layer = nn.Conv1d(2, 1, kernel_size, 1, kernel_size // 2)

    def forward(self, x, t, c):
        c = (c + self.time_embedder(t))[:, None, :]
        x = x[:, None, :]
        x_list = []
        for i, (module, activation) in enumerate(self.encoder_list):
            x = module(x + c)
            x = activation(x)
            if i < len(self.encoder_list) - 2:
                x_list.append(x)
        for i, (module, activation) in enumerate(self.decoder_list):
            x = x + x_list[-i-1]
            x = module(x + c)
            x = activation(x)
        return x[:, 0, :]