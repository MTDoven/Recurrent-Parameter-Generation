import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import functional as F
import math


class TimestepEmbedder(nn.Module):
    def __init__(self, hidden_dim, frequency_embedding_size=256, max_period=10000):
        super().__init__()
        assert frequency_embedding_size % 2 == 0
        self.frequency_embedding_size = frequency_embedding_size
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_dim, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim, bias=True)
        )  # FIXME: this is too big!
        half = frequency_embedding_size // 2
        freqs = torch.exp(-math.log(max_period) * torch.arange(start=0, end=half) / half)
        self.register_buffer("freqs", freqs)

    def forward(self, t):
        args = t[:, None].float() * self.freqs
        t_freq = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        t_emb = self.mlp(t_freq)
        return t_emb


class ConditionalUNet(nn.Module):
    def __init__(self, layer_channels, model_dim, kernel_size):
        super().__init__()
        self.time_embedder = TimestepEmbedder(hidden_dim=model_dim)
        self.condi_embedder = nn.Identity()  # nn.Linear(condition_dim, model_dim)
        self.encoder_list = nn.ModuleList([])
        for i in range(len(layer_channels) // 2 + 1):
            self.encoder_list.append(nn.ModuleList([
                nn.Conv1d(layer_channels[i], layer_channels[i+1], kernel_size, 1, kernel_size // 2),
                nn.Sequential(nn.BatchNorm1d(layer_channels[i+1]), nn.ELU()),
            ]))
        self.decoder_list = nn.ModuleList([])
        for i in range(len(layer_channels) // 2 + 1, len(layer_channels) - 1):
            self.decoder_list.append(nn.ModuleList([
                nn.Conv1d(layer_channels[i], layer_channels[i+1], kernel_size, 1, kernel_size // 2),
                nn.Sequential(nn.BatchNorm1d(layer_channels[i+1]), nn.ELU())
                    if layer_channels[i+1] != 1 else nn.Identity(),
            ]))

    def forward(self, x, t, c):
        x = x[:, None, :]
        t = self.time_embedder(t)[:, None, :]
        c = self.condi_embedder(c)[:, None, :]
        x_list = []
        for i, (module, activation) in enumerate(self.encoder_list):
            x = module((x + c) * t)
            x = activation(x)
            if i < len(self.encoder_list) - 2:
                x_list.append(x)
        for i, (module, activation) in enumerate(self.decoder_list):
            x = x + x_list[-i-1]
            x = module((x + c) * t)
            x = activation(x)
        return x[:, 0, :]




if __name__ == "__main__":
    model = ConditionalUNet(
        layer_channels=(1, 32, 64, 128, 64, 32, 1),
        model_dim=6144,
        kernel_size=65,
    )  # define model
    x = torch.ones((4, 6144))
    t = torch.tensor([1, 2, 3, 4])
    c = torch.ones((4, 6144))
    y = model(x, t, c)
    print(y.shape)
    # param count
    num_param = 0
    for v in model.parameters():
        num_param += v.numel()
    print("num_param:", num_param)