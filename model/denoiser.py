import torch
import torch.nn as nn
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

    def forward(self, x, z):
        shift, scale = self.adaptive_layer_norm_modulation(z).chunk(2, dim=-1)
        x = self.norm(x) * (1 + scale) + shift
        x = self.linear(x)
        return x


class ConditionalMLP(nn.Module):
    def __init__(self, layer_dims: list, condition_dim: int,
                 device, activation=nn.SiLU()):
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

    def forward(self, x, t, z):
        t = self.time_embedder(t).unsqueeze(-2)
        x = x + t
        for module in self.module_list:
            x = module(x, z)
        return x


if __name__ == '__main__':
    model = ConditionalMLP(
            layer_dims=[2048, 1024, 1024, 512, 512],
            condition_dim=4096,
            device="cpu",
            activation=nn.SiLU())
    print(model)
    x = torch.randn((4, 2048))
    t = torch.tensor([4, 104, 1025, 1245])
    z = torch.randn((4, 4096))
    y = model(x, t, z)
    print(y.shape)
