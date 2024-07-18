import torch
from torch import nn
from torch.nn import functional as F
from flash_attn import flash_attn_func
from einops import rearrange
import numpy as np


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, dim),
        )

    def forward(self, x):
        x = self.net(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.norm = nn.LayerNorm(dim)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Linear(inner_dim, dim, bias=False) if inner_dim != dim else nn.Identity()
        alibi_slopes = 2 ** ((torch.tensor([i for i in range(heads)], dtype=torch.float32) - 1) * 2 / heads - 1)
        self.register_buffer("alibi_slopes", alibi_slopes)

    def forward(self, x):
        x = self.norm(x)
        qkv = self.to_qkv(x)
        qkv = qkv.to(torch.bfloat16).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b n h d', h=self.heads), qkv)
        out = flash_attn_func(q=q, k=k, v=v, alibi_slopes=self.alibi_slopes)
        out = out.to(x.dtype).view(out.size(0), out.size(1), -1)
        out = self.to_out(out)
        return out


class Transformer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward, dim_head, num_layers):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(num_layers):
            self.layers.append(nn.ModuleList([
                Attention(d_model, heads=nhead, dim_head=dim_head),
                FeedForward(d_model, dim_feedforward)
            ]))

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x


class TransformerModel(nn.Module):
    config = {
        "d_model": 1024,
        "nhead": 16,
        "dim_feedforward": 4096,
        "dim_head": 64,
        "num_layers": 8,
    }

    def __init__(self):
        super().__init__()
        self.model = Transformer(
            d_model=self.config["d_model"],
            nhead=self.config["nhead"],
            dim_feedforward=self.config["dim_feedforward"],
            dim_head=self.config["dim_head"],
            num_layers=self.config["num_layers"],
        )
        self.next_token = nn.Parameter(nn.init.normal_(torch.empty((1, 1, self.config["d_model"]))))

    def forward(self, x):
        assert len(x.shape) == 3
        assert x.shape[-1] == self.config["d_model"]
        x = torch.cat((x, self.next_token.repeat(x.size(0), 1, 1)), dim=1)
        x = self.model(x)
        return x[:, -1:, :]
