import torch
from torch import nn
from einops import rearrange




class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim),
        )

    def forward(self, x):
        x = self.net(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, mask=None):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.norm = nn.LayerNorm(dim)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Linear(inner_dim, dim, bias=False) if inner_dim != dim else nn.Identity()
        self.softmax = nn.Softmax(dim=-1)
        if mask is not None:
            assert len(mask.shape) == 2
            mask = mask[None, None, :, :]
            self.register_buffer("mask", mask)
            self.use_mask = True
        else:  # not use mask
            self.use_mask = False

    def forward(self, x):
        x = self.norm(x)
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)
        attn = q @ k.transpose(-1, -2)
        if self.use_mask:
            attn = torch.where(self.mask, attn, 1e-8)
        out = self.softmax(attn) @ v
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.to_out(out)
        return out


class Transformer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward, dim_head, num_layers, mask=None):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(num_layers):
            self.layers.append(nn.ModuleList([
                Attention(d_model, heads=nhead, dim_head=dim_head, mask=mask),
                FeedForward(d_model, dim_feedforward)
            ]))

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x


class TransformerModel(nn.Module):
    config = {}

    def __init__(self, positional_embedding):
        super().__init__()
        self.transformer_forward = Transformer(
            d_model=self.config["d_model"],
            nhead=self.config["nhead"],
            dim_feedforward=self.config["dim_feedforward"],
            dim_head=self.config["dim_head"],
            num_layers=self.config["num_layers"],
            mask=self.config.get("mask"),
        )
        pe = positional_embedding[None, :, :]
        if self.config.get("trainable_pe"):
            self.pe = nn.Parameter(pe)
        else:  # fixed positional embedding
            self.register_buffer("pe", pe)

    def forward(self, output_shape, condition=None):
        assert len(condition.shape) == 3
        x = self.transformer_forward(self.pe.repeat(output_shape[0], 1, 1) + condition)
        return x
