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
                nn.Linear(frequency_embedding_size, hidden_dim, bias=True), nn.SiLU(),
                nn.Linear(hidden_dim, hidden_dim, bias=True))
        half = frequency_embedding_size // 2
        freqs = torch.exp(-math.log(max_period) * torch.arange(start=0, end=half) / half)
        self.register_buffer("freqs", freqs)

    def forward(self, t):
        args = t[:, None].float() * self.freqs
        t_freq = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        t_emb = self.mlp(t_freq)
        return t_emb


class AdaptiveLayerNorm(nn.Module):
    def __init__(self, num_features, out_channel, condition_dim, kernel_size):
        super(AdaptiveLayerNorm, self).__init__()
        assert condition_dim >= num_features, f"c:{condition_dim}, n:{num_features}"
        assert condition_dim % num_features == 0, f"c:{condition_dim}, n:{num_features}"
        self.scale_proj = nn.Sequential(
            nn.Conv1d(1, out_channel, kernel_size, condition_dim // num_features, kernel_size // 2),
            nn.SiLU(),
            nn.Conv1d(out_channel, out_channel, kernel_size, 1, kernel_size // 2),
        )
        self.bias_proj = nn.Sequential(
            nn.Conv1d(1, out_channel, kernel_size, condition_dim // num_features, kernel_size // 2),
            nn.SiLU(),
            nn.Conv1d(out_channel, out_channel, kernel_size, 1, kernel_size // 2),
        )
        self.layer_norm = nn.LayerNorm(num_features, elementwise_affine=False)

    def forward(self, x, c):
        scale, bias = self.scale_proj(c), self.bias_proj(c)
        x = self.layer_norm(x) * scale + bias
        return x


class Identity(nn.Identity):
    def forward(self, input, *args, **kwargs):
        return input


class ConditionalUNet(nn.Module):
    def __init__(self, layer_channels, model_dim, condition_dim, kernel_size, condition_kernel_size=33):
        super().__init__()
        encode_channels = layer_channels[:len(layer_channels) // 2 + 1]
        decode_channels = layer_channels[len(layer_channels) // 2:]
        assert len(encode_channels) == len(decode_channels)
        middle_dim = model_dim // int(2**(len(encode_channels)-1))
        self.time_embedder = TimestepEmbedder(hidden_dim=middle_dim)
        self.encoder_list = nn.ModuleList([])
        for i in range(len(encode_channels) - 1):
            self.encoder_list.append(nn.ModuleList([
                nn.Conv1d(encode_channels[i],
                          encode_channels[i+1],
                          kernel_size, 2, kernel_size // 2),
                nn.Sequential(nn.BatchNorm1d(encode_channels[i+1]), nn.SiLU()),
                AdaptiveLayerNorm(model_dim // int(2**(i+1)),
                                  encode_channels[i+1], condition_dim, condition_kernel_size),
            ]))
        self.decoder_list = nn.ModuleList([])
        for i in range(len(decode_channels) - 1):
            self.decoder_list.append(nn.ModuleList([
                nn.ConvTranspose1d(decode_channels[i] if i == 0 else decode_channels[i]+encode_channels[-i-1],
                                   decode_channels[i+1],
                                   kernel_size+1, 2, kernel_size // 2),
                nn.Sequential(nn.BatchNorm1d(decode_channels[i+1]), nn.ELU()),
                AdaptiveLayerNorm(model_dim // int(2**(len(decode_channels)-2-i)),
                                  decode_channels[i+1], condition_dim, condition_kernel_size)
                        if i != len(decode_channels) - 2 else Identity(),
            ]))
        self.output_layer = nn.Conv1d(decode_channels[-1], 1, kernel_size, 1, kernel_size // 2)

    def forward(self, x, t, c):
        x = x[:, None, :]
        c = c[:, None, :]
        x_list = []
        for i, (module, activation, norm) in enumerate(self.encoder_list):
            x = module(x)
            x = activation(x)
            x = norm(x, c)
            x_list.append(x)
        x = x * self.time_embedder(t)[:, None, :]
        for i, (module, activation, norm) in enumerate(self.decoder_list):
            x = x if i == 0 else torch.cat((x, x_list[-1-i]), dim=-2)
            x = module(x)
            x = activation(x)
            x = norm(x, c)
        x = self.output_layer(x)
        assert x.size(-2) == 1
        return x[:, 0, :]




if __name__ == "__main__":
    model = ConditionalUNet(
        layer_channels=(1, 32, 64, 128, 64, 32, 16),
        model_dim=16384,
        condition_dim=8192,
        kernel_size=9,
    )  # define model
    x = torch.ones((4, 16384))
    t = torch.tensor([1, 2, 3, 4])
    c = torch.ones((4, 8192))
    y = model(x, t, c)
    print(y.shape)
    # param count
    num_param = 0
    for v in model.parameters():
        num_param += v.numel()
    print("num_param:", num_param)