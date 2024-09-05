import torch
from torch import nn
from mamba_ssm import Mamba2 as Mamba
import math


class MambaModel(nn.Module):
    config = {}

    def __init__(self, positional_embedding):
        super().__init__()
        mamba_config = {
            "d_model": self.config["d_model"],
            "d_state": self.config["d_state"],
            "d_conv": self.config["d_conv"],
            "expand": self.config["expand"],
        }
        self.mamba_forward = nn.Sequential(*[Mamba(**mamba_config) for _ in range(self.config["num_layers"])])
        pe = positional_embedding[None, :, :]
        if self.config.get("trainable_pe"):
            self.pe = nn.Parameter(pe)
        else:  # fixed positional embedding
            self.register_buffer("pe", pe)

    def forward(self, output_shape, condition=None):
        condition = condition[:, None, :]
        x = self.mamba_forward(self.pe.repeat(output_shape[0], 1, 1) + condition)
        return x
