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
        if self.config.get("trainable_pe"):
            self.pe = nn.Parameter(positional_embedding)
        else:  # fixed positional embedding
            self.register_buffer("pe", positional_embedding)

    def forward(self, output_shape, condition):
        assert len(condition.shape) == 3
        assert condition.shape[-1] == self.config["d_model"]
        x = self.mamba_forward(self.pe.repeat(output_shape[0], 1, 1) + condition)
        return x
