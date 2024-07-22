import torch
from torch import nn
from mamba_ssm import Mamba
import math


class MambaModel(nn.Module):
    config = {
        "d_output": 1024,
        "d_model": 2048,
        "d_state": 16,
        "d_conv": 4,
        "expand": 2,
    }

    def __init__(self, sequence_length):
        super().__init__()
        self.model = Mamba(
            d_model=self.config["d_model"],
            d_state=self.config["d_state"],
            d_conv=self.config["d_conv"],
            expand=self.config["expand"],
        )
        self.input = nn.Parameter(nn.init.normal_(torch.empty(1, sequence_length, self.config["d_model"])))
        self.to_out = nn.Linear(self.config["d_model"], self.config["d_output"])

    def forward(self, output_shape):
        x = self.model(self.input.repeat(output_shape[0], 1, 1))
        x = self.to_out(x)
        return x
