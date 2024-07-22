import torch
from torch import nn
from mamba_ssm import Mamba
import math


class MambaModel(nn.Module):
    config = {
        "d_model": 1024,
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
        pe = self.get_sinusoid(sequence_length, self.config["d_model"]).unsqueeze(0)
        self.register_buffer("pe", pe)

    @staticmethod
    def get_sinusoid(max_len, d_model):
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe

    def forward(self, output_shape):
        x = self.model(self.pe.repeat(output_shape[0], 1, 1))
        return x
