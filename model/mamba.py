import torch
from torch import nn
from mamba_ssm import Mamba2 as Mamba
import math


class MambaModel(nn.Module):
    config = {}

    def __init__(self, sequence_length):
        super().__init__()
        mamba_config = {
            "d_model": self.config["d_model"],
            "d_state": self.config["d_state"],
            "d_conv": self.config["d_conv"],
            "expand": self.config["expand"],
        }
        self.mamba_forward = nn.Sequential(*[Mamba(**mamba_config) for _ in range(self.config["num_layers"])])
        self.to_condition = nn.Linear(self.config["d_condition"], self.config["d_model"])
        pe = self.get_sinusoid(sequence_length, self.config["d_model"])[None, :, :]
        self.register_buffer("pe", pe)

    @staticmethod
    def get_sinusoid(max_len, d_model):
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe

    def forward(self, output_shape, condition=torch.tensor([0.])):
        condition = self.to_condition(condition.view(-1, 1, self.config["d_condition"]).to(self.pe.device))
        x = self.mamba_forward(self.pe.repeat(output_shape[0], 1, 1) + condition)
        return x
