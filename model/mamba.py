import torch
from torch import nn
from mamba_ssm import Mamba2 as Mamba
import math


class Condition(nn.Module):
    def __init__(self, d_condition, d_model, sequence_length):
        super().__init__()
        self.d_condition = d_condition
        self.d_model = d_model
        self.gate = nn.Parameter(torch.ones(1, sequence_length, 1))
        if d_condition == 1:
            self.gate.requires_grad = False
        self.linear = nn.Linear(d_condition, d_model)

    def forward(self, condition):
        assert len(condition.shape) == 2
        assert condition.shape[-1] == self.d_condition
        c = self.linear(condition)[:, None, :] * torch.sigmoid(self.gate)
        return c


class MambaModel(nn.Module):
    config = {}

    def __init__(self, positional_embedding):
        super().__init__()
        if self.config.get("d_model_1") is None:
            assert self.config.get("d_model_2") is None
            self.config["d_model_1"] = self.config["d_model"]
            self.config["d_model_2"] = self.config["d_model"]
        mamba1 = Mamba(d_model=self.config["d_model_1"],
                       d_state=self.config["d_state"],
                       d_conv=self.config["d_conv"],
                       expand=self.config["expand"])
        mamba2 = Mamba(d_model=self.config["d_model_2"],
                       d_state=self.config["d_state"],
                       d_conv=self.config["d_conv"],
                       expand=self.config["expand"])
        mamba2.in_proj = nn.Linear(mamba1.out_proj.out_features, mamba2.in_proj.out_features, bias=False)
        mamba2.out_proj = nn.Linear(mamba2.out_proj.in_features, self.config["dim_per_token"], bias=True)
        self.mamba_forward = nn.Sequential(*[mamba1, mamba2])
        self.to_condition = Condition(d_condition=self.config["d_condition"],
                                      d_model=self.config["d_model_1"],
                                      sequence_length=positional_embedding.shape[-2])
        pe = positional_embedding[None]
        if self.config.get("trainable_pe"):
            self.pe = nn.Parameter(pe)
        else:  # fixed positional embedding
            self.register_buffer("pe", pe)

    def forward(self, output_shape, condition=None):
        if condition is None:
            assert self.config["d_condition"] == 1
            condition = torch.zeros(size=(1, 1), device=self.pe.device)
        condition = self.to_condition(condition)
        x = self.mamba_forward(self.pe + condition)
        return x.repeat(output_shape[0], 1, 1) if x.size(0) == 1 else x
