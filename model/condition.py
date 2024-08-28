import torch
from torch import nn


class ClassToCondition(nn.Module):
    def __init__(self, input_class, d_condition):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_class, d_condition),
            nn.SiLU(),
            nn.Linear(d_condition, d_condition),
            nn.LayerNorm(d_condition),
        )

    def forward(self, indicator):
        assert len(indicator.shape) == 2
        condition = self.mlp(indicator)
        return condition