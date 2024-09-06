import torch
from torch import nn
import math


class LstmModel(nn.Module):
    config = {}

    def __init__(self, positional_embedding):
        super().__init__()
        self.lstm_forward = nn.LSTM(
            input_size=self.config["d_model"],
            hidden_size=self.config["d_model"],
            num_layers=self.config["num_layers"],
            dropout=self.config["dropout"],
            bias=True,
            batch_first=True,)
        pe = positional_embedding[None, :, :]
        if self.config.get("trainable_pe"):
            self.pe = nn.Parameter(pe)
        else:  # fixed positional embedding
            self.register_buffer("pe", pe)

    def forward(self, output_shape, condition=None):
        assert len(condition.shape) == 3
        x, _ = self.lstm_forward(self.pe.repeat(output_shape[0], 1, 1) + condition)
        return x.contiguous()
