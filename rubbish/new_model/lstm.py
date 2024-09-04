import torch
from torch import nn
import math


class LstmModel(nn.Module):
    config = {}

    def __init__(self, sequence_length):
        super().__init__()
        self.lstm_forward = nn.LSTM(
            input_size=self.config["d_model"],
            hidden_size=self.config["d_model"],
            num_layers=self.config["num_layers"],
            dropout=self.config["dropout"],
            bias=True,
            batch_first=True,)
        self.to_condition = nn.Linear(self.config["d_condition"], self.config["d_model"])
        pe = self.get_sinusoid(sequence_length, self.config["d_model"])[None, :, :]
        if self.config.get("trainable_pe"):
            self.pe = nn.Parameter(pe)
        else:  # fixed positional embedding
            self.register_buffer("pe", pe)

    def forward(self, output_shape, condition=torch.tensor([0.])):
        condition = self.to_condition(condition.view(-1, 1, self.config["d_condition"]).to(self.pe.device))
        x, _ = self.lstm_forward(self.pe.repeat(output_shape[0], 1, 1) + condition)
        return x.contiguous()

    @staticmethod
    def get_sinusoid(max_len, d_model):
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe
