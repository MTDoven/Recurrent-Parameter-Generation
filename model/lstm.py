import torch
from torch import nn


class LstmModel(nn.Module):
    config = {}

    def __init__(self, sequence_length):
        super().__init__()
        self.lstm_forward = nn.LSTM(
            input_size=self.config["input_size"],
            hidden_size=self.config["hidden_size"],
            num_layers=self.config["num_layers"],
            dropout=self.config["dropout"],
            bias=True,
            batch_first=True,)
        self.to_condition = nn.Linear(self.config["d_condition"], self.config["d_model"])
        pe = get_sinusoid(sequence_length, self.config["d_model"])[None, :, :]
        if self.config.get("trainable_pe"):
            self.pe = nn.Parameter(pe)
        else:  # fixed positional embedding
            self.register_buffer("pe", pe)

    def forward(self, output_shape, condition=torch.tensor([0.])):
        condition = self.to_condition(condition.view(-1, 1, self.config["d_condition"]).to(self.pe.device))
        x = self.lstm_forward(self.pe.repeat(output_shape[0], 1, 1) + condition)
        return x

