import torch
from torch import nn
from .diffusion import DiffusionLoss


class LstmModel(nn.Module):
    config = {
        "input_size": 32,
        "hidden_size": 4096,
        "output_size": 1024,
        "num_layers": 2,
        "dropout": 0.,
    }

    def __init__(self, sequence_length):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=self.config["input_size"],
            hidden_size=self.config["hidden_size"],
            num_layers=self.config["num_layers"],
            dropout=self.config["dropout"],
            bias=True,
            batch_first=True,)
        self.input = nn.Parameter(nn.init.normal_(torch.empty((1, sequence_length, self.config["input_size"]))))
        self.to_out = nn.Linear(self.config["hidden_size"], self.config["output_size"])

    def forward(self, output_shape):
        assert len(output_shape) == 3
        output, _ = self.lstm(self.input.repeat(output_shape[0], 1, 1))
        output = self.to_out(output)
        return output




class LstmDiffusion(nn.Module):
    def __init__(self, sequence_length, device):
        super().__init__()
        self.model = LstmModel(sequence_length=sequence_length)
        self.criteria = DiffusionLoss(device=device)
        assert self.model.config["output_size"] == self.criteria.config["condition_dim"]
        self.dim_per_token = self.criteria.config["condition_dim"]
        self.sequence_length = sequence_length

    def forward(self, output_shape, x_0, **kwargs):
        c = self.model(output_shape)
        # Given condition c and ground truth token x, compute loss
        loss = self.criteria(x=x_0, c=c, **kwargs)
        return loss

    @torch.no_grad()
    def sample(self, x=None, **kwargs):
        z = self.model(output_shape=[1, self.sequence_length, self.dim_per_token])
        if x is None:
            x = torch.randn((1, self.sequence_length, self.dim_per_token), device=self.criteria.device)
        x = self.criteria.sample(x, z, **kwargs)
        return x

