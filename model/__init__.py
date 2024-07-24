import torch
from torch import nn
from .lstm import LstmModel
from .mamba import MambaModel
from .diffusion import DiffusionLoss, DDIMSampler, DDPMSampler


class LstmDiffusion(nn.Module):
    config = {
        # lstm config
        "input_size": 4096,
        "hidden_size": 4096,
        "output_size": 1024,
        "num_layers": 2,
        "dropout": 0.,
        # diffusion config
        "layer_channels": [1, 32, 64, 96, 64, 32, 1],
        "condition_dim": 1024,
        "kernel_size": 5,
        "sample_mode": DDPMSampler,
        "beta": (0.0001, 0.02),
        "T": 1000,
    }

    def __init__(self, sequence_length, device):
        super().__init__()
        # pass config
        LstmModel.config = self.config
        DiffusionLoss.config = self.config
        # this module init
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


class MambaDiffusion(nn.Module):
    config = {
        # mamba config
        "d_output": 1024,
        "d_model": 4096,
        "d_state": 64,
        "d_conv": 4,
        "expand": 2,
        # diffusion config
        "layer_dims": [1024, 1024, 1024, 1024],
        "condition_dim": 1024,
        "sample_mode": DDPMSampler,
        "beta": (0.0001, 0.02),
        "T": 1000,
    }

    def __init__(self, sequence_length, device):
        super().__init__()
        # pass config
        MambaModel.config = self.config
        DiffusionLoss.config = self.config
        # this module init
        self.model = MambaModel(sequence_length=sequence_length)
        self.criteria = DiffusionLoss(device=device)
        assert self.model.config["d_output"] == self.criteria.config["condition_dim"]
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