import torch
from torch import nn
from .mamba import MambaModel
from .diffusion import DiffusionLoss, DDIMSampler, DDPMSampler


class MambaDiffusion(nn.Module):
    config = {}

    def __init__(self, sequence_length):
        super().__init__()
        # pass config
        MambaModel.config = self.config
        DiffusionLoss.config = self.config
        # this module init
        self.model = MambaModel(sequence_length=sequence_length)
        self.criteria = DiffusionLoss()
        assert self.config["d_model"] == self.config["condition_dim"]
        self.sequence_length = sequence_length

    def forward(self, output_shape=None, x_0=None, condition: torch.Tensor = torch.tensor(0.), **kwargs):
        if kwargs.get("sample"):
            return self.sample(x=None, condition=condition)
        c = self.model(output_shape, condition)
        # Given condition c and ground truth token x, compute loss
        loss = self.criteria(x=x_0, c=c)
        return loss

    @torch.no_grad()
    def sample(self, x=None, condition: torch.Tensor = torch.tensor(0.)):
        z = self.model([1, self.sequence_length, self.config["d_model"]], condition)
        if x is None:
            x = torch.randn((1, self.sequence_length, self.config["model_dim"]), device=z.device)
        x = self.criteria.sample(x, z)
        return x