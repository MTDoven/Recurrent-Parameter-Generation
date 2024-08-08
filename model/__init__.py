import torch
import timm
from torch import nn
from .mamba import MambaModel
from .diffusion import DiffusionLoss, DDIMSampler, DDPMSampler
from .extractor import ResNet18


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
        if kwargs.get("parameter_weight_decay"):
            loss += torch.square(c).mean() * kwargs["parameter_weight_decay"]
        return loss

    @torch.no_grad()
    def sample(self, x=None, condition: torch.Tensor = torch.tensor(0.)):
        z = self.model([1, self.sequence_length, self.config["d_model"]], condition)
        if x is None:
            x = torch.randn((1, self.sequence_length, self.config["model_dim"]), device=z.device)
        x = self.criteria.sample(x, z)
        return x


class ConditionalMambaDiffusion(MambaDiffusion):
    config = {}

    def __init__(self, sequence_length):
        super().__init__(sequence_length)
        self.condition_extractor, output_dim = ResNet18()
        assert self.config["d_condition"] == output_dim
        self.register_buffer("device_sign_buffer", torch.zeros(1))

    def forward(self, output_shape=None, x_0=None, condition=None, **kwargs):
        condition = self.condition_extractor(condition.to(self.device_sign_buffer.device))
        return super().forward(output_shape=output_shape, x_0=x_0, condition=condition, **kwargs)
