import torch
from abc import ABC
from torch import nn
from .lstm import LstmModel
from .mamba import MambaModel
from .transformer import TransformerModel
from .diffusion import DiffusionLoss, DDIMSampler, DDPMSampler
from .condition import ClassToCondition




class ModelDiffusion(nn.Module, ABC):
    config = {}

    def __init__(self, sequence_length):
        super().__init__()
        DiffusionLoss.config = self.config
        self.criteria = DiffusionLoss()
        self.sequence_length = sequence_length
        # to define model after this function

    def forward(self, output_shape=None, x_0=None, condition=None, **kwargs):
        if kwargs.get("sample"):
            return self.sample(x=None, condition=condition)
        c = self.model(output_shape, condition)
        # Given condition c and ground truth token x, compute loss
        loss = self.criteria(x=x_0, c=c, **kwargs)
        return loss

    @torch.no_grad()
    def sample(self, x=None, condition=None):
        z = self.model([1, self.sequence_length, self.config["d_model"]], condition)
        if x is None:
            x = torch.randn((1, self.sequence_length, self.config["dim_per_token"]), device=z.device)
        x = self.criteria.sample(x, z)
        return x


class MambaDiffusion(ModelDiffusion):
    def __init__(self, sequence_length, positional_embedding):
        super().__init__(sequence_length=sequence_length)
        MambaModel.config = self.config
        self.model = MambaModel(positional_embedding=positional_embedding)


class TransformerDiffusion(ModelDiffusion):
    def __init__(self, sequence_length, positional_embedding):
        super().__init__(sequence_length=sequence_length)
        TransformerModel.config = self.config
        self.model = TransformerModel(positional_embedding=positional_embedding)


class LstmDiffusion(ModelDiffusion):
    def __init__(self, sequence_length, positional_embedding):
        super().__init__(sequence_length=sequence_length)
        LstmModel.config = self.config
        self.model = LstmModel(positional_embedding=positional_embedding)




class ClassConditionMambaDiffusion(MambaDiffusion):
    def __init__(self, sequence_length, positional_embedding, input_class=10):
        super().__init__(sequence_length, positional_embedding)
        self.get_condition = ClassToCondition(input_class, self.config["d_condition"])

    def forward(self, output_shape=None, x_0=None, condition=None, **kwargs):
        condition = self.get_condition(condition)
        return super().forward(output_shape=output_shape, x_0=x_0, condition=condition, **kwargs)
