import torch
from abc import ABC
from torch import nn
from .diffusion import DiffusionLoss, DDIMSampler, DDPMSampler
# from .transformer import TransformerModel
from .mamba import MambaModel
# from .lstm import LstmModel




class ModelDiffusion(nn.Module, ABC):
    config = {}

    def __init__(self, sequence_length):
        super().__init__()
        DiffusionLoss.config = self.config
        self.criteria = DiffusionLoss()
        assert self.config["d_model"] == self.config["condition_dim"]
        self.sequence_length = sequence_length
        # to define model after this function
        self.to_condition = nn.Linear(self.config["d_condition"], self.config["d_model"])
        self.to_permutation_state = nn.Embedding(self.config["num_permutation"], self.config["d_model"])
        self.to_permutation_state.weight = \
                nn.Parameter(torch.ones_like(self.to_permutation_state.weight) / self.config["d_model"])

    def forward(self, output_shape=None, x_0=None, condition=None, permutation_state=None, **kwargs):
        # condition
        if condition is not None:
            assert len(condition.shape) == 2
            assert condition.shape[-1] == self.config["d_model"]
            condition = self.to_condition(condition.to(self.device))
        else:  # not use condition
            condition = self.to_condition(torch.zeros(size=(1, 1), device=self.device))
        # process
        if kwargs.get("sample"):
            if permutation_state is not False:
                permutation_state = torch.randint(0, self.to_permutation_state.num_embeddings, (1,), device=self.device)
                permutation_state = self.to_permutation_state(permutation_state)
            else:  # permutation state == False
                permutation_state = 0.
            return self.sample(x=None, condition=condition+permutation_state)
        else:  # train
            if permutation_state is not None:
                permutation_state = self.to_permutation_state(permutation_state)
            else:  # not use permutation state
                permutation_state = 0.
            # Given condition c and ground truth token x, compute loss
            c = self.model(output_shape, condition+permutation_state)
            loss = self.criteria(x=x_0, c=c, **kwargs)
            return loss

    @torch.no_grad()
    def sample(self, x=None, condition=None):
        z = self.model([1, self.sequence_length, self.config["d_model"]], condition)
        if x is None:
            x = torch.randn((1, self.sequence_length, self.config["model_dim"]), device=z.device)
        x = self.criteria.sample(x, z)
        return x

    @property
    def device(self):
        return next(self.parameters()).device




class MambaDiffusion(ModelDiffusion):
    def __init__(self, sequence_length, positional_embedding):
        super().__init__(sequence_length=sequence_length)
        MambaModel.config = self.config
        self.model = MambaModel(positional_embedding=positional_embedding)


# class TransformerDiffusion(ModelDiffusion):
#     def __init__(self, sequence_length):
#         super().__init__(sequence_length=sequence_length)
#         TransformerModel.config = self.config
#         self.model = TransformerModel(sequence_length=sequence_length)
#
#
# class LstmDiffusion(ModelDiffusion):
#     def __init__(self, sequence_length):
#         super().__init__(sequence_length=sequence_length)
#         LstmModel.config = self.config
#         self.model = LstmModel(sequence_length=sequence_length)