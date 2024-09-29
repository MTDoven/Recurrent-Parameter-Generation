import torch
from abc import ABC
from torch import nn
from .diffusion import DiffusionLoss, DDIMSampler, DDPMSampler
from .transformer import TransformerModel
from .mamba import MambaModel
from .lstm import LstmModel
from .gatemlp import GMLPModel




class ModelDiffusion(nn.Module, ABC):
    config = {}

    def __init__(self, sequence_length):
        super().__init__()
        DiffusionLoss.config = self.config
        self.criteria = DiffusionLoss()
        if self.config.get("post_d_model") is None:
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
            assert condition.shape[-1] == self.config["d_condition"]
            condition = self.to_condition(condition.to(self.device)[:, None, :])
        else:  # not use condition
            condition = self.to_condition(torch.zeros(size=(1, 1, 1), device=self.device))
        # process
        if kwargs.get("sample"):
            if permutation_state is not False:
                permutation_state = torch.randint(0, self.to_permutation_state.num_embeddings, (1,), device=self.device)
                permutation_state = self.to_permutation_state(permutation_state)[:, None, :]
            else:  # permutation state == False
                permutation_state = 0.
            return self.sample(x=None, condition=condition+permutation_state)
        else:  # train
            if permutation_state is not None:
                permutation_state = self.to_permutation_state(permutation_state)[:, None, :]
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


class GMLPDiffusion(ModelDiffusion):
    def __init__(self, sequence_length, positional_embedding):
        super().__init__(sequence_length=sequence_length)
        GMLPModel.config = self.config
        self.model = GMLPModel(positional_embedding=positional_embedding)




class ClassConditionMambaDiffusion(MambaDiffusion):
    def __init__(self, sequence_length, positional_embedding, input_class=10):
        super().__init__(sequence_length, positional_embedding)
        self.get_condition = nn.Sequential(
            nn.Linear(input_class, self.config["d_condition"]),
            nn.SiLU(),
        )  # to condition
        self.to_permutation_state = nn.Embedding(self.config["num_permutation"], self.config["d_model"])
        # condition module
        self.to_condition_linear = nn.Linear(self.config["d_condition"], self.config["d_model"])
        to_condition_gate = torch.zeros(size=(1, sequence_length, 1))
        to_condition_gate[:, -8:, :] = 1.
        self.register_buffer("to_condition_gate", to_condition_gate)
        # reset to_condition
        del self.to_condition
        self.to_condition = self._to_condition

    def forward(self, output_shape=None, x_0=None, condition=None, **kwargs):
        condition = self.get_condition(condition.to(self.device))
        return super().forward(output_shape=output_shape, x_0=x_0, condition=condition, **kwargs)

    def _to_condition(self, x):
        assert len(x.shape) == 3
        x = self.to_condition_linear(x)
        x = x * self.to_condition_gate
        return x


class ClassConditionMambaDiffusionFull(MambaDiffusion):
    def __init__(self, sequence_length, positional_embedding, input_class=10, init_noise_intensity=1e-4):
        super().__init__(sequence_length, positional_embedding)
        self.get_condition = nn.Sequential(
            nn.Linear(input_class, self.config["d_condition"]),
            nn.LayerNorm(self.config["d_condition"]),
        )  # to condition
        self.to_permutation_state = nn.Embedding(self.config["num_permutation"], self.config["d_model"])
        # condition module
        self.to_condition_linear = nn.Linear(self.config["d_condition"], self.config["d_model"])
        self.to_condition_conv = nn.Sequential(
            nn.Conv1d(1, sequence_length, 9, 1, 4),
            nn.GroupNorm(num_groups=1, num_channels=sequence_length),
            nn.Conv1d(sequence_length, sequence_length, 9, 1, 4),
        )  # [batch_size, sequence_length, d_model]
        # reset to_condition
        del self.to_condition

    def forward(self, output_shape=None, x_0=None, condition=None, **kwargs):
        if kwargs.get("pre_training"):
            self.to_condition = self._zero_condition
            condition = None
        else:  # train with condition
            self.to_condition = self._to_condition
            condition = self.get_condition(condition.to(self.device))
        return super().forward(output_shape=output_shape, x_0=x_0, condition=condition, **kwargs)

    def _to_condition(self, x):
        assert len(x.shape) == 3
        x = self.to_condition_linear(x)  # [batch_size, 1, d_model]
        x = self.to_condition_conv(x)  # [batch_size, sequence_length, d_model]
        return x

    def _zero_condition(self, x):
        return torch.zeros(size=(x.shape[0], self.sequence_length, self.config["d_model"]), device=x.device)