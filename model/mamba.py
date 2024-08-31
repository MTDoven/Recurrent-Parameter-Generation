import torch
from torch import nn
from mamba_ssm import Mamba2 as Mamba


class Condition(nn.Module):
    def __init__(self, d_condition, d_model, sequence_length):
        super().__init__()
        self.d_condition = d_condition
        self.d_model = d_model
        self.gate = nn.Parameter(torch.ones(1, sequence_length, 1))
        self.linear = nn.Linear(d_condition, d_model)

    def forward(self, condition):
        assert len(condition.shape) == 2
        return self.linear(condition)[:, None, :] * torch.sigmoid(self.gate)


class PermutationState(nn.Module):
    def __init__(self, num_embedding, d_model, sequence_length):
        super().__init__()
        self.embedding = nn.Embedding(num_embedding, d_model)
        self.gate = nn.Parameter(torch.ones(1, sequence_length, 1))

    def forward(self, permutation_state):
        assert len(permutation_state.shape) == 1, f"{permutation_state.shape}"
        return self.embedding(permutation_state)[:, None, :] * torch.sigmoid(self.gate)




class MambaModel(nn.Module):
    config = {}

    def __init__(self, positional_embedding):
        super().__init__()
        if self.config.get("d_model_1") is None:
            assert self.config.get("d_model_2") is None
            self.config["d_model_1"] = self.config["d_model"]
            self.config["d_model_2"] = self.config["d_model"]
        self.mamba1 = Mamba(d_model=self.config["d_model_1"],
                            d_state=self.config["d_state"],
                            d_conv=self.config["d_conv"],
                            expand=self.config["expand"],)
        self.project_middle = nn.Sequential(nn.LayerNorm(self.config["d_model_1"]),
                                            nn.Linear(self.config["d_model_1"], self.config["d_model_2"]),
                                            nn.SiLU(),
                                            nn.Linear(self.config["d_model_2"], self.config["d_model_2"]),
                                            nn.LayerNorm(self.config["d_model_2"]),)
        self.mamba2 = Mamba(d_model=self.config["d_model_2"],
                            d_state=self.config["d_state"],
                            d_conv=self.config["d_conv"],
                            expand=self.config["expand"],)
        self.project_out = nn.Sequential(nn.LayerNorm(self.config["d_model_2"]),
                                         nn.Linear(self.config["d_model_2"], self.config["dim_per_token"]),
                                         nn.SiLU(),
                                         nn.Linear(self.config["dim_per_token"], self.config["dim_per_token"]),)
        self.to_condition = Condition(d_condition=self.config["d_condition"],
                                      d_model=self.config["d_model_1"],
                                      sequence_length=positional_embedding.shape[-2],)
        self.to_permutation_state = PermutationState(num_embedding=self.config["num_permutation_state"],
                                                     d_model=self.config["d_model_1"],
                                                     sequence_length=positional_embedding.shape[-2],)
        pe = positional_embedding[None]
        if self.config.get("trainable_pe"):
            self.pe = nn.Parameter(pe)
        else:  # fixed positional embedding
            self.register_buffer("pe", pe)

    def mamba_forward(self, x):
        x = self.mamba1(x) + x
        x = self.project_middle(x)
        x = self.mamba2(x) + x
        x = self.project_out(x)
        return x

    def forward(self, output_shape, condition=None, permutation_state=None):
        if condition is None:
            assert self.config["d_condition"] == 1
            condition = torch.zeros(size=(1, 1), device=self.pe.device)
        condition = self.to_condition(condition.to(self.pe.device)) + \
                    self.to_permutation_state(permutation_state.to(self.pe.device))
        x = self.mamba_forward(self.pe + condition)
        return x.repeat(output_shape[0], 1, 1) if x.size(0) == 1 else x
