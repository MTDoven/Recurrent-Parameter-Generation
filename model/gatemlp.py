import torch
from torch import nn




class GateMLP(nn.Module):
    def __init__(self, d_model, expand):
        super().__init__()
        self.proj_1 = nn.Linear(d_model, d_model * expand, bias=False)
        self.proj_2 = nn.Linear(d_model, d_model * expand, bias=False)
        self.proj_3 = nn.Linear(d_model * expand, d_model, bias=True)
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, x):
        x, x1 = self.proj_1(x), self.proj_2(x)
        x = x * torch.sigmoid(x1)
        x = self.proj_3(x)
        x = self.layer_norm(x)
        return x


class GMLPModel(nn.Module):
    config = {}

    def __init__(self, positional_embedding):
        super().__init__()
        gmlp_config = {
            "d_model": self.config["d_model"],
            "expand": self.config["expand"],
        }
        self.gmlp_forward = nn.Sequential(*[GateMLP(**gmlp_config) for _ in range(self.config["num_layers"])])
        pe = positional_embedding[None, :, :]
        if self.config.get("trainable_pe"):
            self.pe = nn.Parameter(pe)
        else:  # fixed positional embedding
            self.register_buffer("pe", pe)

    def forward(self, output_shape, condition=None):
        assert len(condition.shape) == 3
        x = self.gmlp_forward(self.pe.repeat(output_shape[0], 1, 1) + condition)
        return x
