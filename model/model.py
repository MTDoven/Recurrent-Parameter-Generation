import torch
from torch import nn
from model.mamba import MambaModel


class Model(nn.Module):
    def __init__(
        self,
        # mamba setting
        dim=256,  # Dimension of the model
        dt_rank=32,  # Rank of the dynamic routing matrix
        dim_inner=256,  # Inner dimension of the model
        d_state=256,  # Dimension of the state vector
        dropout=0.1,  # Dropout rate
        depth=12,  # Depth of the model
        # diffusion setting
    ):
        super().__init__()
        self.mamba = MambaModel(
            dim=dim,  # Dimension of the model
            dt_rank=dt_rank,  # Rank of the dynamic routing matrix
            dim_inner=dim_inner,  # Inner dimension of the model
            d_state=d_state,  # Dimension of the state vector
            dropout=dropout,  # Dropout rate
            depth=depth,  # Depth of the model
        )
        self.next_token = nn.Parameter(torch.nn.init.normal_(torch.empty(size=(1, 1, dim)), std=1e-8))

    def forward_mamba(self, x):
        # b, s, d = x.shape
        x = torch.cat((x, self.next_token.repeat(x.size(0), 1, 1)), dim=1)
        x = self.mamba(x)
        return x  # b, 1, d
