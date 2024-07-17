import torch
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from einops.layers.torch import Reduce
from torch import nn, Tensor
from zeta.nn import SSM


class EncoderMambaBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        dt_rank: int,
        dim_inner: int,
        d_state: int,
    ):
        super().__init__()
        self.dim = dim
        self.dt_rank = dt_rank
        self.dim_inner = dim_inner
        self.d_state = d_state
        # modules
        self.forward_conv1d = nn.Conv1d(in_channels=dim, out_channels=dim, kernel_size=1)
        self.backward_conv1d = nn.Conv1d(in_channels=dim, out_channels=dim, kernel_size=1)
        self.norm = nn.LayerNorm(dim)
        self.silu = nn.SiLU()
        self.ssm = SSM(dim, dt_rank, dim_inner, d_state)
        self.proj = nn.Linear(dim, dim)  # Linear layer for z and x
        self.softplus = nn.Softplus()  # Softplus

    def forward(self, x: torch.Tensor):
        # b, s, d = x.shape
        skip = x  # Skip connection
        x = self.norm(x)  # Normalization
        # Split x into x1 and x2 with linears
        x = self.proj(x)
        x1 = self.process_direction(x, self.forward_conv1d, self.ssm)  # forward conv1d
        x2 = self.process_direction(x, self.backward_conv1d, self.ssm)  # backward conv1d
        # z project and activation
        z = self.silu(x)
        # Matmul & Residual connection
        return x1 * z + x2 * z + skip

    def process_direction(self, x: Tensor, conv1d: nn.Conv1d, ssm: SSM):
        x = rearrange(x, "b s d -> b d s")
        x = self.softplus(conv1d(x))
        # print(f"Conv1d: {x}")
        x = rearrange(x, "b d s -> b s d")
        x = ssm(x)
        return x


class Mamba(nn.Module):
    def __init__(
        self,
        dim: int,
        dt_rank: int = 32,
        dim_inner: int = None,
        d_state: int = None,
        depth: int = 12,
    ):
        super().__init__()
        self.dim = dim
        self.dt_rank = dt_rank
        self.dim_inner = dim_inner
        self.d_state = d_state
        self.depth = depth
        # encoder layers
        self.layers = nn.ModuleList()
        for _ in range(depth):
            self.layers.append(
                EncoderMambaBlock(
                    dim=dim,
                    dt_rank=dt_rank,
                    dim_inner=dim_inner,
                    d_state=d_state,
                )
            )

    def forward(self, x: Tensor):
        # b, s, d = x.shape
        for layer in self.layers:
            x = layer(x)
        return x




class MambaModel(nn.Module):
    config = {
        "dim": 1024,
        "dt_rank": 16,
        "dim_inner": 1024,
        "d_state": 64,
        "depth": 3,
    }

    def __init__(self):
        super().__init__()
        self.model = Mamba(
            dim=self.config["dim"],
            dt_rank=self.config["dt_rank"],
            dim_inner=self.config["dim_inner"],
            d_state=self.config["d_state"],
            depth=self.config["depth"],
        )
        self.next_token = nn.Parameter(nn.init.normal_(torch.empty((1, 1, self.config["dim"]))))

    def forward(self, x):
        assert len(x.shape) == 3
        assert x.shape[-1] == self.config["dim"]
        x = torch.cat((x, self.next_token.repeat(x.size(0), 1, 1)), dim=1)
        x = self.model(x)
        return x[:, -1:, :]
