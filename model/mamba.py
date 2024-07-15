import torch
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from einops.layers.torch import Reduce
from torch import nn, Tensor
from zeta.nn import SSM


def output_head(dim: int, num_classes: int):
    return nn.Sequential(
        Reduce("b s d -> b d", "mean"),
        nn.LayerNorm(dim),
        nn.Linear(dim, num_classes),
    )


class EncoderMambaBlock(nn.Module):
    """
    EncoderMambaBlock is a module that implements the Mamba block from the paper
    Vision Mamba: Efficient Visual Representation Learning with Bidirectional
    State Space Model

    Args:
        dim (int): The input dimension of the input tensor.
        dt_rank (int): The rank of the state space model.
        dim_inner (int): The dimension of the inner layer of the
            multi-head attention.
        d_state (int): The dimension of the state space model.

    Example:
    >>> block = VisionMambaBlock(dim=256, heads=8, dt_rank=32, dim_inner=512, d_state=256)
    >>> x = torch.randn(1, 32, 256)
    >>> out = block(x)
    >>> out.shape
    torch.Size([1, 32, 256])
    """

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


class MambaModel(nn.Module):

    def __init__(
        self,
        dim: int,
        dt_rank: int = 32,
        dim_inner: int = None,
        d_state: int = None,
        dropout: float = 0.1,
        depth: int = 12,
    ):
        super().__init__()
        self.dim = dim
        self.dt_rank = dt_rank
        self.dim_inner = dim_inner
        self.d_state = d_state
        self.dropout = dropout
        self.depth = depth

        # Dropout
        self.dropout = nn.Dropout(dropout)
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
        # Patch embedding
        # b, s, d = x.shape
        x = self.dropout(x)  # Dropout
        # Forward pass with the layers
        for layer in self.layers:
            x = layer(x)
        # Output head with the cls tokens
        return x[:, -1:, :]




if __name__ == "__main__":
    model = MambaModel(
        dim=256,  # Dimension of the model
        dt_rank=32,  # Rank of the dynamic routing matrix
        dim_inner=256,  # Inner dimension of the model
        d_state=256,  # Dimension of the state vector
        dropout=0.1,  # Dropout rate
        depth=12,  # Depth of the model
    )
    x = torch.randn(4, 100, 256)
    out = model(x)
    print(out.shape)