import torch
from torch import nn


class SimpleMLP(nn.Module):
    def __init__(self, layer_dims: list):
        super().__init__()
        module_list = nn.ModuleList()
        for i in range(len(layer_dims) - 1):
            module_list.append(nn.Linear(layer_dims[i], layer_dims[i+1]))
            module_list.append(nn.LayerNorm(layer_dims[i+1], eps=1e-6))
            module_list.append(nn.SiLU())
        del module_list[-2], module_list[-1]
        self.model = nn.Sequential(*module_list)

    def forward(self, x):
        return self.model(x)


def simple_mlp_for_cifar10_classify():
    """ this model size is 0.021B parameters (0,020,992,000) """
    return SimpleMLP([3072, 4096, 2048, 10])


if __name__ == "__main__":
    model = simple_mlp_for_cifar10_classify()
    x = torch.randn((4, 3072))
    y = model(x)
    print(y.shape)