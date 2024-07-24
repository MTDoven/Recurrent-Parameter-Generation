import torch
from torch import nn


class SimpleMLP(nn.Module):
    def __init__(self, layer_dims: list):
        super().__init__()
        module_list = nn.ModuleList()
        for i in range(len(layer_dims) - 1):
            module_list.append(nn.Linear(layer_dims[i], layer_dims[i+1],
                                         bias=i != len(layer_dims) - 2))
            module_list.append(nn.LayerNorm(layer_dims[i+1], eps=1e-6))
            module_list.append(nn.SiLU())
        del module_list[-1]
        del module_list[-1]
        self.model = nn.Sequential(*module_list)

    def forward(self, x):
        return self.model(x)


def cifar10_classifier():
    """ this model size is 0.001B parameters (0,000,992,768) """
    return SimpleMLP([3072, 256, 768, 10])




if __name__ == "__main__":
    model = cifar10_classifier()
    size = 0
    for name, param in model.named_parameters():
        size += len(param.flatten())
    print(size)