import torch
import torch.nn as nn
from torch.nn import functional as F
import timm


class CNNSmall(nn.Module):
    def __init__(self):
        super().__init__()
        self.module = nn.Sequential(
            nn.Conv2d(3, 8, 5),
            nn.MaxPool2d(2, 2),
            nn.LeakyReLU(),
            nn.Conv2d(8, 6, 5),
            nn.MaxPool2d(2, 2),
            nn.LeakyReLU(),
            nn.Conv2d(6, 4, 2),
            nn.LeakyReLU(),
            nn.Flatten(start_dim=1),
        )
        self.head = nn.Sequential(
            nn.Linear(36, 20),
            nn.LeakyReLU(),
            nn.Linear(20, 10),
        )

    def forward(self, x):
        x = F.interpolate(x, (28, 28), mode='bilinear')
        x = self.module(x)
        x = self.head(x)
        return x


def Model():
    model = CNNSmall()
    return model, model.head


if __name__ == "__main__":
    model, _ = Model()
    x = torch.ones([4, 3, 28, 28])
    y = model(x)
    print(y.shape)
    print(model)
    num_param = 0
    for v in model.parameters():
        num_param += v.numel()
    print("num_param:", num_param)
