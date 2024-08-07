import timm
from torch import nn


def ResNet18(output_dim):
    model = timm.create_model("resnet18", pretrained=True)
    model.fc = nn.Linear(512, output_dim)
    return model
