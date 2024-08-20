import timm
import torch
from torch import nn

def binary_classify():
    model = timm.create_model('resnet50', pretrained=True)
    model.fc = nn.Sequential(nn.Linear(2048, 1), nn.Flatten(start_dim=0))
    nn.init.zeros_(model.fc[0].weight)
    nn.init.zeros_(model.fc[0].bias)
    return model