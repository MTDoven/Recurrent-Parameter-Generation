import timm
import torch
from torch import nn

def imagenet_classify():
    model = timm.create_model('vit_tiny_patch16_224', pretrained=True)
    model.head = nn.Sequential(nn.Linear(192, 1), nn.Flatten(start_dim=0))
    return model