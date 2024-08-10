import timm
import torch
from torch import nn

def imagenet_classify():
    model = timm.create_model('vit_tiny_patch16_224', pretrained=True)
    model.head = nn.Linear(192, 10)
    nn.init.zeros_(model.head.weight)
    nn.init.zeros_(model.head.bias)
    return model
