import timm
import torch
from torch import nn

def imagenet_classify():
    model = timm.create_model('convnext_atto', pretrained=True)
    model.head.fc = nn.Linear(320, 1000)
    return model, model.head.fc
