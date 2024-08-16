import timm
import torch
from torch import nn

def imagenet_classify():
    model = timm.create_model('resnet18', pretrained=True)
    return model
