import timm
import torch
from torch import nn

def imagenet_classify():
    model = timm.create_model('convnext_large', pretrained=True)
    return model