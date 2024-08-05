import timm
import torch
from torch import nn

def imagenet_classify():
    model = timm.create_model('convnext_atto', pretrained=True)
    return model