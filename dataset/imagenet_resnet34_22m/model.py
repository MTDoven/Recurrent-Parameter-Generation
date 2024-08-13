import timm
import torch
from torch import nn

def imagenet_classify():
    model = timm.create_model('resnet34', pretrained=True)
    return model
