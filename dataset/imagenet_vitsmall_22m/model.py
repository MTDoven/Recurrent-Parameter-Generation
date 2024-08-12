import timm
import torch
from torch import nn

def imagenet_classify():
    model = timm.create_model('vit_small_patch16_224', pretrained=True)
    return model