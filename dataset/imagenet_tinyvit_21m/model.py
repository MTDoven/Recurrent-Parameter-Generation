import timm
import torch
from torch import nn

def imagenet_classify():
    model = timm.create_model('tiny_vit_21m_224', pretrained=True)
    model.head.fc = nn.Linear(576, 1000)
    return model, model.head.fc

