import torch
import torch.nn as nn
import timm


def Model():
    model = timm.create_model("vit_tiny_patch16_224", pretrained=False)
    model.head = nn.Linear(192, 10)
    return model, model.head


if __name__ == "__main__":
    model, _ = Model()
    print(model)
    num_param = 0
    for v in model.parameters():
        num_param += v.numel()
    print("num_param:", num_param)
