import torch.nn as nn
import torch
import timm
import os


def Model():
    model = timm.create_model("resnet18", pretrained=True)
    model.fc = nn.Linear(512, 100)
    if os.path.exists(os.path.join(os.path.dirname(__file__), "full_model.pth")):
        model.load_state_dict(torch.load(os.path.join(os.path.dirname(__file__), "full_model.pth"), map_location="cpu"))
        for k, v in model.named_parameters():
            if k in ["layer4.1.bn1.weight", "layer4.1.bn1.bias", "layer4.1.bn2.weight", "layer4.1.bn2.bias"]:
                v.requires_grad = True
            else:  # requires_grad = False
                v.requires_grad = False
    return model, model.fc


if __name__ == "__main__":
    model, _ = Model()
    print(model)
    num_param = 0
    for k, v in model.named_parameters():
        num_param += v.numel()
        print(k)
    print("num_param:", num_param)