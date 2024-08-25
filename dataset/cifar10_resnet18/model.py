import torch.nn as nn
import timm


def Model():
    model = timm.create_model("resnet18", pretrained=True)
    model.fc = nn.Linear(512, 10, bias=True)
    # nn.init.zeros_(model.fc.weight)
    # nn.init.zeros_(model.fc.bias)
    return model, model.fc


if __name__ == "__main__":
    model, _ = Model()
    print(model)
    num_param = 0
    for v in model.parameters():
        num_param += v.numel()
    print("num_param:", num_param)
