import torch.nn as nn
import timm


def Model():
    model = timm.create_model("mobilenetv3_large_100", pretrained=True)
    model.classifier = nn.Linear(1280, 10)
    for name, param in model.named_parameters():
        if "bn" in name:
            # print(f"freeze {name}")
            param.requires_grad = False
    return model, model.classifier


if __name__ == "__main__":
    model, _ = Model()
    print(model)
    num_param = 0
    for v in model.parameters():
        num_param += v.numel()
    print("num_param:", num_param)
