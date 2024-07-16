import torch

#diction1 = torch.load("dataset/cifar10_MLP_1M/generated/generated_classifier.pth")
diction2 = torch.load("dataset/cifar10_mlp_1m/checkpoint/0110.pth")

# for key, value in diction1.items():
#     print(value.flatten()[:50])
#     break

values = []
for key, value in diction2.items():
    print(value.abs().mean())
    values.append(value.flatten())
values = torch.cat(values, dim=0)
print()
print(values.abs().mean())
