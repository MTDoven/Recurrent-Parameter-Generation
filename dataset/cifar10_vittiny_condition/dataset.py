from torchvision.datasets import CIFAR10
from torch.utils.data import Dataset
from torchvision import transforms
import torch
import os


class OneClassCIFAR10(Dataset):
    def __init__(self, dataset_root, optim_class, train):
        self.images = []
        temp_dataset = CIFAR10(root=dataset_root, train=train, transform=None)
        self.labels = [(1 if label == optim_class else 0) for i, (_, label) in enumerate(temp_dataset)]
        self.dataset = CIFAR10(
            root=dataset_root,
            train=train,
            transform=transforms.Compose([
                transforms.Resize(224),
                transforms.RandomCrop(224, padding=32),
                transforms.RandomHorizontalFlip(),
                transforms.RandAugment(),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2471, 0.2435, 0.2616)),
            ]) if train else transforms.Compose([
                transforms.Resize(224),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2471, 0.2435, 0.2616)),
            ]),
        )

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        label = self.labels[index]
        img, _ = self.dataset[index]
        return img, torch.tensor(label, dtype=torch.float32)



