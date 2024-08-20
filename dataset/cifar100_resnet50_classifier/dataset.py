from torchvision.datasets import CIFAR100
from torch.utils.data import Dataset
from torchvision import transforms
import torch
import os


class OneClassCIFAR100(Dataset):
    def __init__(self, dataset_root, optim_class, train):
        self.images = []
        temp_dataset = CIFAR100(root=dataset_root, train=train, transform=None, download=True)
        self.labels = [(1 if label == optim_class else 0) for i, (_, label) in enumerate(temp_dataset)]
        self.dataset = CIFAR100(
            root=dataset_root,
            train=train,
            transform=transforms.Compose([
                transforms.Resize(64),
                transforms.RandomCrop(64, padding=8),
                transforms.RandomHorizontalFlip(),
                transforms.AutoAugment(policy=transforms.AutoAugmentPolicy("cifar10")),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ]) if train else transforms.Compose([
                transforms.Resize(64),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ]),
            download=True,
        )

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        label = self.labels[index]
        img, _ = self.dataset[index]
        return img, torch.tensor(label, dtype=torch.float32)



