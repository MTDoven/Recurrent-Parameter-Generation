import os
import torch
import pickle
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset


class ImageNet1k(Dataset):
    def __init__(self, image_root, mapping_dict):
        # mapping
        new_mapping = []
        with open(mapping_dict, "rb") as f:
            mapping_dict = pickle.load(f)
        for class_index, (key, name_list) in enumerate(mapping_dict.items()):
            for name in name_list:
                path = os.path.join(image_root, name)
                new_mapping.append((path, class_index))
        self.mapping = new_mapping
        # transforms
        self.transform = transforms.Compose([
            transforms.AutoAugment(),
            transforms.Resize(224),
            transforms.RandomCrop(224),
            transforms.ToTensor(),
            transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
        ])

    def __getitem__(self, index):
        path, target = self.mapping[index]
        img = Image.open(path).convert("RGB")
        img = self.transform(img)
        return img, torch.tensor(target, dtype=torch.long)

    def __len__(self):
        return len(self.mapping)
