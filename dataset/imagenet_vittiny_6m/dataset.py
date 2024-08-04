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
        file_list = {}
        for item in os.listdir(image_root):
            item = os.path.join(image_root, item)
            if os.path.isdir(item):
                for file in os.listdir(item):
                    file = os.path.join(item, file)
                    file_list[os.path.basename(file)] = file
            elif os.path.isfile(item):
                file_list[os.path.basename(item)] = item
        for class_index, (key, name_list) in enumerate(mapping_dict.items()):
            for name in name_list:
                path = file_list[name]
                new_mapping.append((path, class_index))
        self.mapping = new_mapping
        # transforms
        self.transform = transforms.Compose([
            transforms.AutoAugment(),
            transforms.Resize(224),
            transforms.RandomCrop(224),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ]) if image_root[-5:] == "train" else transforms.Compose([
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])  # evaluate

    def __getitem__(self, index):
        path, target = self.mapping[index]
        img = Image.open(path).convert("RGB")
        img = self.transform(img)
        return img, torch.tensor(target, dtype=torch.long)

    def __len__(self):
        return len(self.mapping)
