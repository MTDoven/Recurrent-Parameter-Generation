import torch
import einops
from torch.utils.data import Dataset
from torchvision.datasets import CIFAR10
from torchvision import transforms
import os
import math
import random
import json
from abc import ABC


def pad_to_length(x, common_factor):
    if len(x.flatten()) % common_factor == 0:
        return x.flatten()
    # print(f"padding {x.shape} according to {common_factor}")
    full_length = (len(x.flatten()) // common_factor + 1) * common_factor
    padding_length = full_length - len(x.flatten())
    # padding = torch.zeros([padding_length, ], dtype=x.dtype, device=x.device)
    padding = torch.full([padding_length, ], dtype=x.dtype, device=x.device, fill_value=torch.nan)
    x = torch.cat((x.flatten(), padding), dim=0)
    return x


class BaseDataset(Dataset, ABC):
    data_path = None
    generated_path = None
    test_command = None

    def __init__(self, checkpoint_path=None, dim_per_token=8192, **kwargs):
        checkpoint_path = self.data_path if checkpoint_path is None else checkpoint_path
        assert os.path.exists(checkpoint_path)
        self.dim_per_token = dim_per_token
        self.structure = None  # set in get_structure()
        self.sequence_length = None  # set in get_structure()
        # load checkpoint_list
        checkpoint_list = os.listdir(checkpoint_path)
        self.checkpoint_list = list([os.path.join(checkpoint_path, item) for item in checkpoint_list])
        self.length = self.real_length = len(self.checkpoint_list)
        self.set_infinite_dataset()
        self.get_structure()
        # other kwargs
        self.kwargs = kwargs

    def get_structure(self):
        # get structure
        checkpoint_list = self.checkpoint_list
        structures = [{} for _ in range(len(checkpoint_list))]
        for i, checkpoint in enumerate(checkpoint_list):
            diction = torch.load(checkpoint, map_location="cpu")
            for key, value in diction.items():
                if ("num_batches_tracked" in key) or (value.numel() == 1):
                    structures[i][key] = (value.shape, value, None)
                elif "running_var" in key:
                    pre_mean = value.mean() * 0.9
                    value = torch.log(value / pre_mean + 0.1)
                    structures[i][key] = (value.shape, pre_mean, value.mean(), value.std())
                else:  # conv & linear
                    structures[i][key] = (value.shape, value.mean(), value.std())
        final_structure = {}
        structure_diction = torch.load(checkpoint_list[0], map_location="cpu")
        for key, param in structure_diction.items():
            if ("num_batches_tracked" in key) or (param.numel() == 1):
                final_structure[key] = (param.shape, param, None)
            elif "running_var" in key:
                value = [param.shape, 0., 0., 0.]
                for structure in structures:
                    for i in [1, 2, 3]:
                        value[i] += structure[key][i]
                for i in [1, 2, 3]:
                    value[i] /= len(structures)
                final_structure[key] = tuple(value)
            else:  # conv & linear
                value = [param.shape, 0., 0.]
                for structure in structures:
                    for i in [1, 2]:
                        value[i] += structure[key][i]
                for i in [1, 2]:
                    value[i] /= len(structures)
                final_structure[key] = tuple(value)
        self.structure = final_structure
        # get sequence_length
        param = self.preprocess(structure_diction)
        self.sequence_length = param.size(0)

    def set_infinite_dataset(self, max_num=None):
        if max_num is None:
            max_num = self.length * 1000000
        self.length = max_num
        return self

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        index = index % self.real_length
        diction = torch.load(self.checkpoint_list[index], map_location="cpu")
        param = self.preprocess(diction)
        return param

    def save_params(self, params, save_path):
        diction = self.postprocess(params.cpu().to(torch.float32))
        torch.save(diction, save_path)

    def preprocess(self, diction: dict, **kwargs) -> torch.Tensor:
        param_list = []
        for key, value in diction.items():
            if ("num_batches_tracked" in key) or (value.numel() == 1):
                continue
            elif "running_var" in key:
                shape, pre_mean, mean, std = self.structure[key]
                value = torch.log(value / pre_mean + 0.1)
            else:  # normal
                shape, mean, std = self.structure[key]
            value = value.flatten()
            value = (value - mean) / std
            value = pad_to_length(value, self.dim_per_token)
            param_list.append(value)
        param = torch.cat(param_list, dim=0)
        param = pad_to_length(param, self.dim_per_token)
        param = param.view(-1, self.dim_per_token)
        # print("Sequence length:", param.size(0))
        return param

    def postprocess(self, params: torch.Tensor, **kwargs) -> dict:
        diction = {}
        params = params.flatten()
        for key, item in self.structure.items():
            if ("num_batches_tracked" in key) or (item[-1] is None):
                shape, mean, std = item
                diction[key] = mean
                continue
            elif "running_var" in key:
                shape, pre_mean, mean, std = item
            else:  # conv & linear
                shape, mean, std = item
            num_elements = math.prod(shape)
            this_param = params[:num_elements].view(*shape)
            this_param = this_param * std + mean
            if "running_var" in key:
                this_param = torch.clip(torch.exp(this_param) - 0.1, min=0.001) * pre_mean
            diction[key] = this_param
            cutting_length = num_elements if num_elements % self.dim_per_token == 0 \
                    else (num_elements // self.dim_per_token + 1) * self.dim_per_token
            params = params[cutting_length:]
        return diction




class Cifar10_MLPTesting(BaseDataset):
    data_path = "./dataset/cifar10_mlptesting_1m/checkpoint"
    generated_path = "./dataset/cifar10_mlptesting_1m/generated/generated_model.pth"
    test_command = "python ./dataset/cifar10_mlptesting_1m/test.py " + \
                   "./dataset/cifar10_mlptesting_1m/generated/generated_model.pth"


class Cifar10_GoogleNet(BaseDataset):
    data_path = "./dataset/cifar10_googlenet_6m/checkpoint"
    generated_path = "./dataset/cifar10_googlenet_6m/generated/generated_model.pth"
    test_command = "python ./dataset/cifar10_googlenet_6m/test.py " + \
                   "./dataset/cifar10_googlenet_6m/generated/generated_model.pth"


class Cifar10_ResNet18(BaseDataset):
    data_path = "./dataset/cifar10_resnet18_11m/checkpoint"
    generated_path = "./dataset/cifar10_resnet18_11m/generated/generated_model.pth"
    test_command = "python ./dataset/cifar10_resnet18_11m/test.py " + \
                   "./dataset/cifar10_resnet18_11m/generated/generated_model.pth"


class ImageNet_ConvNeXt(BaseDataset):
    data_path = "./dataset/imagenet_convnext_4m/checkpoint"
    generated_path = "./dataset/imagenet_convnext_4m/generated/generated_model.pth"
    test_command = "python ./dataset/imagenet_convnext_4m/test.py " + \
                   "./dataset/imagenet_convnext_4m/generated/generated_model.pth"


class ImageNet_ViTTiny(BaseDataset):
    data_path = "./dataset/imagenet_vittiny_6m/checkpoint"
    generated_path = "./dataset/imagenet_vittiny_6m/generated/generated_model.pth"
    test_command = "python ./dataset/imagenet_vittiny_6m/test.py " + \
                   "./dataset/imagenet_vittiny_6m/generated/generated_model.pth"


class ImageNet_ViTBase(BaseDataset):
    data_path = "./dataset/imagenet_vitbase_86m/checkpoint"
    generated_path = "./dataset/imagenet_vitbase_86m/generated/generated_model.pth"
    test_command = "python ./dataset/imagenet_vitbase_86m/test.py " + \
                   "./dataset/imagenet_vitbase_86m/generated/generated_model.pth"


class ImageNet_ResNet50(BaseDataset):
    data_path = "./dataset/imagenet_resnet50_26m/checkpoint"
    generated_path = "./dataset/imagenet_resnet50_26m/generated/generated_model.pth"
    test_command = "python ./dataset/imagenet_resnet50_26m/test.py " + \
                   "./dataset/imagenet_resnet50_26m/generated/generated_model.pth"




class ConditionalDataset(BaseDataset):
    def _extract_condition(self, index: int):
        name = self.checkpoint_list[index]
        condition_list = os.path.basename(name).split("_")
        return condition_list

    def __getitem__(self, index):
        index = index % self.real_length
        diction = torch.load(self.checkpoint_list[index], map_location="cpu")
        condition = self._extract_condition(index)
        param = self.preprocess(diction)
        return param, condition




class Cifar10_ViTTiny_Performance(ConditionalDataset):
    dataset_config = "./dataset/Cifar10/config.json"
    data_path = "./dataset/cifar10_vittiny_performance/checkpoint"
    generated_path = "./dataset/cifar10_vittiny_performance/generated/generated_model_acc{}.pth"
    test_command = "python ./dataset/cifar10_vittiny_performance/test.py " + \
                   "./dataset/cifar10_vittiny_performance/generated/generated_model_acc{}.pth"

    def _extract_condition(self, index: int):
        acc = float(super()._extract_condition(index)[1][3:])
        return torch.tensor(acc, dtype=torch.float32)


class Cifar10_ViTTiny_OneClass(ConditionalDataset):
    dataset_config = "./dataset/Cifar10/config.json"
    data_path = "./dataset/cifar10_vittiny_condition/checkpoint"
    generated_path = "./dataset/cifar10_vittiny_condition/generated/generated_model_class{}.pth"
    test_command = "python ./dataset/cifar10_vittiny_condition/test.py " + \
                   "./dataset/cifar10_vittiny_condition/generated/generated_model_class{}.pth"

    def __init__(self, checkpoint_path=None, dim_per_token=8192, **kwargs):
        super().__init__(checkpoint_path=checkpoint_path, dim_per_token=dim_per_token, **kwargs)
        # load dataset_config
        with open(self.dataset_config, "r") as f:
            dataset_config = json.load(f)
        # train dataset
        self.dataset = CIFAR10(root=dataset_config["dataset_root"], train=True, transform=None)
        self.indices = [[] for _ in range(10)]
        for index, (_, label) in enumerate(self.dataset):
            self.indices[label].append(index)
        self.transform = transforms.Compose([
            transforms.Resize(64),
            transforms.RandomCrop(64, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.RandAugment(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2471, 0.2435, 0.2616)),
        ])
        # test dataset
        self.test_dataset = CIFAR10(root=dataset_config["dataset_root"], train=False, transform=None)
        self.test_indices = [[] for _ in range(10)]
        for index, (_, label) in enumerate(self.test_dataset):
            self.test_indices[label].append(index)
        self.test_transform = transforms.Compose([
            transforms.Resize(64),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2471, 0.2435, 0.2616)),
        ])

    def _extract_condition(self, index: int):
        optim_class = int(super()._extract_condition(index)[1][5:])
        img_index = random.choice(self.indices[optim_class])
        img = self.transform(self.dataset[img_index][0])
        return img

    def get_image_by_class_index(self, class_index):
        img_index = random.choice(self.test_indices[class_index])
        img = self.test_transform(self.test_dataset[img_index][0])
        return img

