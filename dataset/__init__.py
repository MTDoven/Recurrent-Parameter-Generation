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
    if x.numel() % common_factor == 0:
        return x.flatten()
    assert x.numel() < common_factor
    # print(f"padding {x.shape} according to {common_factor}")
    full_length = (x.numel() // common_factor + 1) * common_factor
    padding_length = full_length - len(x.flatten())
    padding = torch.full([padding_length, ], dtype=x.dtype, device=x.device, fill_value=torch.nan)
    x = torch.cat((x.flatten(), padding), dim=0)
    return x

def layer_to_token(x, common_factor):
    if x.numel() <= common_factor:
        return pad_to_length(x.flatten(), common_factor)[None]
    dim2 = x[0].numel()
    dim1 = x.shape[0]
    if dim2 <= common_factor:
        i = int(dim1 / (common_factor / dim2))
        while True:
            if dim1 % i == 0 and dim2 * (dim1 // i) <= common_factor:
                output = x.view(-1, dim2 * (dim1 // i))
                output = [pad_to_length(item, common_factor) for item in output]
                return torch.stack(output, dim=0)
            i += 1
    else:  # dim2 > common_factor
        output = [layer_to_token(item, common_factor) for item in x]
        return torch.cat(output, dim=0)

def token_to_layer(tokens, shape):
    common_factor = tokens.shape[-1]
    num_element = math.prod(shape)
    if num_element <= common_factor:
        param = tokens[0][:num_element].view(shape)
        tokens = tokens[1:]
        return param, tokens
    dim2 = num_element // shape[0]
    dim1 = shape[0]
    if dim2 <= common_factor:
        i = int(dim1 / (common_factor / dim2))
        while True:
            if dim1 % i == 0 and dim2 * (dim1 // i) <= common_factor:
                item_per_token = dim2 * (dim1 // i)
                length = num_element // item_per_token
                output = [item[:item_per_token] for item in tokens[:length]]
                param = torch.cat(output, dim=0).view(shape)
                tokens = tokens[length:]
                return param, tokens
            i += 1
    else:  # dim2 > common_factor
        output = []
        for i in range(shape[0]):
            param, tokens = token_to_layer(tokens, shape[1:])
            output.append(param.flatten())
        param = torch.cat(output, dim=0).view(shape)
        return param, tokens

def positional_embedding_2d(dim1, dim2, d_model):
    assert d_model % 4 == 0, f"Cannot use sin/cos positional encoding with odd dimension {d_model}"
    pe = torch.zeros(d_model, dim1, dim2)
    # Each dimension use half of d_model
    d_model = int(d_model / 2)
    div_term = torch.exp(torch.arange(0., d_model, 2) * -(math.log(10000.0) / d_model))
    pos_w = torch.arange(0., dim2).unsqueeze(1)
    pos_h = torch.arange(0., dim1).unsqueeze(1)
    pe[0:d_model:2, :, :] = torch.sin(pos_w * div_term).transpose(0, 1).unsqueeze(1).repeat(1, dim1, 1)
    pe[1:d_model:2, :, :] = torch.cos(pos_w * div_term).transpose(0, 1).unsqueeze(1).repeat(1, dim1, 1)
    pe[d_model::2, :, :] = torch.sin(pos_h * div_term).transpose(0, 1).unsqueeze(2).repeat(1, 1, dim2)
    pe[d_model+1::2, :, :] = torch.cos(pos_h * div_term).transpose(0, 1).unsqueeze(2).repeat(1, 1, dim2)
    return pe.permute(1, 2, 0)




class BaseDataset(Dataset, ABC):
    data_path = None
    generated_path = None
    test_command = None
    config = {}

    def __init__(self, checkpoint_path=None, dim_per_token=8192, **kwargs):
        self.config.update(kwargs)
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

    def get_position_embedding(self, positional_embedding_dim=None):
        if positional_embedding_dim is None:
            positional_embedding_dim = self.dim_per_token
        assert self.structure is not None
        positional_embedding_index = []
        for key, item in self.structure.items():
            if ("num_batches_tracked" in key) or (item[-1] is None):
                continue
            else:  # conv & linear
                shape, *_ = item
            fake_param = torch.ones(size=shape)
            fake_param = layer_to_token(fake_param, self.dim_per_token)
            positional_embedding_index.append(list(range(fake_param.size(0))))
        dim1 = len(positional_embedding_index)
        dim2 = max([len(token_per_layer) for token_per_layer in positional_embedding_index])
        full_pe = positional_embedding_2d(dim1, dim2, positional_embedding_dim)
        positional_embedding = []
        for layer_index, token_indexes in enumerate(positional_embedding_index):
            for token_index in token_indexes:
                this_pe = full_pe[layer_index, token_index]
                positional_embedding.append(this_pe)
        positional_embedding = torch.stack(positional_embedding)
        return positional_embedding

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
            value = (value - mean) / std
            value = layer_to_token(value, self.dim_per_token)
            param_list.append(value)
        param = torch.cat(param_list, dim=0)
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
            this_param, params = token_to_layer(params, shape)
            this_param = this_param * std + mean
            if "running_var" in key:
                this_param = torch.clip(torch.exp(this_param) - 0.1, min=0.001) * pre_mean
            diction[key] = this_param
        return diction




class Cifar10_ResNet18(BaseDataset):
    data_path = "./dataset/cifar10_resnet18/checkpoint"
    generated_path = "./dataset/cifar10_resnet18/generated/generated_model.pth"
    test_command = "python ./dataset/cifar10_resnet18/test.py " + \
                   "./dataset/cifar10_resnet18/generated/generated_model.pth"

class Cifar10_ResNet50(BaseDataset):
    data_path = "./dataset/cifar10_resnet50/checkpoint"
    generated_path = "./dataset/cifar10_resnet50/generated/generated_model.pth"
    test_command = "python ./dataset/cifar10_resnet50/test.py " + \
                   "./dataset/cifar10_resnet50/generated/generated_model.pth"

class Cifar10_ResNet101(BaseDataset):
    data_path = "./dataset/cifar10_resnet101/checkpoint"
    generated_path = "./dataset/cifar10_resnet101/generated/generated_model.pth"
    test_command = "python ./dataset/cifar10_resnet101/test.py " + \
                   "./dataset/cifar10_resnet101/generated/generated_model.pth"

class ImageNet_ResNet18(BaseDataset):
    data_path = "./dataset/imagenet_resnet18/checkpoint"
    generated_path = "./dataset/imagenet_resnet18/generated/generated_model.pth"
    test_command = "python ./dataset/imagenet_resnet18/test.py " + \
                   "./dataset/imagenet_resnet18/generated/generated_model.pth"

class ImageNet_ResNet50(BaseDataset):
    data_path = "./dataset/imagenet_resnet50/checkpoint"
    generated_path = "./dataset/imagenet_resnet50/generated/generated_model.pth"
    test_command = "python ./dataset/imagenet_resnet50/test.py " + \
                   "./dataset/imagenet_resnet50/generated/generated_model.pth"

class ImageNet_ResNet101(BaseDataset):
    data_path = "./dataset/imagenet_resnet101/checkpoint"
    generated_path = "./dataset/imagenet_resnet101/generated/generated_model.pth"
    test_command = "python ./dataset/imagenet_resnet101/test.py " + \
                   "./dataset/imagenets_resnet101/generated/generated_model.pth"

class ImageNet_ViTTiny(BaseDataset):
    data_path = "./dataset/imagenet_vittiny/checkpoint"
    generated_path = "./dataset/imagenet_vittiny/generated/generated_model.pth"
    test_command = "python ./dataset/imagenet_vittiny/test.py " + \
                   "./dataset/imagenet_vittiny/generated/generated_model.pth"

class ImageNet_ViTSmall(BaseDataset):
    data_path = "./dataset/imagenet_vitsmall/checkpoint"
    generated_path = "./dataset/imagenet_vitsmall/generated/generated_model.pth"
    test_command = "python ./dataset/imagenet_vitsmall/test.py " + \
                   "./dataset/imagenet_vitsmall/generated/generated_model.pth"

class ImageNet_ViTBase(BaseDataset):
    data_path = "./dataset/imagenet_vitbase/checkpoint"
    generated_path = "./dataset/imagenet_vitbase/generated/generated_model.pth"
    test_command = "python ./dataset/imagenet_vitbase/test.py " + \
                   "./dataset/imagenet_vitbase/generated/generated_model.pth"




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




if __name__ == "__main__":
    dataset = Cifar10_ResNet18(
        dim_per_token=8192,
        checkpoint_path="./cifar10_resnet18/checkpoint")
    example = dataset[0]
    print(example.shape, dataset.get_position_embedding(positional_embedding_dim=4096).shape)
    useful_rate = torch.where(torch.isnan(example), 0., 1.).mean()
    print("useful rate:", useful_rate)