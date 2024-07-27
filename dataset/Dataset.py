import torch
import einops
from torch.utils.data import Dataset
import os
import math
from abc import ABC


def pad_to_length(x, common_factor):
    if len(x.flatten()) % common_factor == 0:
        return x.flatten()
    # print(f"padding {x.shape} according to {common_factor}")
    full_length = (len(x.flatten()) // common_factor + 1) * common_factor
    padding_length = full_length - len(x.flatten())
    padding = torch.zeros([padding_length, ], dtype=x.dtype, device=x.device)
    x = torch.cat((x.flatten(), padding), dim=0)
    return x


class BaseDataset(Dataset, ABC):
    data_path = None
    generated_path = None
    test_command = None

    def __init__(self, checkpoint_path=None, dim_per_token=1024, **kwargs):
        checkpoint_path = self.data_path if checkpoint_path is None else checkpoint_path
        assert os.path.exists(checkpoint_path)
        self.dim_per_token = dim_per_token
        self.sequence_length = None
        self.return_full_param = False
        checkpoint_list = os.listdir(checkpoint_path)
        self.checkpoint_list = list([os.path.join(checkpoint_path, item) for item in checkpoint_list])
        self.length = self.real_length = len(self.checkpoint_list)
        self.structure = {}
        diction = torch.load(self.checkpoint_list[0], map_location="cpu")
        for key, value in diction.items():
            if len(value.shape) == 0:
                self.structure[key] = (value.shape, value, None)
                continue
            self.structure[key] = (value.shape, value.mean(), value.std())
        self.kwargs = kwargs

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        index = index % self.real_length
        diction = torch.load(self.checkpoint_list[index], map_location="cpu")
        param = self.preprocess(diction)
        self.sequence_length = param.size(0)
        if self.return_full_param:
            return param
        if self.kwargs.get("use_pe"):
            from model.transformer import get_sinusoid
            param += get_sinusoid(self.sequence_length, self.dim_per_token)
        inputs, targets = param[:-1], param[:]
        return inputs, targets

    def preprocess_data(self, datas):
        max_input_length = config["max_input_length"]
        sequence_length = config["sequence_length"]
        predict_length = 1
        assert max_input_length % predict_length == 0
        assert sequence_length % predict_length == 0
        random_cutting = random.randint(0, sequence_length - 1)
        inputs = datas[:, max(random_cutting - max_input_length, 0):random_cutting, :]
        targets = datas[:, random_cutting:random_cutting + predict_length, :]
        inputs, targets = inputs.to(config["device"], torch.float32), targets.to(config["device"], torch.float32)
        return inputs, targets

    def save_params(self, params, save_path):
        diction = self.postprocess(params.cpu())
        torch.save(diction, save_path)

    def set_infinite_dataset(self, max_num=None):
        if max_num is None:
            max_num = self.length * 10000000
        self.length = max_num
        return self

    def set_return_full_param(self, mode=True):
        self.return_full_param = mode

    def preprocess(self, diction: dict, **kwargs) -> torch.Tensor:
        param_list = []
        for key, value in diction.items():
            shape, mean, std = self.structure[key]
            if std is None:
                continue
            value = value.flatten()
            value = (value - mean) / std
            param_list.append(value)
        param = torch.cat(param_list, dim=0)
        param = pad_to_length(param, self.dim_per_token)
        param = param.view(-1, self.dim_per_token)
        # print("Sequence length:", param.size(0))
        return param

    def postprocess(self, params: torch.Tensor, **kwargs) -> dict:
        diction = {}
        params = params.flatten()
        for key, (shape, mean, std) in self.structure.items():
            if std is None:
                diction[key] = mean
                continue
            num_elements = math.prod(shape)
            this_param = params[:num_elements].view(*shape)
            this_param = this_param * std + mean
            diction[key] = this_param
            if "running_var" in key:
                diction[key] = torch.clip(this_param, min=1e-6)
            params = params[num_elements:]
        return diction


class RandomDebugDataset(Dataset):
    data_path = "do not need a data_path."
    generated_path = "do not need a generated_path."
    test_command = "echo ''"

    def __init__(self, dim_per_token, max_input_length, test_tensor, **kwargs):
        self.real_length = 1
        self.length = 10000000
        self.dim_per_token = dim_per_token
        self.max_input_length = max_input_length
        self.sequence_length = None
        self.return_full_param = False
        self.kwargs = kwargs
        assert isinstance(test_tensor, torch.Tensor), "input a test tensor"
        param = test_tensor
        # make sure image have the same mse loss scale with params
        self.param = (param - param.mean()) / param.std()
        self.param = pad_to_length(self.param, self.dim_per_token).view(-1, self.dim_per_token)


    def __len__(self):
        return self.length

    def __getitem__(self, index):
        index = index % self.real_length
        param = self.param
        max_input_length = self.max_input_length
        self.sequence_length = param.size(0)
        if self.return_full_param:
            return self.param
        random_cutting = index % self.sequence_length
        inputs = param[max(random_cutting - max_input_length, 0):random_cutting, :]
        if inputs.size(0) < max_input_length:
            padding = torch.zeros((max_input_length-inputs.size(0), self.dim_per_token))
            inputs = torch.cat((padding, inputs), dim=0)
        assert inputs.size(0) == max_input_length
        targets = param[random_cutting:random_cutting + 1, :]
        inputs, targets = inputs.to(torch.float32), targets.to(torch.float32)
        return inputs, targets

    def save_params(self, params, save_path):
        print("Contrast:")
        print("ground_truth:", self.param.flatten()[-5:])
        print("prediction:", params.flatten()[-5:])




class Cifar10_MLP(BaseDataset):
    data_path = "./dataset/cifar10_mlp_1m/checkpoint"
    generated_path = "./dataset/cifar10_mlp_1m/generated/generated_classifier.pth"
    test_command = "CUDA_VISIBLE_DEVICE=0 python " + \
                   "./dataset/cifar10_mlp_1m/test.py " + \
                   "./dataset/cifar10_mlp_1m/generated/generated_classifier.pth"


class Cifar10_GoogleNet(BaseDataset):
    data_path = "./dataset/cifar10_googlenet_6m/checkpoint"
    generated_path = "./dataset/cifar10_googlenet_6m/generated/generated_classifier.pth"
    test_command = "CUDA_VISIBLE_DEVICE=0 python " + \
                   "./dataset/cifar10_googlenet_6m/test.py " + \
                   "./dataset/cifar10_googlenet_6m/generated/generated_classifier.pth"


class Cifar10_ResNet18(BaseDataset):
    data_path = "./dataset/cifar10_resnet18_11m/checkpoint-single"
    generated_path = "./dataset/cifar10_resnet18_11m/generated/generated_classifier.pth"
    test_command = "CUDA_VISIBLE_DEVICE=0 python " + \
                   "./dataset/cifar10_resnet18_11m/test.py " + \
                   "./dataset/cifar10_resnet18_11m/generated/generated_classifier.pth"


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
        self.sequence_length = param.size(0)
        if self.return_full_param:
            return param, condition
        if self.kwargs.get("use_pe"):
            from model.transformer import get_sinusoid
            param += get_sinusoid(self.sequence_length, self.dim_per_token)
        inputs, targets = param[:-1], param[:]
        return inputs, targets, condition



class Cifar10_ResNet18_MultiSeed(ConditionalDataset):
    data_path = "./dataset/cifar10_resnet18_11m/checkpoint-92-94"
    generated_path = "./dataset/cifar10_resnet18_11m/generated/generated_seed{}.pth"
    test_command = "CUDA_VISIBLE_DEVICE=0 python " + \
                   "./dataset/cifar10_resnet18_11m/test.py " + \
                   "./dataset/cifar10_resnet18_11m/generated/generated_seed{}.pth"

    def _extract_condition(self, index: int):
        float_number = float(super()._extract_condition(index)[2][4:])
        return (torch.tensor(float_number, dtype=torch.float32) - 15.) / 5.


class Cifar10_ResNet18_MultiAbility(ConditionalDataset):
    data_path = "./dataset/cifar10_resnet18_11m/checkpoint-20-85"
    generated_path = "./dataset/cifar10_resnet18_11m/generated/generated_acc{}.pth"
    test_command = "CUDA_VISIBLE_DEVICE=0 python " + \
                   "./dataset/cifar10_resnet18_11m/test.py " + \
                   "./dataset/cifar10_resnet18_11m/generated/generated_acc{}.pth"

    def _extract_condition(self, index: int):
        float_number = float(super()._extract_condition(index)[1][3:])
        return (torch.tensor(float_number, dtype=torch.float32) - 0.5) / 0.5