import torch
import torch.nn as nn
import math


class Mixer(nn.Module):
    def __init__(self, input_dim, output_dim, kernel_size, activation):
        super().__init__()
        self.pre_norm = nn.BatchNorm1d(input_dim)
        self.middle_norm = nn.BatchNorm1d(input_dim)
        self.activate = activation
        self.conv1 = nn.Conv1d(input_dim, input_dim, kernel_size,
                               1, kernel_size//2)
        self.conv2 = nn.Conv1d(input_dim, input_dim, kernel_size,
                               1, kernel_size//2)
        self.conv3 = nn.Conv1d(input_dim, input_dim, kernel_size,
                               1, kernel_size//2)
        self.conv4 = nn.Conv1d(input_dim, output_dim, kernel_size,
                               1, kernel_size//2)

    def forward(self, x0):
        x1 = self.pre_norm(x0)
        x1 = self.activate(self.conv1(x1))
        x1 = self.activate(self.conv2(x1))
        x1 = x1 + x0
        x2 = self.middle_norm(x1)
        x2 = self.activate(self.conv3(x2))
        x2 = self.activate(self.conv4(x2))
        return x2


class EncoderLayer(nn.Module):
    def __init__(self, hidden_dim, input_length, predict_length,
                 token_mixer_ks, feed_forward_ks, activation):
        super().__init__()
        self.token_mixer = Mixer(input_length, predict_length, token_mixer_ks, activation)
        self.feed_forward = Mixer(hidden_dim, hidden_dim, feed_forward_ks, activation)
        self.predict_length = predict_length

    def forward(self, x):
        src = x
        prediction = self.token_mixer(x)
        x = torch.cat((x[:, :-self.predict_length, :], prediction), dim=1)
        x = src = torch.permute(x + src, (0, 2, 1))
        x = self.feed_forward(x) + src
        x = torch.permute(x, (0, 2, 1))
        return x


class BiARModule(nn.Module):
    def __init__(self, num_layers,
                 hidden_dim, input_length, predict_length,
                 token_mixer_ks, feed_forward_ks, activation):
        super().__init__()
        # define
        self.hidden_dim = hidden_dim
        self.input_length = input_length
        self.predict_length = predict_length
        # define encoder
        module_list = [EncoderLayer(hidden_dim, input_length + predict_length, predict_length,
                                    token_mixer_ks, feed_forward_ks, activation) for _ in range(num_layers)]
        self.encoder = nn.Sequential(*module_list)
        # define next_padding token
        next_padding = torch.empty(size=(1, self.predict_length, self.hidden_dim))
        self.next_padding = nn.Parameter(nn.init.normal_(next_padding))
        start_padding = torch.empty(size=(1, self.input_length, self.hidden_dim))
        self.start_padding = nn.Parameter(nn.init.normal_(start_padding))

    def forward(self, x):
        if x.size(1) < self.input_length:
            x = torch.cat((self.start_padding.repeat(x.size(0), 1, 1), x), dim=1)
            x = x[:, -self.input_length:, :]
        x = torch.cat((x, self.next_padding.repeat(x.size(0), 1, 1)), dim=1)
        x = self.encoder(x)
        return x[:, -self.predict_length:, :]


class RNNModule(nn.Module):
    CONV_EXPAND = 4

    def __init__(self, hidden_dim, book_size, out_conv_ks):
        super().__init__()
        from dataset.Dataset import Cifar10_MLP
        dataset = Cifar10_MLP(checkpoint_path="/home/wangkai/AR-Param-Generation/AR-Param-Generation/dataset/cifar10_MLP_middle/checkpoint",
                              dim_per_token=2048, predict_length=64)
        model_param = dataset[0].to(torch.float32)
        model_param = model_param.view((1, book_size, hidden_dim * 64))
        self.code_book = nn.Parameter(model_param)
        # self.code_book.requires_grad = False
        # self.code_book = nn.Parameter(
        #         nn.init.zeros_(torch.empty(size=(1, book_size, hidden_dim * 64))))
        self.next_state = nn.Sequential(
                nn.Linear(book_size, book_size),)
                #nn.Softmax(dim=-1),)
        self.start_state = nn.Parameter(
                nn.init.zeros_(torch.empty(size=(1, 1, book_size))))
        self.out_conv = nn.Identity()
        # self.out_conv = nn.Sequential(
        #         nn.Conv1d(1, self.CONV_EXPAND, out_conv_ks, 1, out_conv_ks//2),
        #         nn.ELU(),
        #         nn.Conv1d(self.CONV_EXPAND, 1, out_conv_ks, 1, out_conv_ks//2),)

    def forward(self, last_state, batch_size):
        if last_state is None:
            last_state = self.start_state.repeat(batch_size, 1, 1)
        this_state = self.next_state(last_state)
        output = this_state @ self.code_book
        output = self.out_conv(output)
        return output, this_state


if __name__ == '__main__':
    model = BiARModule(num_layers=6,
                       hidden_dim=1024, input_length=512, predict_length=64,
                       token_mixer_ks=65, feed_forward_ks=65, activation=nn.ELU()).cuda()
    # for key, value in model.named_parameters():
    #     print(key, value)
    src = torch.randn((2, 64, 1024)).cuda()
    out = model(src)
    print(out.shape)
