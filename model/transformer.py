import torch
import torch.nn as nn
from .attention import MHA
import math


class PositionalEncoding(torch.nn.Module):
    def __init__(self, d_model, max_len=5000):
        """
        :param d_model: 模型的维度 (word embedding的维度)
        :param max_len: 最大序列长度
        """
        super(PositionalEncoding, self).__init__()

        # 创建位置编码张量
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)

        # 将位置编码注册为缓冲区
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        :param x: 输入张量 (seq_len, batch_size, d_model)
        """
        x = x + self.pe[:x.size(0), :]
        return x


class MaskFreeTransformerEncoderLayer(nn.Module):
    def __init__(self, hidden_dim, num_heads, feedforward_dim,
                 dropout=0.1, activation=nn.GELU(), use_flash_attn=True):
        super().__init__()
        self.attention = MHA(embed_dim=hidden_dim,
                             num_heads=num_heads,
                             dropout=dropout,
                             use_alibi=False,
                             use_flash_attn=use_flash_attn,
                             dtype=torch.bfloat16,
                             rotary_emb_dim=num_heads,)
        self.feed_forward = nn.Sequential(
                nn.Linear(hidden_dim, feedforward_dim),
                activation,
                nn.Dropout(dropout),
                nn.Linear(feedforward_dim, hidden_dim),
                activation,
                nn.Dropout(dropout),)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)

    def forward(self, x):
        dtype = x.dtype
        x = self.attention(x.to(torch.bfloat16)).to(dtype)
        x = self.norm1(x) + x
        x = self.feed_forward(x)
        x = self.norm2(x) + x
        return x


class MaskFreeTransformerEncoder(nn.Module):
    def __init__(self, hidden_dim, num_layers, num_heads, feedforward_dim,
                 dropout=0.1, activation=nn.GELU(), use_flash_attn=True):
        super().__init__()
        module_list = [MaskFreeTransformerEncoderLayer(
                           hidden_dim=hidden_dim,
                           num_heads=num_heads,
                           feedforward_dim=feedforward_dim,
                           dropout=dropout,
                           activation=activation,
                           use_flash_attn=use_flash_attn
                       ) for _ in range(num_layers)]
        self.encoder = nn.Sequential(*module_list)

    def forward(self, x):
        return self.encoder(x)


class BiARTransformer(nn.Module):
    def __init__(self, hidden_dim: int, stage1_layers: int, stage2_layers: int, predict_length: int,
                 num_heads=8, feedforward_dim=2048, dropout=0.1, activation=nn.GELU()):
        super().__init__()
        self.stage1 = MaskFreeTransformerEncoder(hidden_dim=hidden_dim, num_layers=stage1_layers,
                num_heads=num_heads, feedforward_dim=feedforward_dim, dropout=dropout, activation=activation)
        self.stage2 = MaskFreeTransformerEncoder(hidden_dim=hidden_dim, num_layers=stage2_layers,
                num_heads=num_heads, feedforward_dim=feedforward_dim, dropout=dropout, activation=activation)
        self.hidden_dim = hidden_dim
        self.predict_length = predict_length
        self.start_padding = nn.Parameter(torch.randn(size=(1, 1, self.hidden_dim)))
        self.next_padding = nn.Parameter(torch.randn(size=(1, self.predict_length, self.hidden_dim)))

    def forward(self, src, predict_length=None):
        if src is None and not self.training:
            src = self.start_padding
        if predict_length is not None:
            assert predict_length <= self.predict_length
            next_padding = self.next_padding[:, :predict_length, :]
        else:  # enough tokens to predict
            next_padding = self.next_padding
        batch_size = src.size(0)
        middle = self.stage1(src)
        middle = torch.cat([middle, next_padding.repeat(batch_size, 1, 1)], dim=1)
        out = self.stage2(middle)
        return out[:, -self.predict_length:, :]


if __name__ == '__main__':
    model = BiARTransformer(hidden_dim=1024,
                            stage1_layers=2,
                            stage2_layers=2,
                            predict_length=32).cuda()
    for key, value in model.named_parameters():
        print(key, value)
    src = torch.randn((2, 200, 1024)).cuda()
    out = model(src)
    print(out.shape)
