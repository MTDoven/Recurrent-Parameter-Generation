import torch
import torch.nn as nn
from attention import MHA


class FlashMultiHeadAttention(nn.MultiheadAttention):
    def __init__(self, d_model, nhead, dropout,
                 bias, batch_first, **kwargs):
        assert batch_first is True
        super().__init__(d_model, nhead, dropout, bias, batch_first, **kwargs)
        self.MHA = MHA(embed_dim=d_model, num_heads=nhead, dropout=dropout,
                       qkv_proj_bias=bias, out_proj_bias=bias)

    def forward(self, query, key, value, **kwargs):
        # assert query == key == value
        return self.MHA(query), None


class MaskFreeTransformerEncoder(nn.Module):
    def __init__(self, hidden_dim, num_layers,
                 num_heads, feedforward_dim, dropout, activation):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=num_heads,
                dim_feedforward=feedforward_dim, dropout=dropout, activation=activation,  batch_first=True)
        encoder_layer.self_attn = FlashMultiHeadAttention(d_model=hidden_dim, nhead=num_heads,
                dropout=dropout, bias=True, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer=encoder_layer, num_layers=num_layers,
                enable_nested_tensor=False)

    def forward(self, src):
        return self.encoder(src)


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
        self.padding = nn.Parameter(torch.randn(size=(1, self.predict_length, self.hidden_dim)))

    def forward(self, src):
        batch_size = src.size(0)
        middle = self.stage1(src)
        middle = torch.cat([middle, self.padding.repeat(batch_size, 1, 1)], dim=1)
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
