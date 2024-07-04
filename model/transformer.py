import torch
import torch.nn as nn


class MaskFreeTransformerEncoder(nn.Module):
    def __init__(self, hidden_dim, num_layers,
                 num_heads, feedforward_dim, dropout, activation):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=num_heads,
                dim_feedforward=feedforward_dim, dropout=dropout, activation=activation,  batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer=encoder_layer, num_layers=num_layers)

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
        self.padding_number = torch.tensor(1/hidden_dim)

    def forward(self, src):
        batch_size = src.size(0)
        middle = self.stage1(src)
        padding = torch.full(size=(batch_size, self.predict_length, self.hidden_dim), fill_value=self.padding_number,
                             dtype=src.dtype, device=src.device)
        middle = torch.cat([middle, padding], dim=1)
        out = self.stage2(middle)
        return out[:, -self.predict_length:, :]


if __name__ == '__main__':
    model = BiARTransformer(hidden_dim=1024,
                            stage1_layers=2,
                            stage2_layers=2,
                            predict_length=32)
    src = torch.randn((2, 200, 1024))
    out = model(src)
    print(out.shape)
