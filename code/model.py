import torch
import torch.nn as nn
from torch import Tensor


class SineActivation(nn.Module):
    def __init__(self, in_features, periodic_features, out_features, dropout):
        super(SineActivation, self).__init__()
        self.w0 = nn.parameter.Parameter(torch.randn(in_features, out_features - in_features - periodic_features))
        self.b0 = nn.parameter.Parameter(torch.randn(1, out_features - in_features - periodic_features))
        self.w = nn.parameter.Parameter(torch.randn(in_features, periodic_features))
        self.b = nn.parameter.Parameter(torch.randn(1, periodic_features))
        self.activation = torch.sin
        self.dropout = nn.Dropout(dropout)

    def Time2Vector(self, data):
        v_linear = torch.matmul(data, self.w0) + self.b0  # [B, T, D']
        v_sin = self.activation(torch.matmul(data, self.w) + self.b)  # [B, T, periodic_features]
        data = torch.cat([v_linear, v_sin, data], dim=2)  # [B, T, out_features]
        return data

    def forward(self, data):
        data = self.Time2Vector(data)
        data = self.dropout(data)
        return data


class BTC_Transformer(nn.Module):
    def __init__(self,
                 num_encoder_layers: int,
                 # num_decoder_layers: int,
                 in_features: int,
                 periodic_features: int,
                 # out_features: int,
                 hidden_dim: int,
                 nhead: int,
                 dim_feedforward: int = 512,
                 dropout: float = 0.1,
                 activation: str = 'relu',
                 num_classes: int = 2):
        super(BTC_Transformer, self).__init__()

        self.sine_activation = SineActivation(in_features=in_features,
                                              periodic_features=periodic_features,
                                              out_features=hidden_dim,  # 当分类时为out_features
                                              dropout=dropout)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=nhead, dim_feedforward=dim_feedforward,
            dropout=dropout, activation=activation, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)
        # 用于回归任务
        # self.generator = nn.Linear(out_features, in_features)
        # 用于分类任务
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def encode(self, src: Tensor, src_mask: Tensor = None, src_padding_mask: Tensor = None):
        return self.encoder(self.sine_activation(src), mask=src_mask, src_key_padding_mask=src_padding_mask)

    def forward(self, src, src_mask=None):
        src_emb = self.sine_activation(src)
        memory = self.encoder(src_emb, mask=src_mask)
        return self.classifier(memory[:, -1, :])

