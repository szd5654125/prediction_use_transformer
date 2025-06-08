import torch
import torch.nn as nn
from torch import Tensor


class SineActivation(nn.Module):
    def __init__(self, in_features, periodic_features, out_features, dropout):
        super(SineActivation, self).__init__()
        # weights and biases for the periodic features
        # print('aaa', out_features, in_features, periodic_features)
        self.w0 = nn.parameter.Parameter(torch.randn(in_features, out_features - in_features - periodic_features))
        self.b0 = nn.parameter.Parameter(torch.randn(1, out_features - in_features - periodic_features))
        # weights and biases for the linear features
        self.w = nn.parameter.Parameter(torch.randn(in_features, periodic_features))
        self.b = nn.parameter.Parameter(torch.randn(1, periodic_features))
        self.activation = torch.sin
        self.dropout = nn.Dropout(dropout)

    def Time2Vector(self, data):
        """Add features to data:
            1. keep the original features numbered by - in_features
            2. add more periodic features numbered by - periodic_features
            3. add more linear feature to end up with total of features numbered by - out_features

        Args:
            data: Tensor, shape [N, batch_size, in_features]

        Returns:
            data: Tensor, shape [N, batch_size, out_features]
        """
        '''v_linear = torch.matmul(data, self.w0) + self.b0
        v_sin = self.activation(torch.matmul(self.w.t(), data.transpose(1, 2)).transpose(1, 2) + self.b)
        data = torch.cat([v_linear, v_sin, data], 2)'''
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
                 num_decoder_layers: int,
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

        '''self.transformer = nn.Transformer(d_model=hidden_dim,  # 当分类时为out_features
                                          nhead=nhead,
                                          num_encoder_layers=num_encoder_layers,
                                          num_decoder_layers=num_decoder_layers,
                                          dim_feedforward=dim_feedforward,
                                          dropout=dropout,
                                          activation=activation)'''
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=nhead, dim_feedforward=dim_feedforward,
            dropout=dropout, activation=activation, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_dim, nhead=nhead, dim_feedforward=dim_feedforward,
            dropout=dropout, activation=activation, batch_first=True
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_decoder_layers)
        # 用于回归任务
        # self.generator = nn.Linear(out_features, in_features)
        # 用于分类任务
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def encode(self, src: Tensor, src_mask: Tensor = None, src_padding_mask: Tensor = None):
        return self.encoder(self.sine_activation(src), mask=src_mask, src_key_padding_mask=src_padding_mask)

    def decode(self, tgt: Tensor, memory: Tensor, tgt_mask: Tensor = None, tgt_padding_mask: Tensor = None,
               memory_key_padding_mask: Tensor = None):
        return self.decoder(self.sine_activation(tgt), memory, tgt_mask=tgt_mask,
                            tgt_key_padding_mask=tgt_padding_mask,
                            memory_key_padding_mask=memory_key_padding_mask)

    def forward(self, src, tgt, src_mask=None, tgt_mask=None, mem_mask=None,
                src_padding_mask=None, tgt_padding_mask=None, memory_key_padding_mask=None):
        src_emb = self.sine_activation(src)
        tgt_emb = self.sine_activation(tgt)
        memory = self.encoder(src_emb, mask=src_mask, src_key_padding_mask=src_padding_mask)
        output = self.decoder(tgt_emb, memory,
                              tgt_mask=tgt_mask,
                              memory_mask=mem_mask,
                              tgt_key_padding_mask=tgt_padding_mask,
                              memory_key_padding_mask=memory_key_padding_mask)
        return self.classifier(output[:, -1, :])  # 因为用了 batch_first

    '''def encode(self, src: Tensor, src_mask: Tensor):
        return self.transformer.encoder(self.sine_activation(src), src_mask)

    def decode(self, tgt: Tensor, memory: Tensor, tgt_mask: Tensor):
        return self.transformer.decoder(self.sine_activation(tgt), memory, tgt_mask)

    def forward(self,
                src: Tensor,
                trg: Tensor,
                src_mask: Tensor = None,
                tgt_mask: Tensor = None,
                mem_mask: Tensor = None,
                src_padding_mask: Tensor = None,
                tgt_padding_mask: Tensor = None,
                memory_key_padding_mask: Tensor = None):
        src_emb = self.sine_activation(src)
        tgt_emb = self.sine_activation(trg)
        outs = self.transformer(src_emb, tgt_emb, src_mask, tgt_mask, mem_mask,
                                src_padding_mask, tgt_padding_mask, memory_key_padding_mask)
        # 用于回归任务
        # return self.generator(outs)
        final_output = outs[-1]
        return self.classifier(final_output)'''

