import math
import torch
import torch.nn as nn
from blocks.multihead_attn import MultiHeadAttention
from blocks.positionwise_ff import PositionWiseFeedForward
from blocks.layer_norm import LayerNorm


class EncoderLayer(nn.Module):

    def __init__(self, d_model, ff_dim, n_head, dropout):
        super(EncoderLayer, self).__init__()


        self.d_model = d_model
        self.ff_dim = ff_dim
        self.n_head = n_head
        self.dropout = dropout

        self.attention = MultiHeadAttention(d_model, n_head)
        self.layernorm1 = LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)

        self.ff = PositionWiseFeedForward(d_model, ff_dim, dropout)
        self.layernorm2 = LayerNorm(d_model)
        self.dropout2 = nn.Dropout(dropout)

    
    def forward(self, inputs, source_mask):

        _x = inputs

        x = self.attention(inputs, inputs, inputs, source_mask)
        inputs = self.layernorm1(_x + x)
        x = self.dropout1(x)

        _x = x

        x = self.ff(x)
        x = self.layernorm2(_x + x)
        x = self.dropout2(x)

        return x


