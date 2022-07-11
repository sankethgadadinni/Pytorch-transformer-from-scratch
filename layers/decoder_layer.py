import math
import torch
import torch.nn as nn
from blocks.multihead_attn import MultiHeadAttention
from blocks.positionwise_ff import PositionWiseFeedForward
from blocks.layer_norm import LayerNorm



class DecoderLayer(nn.Module):

    def __init__(self, d_model, ff_dim, n_head, dropout):
        super(DecoderLayer, self).__init__()

        self.d_model = d_model
        self.ff_dim = ff_dim
        self.n_head = n_head
        self.dropout = dropout

        self.attention = MultiHeadAttention(d_model, n_head)
        self.layernorm1 = LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)

        self.enc_dec_attention = MultiHeadAttention(d_model, n_head)
        self.layernorm2 = LayerNorm(d_model)
        self.dropout2 = nn.Dropout(dropout)

        self.ff = PositionWiseFeedForward(d_model, ff_dim, dropout)
        self.layernorm3 = LayerNorm(d_model)
        self.dropout3 = nn.Dropout(dropout)


    def forward(self, decoder_inputs, encoder_outputs, target_mask, source_mask):

        _x = decoder_inputs

        x = self.attention(decoder_inputs, decoder_inputs, decoder_inputs, target_mask)
        x = self.layernorm1(_x + x)
        x = self.dropout1(x)

        if encoder_outputs is not None:

            _x = x
            x = self.attention(x, encoder_outputs, encoder_outputs, source_mask)
            x = self.layernorm2(_x + x)
            x = self.dropout2(x)

        
        _x = x
        x = self.ff(x)
        x = self.layernorm3(_x + x)
        x = self.dropout3(x)

        return x

