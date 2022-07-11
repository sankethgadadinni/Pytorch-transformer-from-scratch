import math
import torch
import torch.nn as nn

from embeddings.transformer_emb import TransformerEmbeddings
from layers.encoder_layer import EncoderLayer



class Encoder(nn.Module):

    def __init__(self, enc_vocab_size, d_model, ff_dim, max_length, n_head, n_layers, dropout):
        super(Encoder, self).__init__()

        self.emb = TransformerEmbeddings(enc_vocab_size, max_length, d_model, dropout)

        self.layers = nn.ModuleList([EncoderLayer(d_model, ff_dim, n_head, dropout)for _ in range(n_layers)])


    def forward(self, inputs, source_mask):

        x = self.emb(inputs)

        for layer in self.layers:
            x = layer(x, source_mask)
        
        return x