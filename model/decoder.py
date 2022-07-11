import math
import torch
import torch.nn as nn


from embeddings.transformer_emb import TransformerEmbeddings
from layers.decoder_layer import DecoderLayer



class Decoder(nn.Module):

    def __init__(self, dec_vocab_size, d_model, ff_dim, max_length, n_head, n_layers, dropout):
        super(Decoder, self).__init__()

        self.emb = TransformerEmbeddings(dec_vocab_size, max_length, d_model, dropout)

        self.layers = nn.ModuleList([DecoderLayer(d_model, ff_dim, n_head, dropout) for _ in range(n_layers)])

        self.linear = nn.Linear(d_model, dec_vocab_size)

    
    def forward(self, target, encoder_output, target_mask, source_mask):

        target = self.emb(target)

        for layer in self.layers:
            target = layer(target)
        
        target = self.linear(target)

        return target

