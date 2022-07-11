import torch 
import torch.nn as nn
from token_emb import TokenEmbeddings
from positional_emb import PositionalEmbeddings


class TransformerEmbeddings(nn.Module):

    def __init__(self, vocab_size, max_length, d_model, drop_prob):
        super(TransformerEmbeddings, self).__init__()

        self.vocab_size = vocab_size
        self.max_length = max_length
        self.d_model = d_model

        self.tok_emb = TokenEmbeddings(vocab_size, d_model)
        self.pos_emb = PositionalEmbeddings(max_length, d_model)

        self.drop_out = nn.Dropout(p=drop_prob)


    def forward(self, inputs):

        token_embeddings = self.tok_emb(inputs)
        positional_embeddings = self.pos_emb(inputs)

        transformer_embeddings = token_embeddings + positional_embeddings

        return self.drop_out(transformer_embeddings)