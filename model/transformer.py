import math
import torch
import torch.nn as nn

from encoder import Encoder
from decoder import Decoder

class Transformer(nn.Module):

    def __init__(self, src_pad_idx, trg_pad_idx, src_sos_idx, trg_sos_idx, enc_vocab_size, dec_vocab_size, d_model, n_head,
                max_length, ff_dim, n_layers, dropout):
        super(Transformer, self).__init__()


        self.src_pad_idx = src_pad_idx
        self.trg_pad_idx = trg_pad_idx
        self.src_sos_idx = src_sos_idx
        self.trg_sos_idx = trg_sos_idx

        self.encoder = Encoder(enc_vocab_size, d_model, ff_dim, max_length, n_head, n_layers, dropout)
        self.decoder = Decoder(dec_vocab_size, d_model, ff_dim, max_length, n_head, n_layers, dropout)


    
    def forward(self, source, target):

        src_mask = self.make_pad_mask(source, source)

        src_trg_mask = self.make_pad_mask(target, source)

        trg_mask = self.make_pad_mask(target, target) * \
                   self.make_no_peak_mask(target, target)

        enc_src = self.encoder(source, src_mask)
        output = self.decoder(target, enc_src, trg_mask, src_trg_mask)
        return output


    def make_pad_mask(self, q, k):

        len_q, len_k = q.size(1), k.size(1)

        k = k.ne(self.src_pad_idx).unsqueeze(1).unsqueeze(2)
        k = k.repeat(1, 1, len_q, 1)

        q = q.ne(self.src_pad_idx).unsqueeze(1).unsqueeze(3)
        q = q.repeat(1, 1, 1, len_k)

        mask = k & q

        return mask


    def make_no_peak_mask(self, q, k):

        len_q, len_k = q.size(1), k.size(1)

        mask = torch.tril(torch.ones(len_q, len_k)).type(torch.BoolTensor).to(self.device)

        return mask