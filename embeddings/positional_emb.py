import torch
import torch.nn as nn



class PositionalEmbeddings(nn.Module):

    def __init__(self, max_length, d_model):
        super(PositionalEmbeddings, self).__init__()

        self.max_length = max_length
        self.d_model = d_model

        self.encoding = torch.zeros(max_length, d_model)
        self.encoding.requires_grad = False

        pos = torch.arange(0, max_length)
        pos = pos.float().unsqueeze(1)

        _2i = torch.arange(0, d_model, step=2).float()


        self.encoding[:, 0::2] = torch.sin( pos / 10000 ** (_2i/d_model))
        self.encoding[:, 1::2] = torch.cos( pos / 10000 ** (_2i/d_model))


    
    def forward(self, inputs):

        batch_size, seq_length = inputs.size()

        return self.encoding[:seq_length, :]


