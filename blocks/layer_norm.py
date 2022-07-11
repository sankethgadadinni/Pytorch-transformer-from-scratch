import torch
import torch.nn as nn


class LayerNorm(nn.Module):

    def __init__(self, d_model, eps = 1e-12):
        super(LayerNorm, self).__init__()

        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(d_model))
        self.beta = nn.Parameter(torch.zeros(d_model))

    
    def forward(self, inputs):

        mean = inputs.mean(-1, keepdim=True)
        std = inputs.std(-1, keepdim=True)

        out = (inputs - mean) / (std + self.eps)
        out = self.gamma * out + self.beta

        return out


