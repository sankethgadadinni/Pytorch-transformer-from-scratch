import torch
import torch.nn as nn


class PositionWiseFeedForward(nn.Module):

    def __init__(self, d_model, hidden_dim, dropout):
        super(PositionWiseFeedForward, self).__init__()

        self.d_model = d_model
        self.hidden_dim = hidden_dim

        self.fc1 = nn.Linear(d_model, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, d_model)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout)

    
    def forward(self, inputs):

        out = self.fc1(inputs)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.dropout(out)

        return out