import torch 
import torch.nn as nn

class TokenEmbeddings(nn.Module):

    def __init__(self, vocab_size, d_model):
        super(TokenEmbeddings, self).__init__()

        self.vocab_size = vocab_size
        self.d_model = d_model

        self.token_emb = nn.Embedding(vocab_size, d_model)


    
    def forward(self, inputs):

        token_embeddings = self.token_emb(inputs)

        return token_embeddings


