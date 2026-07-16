import math
import torch
import torch.nn as nn

class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size, d_model):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.d_model = d_model 

    # per section 3.4 (Embeddings and Softmax)
    def forward(self, x):
        return self.embedding(x) * math.sqrt(self.d_model)
    
# this embeds positional meaning to the tokens for attention later, keeps tokens from being oblivious 
# of their position and context in the token stream
class PositionalEncoding(nn.Module):
    # PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
    # PE(pos, 2i + 1) = cos(pos / 10000^(2i/d_model))
    def __init__(self, d_model, max_len = 5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len).unsqueeze(1).float()
        div = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, : x.size(1)]