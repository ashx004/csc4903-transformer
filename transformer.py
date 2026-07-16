import torch
import torch.nn as nn
import math
import torch.nn.functional as F
from attention import MultiHeadAttention

class FeedForwardNetwork(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        # W_1 + b_1
        self.linear1 = nn.Linear(d_model, d_ff)
        # W_2 + b_2
        self.linear2 = nn.Linear(d_ff, d_model)

    # formula from paper: FFN(x) = max(0, xW_1 + b_1)W_2 + b_2
    # the function nn.Linear performs a feed forward for us, with learned weights and biases
    def forward(self, x):
        return self.linear2(F.relu(self.linear1(x)))
    
class EncoderLayer(nn.Module):
    # default params are the defaults for the original paper's model
    def __init__(self, d_model=512, heads=8, is_encoded=False):
        super().__init__()
        self.d_model = d_model 
        self.heads = heads 
        self.is_encoded = is_encoded
        # two sublayers: multihead attention, then feed forward network
        self.mh_attn = MultiHeadAttention(d_model, heads, is_encoded) # d_model = 512, heads = 8, is_encoded = False
        self.ffn = FeedForwardNetwork()
        self.norm = nn.LayerNorm(d_model)

    def add_and_norm(self, x, sublayer):
        return self.norm(x + sublayer)

    # what does this need to do:
    # Layer 1: perform mh attention, then add & norm
    # Layer 2: perform ffn, then return the add & norm
    def forward(self, x):
        # perform mh attention
        sublayer1 = self.mh_attn(x)
        # add & norm between sublayers
        middle_layer = self.add_and_norm(x, sublayer1)
        # feed forward
        sublayer2 = self.ffn(middle_layer)
        # return the final add & norm out
        return self.add_and_norm(middle_layer, sublayer2)

class EncoderStack(nn.Module):
    def __init__(self, n=6):
        super().__init__()
        # list to hold the encoderLayer
        self.list = nn.ModuleList()
        for i in range(n):
            self.list.insert(EncoderLayer())
        

        


class DecoderLayer(nn.Module):
    def __init__(self):
        print("TODO")

class DecoderStack:
    def __init__(self):
        print("TODO")

