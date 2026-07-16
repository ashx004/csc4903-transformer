import torch
import torch.nn as nn
import math
import torch.nn.functional as F
from attention import MultiHeadAttention

class FeedForwardNetwork(nn.Module):
    def __init__(self, d_model=512, d_ff=2048):
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
    def __init__(self, d_model=512, heads=8, is_masked=False):
        super().__init__()
        # two sublayers: multihead attention, then feed forward network
        self.mh_attn = MultiHeadAttention(d_model, heads, is_masked) # d_model = 512, heads = 8, is_encoded = False
        self.ffn = FeedForwardNetwork()
        # layernorm objects have their own metaparameters that are adjusted based on the distributions involved
        self.attention_norm = nn.LayerNorm(d_model)
        self.ffn_norm = nn.LayerNorm(d_model)

    def add_and_norm_mh_attn(self, x, sublayer):
        return self.attention_norm(x + sublayer)
    
    def add_and_norm_ffn(self, x, sublayer):
        return self.ffn_norm(x + sublayer)

    # what does this need to do:
    # Layer 1: perform mh attention, then add & norm
    # Layer 2: perform ffn, then return the add & norm
    def forward(self, x):
        # perform mh attention
        sublayer1 = self.mh_attn(x)
        # add & norm between sublayers
        middle_layer = self.add_and_norm_mh_attn(x, sublayer1)
        # feed forward
        sublayer2 = self.ffn(middle_layer)
        # return the final add & norm out
        return self.add_and_norm_ffn(middle_layer, sublayer2)

class EncoderStack(nn.Module):
    def __init__(self, n=6):
        super().__init__()
        # list to hold the encoderLayer
        self.layer_list = nn.ModuleList()
        for _ in range(n):
            self.layer_list.append(EncoderLayer())

    def forward(self, x):
        cur_input = x
        for layer in self.layer_list:
            cur_input = layer(cur_input)
        return cur_input

class DecoderLayer(nn.Module):
    def __init__(self, d_model=512, heads=8):
        super().__init__()
        # two sublayers: multihead attention, then feed forward network
        self.masked_mh_attn = MultiHeadAttention(d_model, heads, True) # d_model = 512, heads = 8, is_masked = True
        self.mh_attn = MultiHeadAttention(d_model, heads, False) # normal multihead attention
        self.ffn = FeedForwardNetwork()
        self.masked_attention_norm = nn.LayerNorm(d_model)
        self.attention_norm = nn.LayerNorm(d_model)
        self.ffn_norm = nn.LayerNorm(d_model)

    def add_and_norm_masked_mh_attn(self, x, sublayer):
        return self.masked_attention_norm(x + sublayer)

    def add_and_norm_mh_attn(self, x, sublayer):
        return self.attention_norm(x + sublayer)
    
    def add_and_norm_ffn(self, x, sublayer):
        return self.ffn_norm(x + sublayer)

    # what does this need to do:
    # masked multihead attention, then add & norm
    # multihead attention with context from the final encoder layer output
    # feed forward, return the final add and norm 
    def forward(self, x, context):
        sublayer1 = self.masked_mh_attn(x)
        middle_layer = self.add_and_norm_masked_mh_attn(x, sublayer1)
        sublayer2 = self.mh_attn(middle_layer, context)
        middle_layer = self.add_and_norm_mh_attn(middle_layer, sublayer2)
        sublayer3 = self.ffn(middle_layer)
        return self.add_and_norm_ffn(middle_layer, sublayer3)
    
class DecoderStack(nn.Module):
    def __init__(self, n=6):
        super().__init__()
        self.layer_list = nn.ModuleList()
        for _ in range(n):
            self.layer_list.append(DecoderLayer())

    def forward(self, x, context):
        cur_input = x
        for layer in self.layer_list:
            cur_input = layer(cur_input, context)
        return cur_input
    
