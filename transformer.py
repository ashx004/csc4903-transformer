import torch
import torch.nn as nn
import math
import torch.nn.functional as F

class FeedForwardNetwork(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        # W_1
        self.linear1 = nn.Linear(d_model, d_ff)
        # W_2
        self.linear2 = nn.Linear(d_ff, d_model)

    # formula from paper: FFN(x) = max(0, xW_1 + b_1)W_2 + b_2
    # the function nn.Linear performs a feed forward for us, with learned weights and biases
    def forward(self, x):
        return self.linear2(F.relu(self.linear1(x)))

class EncoderLayer(nn.Module):
    def __init__(self):
        print("TODO")

class DecoderLayer(nn.Module):
    def __init__(self):
        print("TODO")

