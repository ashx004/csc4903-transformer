import torch
import torch.nn as nn
import math

# is_encoded tells us what layer we are in, True -> decoder layer, False -> encoder layer
def scaled_dot_product_attention(Q, K, V, is_encoded=True):
    d_k = Q.size(-1)
    # first mat mul block on left of diagram
    scores = (Q @ K.mT) / math.sqrt(d_k)

    # applies causal mask if we are in the decoder layer
    if is_encoded:
        seq_len = scores.size(-1)
        causal_mask = torch.triu(torch.ones(seq_len, seq_len, dtype=torch.bool), diagonal=1)
        scores = scores.masked_fill(causal_mask, -1e9)

    # we want to softmax the last position in the tensor (i.e, dim=-1 applies softmax across just the very last dimension)
    weights = torch.softmax(scores, dim=-1)
    return weights @ V

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, is_encoded=True):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_head = d_model // num_heads
        self.is_encoded = is_encoded

        # weight tensors (not allowing a bias, not in the original paper)
        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model,bias=False)
        self.W_o = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x):
        # grab dimensions needed for computation
        batch_size, seq_len, _ = x.size()

        # project Q, K, and V
        Q = self.W_q.forward(x)
        K = self.W_k.forward(x)
        V = self.W_v.forward(x)

        # split into heads
        Q = Q.view(batch_size, seq_len, self.num_heads, self.d_head).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.num_heads, self.d_head).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.num_heads, self.d_head).transpose(1, 2)

        # perform parallel scaled_dot_product_attention across all the heads that Q, K, and V have been split into
        output = scaled_dot_product_attention(Q, K, V, self.is_encoded)

        # concatenating all heads back into the original configuration before split
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        
        output = self.W_o.forward(output)

        return output
