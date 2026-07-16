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

