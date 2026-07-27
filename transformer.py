import torch
import torch.nn as nn
import math
import torch.nn.functional as F
from attention import MultiHeadAttention
from embeddings import TokenEmbedding, PositionalEncoding

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
    # src_key_padding_mask: (batch, src_len) bool, True at <pad> positions in the source sequence
    def forward(self, x, src_key_padding_mask=None):
        # perform mh attention
        sublayer1 = self.mh_attn(x, key_padding_mask=src_key_padding_mask)
        # add & norm between sublayers
        middle_layer = self.add_and_norm_mh_attn(x, sublayer1)
        # feed forward
        sublayer2 = self.ffn(middle_layer)
        # return the final add & norm out
        return self.add_and_norm_ffn(middle_layer, sublayer2)

class EncoderStack(nn.Module):
    def __init__(self, n=6, d_model=512, heads=8, d_ff=2048):
        super().__init__()
        # list to hold the encoderLayer
        self.layer_list = nn.ModuleList()
        for _ in range(n):
            layer = EncoderLayer(d_model=d_model, heads=heads)
            layer.ffn = FeedForwardNetwork(d_model, d_ff)
            self.layer_list.append(layer)

    def forward(self, x, src_key_padding_mask=None):
        cur_input = x
        for layer in self.layer_list:
            cur_input = layer(cur_input, src_key_padding_mask=src_key_padding_mask)
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
    # multihead attention with context from the final encoder layer output (skipped if no context given)
    # feed forward, return the final add and norm 
    # tgt_key_padding_mask: pad mask for the decoder's own input (masked self-attn)
    # memory_key_padding_mask: pad mask for the encoder output (cross-attn)
    # context=None turns this into a plain GPT-style causal decoder block (no cross-attention) —
    # used for standalone next-token language modeling instead of seq2seq translation.
    def forward(self, x, context=None, tgt_key_padding_mask=None, memory_key_padding_mask=None):
        sublayer1 = self.masked_mh_attn(x, key_padding_mask=tgt_key_padding_mask)
        middle_layer = self.add_and_norm_masked_mh_attn(x, sublayer1)
        if context is not None:
            sublayer2 = self.mh_attn(middle_layer, context, key_padding_mask=memory_key_padding_mask)
            middle_layer = self.add_and_norm_mh_attn(middle_layer, sublayer2)
        sublayer3 = self.ffn(middle_layer)
        return self.add_and_norm_ffn(middle_layer, sublayer3)
    
class DecoderStack(nn.Module):
    def __init__(self, n=6, d_model=512, heads=8, d_ff=2048):
        super().__init__()
        self.layer_list = nn.ModuleList()
        for _ in range(n):
            layer = DecoderLayer(d_model=d_model, heads=heads)
            layer.ffn = FeedForwardNetwork(d_model, d_ff)
            self.layer_list.append(layer)

    def forward(self, x, context=None, tgt_key_padding_mask=None, memory_key_padding_mask=None):
        cur_input = x
        for layer in self.layer_list:
            cur_input = layer(
                cur_input,
                context,
                tgt_key_padding_mask=tgt_key_padding_mask,
                memory_key_padding_mask=memory_key_padding_mask,
            )
        return cur_input


# ---------------------------------------------------------------------------
# Full seq2seq Transformer (Vaswani et al., "Attention Is All You Need")
# ---------------------------------------------------------------------------

class Transformer(nn.Module):
    """Ties embeddings + encoder stack + decoder stack + output projection together.

    This is what actually needs to exist to run the model end-to-end: encode a
    source sentence, decode a target sentence conditioned on it, and produce
    vocabulary logits at every target position.
    """

    def __init__(
        self,
        src_vocab_size,
        tgt_vocab_size,
        d_model=512,
        heads=8,
        d_ff=2048,
        n_enc_layers=6,
        n_dec_layers=6,
        max_len=5000,
        pad_idx=0,
        dropout=0.1,
        tie_weights=True,
    ):
        super().__init__()
        self.pad_idx = pad_idx
        self.d_model = d_model

        self.src_tok_emb = TokenEmbedding(src_vocab_size, d_model)
        self.tgt_tok_emb = TokenEmbedding(tgt_vocab_size, d_model)
        self.pos_enc = PositionalEncoding(d_model, max_len=max_len)
        self.emb_dropout = nn.Dropout(dropout)

        self.encoder = EncoderStack(n=n_enc_layers, d_model=d_model, heads=heads, d_ff=d_ff)
        self.decoder = DecoderStack(n=n_dec_layers, d_model=d_model, heads=heads, d_ff=d_ff)

        self.generator = nn.Linear(d_model, tgt_vocab_size, bias=False)

        # per section 3.4: share weights between the target embedding and the pre-softmax
        # linear transformation
        if tie_weights:
            self.generator.weight = self.tgt_tok_emb.embedding.weight

    def make_padding_mask(self, ids):
        # True at positions that are <pad> and therefore should never be attended to
        return ids == self.pad_idx

    def encode(self, src, src_key_padding_mask=None):
        x = self.emb_dropout(self.pos_enc(self.src_tok_emb(src)))
        return self.encoder(x, src_key_padding_mask=src_key_padding_mask)

    def decode(self, tgt, memory, tgt_key_padding_mask=None, memory_key_padding_mask=None):
        x = self.emb_dropout(self.pos_enc(self.tgt_tok_emb(tgt)))
        return self.decoder(
            x, memory,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=memory_key_padding_mask,
        )

    def forward(self, src, tgt):
        """Teacher-forced training forward pass. Returns logits (B, tgt_len, vocab)."""
        src_pad_mask = self.make_padding_mask(src)
        tgt_pad_mask = self.make_padding_mask(tgt)

        memory = self.encode(src, src_key_padding_mask=src_pad_mask)
        out = self.decode(
            tgt, memory,
            tgt_key_padding_mask=tgt_pad_mask,
            memory_key_padding_mask=src_pad_mask,
        )
        return self.generator(out)

    @torch.no_grad()
    def generate(self, src, sos_idx, eos_idx, max_len=128, device=None):
        """Greedy autoregressive decoding for a batch of source sentences.

        src: (B, src_len) token id tensor (already includes <sos>/<eos>, padded).
        Returns: (B, T) generated token ids, including <sos> but stopping (per
        sequence) once <eos> is produced.
        """
        self.eval()
        device = device or src.device
        src = src.to(device)
        batch_size = src.size(0)

        src_pad_mask = self.make_padding_mask(src)
        memory = self.encode(src, src_key_padding_mask=src_pad_mask)

        # start every sequence with <sos>
        ys = torch.full((batch_size, 1), sos_idx, dtype=torch.long, device=device)
        finished = torch.zeros(batch_size, dtype=torch.bool, device=device)

        for _ in range(max_len - 1):
            out = self.decode(ys, memory, memory_key_padding_mask=src_pad_mask)
            logits = self.generator(out[:, -1, :])  # (B, vocab) — only need the last position
            next_tok = logits.argmax(dim=-1)  # greedy

            # once a sequence has hit <eos>, keep padding it with <eos> so it doesn't change further
            next_tok = torch.where(finished, torch.full_like(next_tok, eos_idx), next_tok)
            ys = torch.cat([ys, next_tok.unsqueeze(1)], dim=1)

            finished = finished | (next_tok == eos_idx)
            if finished.all():
                break

        return ys