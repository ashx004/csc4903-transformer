"""Decoder-only (GPT-style) language model for next-token prediction.

This reuses the SAME DecoderStack/DecoderLayer from transformer.py used for
seq2seq translation -- just called with context=None, which skips the
cross-attention sublayer entirely (see transformer.py's DecoderLayer.forward).
So this is not a separate architecture: it's your existing decoder stack,
run standalone instead of conditioned on an encoder.
"""

import torch
import torch.nn as nn

from embeddings import TokenEmbedding, PositionalEncoding
from transformer import DecoderStack


class TransformerLM(nn.Module):
    """GPT-style decoder-only language model, built from the existing DecoderStack."""

    def __init__(
        self,
        vocab_size,
        d_model=256,
        heads=8,
        d_ff=1024,
        n_layers=6,
        max_len=512,
        pad_idx=0,
        dropout=0.1,
        tie_weights=True,
    ):
        super().__init__()
        self.pad_idx = pad_idx
        self.tok_emb = TokenEmbedding(vocab_size, d_model)
        self.pos_enc = PositionalEncoding(d_model, max_len=max_len)
        self.emb_dropout = nn.Dropout(dropout)

        # reuse the same decoder stack as the translation model; context=None below
        # means the cross-attention sublayer in each DecoderLayer is skipped
        self.decoder = DecoderStack(n=n_layers, d_model=d_model, heads=heads, d_ff=d_ff)

        self.generator = nn.Linear(d_model, vocab_size, bias=False)
        if tie_weights:
            self.generator.weight = self.tok_emb.embedding.weight

    def forward(self, x):
        """x: (B, T) token ids. Returns logits (B, T, vocab) -- logits[:, i] predicts token i+1."""
        pad_mask = x == self.pad_idx
        h = self.emb_dropout(self.pos_enc(self.tok_emb(x)))
        # context=None -> DecoderLayer skips cross-attention, runs as a plain causal decoder
        h = self.decoder(h, context=None, tgt_key_padding_mask=pad_mask)
        return self.generator(h)

    @torch.no_grad()
    def generate(self, prompt_ids, max_new_tokens=50, eos_idx=None, temperature=1.0, top_k=None, device=None):
        """Autoregressive next-token generation, one token at a time.

        prompt_ids: (B, T) token ids to condition on.
        temperature: >1.0 = more random, <1.0 = more greedy/confident, 1.0 = unmodified.
        top_k: if set, sample only among the top_k most likely next tokens each step.
        Returns: (B, T + generated) token ids.
        """
        self.eval()
        device = device or prompt_ids.device
        ys = prompt_ids.to(device)

        for _ in range(max_new_tokens):
            logits = self.forward(ys)[:, -1, :]  # (B, vocab) -- only need the last position
            logits = logits / max(temperature, 1e-5)

            if top_k is not None:
                topk_vals, topk_idx = torch.topk(logits, top_k, dim=-1)
                probs = torch.zeros_like(logits).scatter_(-1, topk_idx, torch.softmax(topk_vals, dim=-1))
            else:
                probs = torch.softmax(logits, dim=-1)

            next_tok = torch.multinomial(probs, num_samples=1)  # (B, 1) -- sample, not pure argmax
            ys = torch.cat([ys, next_tok], dim=1)

            if eos_idx is not None and (next_tok == eos_idx).all():
                break

        return ys