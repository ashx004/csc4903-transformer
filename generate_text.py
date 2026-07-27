"""Generate text with the decoder-only TransformerLM: give it a prompt, it predicts
the next most likely token, appends it, and repeats — the actual autoregressive
next-token generation loop.

Usage
-----
    python generate_text.py --checkpoint checkpoints_lm/best.pt --prompt "The weather today is"
"""

from __future__ import annotations

import argparse

import torch

from data import _load_or_train_tokenizer, _load_iwslt, Vocab, EOS_IDX
from lm import TransformerLM


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--prompt", type=str, required=True)
    parser.add_argument("--max-new-tokens", type=int, default=50)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=40)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    train_args = ckpt["args"]

    # rebuild the same tokenizer used at training time (reuses the cached tok_en_lm_*.json file)
    train_pairs, _, _ = _load_iwslt()
    train_texts_en = [p[0] for p in train_pairs]
    if train_args.get("max_train_samples"):
        train_texts_en = train_texts_en[: train_args["max_train_samples"]]
    tok = _load_or_train_tokenizer(train_texts_en, "en_lm", train_args["vocab_size"])
    vocab = Vocab(tok)

    model = TransformerLM(
        vocab_size=vocab.size,
        d_model=train_args["d_model"],
        heads=train_args["heads"],
        d_ff=train_args["d_ff"],
        n_layers=train_args["n_layers"],
        max_len=train_args["block_size"] + 1,
        pad_idx=vocab.pad_idx,
    ).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    # tok.encode() auto-adds <sos>...<eos> via the trained post-processor, so strip
    # the trailing <eos> here -- we want the prompt to look "unfinished", not like a
    # completed sentence, otherwise the model treats it as already ended.
    raw_ids = tok.encode(args.prompt).ids
    if raw_ids and raw_ids[-1] == vocab.eos_idx:
        raw_ids = raw_ids[:-1]
    prompt_ids = raw_ids  # already starts with <sos> from the post-processor
    prompt_tensor = torch.tensor([prompt_ids], dtype=torch.long, device=device)

    out_ids = model.generate(
        prompt_tensor,
        max_new_tokens=args.max_new_tokens,
        eos_idx=EOS_IDX,
        temperature=args.temperature,
        top_k=args.top_k,
        device=device,
    )

    print("Raw generated ids:", out_ids[0].tolist())  # debug: see if it's just <eos>/<pad> repeating
    generated_text = vocab.decode(out_ids[0].tolist(), skip_special=True)
    print(f"Prompt: {args.prompt}")
    print(f"Generated: {generated_text}")


if __name__ == "__main__":
    main()