"""Train the decoder-only TransformerLM for next-token prediction on English text.

Reuses IWSLT'14's English sentences (ignoring the German side entirely) as a
monolingual text corpus. Sentences are tokenized, concatenated into one long
stream separated by <eos>, and chopped into fixed-length blocks — the standard
way to train a GPT-style LM.

Usage
-----
    python train_lm.py --epochs 5 --max-train-samples 50000
"""

from __future__ import annotations

import argparse
import math
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from data import _load_iwslt, _load_or_train_tokenizer, Vocab, PAD_IDX, EOS_IDX, CACHE_DIR
from lm import TransformerLM

CKPT_DIR = Path(__file__).parent / "checkpoints_lm"


class LMBlockDataset(Dataset):
    """Chops a long token stream into fixed-length (input, target) blocks for next-token prediction."""

    def __init__(self, token_stream, block_size):
        self.block_size = block_size
        # only keep full blocks of size block_size + 1 (need one extra token for the target shift)
        n_blocks = (len(token_stream) - 1) // block_size
        self.data = torch.tensor(token_stream[: n_blocks * block_size + 1], dtype=torch.long)
        self.n_blocks = n_blocks

    def __len__(self):
        return self.n_blocks

    def __getitem__(self, idx):
        start = idx * self.block_size
        chunk = self.data[start: start + self.block_size + 1]
        return {"input": chunk[:-1], "target": chunk[1:]}


def build_lm_data(vocab_size, block_size, max_train_samples=None):
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading IWSLT'14 (English side only, for monolingual LM training)...")
    train_pairs, val_pairs, _ = _load_iwslt()
    train_texts_en = [p[0] for p in train_pairs]
    val_texts_en = [p[0] for p in val_pairs]

    if max_train_samples is not None:
        train_texts_en = train_texts_en[:max_train_samples]

    print(f"Training BPE tokenizer (vocab_size={vocab_size}) on {len(train_texts_en)} English sentences...")
    tok = _load_or_train_tokenizer(train_texts_en, "en_lm", vocab_size)
    vocab = Vocab(tok)
    print(f"  Vocab size: {vocab.size}")

    def to_stream(texts):
        stream = []
        for t in texts:
            stream.extend(vocab.encode(t, add_special=True))  # <sos> ... <eos> per sentence
        return stream

    print("Building token streams...")
    train_stream = to_stream(train_texts_en)
    val_stream = to_stream(val_texts_en)
    print(f"  Train tokens: {len(train_stream)}")
    print(f"  Val tokens:   {len(val_stream)}")

    train_ds = LMBlockDataset(train_stream, block_size)
    val_ds = LMBlockDataset(val_stream, block_size)
    print(f"  Train blocks: {len(train_ds)}")
    print(f"  Val blocks:   {len(val_ds)}")

    return train_ds, val_ds, vocab


class NoamScheduler:
    def __init__(self, optimizer, d_model, warmup_steps=2000):
        self.optimizer = optimizer
        self.d_model = d_model
        self.warmup_steps = warmup_steps
        self.step_num = 0

    def step(self):
        self.step_num += 1
        lr = (self.d_model ** -0.5) * min(
            self.step_num ** -0.5, self.step_num * self.warmup_steps ** -1.5
        )
        for group in self.optimizer.param_groups:
            group["lr"] = lr
        self.optimizer.step()
        return lr


def train_one_epoch(model, dataloader, criterion, scheduler, device, log_every=100):
    model.train()
    total_loss, total_tokens = 0.0, 0
    start = time.time()

    for i, batch in enumerate(dataloader):
        x = batch["input"].to(device)
        y = batch["target"].to(device)

        logits = model(x)  # (B, T, vocab)
        loss = criterion(logits.reshape(-1, logits.size(-1)), y.reshape(-1))

        scheduler.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        lr = scheduler.step()

        n_tokens = y.numel()
        total_loss += loss.item() * n_tokens
        total_tokens += n_tokens

        if i % log_every == 0:
            elapsed = time.time() - start
            print(f"  step {i:5d} | loss {loss.item():.4f} | lr {lr:.2e} | {elapsed:.1f}s elapsed")

    return total_loss / max(total_tokens, 1)


@torch.no_grad()
def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss, total_tokens = 0.0, 0
    for batch in dataloader:
        x = batch["input"].to(device)
        y = batch["target"].to(device)
        logits = model(x)
        loss = criterion(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
        n_tokens = y.numel()
        total_loss += loss.item() * n_tokens
        total_tokens += n_tokens
    return total_loss / max(total_tokens, 1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--block-size", type=int, default=128, help="sequence length per training example")
    parser.add_argument("--vocab-size", type=int, default=8000)
    parser.add_argument("--d-model", type=int, default=256)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--d-ff", type=int, default=1024)
    parser.add_argument("--n-layers", type=int, default=4)
    parser.add_argument("--warmup-steps", type=int, default=2000)
    parser.add_argument("--label-smoothing", type=float, default=0.1)
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--resume", type=str, default=None)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_ds, val_ds, vocab = build_lm_data(
        vocab_size=args.vocab_size,
        block_size=args.block_size,
        max_train_samples=args.max_train_samples,
    )

    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=True)
    val_dl = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    model = TransformerLM(
        vocab_size=vocab.size,
        d_model=args.d_model,
        heads=args.heads,
        d_ff=args.d_ff,
        n_layers=args.n_layers,
        max_len=args.block_size + 1,
        pad_idx=PAD_IDX,
    ).to(device)

    criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX, label_smoothing=args.label_smoothing)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0, betas=(0.9, 0.98), eps=1e-9)
    scheduler = NoamScheduler(optimizer, d_model=args.d_model, warmup_steps=args.warmup_steps)

    start_epoch = 0
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.step_num = ckpt["step_num"]
        start_epoch = ckpt["epoch"] + 1
        print(f"Resumed from {args.resume} at epoch {start_epoch}")

    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    best_val_loss = float("inf")

    for epoch in range(start_epoch, args.epochs):
        print(f"\nEpoch {epoch + 1}/{args.epochs}")
        train_loss = train_one_epoch(model, train_dl, criterion, scheduler, device)
        val_loss = evaluate(model, val_dl, criterion, device)
        print(
            f"  train_loss={train_loss:.4f} (ppl={math.exp(train_loss):.2f}) | "
            f"val_loss={val_loss:.4f} (ppl={math.exp(val_loss):.2f})"
        )

        ckpt = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "step_num": scheduler.step_num,
            "epoch": epoch,
            "args": vars(args),
        }
        torch.save(ckpt, CKPT_DIR / "last.pt")
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(ckpt, CKPT_DIR / "best.pt")
            print(f"  saved new best checkpoint (val_loss={val_loss:.4f})")


if __name__ == "__main__":
    main()