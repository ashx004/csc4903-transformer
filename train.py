"""Training loop for the from-scratch Transformer (Vaswani et al.).

Usage
-----
    python train.py --epochs 10 --batch-size 128

This wires together data.py (dataloaders + vocabs) and transformer.py
(the Transformer model) with:
  - label-smoothed cross-entropy loss (section 5.4 of the paper)
  - the Noam warmup/decay learning rate schedule (section 5.3, eq. 3)
  - teacher forcing (decoder input = tgt[:, :-1], target = tgt[:, 1:])
  - checkpointing so you can stop/resume and later run generate.py
"""

from __future__ import annotations

import argparse
import math
import time
from pathlib import Path

import torch
import torch.nn as nn

from data import build_dataloaders, PAD_IDX
from transformer import Transformer

CKPT_DIR = Path(__file__).parent / "checkpoints"


class NoamScheduler:
    """LR schedule from section 5.3: lr = d_model^-0.5 * min(step^-0.5, step*warmup^-1.5)."""

    def __init__(self, optimizer, d_model, warmup_steps=4000):
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
        src = batch["src"].to(device)
        tgt = batch["tgt"].to(device)

        # teacher forcing: shift target right by one
        tgt_in = tgt[:, :-1]
        tgt_out = tgt[:, 1:]

        logits = model(src, tgt_in)  # (B, T-1, vocab)
        loss = criterion(logits.reshape(-1, logits.size(-1)), tgt_out.reshape(-1))

        scheduler.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        lr = scheduler.step()

        n_tokens = (tgt_out != PAD_IDX).sum().item()
        total_loss += loss.item() * n_tokens
        total_tokens += n_tokens

        if i % log_every == 0:
            elapsed = time.time() - start
            print(
                f"  step {i:5d} | loss {loss.item():.4f} | lr {lr:.2e} | "
                f"{elapsed:.1f}s elapsed"
            )

    return total_loss / max(total_tokens, 1)


@torch.no_grad()
def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss, total_tokens = 0.0, 0
    for batch in dataloader:
        src = batch["src"].to(device)
        tgt = batch["tgt"].to(device)
        tgt_in = tgt[:, :-1]
        tgt_out = tgt[:, 1:]
        logits = model(src, tgt_in)
        loss = criterion(logits.reshape(-1, logits.size(-1)), tgt_out.reshape(-1))
        n_tokens = (tgt_out != PAD_IDX).sum().item()
        total_loss += loss.item() * n_tokens
        total_tokens += n_tokens
    return total_loss / max(total_tokens, 1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--max-len", type=int, default=128)
    parser.add_argument("--vocab-size", type=int, default=32000)
    parser.add_argument("--d-model", type=int, default=512)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--d-ff", type=int, default=2048)
    parser.add_argument("--n-layers", type=int, default=6)
    parser.add_argument("--warmup-steps", type=int, default=4000)
    parser.add_argument("--label-smoothing", type=float, default=0.1)
    parser.add_argument("--resume", type=str, default=None, help="path to checkpoint to resume from")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_dl, val_dl, vocab_en, vocab_de = build_dataloaders(
        batch_size=args.batch_size,
        max_len=args.max_len,
        vocab_size=args.vocab_size,
    )

    model = Transformer(
        src_vocab_size=vocab_en.size,
        tgt_vocab_size=vocab_de.size,
        d_model=args.d_model,
        heads=args.heads,
        d_ff=args.d_ff,
        n_enc_layers=args.n_layers,
        n_dec_layers=args.n_layers,
        max_len=args.max_len,
        pad_idx=PAD_IDX,
    ).to(device)

    # label smoothing per section 5.4; ignore_index makes pad positions contribute zero loss
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX, label_smoothing=args.label_smoothing)

    # betas/eps per section 5.3; lr is overwritten every step by the scheduler
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0, betas=(0.9, 0.98), eps=1e-9)
    scheduler = NoamScheduler(optimizer, d_model=args.d_model, warmup_steps=args.warmup_steps)

    start_epoch = 0
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
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