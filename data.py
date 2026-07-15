"""Data pipeline: download, tokenize, and batch IWSLT'14 English→German.

Downloads raw data directly from HuggingFace, trains BPE tokenizers,
and provides ready-to-use DataLoaders.

Usage
-----
    from data import build_dataloaders

    train_dl, val_dl, vocab_en, vocab_de = build_dataloaders(
        batch_size=256,        # sentences per batch
        max_len=128,
        vocab_size=32000,
    )

    for batch in train_dl:
        src = batch["src"]     # (B, src_len)  – English tokens
        tgt = batch["tgt"]     # (B, tgt_len)  – German tokens
        break
"""

from __future__ import annotations

import io
import re
import urllib.request
import zipfile
from pathlib import Path

import torch
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.processors import TemplateProcessing
from tokenizers.trainers import BpeTrainer
from torch.utils.data import DataLoader, Dataset

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SPECIAL_TOKENS = ["<pad>", "<unk>", "<sos>", "<eos>"]
PAD_IDX = 0
UNK_IDX = 1
SOS_IDX = 2
EOS_IDX = 3

IWSLT_URL = (
    "https://huggingface.co/datasets/IWSLT/iwslt2017"
    "/resolve/main/data/2017-01-trnted/texts/en/de/en-de.zip"
)

CACHE_DIR = Path(__file__).parent / "cache"


# ---------------------------------------------------------------------------
# Download + parse IWSLT'14
# ---------------------------------------------------------------------------

def _download_iwslt() -> zipfile.ZipFile:
    """Download IWSLT EN→DE zip and return a ZipFile handle."""
    cache_path = CACHE_DIR / "iwslt_en_de.zip"
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    if not cache_path.exists():
        print(f"Downloading IWSLT'14 EN-DE ({IWSLT_URL}) ...")
        urllib.request.urlretrieve(IWSLT_URL, cache_path)
    else:
        print("Using cached IWSLT zip.")

    return zipfile.ZipFile(cache_path)


def _parse_tagged_lines(zf: zipfile.ZipFile, en_name: str, de_name: str) -> list[tuple[str, str]]:
    """Parse train.tags.* files, stripping XML tags and blank lines."""
    en_raw = zf.read(en_name).decode("utf-8").split("\n")
    de_raw = zf.read(de_name).decode("utf-8").split("\n")

    pairs = []
    for en_l, de_l in zip(en_raw, de_raw):
        en_l = en_l.strip()
        de_l = de_l.strip()
        if not en_l or not de_l:
            continue
        if en_l.startswith("<") or de_l.startswith("<"):
            continue
        pairs.append((en_l, de_l))
    return pairs


def _parse_xml_segments(zf: zipfile.ZipFile, filename: str) -> list[str]:
    """Extract <seg> text from IWSLT XML files."""
    raw = zf.read(filename).decode("utf-8")
    return re.findall(r"<seg[^>]*>(.*?)</seg>", raw, re.DOTALL)


def _load_iwslt() -> tuple[list[tuple[str, str]], list[tuple[str, str]], list[tuple[str, str]]]:
    """Return (train, val, test) as lists of (en, de) string pairs."""
    zf = _download_iwslt()

    # Train
    train_pairs = _parse_tagged_lines(zf, "en-de/train.tags.en-de.en", "en-de/train.tags.en-de.de")

    # Validation (XML format)
    val_en = _parse_xml_segments(zf, "en-de/IWSLT17.TED.dev2010.en-de.en.xml")
    val_de = _parse_xml_segments(zf, "en-de/IWSLT17.TED.dev2010.en-de.de.xml")
    val_pairs = list(zip(val_en, val_de))

    # Test (combine all test years)
    test_pairs = []
    for year in [2010, 2011, 2012, 2013, 2014, 2015]:
        en_name = f"en-de/IWSLT17.TED.tst{year}.en-de.en.xml"
        de_name = f"en-de/IWSLT17.TED.tst{year}.en-de.de.xml"
        try:
            t_en = _parse_xml_segments(zf, en_name)
            t_de = _parse_xml_segments(zf, de_name)
            test_pairs.extend(zip(t_en, t_de))
        except KeyError:
            pass  # some years may not be in the archive

    zf.close()
    return train_pairs, val_pairs, test_pairs


# ---------------------------------------------------------------------------
# Vocabulary wrapper
# ---------------------------------------------------------------------------

class Vocab:
    """Thin wrapper around a BPE tokenizer for convenient encode/decode."""

    def __init__(self, tokenizer: Tokenizer) -> None:
        self.tok = tokenizer
        self.pad_idx = PAD_IDX
        self.unk_idx = UNK_IDX
        self.sos_idx = SOS_IDX
        self.eos_idx = EOS_IDX

    @property
    def size(self) -> int:
        return self.tok.get_vocab_size()

    def encode(self, text: str, add_special: bool = True) -> list[int]:
        """Encode a string to token ids.

        If *add_special* is True, prepends <sos> and appends <eos>.
        """
        ids = self.tok.encode(text).ids
        if add_special:
            ids = [SOS_IDX] + ids + [EOS_IDX]
        return ids

    def decode(self, ids: list[int], skip_special: bool = True) -> str:
        return self.tok.decode(ids, skip_special_tokens=skip_special)


# ---------------------------------------------------------------------------
# Tokenizer training
# ---------------------------------------------------------------------------

def _train_tokenizer(texts: list[str], vocab_size: int, path: Path) -> Tokenizer:
    """Train a BPE tokenizer on a list of strings and save to *path*."""
    tok = Tokenizer(BPE(unk_token="<unk>"))
    tok.pre_tokenizer = Whitespace()
    tok.post_processor = TemplateProcessing(
        single="<sos>:0 $A:0 <eos>:0",
        special_tokens=[("<sos>", SOS_IDX), ("<eos>", EOS_IDX)],
    )

    trainer = BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=SPECIAL_TOKENS,
        min_frequency=2,
    )
    tok.train_from_iterator(texts, trainer=trainer)
    tok.save(str(path))
    return tok


def _load_or_train_tokenizer(
    texts: list[str], lang: str, vocab_size: int,
) -> Tokenizer:
    path = CACHE_DIR / f"tok_{lang}_{vocab_size}.json"
    if path.exists():
        return Tokenizer.from_file(str(path))
    return _train_tokenizer(texts, vocab_size, path)


# ---------------------------------------------------------------------------
# Dataset + collation
# ---------------------------------------------------------------------------

class TranslationDataset(Dataset):
    """Pre-tokenized parallel sentence pairs."""

    def __init__(self, src_ids: list[list[int]], tgt_ids: list[list[int]]) -> None:
        assert len(src_ids) == len(tgt_ids)
        self.src_ids = src_ids
        self.tgt_ids = tgt_ids

    def __len__(self) -> int:
        return len(self.src_ids)

    def __getitem__(self, idx: int) -> dict[str, list[int]]:
        return {"src": self.src_ids[idx], "tgt": self.tgt_ids[idx]}


def _collate_fn(batch: list[dict[str, list[int]]]) -> dict[str, torch.Tensor]:
    """Pad variable-length sequences and stack into tensors."""
    max_src = max(len(b["src"]) for b in batch)
    max_tgt = max(len(b["tgt"]) for b in batch)

    src = torch.full((len(batch), max_src), PAD_IDX, dtype=torch.long)
    tgt = torch.full((len(batch), max_tgt), PAD_IDX, dtype=torch.long)

    for i, b in enumerate(batch):
        src[i, : len(b["src"])] = torch.tensor(b["src"], dtype=torch.long)
        tgt[i, : len(b["tgt"])] = torch.tensor(b["tgt"], dtype=torch.long)

    return {"src": src, "tgt": tgt}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_dataloaders(
    batch_size: int = 256,
    max_len: int = 128,
    vocab_size: int = 32000,
    num_workers: int = 0,
) -> tuple[DataLoader, DataLoader, Vocab, Vocab]:
    """Download IWSLT'14 EN→DE, tokenize, and return DataLoaders.

    Parameters
    ----------
    batch_size : int
        Number of sentences per batch.
    max_len : int
        Discard sentences longer than this (in tokens, including <sos>/<eos>).
    vocab_size : int
        BPE vocabulary size for each language.
    num_workers : int
        DataLoader workers.

    Returns
    -------
    train_dl, val_dl : DataLoader
        Iterating yields ``{"src": (B, S), "tgt": (B, T)}``.
    vocab_en, vocab_de : Vocab
        Source and target vocabulary wrappers.
    """
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    # ── 1. Load raw text ───────────────────────────────────────────────
    print("Loading IWSLT'14 EN-DE dataset...")
    train_pairs, val_pairs, test_pairs = _load_iwslt()
    print(f"  Raw train: {len(train_pairs)} pairs")
    print(f"  Raw val:   {len(val_pairs)} pairs")
    print(f"  Raw test:  {len(test_pairs)} pairs")

    train_texts_en = [p[0] for p in train_pairs]
    train_texts_de = [p[1] for p in train_pairs]

    # ── 2. Train tokenizers ────────────────────────────────────────────
    print(f"Training BPE tokenizers (vocab_size={vocab_size})...")
    tok_en = _load_or_train_tokenizer(train_texts_en, "en", vocab_size)
    tok_de = _load_or_train_tokenizer(train_texts_de, "de", vocab_size)
    vocab_en = Vocab(tok_en)
    vocab_de = Vocab(tok_de)
    print(f"  English vocab: {vocab_en.size} tokens")
    print(f"  German  vocab: {vocab_de.size} tokens")

    # ── 3. Tokenize ────────────────────────────────────────────────────
    def _tokenize_pair(
        src_text: str, tgt_text: str,
    ) -> tuple[list[int], list[int]] | None:
        src_ids = vocab_en.encode(src_text, add_special=True)
        tgt_ids = vocab_de.encode(tgt_text, add_special=True)
        if len(src_ids) > max_len or len(tgt_ids) > max_len:
            return None
        return src_ids, tgt_ids

    print(f"Tokenizing (max_len={max_len})...")
    train_src, train_tgt = [], []
    for s, t in train_pairs:
        pair = _tokenize_pair(s, t)
        if pair is not None:
            train_src.append(pair[0])
            train_tgt.append(pair[1])

    val_src, val_tgt = [], []
    for s, t in val_pairs:
        pair = _tokenize_pair(s, t)
        if pair is not None:
            val_src.append(pair[0])
            val_tgt.append(pair[1])

    print(f"  Train: {len(train_src)} sentence pairs")
    print(f"  Val:   {len(val_src)} sentence pairs")

    # ── 4. DataLoaders ─────────────────────────────────────────────────
    train_ds = TranslationDataset(train_src, train_tgt)
    val_ds = TranslationDataset(val_src, val_tgt)

    train_dl = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=_collate_fn,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )
    val_dl = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=_collate_fn,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_dl, val_dl, vocab_en, vocab_de


# ---------------------------------------------------------------------------
# Quick manual test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    train_dl, val_dl, vocab_en, vocab_de = build_dataloaders(
        batch_size=64,
        max_len=128,
        vocab_size=32000,
    )

    batch = next(iter(train_dl))
    print(f"\nBatch src shape: {batch['src'].shape}")
    print(f"Batch tgt shape: {batch['tgt'].shape}")
    print(f"Sample EN: {vocab_en.decode(batch['src'][0].tolist())}")
    print(f"Sample DE: {vocab_de.decode(batch['tgt'][0].tolist())}")
