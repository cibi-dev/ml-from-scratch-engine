"""Character-level tokenizer and dataset utilities for Nano-Transformer."""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import torch

# Standard 65-character vocabulary from the Tiny Shakespeare dataset
DEFAULT_TINY_SHAKESPEARE_VOCAB = (
    "\n !$&',-.3:;?ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
)

TINY_SHAKESPEARE_SAMPLE = """First Citizen:
Before we proceed any further, hear me speak.

All:
Speak, speak.

First Citizen:
You are all resolved rather to die than to famish?

All:
Resolved. resolved.

First Citizen:
First, you know Caius Marcius is chief enemy to the people.

All:
We know't, we know't.

First Citizen:
Let us kill him, and we'll have corn at our own price.
Is't a verdict?

All:
No more talking on't; let it be done: away, away!

Second Citizen:
One word, good citizens.

First Citizen:
We are accounted poor citizens, the patricians good.
What authority surfeits on would relieve us: if they
would yield us but the superfluity, while it were
wholesome, we might guess they relieved us humanely;
but they think we are too dear: the leanness that
afflicts us, the object of our misery, is as an
inventory to particularise their abundance; our
sufferance is a gain to them. Let us revenge this with
our pikes, ere we become rakes: for the gods know I
speak this in hunger for bread, not in thirst for revenge.

Second Citizen:
Would you proceed especially against Caius Marcius?

All:
Against him first: he's a very dog to the commonalty.

Second Citizen:
Consider you what services he has done for his country?

First Citizen:
Very well; and could be content to give him good
report for't, but that he pays himself with being proud.

Second Citizen:
Nay, but speak not maliciously.

First Citizen:
I say unto you, what he hath done famously, he did
it to that end: though soft-conscienced men can be
content to say it was for his country, he did it to
please his mother and to be partly proud; which he
is, even to the altitude of his virtue.
"""


class CharTokenizer:
    """Character-level tokenizer with strict vocabulary encoding and decoding validation."""

    def __init__(self, chars: Optional[Sequence[str] | str] = None) -> None:
        if chars is None:
            chars_list = sorted(list(set(DEFAULT_TINY_SHAKESPEARE_VOCAB)))
        else:
            chars_list = sorted(list(set(chars)))

        self.chars: List[str] = chars_list
        self.vocab_size: int = len(self.chars)
        self.stoi = {ch: i for i, ch in enumerate(self.chars)}
        self.itos = {i: ch for i, ch in enumerate(self.chars)}

    def encode(self, text: str) -> List[int]:
        """Encode a string into a list of token integers.

        Args:
            text: Input string to encode.

        Returns:
            List of integer token IDs.

        Raises:
            ValueError: If any character in text is not in vocabulary.
        """
        tokens: List[int] = []
        for char in text:
            if char not in self.stoi:
                raise ValueError(f"Unrecognized character in input text: {char!r}")
            tokens.append(self.stoi[char])
        return tokens

    def decode(self, tokens: Sequence[int]) -> str:
        """Decode a sequence of token integers back into a string.

        Args:
            tokens: Sequence of integer token IDs.

        Returns:
            Decoded string.

        Raises:
            ValueError: If any token ID is out of vocabulary bounds [0, vocab_size).
        """
        chars: List[str] = []
        for t in tokens:
            if t not in self.itos:
                raise ValueError(
                    f"Token ID {t} out of range [0, {self.vocab_size})"
                )
            chars.append(self.itos[t])
        return "".join(chars)


def get_tiny_shakespeare_data() -> str:
    """Return synthetic/sample Tiny Shakespeare text."""
    return TINY_SHAKESPEARE_SAMPLE


def create_data_splits(
    text: str,
    tokenizer: Optional[CharTokenizer] = None,
    train_ratio: float = 0.9,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Encode text and split into train and validation 1D tensors.

    Args:
        text: Raw text string.
        tokenizer: Optional CharTokenizer instance (default is 65-char vocab).
        train_ratio: Fraction of data for training (0 < train_ratio < 1).

    Returns:
        Tuple of (train_tensor, val_tensor) of type torch.long.
    """
    if not (0.0 < train_ratio < 1.0):
        raise ValueError(f"train_ratio must be in (0, 1), got {train_ratio}")

    tok = tokenizer if tokenizer is not None else CharTokenizer()
    encoded = tok.encode(text)
    data = torch.tensor(encoded, dtype=torch.long)

    n = int(train_ratio * len(data))
    train_data = data[:n]
    val_data = data[n:]
    return train_data, val_data


def get_batch(
    data: torch.Tensor,
    batch_size: int,
    block_size: int,
    device: str = "cpu",
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Generate a small batch of inputs x and targets y from a 1D dataset tensor.

    Args:
        data: 1D long tensor of token IDs.
        batch_size: Number of sequences in batch (B).
        block_size: Context length of each sequence (T).
        device: Target device ('cpu' or 'cuda').

    Returns:
        Tuple of tensors (x, y) both of shape (batch_size, block_size).

    Raises:
        ValueError: If data tensor length is not strictly greater than block_size.
    """
    if data.dim() != 1:
        raise ValueError(f"Dataset tensor must be 1D, got {data.dim()}D shape {data.shape}")
    if len(data) <= block_size:
        raise ValueError(
            f"Data length ({len(data)}) must be strictly greater than block_size ({block_size})"
        )
    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {batch_size}")

    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i : i + block_size] for i in ix])
    y = torch.stack([data[i + 1 : i + 1 + block_size] for i in ix])

    if device != "cpu":
        x, y = x.to(device), y.to(device)
    return x, y
