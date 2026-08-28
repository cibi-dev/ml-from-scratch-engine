"""BasicTokenizer: Naive byte-level Byte Pair Encoding (BPE) on flat byte streams."""

from __future__ import annotations

from typing import Sequence

from bpe_tokenizer.base import Tokenizer, get_stats, merge


class BasicTokenizer(Tokenizer):
    """Naive byte-level BPE tokenizer operating directly on flat byte sequences.

    Does not split on regex boundaries or handle special tokens.
    """

    def __init__(self) -> None:
        super().__init__()

    def train(self, text: str, vocab_size: int, verbose: bool = False) -> None:
        """Train the BPE tokenizer on raw text.

        Args:
            text: Training corpus string.
            vocab_size: Desired total vocabulary size (must be >= 256).
            verbose: If True, prints merge progress.

        Raises:
            ValueError: If vocab_size < 256 or input exceeds MAX_INPUT_BYTES.
        """
        self._validate_input_bytes(text)
        if vocab_size < 256:
            raise ValueError(f"vocab_size must be at least 256, got {vocab_size}.")

        num_merges = vocab_size - 256
        text_bytes = text.encode("utf-8")
        ids: list[int] = list(text_bytes)

        merges: dict[tuple[int, int], int] = {}
        vocab: dict[int, bytes] = {idx: bytes([idx]) for idx in range(256)}

        for i in range(num_merges):
            stats = get_stats(ids)
            if not stats:
                if verbose:
                    print(f"Early stop at merge {i}: no more adjacent pairs found.")
                break
            # Find the most frequent pair; break ties by smallest pair tuple
            pair = max(stats, key=lambda p: (stats[p], -p[0], -p[1]))
            idx = 256 + i
            ids = merge(ids, pair, idx)
            merges[pair] = idx
            vocab[idx] = vocab[pair[0]] + vocab[pair[1]]
            if verbose:
                print(f"merge {i+1}/{num_merges}: {pair} -> {idx} ({stats[pair]} occurrences)")

        self.merges = merges
        self.vocab = vocab

    def encode(self, text: str) -> list[int]:
        """Encode string into list of token IDs using learned BPE merges.

        Args:
            text: Input string.

        Returns:
            List of integer token IDs.

        Raises:
            ValueError: If input exceeds MAX_INPUT_BYTES.
        """
        self._validate_input_bytes(text)
        text_bytes = text.encode("utf-8")
        ids: list[int] = list(text_bytes)

        while len(ids) >= 2:
            stats = get_stats(ids)
            # Find the pair in stats that was merged earliest in self.merges
            pair = min(stats, key=lambda p: self.merges.get(p, float("inf")))
            if pair not in self.merges:
                break
            idx = self.merges[pair]
            ids = merge(ids, pair, idx)

        return ids
