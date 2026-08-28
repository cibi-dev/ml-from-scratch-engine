"""Base Tokenizer class and core BPE algorithm utilities."""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from typing import Sequence

# Maximum input size allowed for training or encoding (10 MB)
MAX_INPUT_BYTES: int = 10 * 1024 * 1024


def get_stats(
    ids: Sequence[int],
    counts: dict[tuple[int, int], int] | None = None,
) -> dict[tuple[int, int], int]:
    """Count consecutive pairs in a sequence of token IDs.

    Args:
        ids: Sequence of integer token IDs.
        counts: Optional existing counts dictionary to update in-place.

    Returns:
        Dictionary mapping (p0, p1) pairs to their occurrence counts.
    """
    stats = counts if counts is not None else {}
    n = len(ids)
    if n < 2:
        return stats
    for i in range(n - 1):
        pair = (ids[i], ids[i + 1])
        stats[pair] = stats.get(pair, 0) + 1
    return stats


def merge(ids: Sequence[int], pair: tuple[int, int], idx: int) -> list[int]:
    """Merge occurrences of pair (p0, p1) in ids into a new token ID idx.

    Args:
        ids: Sequence of integer token IDs.
        pair: Tuple of two integer token IDs to replace.
        idx: New integer token ID.

    Returns:
        A new list with pair replaced by idx.
    """
    new_ids: list[int] = []
    i = 0
    n = len(ids)
    p0, p1 = pair
    while i < n:
        if i < n - 1 and ids[i] == p0 and ids[i + 1] == p1:
            new_ids.append(idx)
            i += 2
        else:
            new_ids.append(ids[i])
            i += 1
    return new_ids


def render_token(t: bytes) -> str:
    """Safely render raw bytes into a printable, human-readable string.

    Args:
        t: Byte sequence representing a token.

    Returns:
        Printable string representation with control/non-printable bytes escaped.
    """
    s = t.decode("utf-8", errors="replace")
    out: list[str] = []
    for c in s:
        if c.isprintable() and c not in ("\n", "\r", "\t"):
            out.append(c)
        elif c == "\n":
            out.append("\\n")
        elif c == "\r":
            out.append("\\r")
        elif c == "\t":
            out.append("\\t")
        else:
            code = ord(c)
            out.append(f"\\x{code:02x}" if code < 256 else f"\\u{code:04x}")
    return "".join(out)


class Tokenizer(ABC):
    """Abstract base class for Byte Pair Encoding (BPE) tokenizers."""

    def __init__(self) -> None:
        # (p0, p1) -> new_token_id
        self.merges: dict[tuple[int, int], int] = {}
        # regex pattern used for chunking (empty for BasicTokenizer)
        self.pattern: str = ""
        # special_token_string -> special_token_id
        self.special_tokens: dict[str, int] = {}
        # special_token_id -> special_token_string
        self.inverse_special_tokens: dict[int, str] = {}
        # token_id -> byte sequence
        self.vocab: dict[int, bytes] = self._build_vocab()

    def _validate_input_bytes(self, text: str) -> bytes:
        """Validate input text size against MAX_INPUT_BYTES.

        Args:
            text: Input string.

        Returns:
            UTF-8 encoded bytes of text.

        Raises:
            ValueError: If input length exceeds MAX_INPUT_BYTES.
        """
        raw_bytes = text.encode("utf-8")
        if len(raw_bytes) > MAX_INPUT_BYTES:
            raise ValueError(
                f"Input size ({len(raw_bytes)} bytes) exceeds MAX_INPUT_BYTES limit "
                f"({MAX_INPUT_BYTES} bytes / 10MB)."
            )
        return raw_bytes

    def _build_vocab(self) -> dict[int, bytes]:
        """Reconstruct vocabulary mapping from merges and special tokens.

        Returns:
            Dictionary mapping token ID to bytes.
        """
        vocab: dict[int, bytes] = {idx: bytes([idx]) for idx in range(256)}
        # Sort merges by token_id to construct sequentially
        sorted_merges = sorted(self.merges.items(), key=lambda item: item[1])
        for (p0, p1), idx in sorted_merges:
            if p0 in vocab and p1 in vocab:
                vocab[idx] = vocab[p0] + vocab[p1]
            else:
                raise ValueError(
                    f"Invalid merge rule ({p0}, {p1}) -> {idx}: components not in vocab."
                )
        for special_str, special_id in self.special_tokens.items():
            vocab[special_id] = special_str.encode("utf-8")
        return vocab

    @abstractmethod
    def train(self, text: str, vocab_size: int, verbose: bool = False) -> None:
        """Train tokenizer on text up to target vocabulary size.

        Args:
            text: Training corpus string.
            vocab_size: Target vocabulary size (must be >= 256).
            verbose: If True, prints training progress.
        """
        pass

    @abstractmethod
    def encode(self, text: str) -> list[int]:
        """Encode text into a list of token IDs.

        Args:
            text: Input string to tokenize.

        Returns:
            List of integer token IDs.
        """
        pass

    def decode(self, ids: list[int]) -> str:
        """Decode a list of token IDs into a string.

        Args:
            ids: List of integer token IDs.

        Returns:
            Decoded string using UTF-8 with replacement for malformed sequences.

        Raises:
            TypeError: If any token ID is not an integer.
            ValueError: If any token ID is negative or not found in vocabulary.
        """
        part_bytes: list[bytes] = []
        for token_id in ids:
            if not isinstance(token_id, int):
                raise TypeError(
                    f"Token ID must be an integer, got {type(token_id).__name__}."
                )
            if token_id < 0:
                raise ValueError(f"Negative token ID {token_id} is invalid.")
            if token_id in self.vocab:
                part_bytes.append(self.vocab[token_id])
            elif token_id in self.inverse_special_tokens:
                part_bytes.append(self.inverse_special_tokens[token_id].encode("utf-8"))
            else:
                raise ValueError(
                    f"Token ID {token_id} is out of bounds or unknown (vocab size: {len(self.vocab)})."
                )
        raw_bytes = b"".join(part_bytes)
        return raw_bytes.decode("utf-8", errors="replace")

    def save(self, file_prefix: str) -> None:
        """Save model merges and readable vocab to files.

        Writes:
            <file_prefix>.model - Model definition (pattern, special tokens, merges).
            <file_prefix>.vocab - Human-readable vocabulary reference.

        Args:
            file_prefix: Path prefix for output files.
        """
        model_file = f"{file_prefix}.model"
        with open(model_file, "w", encoding="utf-8") as f:
            f.write("bpe_tokenizer v1\n")
            f.write(f"pattern {json.dumps(self.pattern)}\n")
            f.write(f"special_tokens_count {len(self.special_tokens)}\n")
            for special_str, special_id in sorted(self.special_tokens.items(), key=lambda x: x[1]):
                f.write(f"{json.dumps(special_str)} {special_id}\n")
            f.write(f"merges_count {len(self.merges)}\n")
            for (p0, p1), idx in sorted(self.merges.items(), key=lambda x: x[1]):
                f.write(f"{p0} {p1} {idx}\n")

        vocab_file = f"{file_prefix}.vocab"
        with open(vocab_file, "w", encoding="utf-8") as f:
            for idx, token_bytes in sorted(self.vocab.items(), key=lambda x: x[0]):
                rendered = render_token(token_bytes)
                is_special = " (special)" if idx in self.inverse_special_tokens else ""
                f.write(f"[{idx}] {token_bytes!r} -> {rendered}{is_special}\n")

    def load(self, model_file: str) -> None:
        """Load tokenizer model from .model file.

        Args:
            model_file: Path to .model file.

        Raises:
            ValueError: If file format is invalid or corrupted.
        """
        if not model_file.endswith(".model"):
            model_file = f"{model_file}.model"

        merges: dict[tuple[int, int], int] = {}
        special_tokens: dict[str, int] = {}
        pattern: str = ""

        try:
            with open(model_file, "r", encoding="utf-8") as f:
                header = f.readline().strip()
                if header != "bpe_tokenizer v1":
                    raise ValueError(f"Unsupported model header format: '{header}'")

                # Pattern line
                pattern_line = f.readline().strip()
                if not pattern_line.startswith("pattern "):
                    raise ValueError("Malformed model file: missing pattern line.")
                pattern = json.loads(pattern_line[len("pattern ") :])

                # Special tokens count line
                special_count_line = f.readline().strip()
                if not special_count_line.startswith("special_tokens_count "):
                    raise ValueError("Malformed model file: missing special_tokens_count line.")
                num_special = int(special_count_line[len("special_tokens_count ") :])

                for _ in range(num_special):
                    line = f.readline().strip()
                    if not line:
                        raise ValueError("Unexpected EOF while reading special tokens.")
                    parts = line.rsplit(" ", 1)
                    if len(parts) != 2:
                        raise ValueError(f"Malformed special token line: {line}")
                    s_str = json.loads(parts[0])
                    s_id = int(parts[1])
                    special_tokens[s_str] = s_id

                # Merges count line
                merges_count_line = f.readline().strip()
                if not merges_count_line.startswith("merges_count "):
                    raise ValueError("Malformed model file: missing merges_count line.")
                num_merges = int(merges_count_line[len("merges_count ") :])

                for _ in range(num_merges):
                    line = f.readline().strip()
                    if not line:
                        raise ValueError("Unexpected EOF while reading merges.")
                    parts = line.split()
                    if len(parts) != 3:
                        raise ValueError(f"Malformed merge line: {line}")
                    p0, p1, idx = int(parts[0]), int(parts[1]), int(parts[2])
                    merges[(p0, p1)] = idx

        except (json.JSONDecodeError, ValueError) as e:
            raise ValueError(f"Failed to load model file '{model_file}': {e}") from e

        self.pattern = pattern
        self.merges = merges
        self.special_tokens = special_tokens
        self.inverse_special_tokens = {v: k for k, v in special_tokens.items()}
        self.vocab = self._build_vocab()
