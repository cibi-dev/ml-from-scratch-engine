"""RegexTokenizer: Regex-split Byte Pair Encoding (BPE) with special token support."""

from __future__ import annotations

from typing import Any, Sequence, Set

import regex

from bpe_tokenizer.base import Tokenizer, get_stats, merge

# Standard GPT-2 and GPT-4 regex patterns
GPT2_SPLIT_PATTERN: str = (
    r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
)
GPT4_SPLIT_PATTERN: str = (
    r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""
)


class RegexTokenizer(Tokenizer):
    """BPE Tokenizer using regex-based pre-tokenization and special token handling.

    Prevents merges from crossing semantic boundaries (e.g. words, numbers, punctuation)
    and securely manages control / special tokens.
    """

    def __init__(self, pattern: str | None = None) -> None:
        """Initialize RegexTokenizer with an optional regex split pattern.

        Args:
            pattern: Custom regex string. Defaults to GPT4_SPLIT_PATTERN.
        """
        super().__init__()
        self.pattern: str = pattern if pattern is not None else GPT4_SPLIT_PATTERN
        self.compiled_pattern: regex.Pattern[str] = regex.compile(self.pattern)

    def register_special_tokens(self, special_tokens: dict[str, int]) -> None:
        """Register special tokens and their assigned token IDs.

        Args:
            special_tokens: Mapping from special token string to integer token ID.

        Raises:
            ValueError: If token string is empty, ID is invalid/negative, or duplicate IDs.
        """
        seen_ids: dict[int, str] = dict(self.inverse_special_tokens)
        for token_str, token_id in special_tokens.items():
            if not isinstance(token_str, str) or not token_str:
                raise ValueError(f"Special token must be a non-empty string, got {token_str!r}.")
            if not isinstance(token_id, int) or token_id < 0:
                raise ValueError(f"Special token ID must be a non-negative integer, got {token_id}.")
            if token_id in seen_ids and seen_ids[token_id] != token_str:
                raise ValueError(
                    f"Duplicate special token ID: {token_id} is already assigned to '{seen_ids[token_id]}'."
                )
            seen_ids[token_id] = token_str

        for token_str, token_id in special_tokens.items():
            if token_str in self.special_tokens:
                old_id = self.special_tokens[token_str]
                if old_id != token_id:
                    self.inverse_special_tokens.pop(old_id, None)
                    self.vocab.pop(old_id, None)

            self.special_tokens[token_str] = token_id
            self.inverse_special_tokens[token_id] = token_str
            self.vocab[token_id] = token_str.encode("utf-8")

    def train(self, text: str, vocab_size: int, verbose: bool = False) -> None:
        """Train tokenizer on text respecting regex split boundaries.

        Args:
            text: Training corpus string.
            vocab_size: Target vocabulary size (must be >= 256).
            verbose: If True, prints merge progress.

        Raises:
            ValueError: If vocab_size < 256 or input exceeds MAX_INPUT_BYTES.
        """
        self._validate_input_bytes(text)
        if vocab_size < 256:
            raise ValueError(f"vocab_size must be at least 256, got {vocab_size}.")

        num_merges = vocab_size - 256
        # Split text into regex chunks
        text_chunks = self.compiled_pattern.findall(text)
        ids: list[list[int]] = [list(chunk.encode("utf-8")) for chunk in text_chunks]

        merges: dict[tuple[int, int], int] = {}
        vocab: dict[int, bytes] = {idx: bytes([idx]) for idx in range(256)}

        for i in range(num_merges):
            stats: dict[tuple[int, int], int] = {}
            for chunk_ids in ids:
                get_stats(chunk_ids, stats)
            if not stats:
                if verbose:
                    print(f"Early stop at merge {i}: no more adjacent pairs found.")
                break

            # Pick the most frequent pair; break ties by smallest pair tuple
            pair = max(stats, key=lambda p: (stats[p], -p[0], -p[1]))
            idx = 256 + i
            ids = [merge(chunk_ids, pair, idx) for chunk_ids in ids]
            merges[pair] = idx
            vocab[idx] = vocab[pair[0]] + vocab[pair[1]]
            if verbose:
                print(f"merge {i+1}/{num_merges}: {pair} -> {idx} ({stats[pair]} occurrences)")

        self.merges = merges
        self.vocab = vocab
        # Re-apply special tokens to vocab
        for token_str, token_id in self.special_tokens.items():
            self.vocab[token_id] = token_str.encode("utf-8")

    def _encode_chunk(self, chunk_ids: list[int]) -> list[int]:
        """Encode a single chunk of byte IDs using learned merges.

        Args:
            chunk_ids: List of byte IDs for one regex chunk.

        Returns:
            List of merged token IDs.
        """
        ids = list(chunk_ids)
        while len(ids) >= 2:
            stats = get_stats(ids)
            pair = min(stats, key=lambda p: self.merges.get(p, float("inf")))
            if pair not in self.merges:
                break
            idx = self.merges[pair]
            ids = merge(ids, pair, idx)
        return ids

    def encode(
        self,
        text: str,
        allowed_special: set[str] | str = "none_raise",
    ) -> list[int]:
        """Encode text into token IDs with secure special token handling.

        Args:
            text: Input string to tokenize.
            allowed_special: Control policy for special tokens:
                - "none_raise" (default): Raise ValueError if ANY registered special token
                  is present in text.
                - "none": Treat all special tokens as regular text bytes (do not match).
                - "all": Match and encode all registered special tokens as single IDs.
                - set[str]: Match only special tokens in the set. Raise ValueError if any
                  registered special token NOT in this set is present in text.

        Returns:
            List of integer token IDs.

        Raises:
            ValueError: If input exceeds MAX_INPUT_BYTES, an unallowed special token is found,
                        or allowed_special value is invalid.
        """
        self._validate_input_bytes(text)

        special_set: set[str]
        if allowed_special == "none_raise":
            for special in self.special_tokens:
                if special in text:
                    raise ValueError(
                        f"Encountered special token '{special}' in input text, but "
                        f"allowed_special is 'none_raise'. Pass allowed_special='none' to treat "
                        f"as normal text, or allowed_special='all'/set(...) to allow."
                    )
            special_set = set()
        elif allowed_special == "none":
            special_set = set()
        elif allowed_special == "all":
            special_set = set(self.special_tokens.keys())
        elif isinstance(allowed_special, set):
            for special in self.special_tokens:
                if special not in allowed_special and special in text:
                    raise ValueError(
                        f"Encountered disallowed special token '{special}' in input text. "
                        f"Allowed special tokens: {allowed_special}."
                    )
            special_set = set(self.special_tokens.keys()) & allowed_special
        else:
            raise ValueError(
                f"Invalid value for allowed_special: {allowed_special!r}. "
                f"Expected 'none_raise', 'none', 'all', or a set of strings."
            )

        if not special_set:
            # Fast path: no special tokens to match
            text_chunks = self.compiled_pattern.findall(text)
            ids: list[int] = []
            for chunk in text_chunks:
                chunk_bytes = list(chunk.encode("utf-8"))
                ids.extend(self._encode_chunk(chunk_bytes))
            return ids

        # Split text by special tokens (matching longest first to avoid substring conflicts)
        sorted_specials = sorted(special_set, key=len, reverse=True)
        special_pattern = "(" + "|".join(regex.escape(s) for s in sorted_specials) + ")"
        special_chunks = regex.split(special_pattern, text)

        ids = []
        for chunk in special_chunks:
            if not chunk:
                continue
            if chunk in self.special_tokens and chunk in special_set:
                ids.append(self.special_tokens[chunk])
            else:
                text_chunks = self.compiled_pattern.findall(chunk)
                for tc in text_chunks:
                    chunk_bytes = list(tc.encode("utf-8"))
                    ids.extend(self._encode_chunk(chunk_bytes))
        return ids

    def load(self, model_file: str) -> None:
        """Load tokenizer model and recompile regex pattern.

        Args:
            model_file: Path to .model file.
        """
        super().load(model_file)
        if self.pattern:
            self.compiled_pattern = regex.compile(self.pattern)
        else:
            self.pattern = GPT4_SPLIT_PATTERN
            self.compiled_pattern = regex.compile(self.pattern)
