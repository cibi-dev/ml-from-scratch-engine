"""Text preprocessing and shingling module for MinHash deduplication."""

from __future__ import annotations

import re
import unicodedata

# Compiled regex patterns for linear-time scanning without nested quantifiers
_CONTROL_WHITESPACE_RE = re.compile(r"[\r\n\t\v\f]")
_COLLAPSE_WHITESPACE_RE = re.compile(r"\s+")
_WORD_TOKEN_RE = re.compile(r"\b\w+\b")


def normalize_text(text: str) -> str:
    """Normalize text for consistent MinHash shingling and deduplication.

    Steps applied:
    1. Unicode NFKC normalization (canonical decomposition followed by canonical composition).
    2. Lowercasing.
    3. Replacement of standard control whitespace (\\r, \\n, \\t, etc.) with space.
    4. Stripping of non-printable, control, format, surrogate, private-use characters
       (Unicode category 'C', including zero-width spaces and invisible characters).
    5. Collapsing multiple whitespace sequences into a single space and stripping edges.

    Args:
        text: Raw input string.

    Returns:
        Cleaned, normalized string.
    """
    if not text:
        return ""

    # Step 1: Unicode NFKC normalization
    text = unicodedata.normalize("NFKC", text)

    # Step 2: Lowercasing
    text = text.lower()

    # Step 3: Replace control whitespace with space
    text = _CONTROL_WHITESPACE_RE.sub(" ", text)

    # Step 4: Strip all Unicode Category 'C' (Other / Control / Format / Zero-Width / Surrogates)
    text = "".join(c for c in text if not unicodedata.category(c).startswith("C"))

    # Step 5: Collapse multiple spaces to single space and strip
    text = _COLLAPSE_WHITESPACE_RE.sub(" ", text).strip()

    return text


def get_shingles(text: str, k: int = 5, mode: str = "word") -> list[str]:
    """Generate k-gram shingles from normalized or raw text.

    Args:
        text: Input string.
        k: Shingle length (number of words or characters). Must be >= 1.
        mode: Shingling mode: 'word' for word-level k-grams or 'char' for character-level k-grams.

    Returns:
        List of k-gram shingles. If the document length is strictly between 0 and k,
        gracefully falls back to a single shingle representing the entire document.
        If the document is empty, returns an empty list.

    Raises:
        ValueError: If k < 1 or mode is not in ('word', 'char').
    """
    if k < 1:
        raise ValueError(f"k must be a positive integer >= 1, got {k}")

    if mode not in ("word", "char"):
        raise ValueError(f"Invalid mode '{mode}'. Must be 'word' or 'char'")

    if not text:
        return []

    if mode == "word":
        tokens = _WORD_TOKEN_RE.findall(text)
        num_tokens = len(tokens)
        if num_tokens == 0:
            return []
        if num_tokens < k:
            # Fallback for short documents (< k tokens)
            return [" ".join(tokens)]
        return [" ".join(tokens[i : i + k]) for i in range(num_tokens - k + 1)]

    # mode == "char"
    text_len = len(text)
    if text_len == 0:
        return []
    if text_len < k:
        # Fallback for short documents (< k chars)
        return [text]
    return [text[i : i + k] for i in range(text_len - k + 1)]
