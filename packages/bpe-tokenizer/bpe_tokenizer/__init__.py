"""High-performance, secure Byte Pair Encoding (BPE) tokenizer in Python."""

from bpe_tokenizer.base import (
    MAX_INPUT_BYTES,
    Tokenizer,
    get_stats,
    merge,
    render_token,
)
from bpe_tokenizer.basic import BasicTokenizer
from bpe_tokenizer.regex import (
    GPT2_SPLIT_PATTERN,
    GPT4_SPLIT_PATTERN,
    RegexTokenizer,
)

__all__ = [
    "Tokenizer",
    "BasicTokenizer",
    "RegexTokenizer",
    "GPT2_SPLIT_PATTERN",
    "GPT4_SPLIT_PATTERN",
    "MAX_INPUT_BYTES",
    "get_stats",
    "merge",
    "render_token",
]
