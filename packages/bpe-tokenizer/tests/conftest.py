"""Shared test fixtures and sample data for bpe_tokenizer tests."""

import pytest

from bpe_tokenizer.basic import BasicTokenizer
from bpe_tokenizer.regex import RegexTokenizer


@pytest.fixture
def sample_english_text() -> str:
    return (
        "The quick brown fox jumps over the lazy dog. "
        "Byte Pair Encoding is an algorithm originally designed for data compression "
        "that was later adapted for neural network tokenization."
    )


@pytest.fixture
def sample_code_text() -> str:
    return '''def calculate_bpe(text: str, vocab_size: int = 512) -> dict[tuple[int, int], int]:
    """Calculate BPE merges for given text."""
    ids = list(text.encode("utf-8"))
    for i in range(vocab_size - 256):
        stats = get_stats(ids)
        if not stats:
            break
        pair = max(stats, key=stats.get)
        ids = merge(ids, pair, 256 + i)
    return {"length": len(ids)}
'''


@pytest.fixture
def sample_cjk_text() -> str:
    return (
        "你好，世界！这是一段中文测试文本。"
        "こんにちは世界！これは日本語のテストです。"
        "안녕하세요 세계! 이것은 한국어 테스트입니다."
    )


@pytest.fixture
def sample_emoji_text() -> str:
    return (
        "Hello 🌍! Family: 👨‍👩‍👧‍👦, Developer: 👩‍💻, Pride: 🏳️‍🌈, "
        "Handshake: 🧑🏽‍🤝‍🧑🏻, Superhero: 🦸‍♂️, Rockets: 🚀✨🎉"
    )


@pytest.fixture
def sample_multilingual_text() -> str:
    return (
        "El veloz murciélago hindú comía feliz cardillo y kiwi. "
        "Portez ce vieux whisky au juge blond qui fume. "
        "Zwölf Boxkämpfer jagen Viktor quer über den großen Sylter Deich. "
        "Съешь же ещё этих мягких французских булок, да выпей чаю."
    )


@pytest.fixture
def sample_whitespace_text() -> str:
    return "  \t\n  Hello   world!  \r\n\r\n\t  Line 2 with trailing spaces   \n\n"


@pytest.fixture
def training_corpus() -> str:
    return (
        "Byte pair encoding (also known as BPE) is a subword tokenization algorithm. "
        "It was originally described in 1994 as a data compression algorithm. "
        "In modern Large Language Models (LLMs) such as GPT-2, GPT-4, and Llama, "
        "BPE is used to convert strings into sequences of token identifiers. "
        "The algorithm iteratively finds the most frequent pair of adjacent tokens "
        "and merges them into a new token."
    )


@pytest.fixture
def trained_basic_tokenizer(training_corpus: str) -> BasicTokenizer:
    tok = BasicTokenizer()
    tok.train(training_corpus, vocab_size=300)
    return tok


@pytest.fixture
def trained_regex_tokenizer(training_corpus: str) -> RegexTokenizer:
    tok = RegexTokenizer()
    tok.train(training_corpus, vocab_size=300)
    return tok


@pytest.fixture
def trained_regex_with_specials(training_corpus: str) -> RegexTokenizer:
    tok = RegexTokenizer()
    tok.register_special_tokens({
        "<|endoftext|>": 1000,
        "<|fim_prefix|>": 1001,
        "<|fim_middle|>": 1002,
        "<|fim_suffix|>": 1003,
    })
    tok.train(training_corpus, vocab_size=300)
    return tok
