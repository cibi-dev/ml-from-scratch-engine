"""Unit tests for BasicTokenizer and core BPE primitives."""

import os
import tempfile
import pytest

from bpe_tokenizer.base import get_stats, merge, render_token
from bpe_tokenizer.basic import BasicTokenizer


def test_get_stats_empty() -> None:
    assert get_stats([]) == {}
    assert get_stats([1]) == {}


def test_get_stats_basic() -> None:
    ids = [1, 2, 3, 1, 2]
    stats = get_stats(ids)
    assert stats[(1, 2)] == 2
    assert stats[(2, 3)] == 1
    assert stats[(3, 1)] == 1


def test_get_stats_with_existing_counts() -> None:
    existing = {(1, 2): 5}
    stats = get_stats([1, 2, 1, 2], counts=existing)
    assert stats[(1, 2)] == 7


def test_merge_basic() -> None:
    ids = [1, 2, 3, 1, 2, 4]
    merged = merge(ids, (1, 2), 99)
    assert merged == [99, 3, 99, 4]


def test_merge_overlapping() -> None:
    # [1, 1, 1] merged with (1, 1) -> [99, 1]
    ids = [1, 1, 1]
    merged = merge(ids, (1, 1), 99)
    assert merged == [99, 1]


def test_render_token_ascii() -> None:
    assert render_token(b"hello") == "hello"


def test_render_token_escaped_whitespace() -> None:
    assert render_token(b"\n\r\t") == "\\n\\r\\t"


def test_render_token_control_and_multibyte() -> None:
    rendered = render_token(b"\x00")
    assert "\\x00" in rendered
    # Valid utf-8 unicode
    assert render_token("ñ".encode("utf-8")) == "ñ"


def test_basic_tokenizer_init() -> None:
    tok = BasicTokenizer()
    assert len(tok.merges) == 0
    assert len(tok.vocab) == 256
    assert tok.pattern == ""
    assert tok.vocab[65] == b"A"


def test_basic_tokenizer_train_vocab_size_below_256() -> None:
    tok = BasicTokenizer()
    with pytest.raises(ValueError, match="vocab_size must be at least 256"):
        tok.train("hello world", vocab_size=255)


def test_basic_tokenizer_train_vocab_size_256() -> None:
    tok = BasicTokenizer()
    tok.train("hello world", vocab_size=256)
    assert len(tok.merges) == 0
    assert len(tok.vocab) == 256


def test_basic_tokenizer_train_simple() -> None:
    # "aaabdaaabac" -> bytes
    # 'aa' is the most frequent pair
    text = "aaabdaaabac"
    tok = BasicTokenizer()
    tok.train(text, vocab_size=258, verbose=True)
    assert len(tok.merges) == 2
    assert 256 in tok.vocab
    assert 257 in tok.vocab
    # Check that 256 is b'aa'
    assert tok.vocab[256] == b"aa"


def test_basic_tokenizer_early_stopping() -> None:
    tok = BasicTokenizer()
    # Short text with only 2 merges possible
    tok.train("ab", vocab_size=300)
    assert len(tok.merges) <= 1


def test_basic_tokenizer_encode_empty() -> None:
    tok = BasicTokenizer()
    assert tok.encode("") == []
    assert tok.decode([]) == ""


def test_basic_tokenizer_encode_single_byte() -> None:
    tok = BasicTokenizer()
    assert tok.encode("a") == [97]
    assert tok.decode([97]) == "a"


def test_basic_tokenizer_encode_merged() -> None:
    text = "abababab"
    tok = BasicTokenizer()
    tok.train(text, vocab_size=258)
    encoded = tok.encode("abab")
    assert all(isinstance(t, int) for t in encoded)
    assert len(encoded) < 4
    assert tok.decode(encoded) == "abab"


def test_basic_tokenizer_roundtrip_untrained(sample_english_text: str) -> None:
    tok = BasicTokenizer()
    encoded = tok.encode(sample_english_text)
    decoded = tok.decode(encoded)
    assert decoded == sample_english_text
    assert len(encoded) == len(sample_english_text.encode("utf-8"))


def test_basic_tokenizer_roundtrip_trained(trained_basic_tokenizer: BasicTokenizer, sample_english_text: str) -> None:
    encoded = trained_basic_tokenizer.encode(sample_english_text)
    decoded = trained_basic_tokenizer.decode(encoded)
    assert decoded == sample_english_text
    # Compression: tokens count should be less than raw bytes
    assert len(encoded) < len(sample_english_text.encode("utf-8"))


def test_basic_tokenizer_save_and_load(trained_basic_tokenizer: BasicTokenizer, sample_english_text: str) -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        prefix = os.path.join(tmpdir, "test_basic_model")
        trained_basic_tokenizer.save(prefix)

        assert os.path.exists(f"{prefix}.model")
        assert os.path.exists(f"{prefix}.vocab")

        # Load into fresh instance
        loaded_tok = BasicTokenizer()
        loaded_tok.load(f"{prefix}.model")

        assert loaded_tok.merges == trained_basic_tokenizer.merges
        assert loaded_tok.vocab == trained_basic_tokenizer.vocab

        enc_orig = trained_basic_tokenizer.encode(sample_english_text)
        enc_loaded = loaded_tok.encode(sample_english_text)
        assert enc_orig == enc_loaded
        assert loaded_tok.decode(enc_loaded) == sample_english_text
