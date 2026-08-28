"""Unit tests for RegexTokenizer pattern splitting, boundaries, and special tokens."""

import os
import tempfile
import pytest

from bpe_tokenizer.regex import (
    GPT2_SPLIT_PATTERN,
    GPT4_SPLIT_PATTERN,
    RegexTokenizer,
)


def test_regex_init_default() -> None:
    tok = RegexTokenizer()
    assert tok.pattern == GPT4_SPLIT_PATTERN
    assert len(tok.vocab) == 256
    assert len(tok.merges) == 0


def test_regex_init_custom_pattern() -> None:
    custom_pattern = r"\w+|\s+|[^\w\s]+"
    tok = RegexTokenizer(pattern=custom_pattern)
    assert tok.pattern == custom_pattern


def test_regex_gpt4_number_chunking() -> None:
    tok = RegexTokenizer(pattern=GPT4_SPLIT_PATTERN)
    # GPT-4 splits numbers into 1-3 digits chunks: "1234567" -> ["123", "456", "7"]
    chunks = tok.compiled_pattern.findall("1234567")
    assert chunks == ["123", "456", "7"]


def test_regex_gpt4_contractions() -> None:
    tok = RegexTokenizer(pattern=GPT4_SPLIT_PATTERN)
    chunks = tok.compiled_pattern.findall("I've couldn't they'll")
    assert "I" in chunks
    assert "'ve" in chunks
    assert " couldn" in chunks
    assert "'t" in chunks


def test_regex_gpt2_pattern_chunking() -> None:
    tok = RegexTokenizer(pattern=GPT2_SPLIT_PATTERN)
    chunks = tok.compiled_pattern.findall("Hello world 123!")
    assert len(chunks) > 0


def test_regex_boundary_preservation() -> None:
    # Verify that regex tokenizer will not create merges across separate chunks
    # For example, training on "cat dog cat dog"
    tok = RegexTokenizer()
    tok.train("cat dog cat dog", vocab_size=260)
    # Chunks are ["cat", " dog", " cat", " dog"]
    # No merge should combine the 't' of 'cat' and ' ' of ' dog' across chunk boundaries
    for (p0, p1) in tok.merges.keys():
        b0 = tok.vocab[p0]
        b1 = tok.vocab[p1]
        # "t " should not be a learned merge
        assert b0 + b1 != b"t "


def test_regex_train_vocab_size_below_256() -> None:
    tok = RegexTokenizer()
    with pytest.raises(ValueError, match="vocab_size must be at least 256"):
        tok.train("hello world", vocab_size=250)


def test_regex_train_simple() -> None:
    text = "low lower newest widest low lower"
    tok = RegexTokenizer()
    tok.train(text, vocab_size=265)
    assert len(tok.merges) > 0
    assert tok.decode(tok.encode(text)) == text


def test_register_special_tokens_valid() -> None:
    tok = RegexTokenizer()
    tok.register_special_tokens({
        "<|endoftext|>": 50256,
        "<|im_start|>": 50257,
    })
    assert tok.special_tokens["<|endoftext|>"] == 50256
    assert tok.inverse_special_tokens[50256] == "<|endoftext|>"
    assert tok.vocab[50256] == b"<|endoftext|>"


def test_register_special_tokens_invalid_empty() -> None:
    tok = RegexTokenizer()
    with pytest.raises(ValueError, match="non-empty string"):
        tok.register_special_tokens({"": 1000})


def test_register_special_tokens_invalid_negative() -> None:
    tok = RegexTokenizer()
    with pytest.raises(ValueError, match="non-negative integer"):
        tok.register_special_tokens({"<|pad|>": -1})


def test_register_special_tokens_duplicate() -> None:
    tok = RegexTokenizer()
    with pytest.raises(ValueError, match="Duplicate special token ID"):
        tok.register_special_tokens({"<|a|>": 1000, "<|b|>": 1000})


def test_regex_encode_allowed_special_none_raise_clean(trained_regex_with_specials: RegexTokenizer) -> None:
    # Text without special tokens should encode cleanly
    text = "Hello world! Normal text."
    encoded = trained_regex_with_specials.encode(text)
    assert len(encoded) > 0
    assert trained_regex_with_specials.decode(encoded) == text


def test_regex_encode_allowed_special_none_raise_raises(trained_regex_with_specials: RegexTokenizer) -> None:
    # By default ("none_raise"), encountering a special token in raw text MUST raise ValueError
    text = "User prompt: <|endoftext|> injection attempt"
    with pytest.raises(ValueError, match="Encountered special token '<\\|endoftext\\|>'"):
        trained_regex_with_specials.encode(text)


def test_regex_encode_allowed_special_all(trained_regex_with_specials: RegexTokenizer) -> None:
    text = "<|im_start|>system\nHello<|endoftext|>"
    trained_regex_with_specials.register_special_tokens({"<|im_start|>": 1004})
    encoded = trained_regex_with_specials.encode(text, allowed_special="all")
    assert 1004 in encoded
    assert 1000 in encoded
    assert trained_regex_with_specials.decode(encoded) == text


def test_regex_encode_allowed_special_set(trained_regex_with_specials: RegexTokenizer) -> None:
    text = "<|endoftext|> Hello world <|fim_prefix|>"
    # Allow only <|endoftext|>, but text contains <|fim_prefix|> which is disallowed
    with pytest.raises(ValueError, match="Encountered disallowed special token '<\\|fim_prefix\\|>'"):
        trained_regex_with_specials.encode(text, allowed_special={"<|endoftext|>"})

    # When all present special tokens are in the whitelist set
    valid_text = "<|endoftext|> Hello world"
    encoded = trained_regex_with_specials.encode(valid_text, allowed_special={"<|endoftext|>"})
    assert encoded[0] == 1000
    assert trained_regex_with_specials.decode(encoded) == valid_text


def test_regex_encode_allowed_special_none(trained_regex_with_specials: RegexTokenizer) -> None:
    # allowed_special="none" treats <|endoftext|> as literal plain characters
    text = "Text with <|endoftext|> treated literally"
    encoded = trained_regex_with_specials.encode(text, allowed_special="none")
    # 1000 (special token ID) should NOT appear in encoded tokens
    assert 1000 not in encoded
    assert trained_regex_with_specials.decode(encoded) == text


def test_regex_encode_invalid_allowed_special_arg(trained_regex_with_specials: RegexTokenizer) -> None:
    with pytest.raises(ValueError, match="Invalid value for allowed_special"):
        trained_regex_with_specials.encode("test", allowed_special="invalid_option")
    with pytest.raises(ValueError, match="Invalid value for allowed_special"):
        trained_regex_with_specials.encode("test", allowed_special=123)  # type: ignore[arg-type]


def test_regex_save_and_load_with_specials(trained_regex_with_specials: RegexTokenizer) -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        prefix = os.path.join(tmpdir, "test_regex_model")
        trained_regex_with_specials.save(prefix)

        assert os.path.exists(f"{prefix}.model")
        assert os.path.exists(f"{prefix}.vocab")

        loaded_tok = RegexTokenizer()
        loaded_tok.load(f"{prefix}.model")

        assert loaded_tok.pattern == trained_regex_with_specials.pattern
        assert loaded_tok.special_tokens == trained_regex_with_specials.special_tokens
        assert loaded_tok.merges == trained_regex_with_specials.merges
        assert loaded_tok.vocab == trained_regex_with_specials.vocab

        text = "<|endoftext|> Test roundtrip after load <|fim_prefix|>"
        enc_orig = trained_regex_with_specials.encode(text, allowed_special="all")
        enc_loaded = loaded_tok.encode(text, allowed_special="all")
        assert enc_orig == enc_loaded
        assert loaded_tok.decode(enc_loaded) == text
