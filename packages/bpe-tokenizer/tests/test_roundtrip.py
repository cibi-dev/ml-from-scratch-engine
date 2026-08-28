"""Roundtrip property tests: decode(encode(x)) == x across diverse text domains."""

import unicodedata
import pytest

from bpe_tokenizer.basic import BasicTokenizer
from bpe_tokenizer.regex import RegexTokenizer


@pytest.mark.parametrize("tok_fixture", ["trained_basic_tokenizer", "trained_regex_tokenizer"])
def test_roundtrip_english(request: pytest.FixtureRequest, tok_fixture: str, sample_english_text: str) -> None:
    tok = request.getfixturevalue(tok_fixture)
    encoded = tok.encode(sample_english_text)
    decoded = tok.decode(encoded)
    assert decoded == sample_english_text


@pytest.mark.parametrize("tok_fixture", ["trained_basic_tokenizer", "trained_regex_tokenizer"])
def test_roundtrip_code(request: pytest.FixtureRequest, tok_fixture: str, sample_code_text: str) -> None:
    tok = request.getfixturevalue(tok_fixture)
    encoded = tok.encode(sample_code_text)
    decoded = tok.decode(encoded)
    assert decoded == sample_code_text


@pytest.mark.parametrize("tok_fixture", ["trained_basic_tokenizer", "trained_regex_tokenizer"])
def test_roundtrip_cjk(request: pytest.FixtureRequest, tok_fixture: str, sample_cjk_text: str) -> None:
    tok = request.getfixturevalue(tok_fixture)
    encoded = tok.encode(sample_cjk_text)
    decoded = tok.decode(encoded)
    assert decoded == sample_cjk_text


@pytest.mark.parametrize("tok_fixture", ["trained_basic_tokenizer", "trained_regex_tokenizer"])
def test_roundtrip_emoji_zwj(request: pytest.FixtureRequest, tok_fixture: str, sample_emoji_text: str) -> None:
    tok = request.getfixturevalue(tok_fixture)
    encoded = tok.encode(sample_emoji_text)
    decoded = tok.decode(encoded)
    assert decoded == sample_emoji_text


@pytest.mark.parametrize("tok_fixture", ["trained_basic_tokenizer", "trained_regex_tokenizer"])
def test_roundtrip_accents_and_multilingual(
    request: pytest.FixtureRequest, tok_fixture: str, sample_multilingual_text: str
) -> None:
    tok = request.getfixturevalue(tok_fixture)
    encoded = tok.encode(sample_multilingual_text)
    decoded = tok.decode(encoded)
    assert decoded == sample_multilingual_text


@pytest.mark.parametrize("tok_fixture", ["trained_basic_tokenizer", "trained_regex_tokenizer"])
def test_roundtrip_whitespace(request: pytest.FixtureRequest, tok_fixture: str, sample_whitespace_text: str) -> None:
    tok = request.getfixturevalue(tok_fixture)
    encoded = tok.encode(sample_whitespace_text)
    decoded = tok.decode(encoded)
    assert decoded == sample_whitespace_text


@pytest.mark.parametrize("tok_fixture", ["trained_basic_tokenizer", "trained_regex_tokenizer"])
def test_roundtrip_symbols_and_math(request: pytest.FixtureRequest, tok_fixture: str) -> None:
    tok = request.getfixturevalue(tok_fixture)
    math_text = "∀x ∈ ℝ: ∑_{i=1}^n x_i ≤ √n ‖x‖₂ ∧ ∫₀^∞ e^{-x²} dx = √π / 2. Currency: 100€, ¥5000, £20, $50, ₿1.25"
    encoded = tok.encode(math_text)
    decoded = tok.decode(encoded)
    assert decoded == math_text


@pytest.mark.parametrize("tok_fixture", ["trained_basic_tokenizer", "trained_regex_tokenizer"])
def test_roundtrip_unicode_normalization_forms(request: pytest.FixtureRequest, tok_fixture: str) -> None:
    # NFC vs NFD forms must preserve byte-exact identity
    tok = request.getfixturevalue(tok_fixture)
    nfc = unicodedata.normalize("NFC", "Schönheit café naïve")
    nfd = unicodedata.normalize("NFD", "Schönheit café naïve")

    enc_nfc = tok.encode(nfc)
    assert tok.decode(enc_nfc) == nfc

    enc_nfd = tok.encode(nfd)
    assert tok.decode(enc_nfd) == nfd


@pytest.mark.parametrize("tok_fixture", ["trained_basic_tokenizer", "trained_regex_tokenizer"])
def test_roundtrip_boundary_byte_sequences(request: pytest.FixtureRequest, tok_fixture: str) -> None:
    tok = request.getfixturevalue(tok_fixture)
    # Characters spanning 1, 2, 3, and 4 UTF-8 bytes
    test_str = (
        "A"                      # 1-byte (0x41)
        + "\u00e9"              # 2-byte (0xc3 0xa9 - é)
        + "\u4e2d"              # 3-byte (0xe4 0xb8 0xad - 中)
        + "\U0001f600"          # 4-byte (0xf0 0x9f 0x98 0x80 - 😀)
    )
    encoded = tok.encode(test_str)
    assert tok.decode(encoded) == test_str


def test_roundtrip_comprehensive_synthetic_strings(
    trained_basic_tokenizer: BasicTokenizer, trained_regex_tokenizer: RegexTokenizer
) -> None:
    cases = [
        "",
        " ",
        "  ",
        "\n",
        "\r\n",
        "\t\t\t",
        "1234567890",
        "---===+++***///\\\\\\",
        "{\"key\": [1, 2, 3], \"nested\": {\"valid\": true}}",
        "<html><head><title>Test</title></head><body><h1>Hello</h1></body></html>",
        "SELECT * FROM users WHERE age > 18 AND status = 'active' ORDER BY created_at DESC;",
        "fn main() { println!(\"Hello, world!\"); }",
        "/* multiline\n * comment\n */",
    ]
    for case in cases:
        assert trained_basic_tokenizer.decode(trained_basic_tokenizer.encode(case)) == case
        assert trained_regex_tokenizer.decode(trained_regex_tokenizer.encode(case)) == case
