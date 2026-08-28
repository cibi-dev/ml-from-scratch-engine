"""Tests for text preprocessing, normalization, and shingle extraction."""

from __future__ import annotations

import pytest

from minhash_dedup.preprocessing import get_shingles, normalize_text


class TestNormalizeText:
    def test_nfkc_normalization_ligatures(self) -> None:
        raw = "The ﬂoor is clean with ﬁne craft."
        normalized = normalize_text(raw)
        assert "floor" in normalized
        assert "fine" in normalized

    def test_nfkc_normalization_fullwidth(self) -> None:
        raw = "Ｈｅｌｌｏ Ｗｏｒｌｄ"
        normalized = normalize_text(raw)
        assert normalized == "hello world"

    def test_lowercase_conversion(self) -> None:
        raw = "Machine Learning & Natural Language Processing"
        normalized = normalize_text(raw)
        assert normalized == "machine learning & natural language processing"

    def test_whitespace_collapsing(self) -> None:
        raw = "  Line 1  \n\n\t  Line 2   \r\n  Line 3  "
        normalized = normalize_text(raw)
        assert normalized == "line 1 line 2 line 3"

    def test_strip_control_characters(self) -> None:
        raw = "Hello\x00\x07World\x1bTest"
        normalized = normalize_text(raw)
        assert normalized == "helloworldtest"

    def test_strip_zero_width_and_invisible_chars(self) -> None:
        # Zero-width space (\u200b), ZWNJ (\u200c), ZWJ (\u200d), BOM (\ufeff), soft hyphen (\u00ad), LTR/RTL (\u200e, \u200f)
        raw = "d\u200bu\u200cp\u200dl\ufeffi\u00adc\u200ea\u200ft\u200ee"
        normalized = normalize_text(raw)
        assert normalized == "duplicate"

    def test_empty_and_whitespace_only(self) -> None:
        assert normalize_text("") == ""
        assert normalize_text("   \t\n\r   ") == ""
        assert normalize_text("\u200b\u200c\ufeff") == ""

    def test_unicode_accents_and_multilingual(self) -> None:
        raw = "Curaçao and São Paulo résumé"
        normalized = normalize_text(raw)
        assert normalized == "curaçao and são paulo résumé"


class TestGetShingles:
    def test_word_shingles_standard(self) -> None:
        text = "the quick brown fox jumps over the lazy dog"
        shingles = get_shingles(text, k=3, mode="word")
        assert len(shingles) == 7
        assert shingles[0] == "the quick brown"
        assert shingles[1] == "quick brown fox"
        assert shingles[-1] == "the lazy dog"

    def test_word_shingles_k_equals_1(self) -> None:
        text = "apple banana orange"
        shingles = get_shingles(text, k=1, mode="word")
        assert shingles == ["apple", "banana", "orange"]

    def test_word_shingles_short_text_fallback(self) -> None:
        text = "hello world"
        shingles = get_shingles(text, k=5, mode="word")
        # Text has 2 words, k=5 -> fallback to single shingle of all words
        assert shingles == ["hello world"]

    def test_word_shingles_exact_k_tokens(self) -> None:
        text = "one two three"
        shingles = get_shingles(text, k=3, mode="word")
        assert shingles == ["one two three"]

    def test_char_shingles_standard(self) -> None:
        text = "abcdef"
        shingles = get_shingles(text, k=3, mode="char")
        assert shingles == ["abc", "bcd", "cde", "def"]

    def test_char_shingles_short_text_fallback(self) -> None:
        text = "abc"
        shingles = get_shingles(text, k=5, mode="char")
        assert shingles == ["abc"]

    def test_empty_text_returns_empty_list(self) -> None:
        assert get_shingles("", k=3, mode="word") == []
        assert get_shingles("", k=3, mode="char") == []

    def test_no_word_tokens_returns_empty_list(self) -> None:
        assert get_shingles("... !!! ???", k=3, mode="word") == []

    def test_invalid_k_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="k must be a positive integer"):
            get_shingles("some text", k=0, mode="word")
        with pytest.raises(ValueError, match="k must be a positive integer"):
            get_shingles("some text", k=-1, mode="word")

    def test_invalid_mode_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="Invalid mode"):
            get_shingles("some text", k=3, mode="byte")
