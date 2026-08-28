# BPE Tokenizer

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![Type Checked: mypy strict](https://img.shields.io/badge/mypy-strict-brightgreen.svg)](https://mypy-lang.org/)
[![Tests: 74 Passed](https://img.shields.io/badge/tests-74%20passed-brightgreen.svg)](https://pytest.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

A high-performance, secure Byte Pair Encoding (BPE) tokenizer implemented in Python from scratch. Designed with clean architecture, strict static typing (mypy `--strict`), full test coverage, and comprehensive security hardening.

---

## 🏗️ Architecture & Data Flow

```
+-----------------------------------------------------------------------------+
|                                 RAW INPUT STRING                            |
+-----------------------------------------------------------------------------+
                                       |
                   [Security Check: len(bytes) <= 10MB]
                                       |
                                       v
                     +-----------------------------------+
                     | Special Token Injection Guard     |
                     | (allowed_special="none_raise")    |
                     +-----------------------------------+
                                       |
                +----------------------+----------------------+
                |                                             |
                v                                             v
     [BasicTokenizer Path]                         [RegexTokenizer Path]
                |                                             |
        Flat Byte Sequence                            Regex Chunking (GPT-4)
       [b"H", b"e", b"l", ...]                    ["Hello", " world", "!"]
                |                                             |
                |                                     Per-Chunk Byte IDs
                |                                   [[72, 101, ...], ...]
                |                                             |
                +----------------------+----------------------+
                                       |
                                       v
                     +-----------------------------------+
                     |  Iterative BPE Merge Processing   |
                     |  (min merge rank priority queue)  |
                     +-----------------------------------+
                                       |
                                       v
                             +-------------------+
                             |     Token IDs     |
                             |  [1004, 256, 89]  |
                             +-------------------+
                                       |
                                [Decode Step]
               [Bounds Check 0 <= id < vocab_size & Special Tokens]
                                       |
                                       v
                              Raw Byte Concatenation
                                       |
                             [UTF-8 errors="replace"]
                                       |
                                       v
                             Lossless Output String
```

---

## ✨ Features

- **Byte-Level BPE Core**: Operates on raw UTF-8 bytes (vocabulary initialized to 256 byte tokens `0..255`), ensuring universal coverage of any text domain without out-of-vocabulary (OOV) tokens.
- **Regex Boundary Enforcement**: Implements GPT-2 and GPT-4 pre-tokenization regex patterns to isolate contractions, numbers, punctuation, and whitespace, preventing cross-boundary merges.
- **Strict Security Guardrails**:
  - `MAX_INPUT_BYTES = 10MB` limit enforced on training and encoding.
  - Safe default `allowed_special="none_raise"` to prevent LLM prompt/control token injection attacks.
  - ReDoS-hardened regular expressions with linear-time backtracking bounds.
  - Safe plain-text serialization (no `pickle` or `eval`).
  - Graceful UTF-8 handling with `errors="replace"` and strict token ID boundary verification.
- **Full Domain Roundtrip Guarantee**: Tested losslessly across English prose, multi-language code (Python, Rust, C++, JS), CJK, emojis with ZWJ sequences, accents, diacritics, and whitespace variations.
- **Type-Safe**: 100% type-annotated, passing `mypy --strict`.

---

## 📦 Installation

Using `uv`:
```bash
uv sync --all-extras
```

Or using `pip`:
```bash
pip install -e ".[dev]"
```

---

## 🚀 Quickstart

```python
from bpe_tokenizer import RegexTokenizer, GPT4_SPLIT_PATTERN

# 1. Initialize tokenizer with GPT-4 split pattern
tok = RegexTokenizer(pattern=GPT4_SPLIT_PATTERN)

# 2. Register control / special tokens
tok.register_special_tokens({
    "<|endoftext|>": 50000,
    "<|im_start|>": 50001,
    "<|im_end|>": 50002,
})

# 3. Train on corpus
corpus = "Byte pair encoding is a subword tokenization algorithm used in LLMs."
tok.train(corpus, vocab_size=300)

# 4. Secure encoding (default: raises ValueError if unallowed special tokens appear)
text = "Byte pair encoding in Python!"
token_ids = tok.encode(text)
print("Token IDs:", token_ids)

# 5. Explicitly allow special tokens when formatting prompts
prompt = "<|im_start|>user\nHello!<|im_end|>"
prompt_ids = tok.encode(prompt, allowed_special="all")
print("Prompt IDs:", prompt_ids)

# 6. Lossless decoding
decoded_text = tok.decode(prompt_ids)
assert decoded_text == prompt
```

---

## 📖 API Reference

### `Tokenizer` (Abstract Base Class)
- `merges: dict[tuple[int, int], int]`: Map of merged token pairs `(p0, p1)` to new token ID.
- `vocab: dict[int, bytes]`: Map of token ID to byte sequence.
- `special_tokens: dict[str, int]`: Map of special token string to assigned token ID.
- `train(text: str, vocab_size: int, verbose: bool = False) -> None`: Train BPE model.
- `encode(text: str) -> list[int]`: Encode string into token IDs.
- `decode(ids: list[int]) -> str`: Decode token IDs into UTF-8 string.
- `save(file_prefix: str) -> None`: Save `<file_prefix>.model` and human-readable `<file_prefix>.vocab`.
- `load(model_file: str) -> None`: Load model definition from `.model` file.

### `BasicTokenizer`
Naive byte-level BPE tokenizer operating directly on flat byte streams without regex boundary isolation.
- `train(text: str, vocab_size: int, verbose: bool = False) -> None`
- `encode(text: str) -> list[int]`

### `RegexTokenizer(pattern: str | None = None)`
Advanced BPE tokenizer using regex chunking (defaults to `GPT4_SPLIT_PATTERN`) and special token handling.
- `register_special_tokens(special_tokens: dict[str, int]) -> None`: Register custom control tokens.
- `encode(text: str, allowed_special: set[str] | str = "none_raise") -> list[int]`:
  - `"none_raise"` *(default)*: Raises `ValueError` if any registered special token is found in `text`.
  - `"none"`: Ignores special tokens, tokenizing their characters as normal text.
  - `"all"`: Parses all registered special tokens as dedicated token IDs.
  - `set[str]`: Parses only the special tokens in the whitelist set, raising `ValueError` for any unlisted special token.

### Utilities
- `get_stats(ids: Sequence[int], counts: dict[tuple[int, int], int] | None = None) -> dict[tuple[int, int], int]`: Count consecutive token pairs.
- `merge(ids: Sequence[int], pair: tuple[int, int], idx: int) -> list[int]`: Merge token pair into `idx`.
- `render_token(t: bytes) -> str`: Safely render raw bytes into an escaped printable string.

---

## 🔒 Security Considerations

| Vector | Mitigation Strategy |
| :--- | :--- |
| **Denial of Service (Oversized Payloads)** | Strict `MAX_INPUT_BYTES = 10 * 1024 * 1024` (10 MB) enforcement on both training and inference; raises `ValueError` before processing. |
| **Control Token Injection** | Default `allowed_special="none_raise"` ensures untrusted user input cannot inject special tokens like `<|endoftext|>` or `<|im_start|>` without explicit application approval. |
| **Regular Expression DoS (ReDoS)** | Uses atomic matching `?+` and bounded quantifier patterns (`\p{N}{1,3}`) from standard GPT-4 tokenization specs, preventing exponential backtracking on pathological inputs. |
| **Arbitrary Code Execution** | Model persistence uses a simple, deterministic line-based format (`.model` and `.vocab`) with JSON string validation, strictly avoiding insecure serializers like `pickle` or `yaml.load`. |
| **Malformed Unicode & Out-of-Bounds IDs** | Bounds checking rejects negative or unknown token IDs with `ValueError`. Decoding utilizes `errors="replace"` to gracefully recover from split multi-byte byte streams. |

---

## 🧪 Testing & Verification

Run the test suite with pytest:
```bash
export PATH="$HOME/.local/bin:$PATH"
uv run pytest -v --tb=short
```

Run static type checking with strict mypy:
```bash
uv run mypy bpe_tokenizer/ tests/ --strict
```

Run the interactive demonstration script:
```bash
uv run python examples/train_example.py
```

---

## 📄 License

MIT License. Copyright (c) 2026 cibi-dev.
