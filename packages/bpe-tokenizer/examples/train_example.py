#!/usr/bin/env python3
"""Example script demonstrating training, encoding, decoding, and saving with bpe_tokenizer."""

import os
import tempfile

from bpe_tokenizer import (
    GPT4_SPLIT_PATTERN,
    BasicTokenizer,
    RegexTokenizer,
    render_token,
)

SAMPLE_TRAINING_CORPUS = """
Byte pair encoding (BPE) is a data compression technique adapted for subword tokenization
in neural natural language processing models. Instead of splitting text into fixed words or
characters, BPE starts with individual bytes (0-255) and iteratively merges the most
frequently occurring adjacent pairs into new tokens.

Modern LLMs such as GPT-2, GPT-4, LLaMA, and Claude use variants of byte-level BPE with regex
pre-tokenization to prevent merges across whitespace, punctuation, and linguistic boundaries.

Special control tokens like <|endoftext|>, <|im_start|>, and <|fim_prefix|> are used to structure
conversations, demarcate document boundaries, and enable fill-in-the-middle code completion.
"""


def main() -> None:
    print("=" * 70)
    print("1. Training BasicTokenizer (Naive Byte-Level BPE)")
    print("=" * 70)

    basic_tok = BasicTokenizer()
    target_vocab_size = 300
    print(f"Initial vocab size: {len(basic_tok.vocab)} (bytes 0..255)")
    print(f"Training on sample corpus up to vocab_size = {target_vocab_size}...")
    basic_tok.train(SAMPLE_TRAINING_CORPUS, vocab_size=target_vocab_size, verbose=False)
    print(f"Learned {len(basic_tok.merges)} merges. Total vocab: {len(basic_tok.vocab)}")

    sample_input = "Byte pair encoding is an efficient subword tokenizer!"
    encoded_basic = basic_tok.encode(sample_input)
    decoded_basic = basic_tok.decode(encoded_basic)
    raw_bytes_len = len(sample_input.encode("utf-8"))

    print(f"\nOriginal text: '{sample_input}'")
    print(f"Raw UTF-8 bytes: {raw_bytes_len} bytes")
    print(f"Encoded tokens ({len(encoded_basic)} tokens): {encoded_basic}")
    print(f"Compression ratio: {raw_bytes_len / len(encoded_basic):.2f}x")
    print(f"Decoded text:  '{decoded_basic}'")
    assert decoded_basic == sample_input, "Roundtrip verification failed!"
    print("Roundtrip check: PASSED (lossless reconstruction)")

    print("\n" + "=" * 70)
    print("2. Training RegexTokenizer with GPT-4 Split Pattern & Special Tokens")
    print("=" * 70)

    regex_tok = RegexTokenizer(pattern=GPT4_SPLIT_PATTERN)

    # Register special tokens
    special_tokens = {
        "<|endoftext|>": 50000,
        "<|im_start|>": 50001,
        "<|im_end|>": 50002,
        "<|fim_prefix|>": 50003,
        "<|fim_middle|>": 50004,
        "<|fim_suffix|>": 50005,
    }
    regex_tok.register_special_tokens(special_tokens)
    print(f"Registered {len(special_tokens)} special tokens: {list(special_tokens.keys())}")

    regex_tok.train(SAMPLE_TRAINING_CORPUS, vocab_size=320, verbose=False)
    print(f"Learned {len(regex_tok.merges)} regex-bounded merges.")

    # Display top 5 learned merges
    print("\nTop 5 learned merges:")
    for i, (pair, idx) in enumerate(list(regex_tok.merges.items())[:5]):
        p0_str = render_token(regex_tok.vocab[pair[0]])
        p1_str = render_token(regex_tok.vocab[pair[1]])
        merged_str = render_token(regex_tok.vocab[idx])
        print(f"  Merge {i+1}: ({pair[0]}: '{p0_str}', {pair[1]}: '{p1_str}') -> Token {idx}: '{merged_str}'")

    print("\n" + "=" * 70)
    print("3. Special Token Handling and Injection Prevention")
    print("=" * 70)

    prompt = "<|im_start|>user\nWrite a hello world program in Python.<|im_end|><|endoftext|>"

    # 1. Default safe mode: allowed_special="none_raise" prevents untrusted injections
    try:
        regex_tok.encode(prompt)
    except ValueError as e:
        print(f"Default security behavior (allowed_special='none_raise'):\n  Blocked injection: {e}\n")

    # 2. Explicitly allowed special tokens
    encoded_specials = regex_tok.encode(prompt, allowed_special="all")
    print(f"Explicitly allowed special tokens (allowed_special='all'):")
    print(f"  Encoded token IDs: {encoded_specials}")
    decoded_specials = regex_tok.decode(encoded_specials)
    print(f"  Decoded text: '{decoded_specials}'")
    assert decoded_specials == prompt

    # 3. Whitelisted subset of special tokens
    doc = "<|fim_prefix|>def add(a, b):<|fim_suffix|>return result<|fim_middle|>\n    result = a + b\n"
    encoded_subset = regex_tok.encode(
        doc,
        allowed_special={"<|fim_prefix|>", "<|fim_middle|>", "<|fim_suffix|>"},
    )
    print(f"\nFIM code template with whitelisted tokens:")
    print(f"  Tokens: {encoded_subset}")
    print(f"  Decoded successfully: {regex_tok.decode(encoded_subset) == doc}")

    print("\n" + "=" * 70)
    print("4. Safe Model Serialization & Deserialization (.model / .vocab)")
    print("=" * 70)

    with tempfile.TemporaryDirectory() as tmpdir:
        model_prefix = os.path.join(tmpdir, "demo_model")
        regex_tok.save(model_prefix)
        print(f"Saved model to: {model_prefix}.model")
        print(f"Saved human-readable vocabulary to: {model_prefix}.vocab")

        # Load into a new tokenizer instance
        loaded_tok = RegexTokenizer()
        loaded_tok.load(f"{model_prefix}.model")
        print(f"Loaded tokenizer successfully. Merges: {len(loaded_tok.merges)}, Vocab: {len(loaded_tok.vocab)}")

        # Verify exact equivalence
        test_text = "Testing persistence across model save and load cycles. 🚀"
        enc1 = regex_tok.encode(test_text)
        enc2 = loaded_tok.encode(test_text)
        assert enc1 == enc2, "Loaded tokenizer produced different encoding!"
        assert loaded_tok.decode(enc2) == test_text, "Decoded text did not match original!"
        print("Model verification: SUCCESS (loaded model is byte-identical)")

    print("\n" + "=" * 70)
    print("All demonstrations completed successfully!")
    print("=" * 70)


if __name__ == "__main__":
    main()
