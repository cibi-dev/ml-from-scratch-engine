import pytest

torch = pytest.importorskip("torch", reason="extra opcional [torch]")

from nano_transformer.data import CharTokenizer, create_data_splits
from nano_transformer.generate import generate
from nano_transformer.model import GPT, GPTConfig


def test_oob_token_id_rejection_negative(small_model: GPT) -> None:
    """Test that negative token IDs are rejected with ValueError."""
    bad_input = torch.tensor([[0, 5, -1, 10]], dtype=torch.long)
    with pytest.raises(ValueError, match="Token ID out of bounds"):
        small_model(bad_input)


def test_oob_token_id_rejection_too_large(small_model: GPT) -> None:
    """Test that token IDs >= vocab_size are rejected with ValueError."""
    vocab_size = small_model.config.vocab_size
    bad_input = torch.tensor([[0, 5, vocab_size, 10]], dtype=torch.long)
    with pytest.raises(ValueError, match="Token ID out of bounds"):
        small_model(bad_input)


def test_oob_target_token_id_rejection(small_model: GPT) -> None:
    """Test that target token IDs outside [0, vocab_size) (except -1) raise ValueError."""
    valid_input = torch.tensor([[0, 1, 2, 3]], dtype=torch.long)
    bad_target = torch.tensor([[0, 1, small_model.config.vocab_size + 10, 3]], dtype=torch.long)
    with pytest.raises(ValueError, match="Target token ID out of bounds"):
        small_model(valid_input, targets=bad_target)


def test_ignore_index_in_targets_allowed(small_model: GPT) -> None:
    """Test that target index -1 (standard ignore_index) is allowed and computed properly."""
    valid_input = torch.tensor([[0, 1, 2, 3]], dtype=torch.long)
    target_with_ignore = torch.tensor([[-1, 1, 2, -1]], dtype=torch.long)
    logits, loss = small_model(valid_input, targets=target_with_ignore)
    assert loss is not None
    assert not torch.isnan(loss)


def test_sequence_length_overflow_truncation(small_model: GPT) -> None:
    """Test that input sequence longer than block_size is safely truncated to block_size."""
    block_size = small_model.config.block_size
    long_input = torch.randint(0, small_model.config.vocab_size, (2, block_size + 50))

    logits, _ = small_model(long_input)
    assert logits.shape == (2, block_size, small_model.config.vocab_size)


def test_target_length_overflow_truncation(small_model: GPT) -> None:
    """Test that target sequence longer than block_size is safely truncated to block_size."""
    block_size = small_model.config.block_size
    long_input = torch.randint(0, small_model.config.vocab_size, (2, block_size + 30))
    long_targets = torch.randint(0, small_model.config.vocab_size, (2, block_size + 30))

    logits, loss = small_model(long_input, targets=long_targets)
    assert logits.shape == (2, block_size, small_model.config.vocab_size)
    assert loss is not None
    assert not torch.isnan(loss)


def test_infinite_generation_loop_prevention(small_model: GPT) -> None:
    """Test that autoregressive generate strictly terminates at max_new_tokens hard limit."""
    prompt = torch.tensor([[0, 1, 2]], dtype=torch.long)
    max_new = 25
    out = generate(small_model, prompt, max_new_tokens=max_new, temperature=1.0)
    assert out.shape == (1, 3 + max_new)


def test_generate_eos_early_stopping(small_model: GPT) -> None:
    """Test that generation stops immediately upon producing the designated EOS token."""
    prompt = torch.tensor([[0, 1]], dtype=torch.long)

    # Determine what token the model predicts at step 1 under greedy decoding
    with torch.no_grad():
        logits, _ = small_model(prompt)
        predicted_first_token = int(torch.argmax(logits[:, -1, :], dim=-1).item())

    # Set eos_token_id to the predicted token
    out = generate(
        small_model,
        prompt,
        max_new_tokens=50,
        temperature=0.0,
        eos_token_id=predicted_first_token,
    )
    # Output must have exactly 1 generated token (total length 3)
    assert out.shape == (1, 3)
    assert out[0, -1].item() == predicted_first_token


def test_generate_invalid_arguments(small_model: GPT) -> None:
    """Test that invalid generation parameters raise ValueError."""
    prompt = torch.tensor([[0, 1]], dtype=torch.long)
    with pytest.raises(ValueError, match="max_new_tokens must be positive"):
        generate(small_model, prompt, max_new_tokens=0)

    with pytest.raises(ValueError, match="temperature must be non-negative"):
        generate(small_model, prompt, max_new_tokens=10, temperature=-0.5)

    with pytest.raises(ValueError, match="Input idx must be 2D tensor"):
        generate(small_model, torch.tensor([0, 1]))


def test_generate_preserves_prompt(small_model: GPT) -> None:
    """Test that generated sequence strictly begins with original prompt."""
    prompt = torch.tensor([[3, 7, 11, 15]], dtype=torch.long)
    out = generate(small_model, prompt, max_new_tokens=10, temperature=0.7)
    assert torch.equal(out[:, :4], prompt)


def test_tokenizer_oob_decode_rejection(tokenizer: CharTokenizer) -> None:
    """Test that tokenizer rejects invalid token IDs during decoding."""
    with pytest.raises(ValueError, match="out of range"):
        tokenizer.decode([-1])

    with pytest.raises(ValueError, match="out of range"):
        tokenizer.decode([tokenizer.vocab_size + 5])


def test_tokenizer_unknown_char_rejection(tokenizer: CharTokenizer) -> None:
    """Test that tokenizer rejects characters not in vocabulary."""
    with pytest.raises(ValueError, match="Unrecognized character"):
        tokenizer.encode("Hello 🚀 World")


def test_tokenizer_roundtrip_integrity(tokenizer: CharTokenizer) -> None:
    """Test lossless roundtrip encoding and decoding of valid text."""
    sample = "First Citizen:\nBefore we proceed any further, hear me speak."
    encoded = tokenizer.encode(sample)
    decoded = tokenizer.decode(encoded)
    assert decoded == sample
    assert len(encoded) == len(sample)


def test_tokenizer_custom_vocabulary() -> None:
    """Test CharTokenizer with a custom character set."""
    custom_tok = CharTokenizer(chars="ABCD")
    assert custom_tok.vocab_size == 4
    encoded = custom_tok.encode("DCBA")
    assert encoded == [3, 2, 1, 0]
    assert custom_tok.decode(encoded) == "DCBA"


def test_create_data_splits_validation() -> None:
    """Test train_ratio validation in create_data_splits."""
    with pytest.raises(ValueError, match="train_ratio must be in"):
        create_data_splits("Hello world", train_ratio=0.0)

    with pytest.raises(ValueError, match="train_ratio must be in"):
        create_data_splits("Hello world", train_ratio=1.5)


def test_generate_greedy_temperature_zero(small_model: GPT) -> None:
    """Test that temperature=0.0 produces deterministic greedy output."""
    prompt = torch.tensor([[5, 12, 18]], dtype=torch.long)
    out1 = generate(small_model, prompt, max_new_tokens=15, temperature=0.0)
    out2 = generate(small_model, prompt, max_new_tokens=15, temperature=0.0)
    assert torch.equal(out1, out2)


def test_top_k_filtering_bounds(small_model: GPT) -> None:
    """Test top_k filtering with k exceeding vocab_size and k=1."""
    prompt = torch.tensor([[1, 2, 3]], dtype=torch.long)
    out_large_k = generate(
        small_model, prompt, max_new_tokens=5, temperature=1.0, top_k=1000
    )
    assert out_large_k.shape == (1, 8)

    out_k1 = generate(small_model, prompt, max_new_tokens=5, temperature=1.0, top_k=1)
    assert out_k1.shape == (1, 8)
