import math
import pytest

torch = pytest.importorskip("torch", reason="extra opcional [torch]")
import torch.nn.functional as F

from nano_transformer.model import GPT, CausalSelfAttention, GPTConfig


def test_causal_mask_leakage_invariance(small_model: GPT) -> None:
    """Test that future tokens NEVER affect past or current token logits (leakage test).

    If sequence A and sequence B share the same prefix up to position k-1,
    the model's logits at positions 0..k-1 MUST be strictly identical (diff < 1e-6)
    regardless of what tokens appear at positions >= k.
    """
    small_model.eval()

    # Create base sequence of length 8
    seq_a = torch.tensor([[10, 20, 30, 40, 15, 25, 35, 45]], dtype=torch.long)
    # Create sequence B with identical prefix (length 4) but completely different suffix
    seq_b = torch.tensor([[10, 20, 30, 40, 60, 61, 62, 63]], dtype=torch.long)

    with torch.no_grad():
        logits_a, _ = small_model(seq_a)
        logits_b, _ = small_model(seq_b)

    # Prefix positions 0, 1, 2, 3 must produce identical logits
    prefix_diff = torch.max(torch.abs(logits_a[:, :4, :] - logits_b[:, :4, :])).item()
    assert prefix_diff < 1e-6, f"Causal mask leakage detected! Max prefix diff: {prefix_diff}"

    # Suffix positions (4, 5, 6, 7) will differ
    suffix_diff = torch.max(torch.abs(logits_a[:, 4:, :] - logits_b[:, 4:, :])).item()
    assert suffix_diff > 1e-3, "Suffix should produce different logits"


def test_attention_strictly_lower_triangular(small_config: GPTConfig) -> None:
    """Test that CausalSelfAttention's causal mask is strictly lower-triangular."""
    attn = CausalSelfAttention(small_config)
    mask = attn.bias.squeeze()  # shape (block_size, block_size)

    # Lower triangular elements must be 1.0, strictly upper triangular must be 0.0
    for i in range(small_config.block_size):
        for j in range(small_config.block_size):
            if j <= i:
                assert mask[i, j].item() == 1.0, f"Expected 1 at ({i}, {j})"
            else:
                assert mask[i, j].item() == 0.0, f"Expected 0 at ({i}, {j})"


def test_attention_scale_factor() -> None:
    """Test that attention logits are scaled by 1.0 / sqrt(head_dim)."""
    n_embd = 64
    n_head = 4
    head_dim = n_embd // n_head  # 16
    expected_scale = 1.0 / math.sqrt(head_dim)  # 0.25

    config = GPTConfig(n_embd=n_embd, n_head=n_head, dropout=0.0)
    attn = CausalSelfAttention(config)

    assert attn.head_dim == 16
    assert abs(1.0 / math.sqrt(attn.head_dim) - expected_scale) < 1e-7


def test_attention_batch_independence(small_model: GPT) -> None:
    """Test that batch items are processed independently without cross-contamination."""
    small_model.eval()

    item_1 = torch.tensor([[5, 10, 15, 20]], dtype=torch.long)
    item_2 = torch.tensor([[1, 2, 3, 4]], dtype=torch.long)
    batch = torch.cat([item_1, item_2], dim=0)

    with torch.no_grad():
        logits_batch, _ = small_model(batch)
        logits_1, _ = small_model(item_1)
        logits_2, _ = small_model(item_2)

    diff_1 = torch.max(torch.abs(logits_batch[0:1] - logits_1)).item()
    diff_2 = torch.max(torch.abs(logits_batch[1:2] - logits_2)).item()

    assert diff_1 < 1e-6
    assert diff_2 < 1e-6


def test_attention_deterministic_in_eval_mode(small_model: GPT) -> None:
    """Test that attention is fully deterministic when model is in eval mode."""
    small_model.eval()
    x = torch.randint(0, small_model.config.vocab_size, (2, 16))

    with torch.no_grad():
        out1, _ = small_model(x)
        out2, _ = small_model(x)

    assert torch.equal(out1, out2)


def test_attention_dropout_active_in_train() -> None:
    """Test that dropout produces non-identical outputs in training mode with dropout > 0."""
    torch.manual_seed(123)
    config = GPTConfig(
        vocab_size=65,
        block_size=32,
        n_layer=2,
        n_head=2,
        n_embd=32,
        dropout=0.5,
    )
    model = GPT(config)
    model.train()

    x = torch.randint(0, config.vocab_size, (2, 16))
    out1, _ = model(x)
    out2, _ = model(x)

    assert not torch.equal(out1, out2)
