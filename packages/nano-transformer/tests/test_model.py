"""Tests for model architecture, shape propagation, weight tying, and parameter counts."""

import math
import pytest
import torch

from nano_transformer.model import GPT, Block, CausalSelfAttention, GPTConfig, MLP


def test_config_defaults() -> None:
    """Test standard GPTConfig default values."""
    config = GPTConfig()
    assert config.vocab_size == 65
    assert config.block_size == 256
    assert config.n_layer == 4
    assert config.n_head == 4
    assert config.n_embd == 128
    assert config.dropout == 0.0
    assert config.bias is True


def test_config_validation_divisibility() -> None:
    """Test that CausalSelfAttention rejects n_embd not divisible by n_head."""
    invalid_config = GPTConfig(n_embd=65, n_head=4)
    with pytest.raises(ValueError, match="must be divisible by n_head"):
        CausalSelfAttention(invalid_config)


def test_model_instantiation(small_config: GPTConfig) -> None:
    """Test instantiation of GPT model."""
    model = GPT(small_config)
    assert isinstance(model, torch.nn.Module)
    blocks = model.transformer["h"]
    assert isinstance(blocks, torch.nn.ModuleList)
    assert len(blocks) == small_config.n_layer


def test_invalid_input_dimensions(small_model: GPT) -> None:
    """Test that 1D or 3D input tensor raises ValueError."""
    with pytest.raises(ValueError, match="Input tensor must be 2D"):
        small_model(torch.tensor([1, 2, 3]))

    with pytest.raises(ValueError, match="Input tensor must be 2D"):
        small_model(torch.zeros(2, 3, 4, dtype=torch.long))


def test_invalid_target_dimensions(small_model: GPT) -> None:
    """Test that 1D or 3D targets tensor raises ValueError."""
    x = torch.zeros(2, 4, dtype=torch.long)
    with pytest.raises(ValueError, match="Targets tensor must be 2D"):
        small_model(x, targets=torch.tensor([1, 2, 3]))


def test_shape_propagation_without_targets(small_model: GPT) -> None:
    """Test forward pass shape propagation (B, T) -> (B, T, vocab_size)."""
    batch_size, seq_len = 4, 16
    x = torch.randint(0, small_model.config.vocab_size, (batch_size, seq_len))
    logits, loss = small_model(x)

    assert logits.shape == (batch_size, seq_len, small_model.config.vocab_size)
    assert loss is None


def test_shape_propagation_with_targets(small_model: GPT) -> None:
    """Test forward pass returning valid scalar cross-entropy loss."""
    batch_size, seq_len = 2, 8
    x = torch.randint(0, small_model.config.vocab_size, (batch_size, seq_len))
    y = torch.randint(0, small_model.config.vocab_size, (batch_size, seq_len))
    logits, loss = small_model(x, targets=y)

    assert logits.shape == (batch_size, seq_len, small_model.config.vocab_size)
    assert loss is not None
    assert loss.dim() == 0  # Scalar loss
    assert not torch.isnan(loss)
    assert not torch.isinf(loss)


def test_weight_tying(standard_model: GPT) -> None:
    """Test weight tying between wte.weight and lm_head.weight."""
    wte_weight = standard_model.transformer["wte"].weight
    lm_head_weight = standard_model.lm_head.weight

    # Ensure identity equality (exact same underlying tensor storage)
    assert wte_weight is lm_head_weight
    assert wte_weight.data_ptr() == lm_head_weight.data_ptr()
    assert wte_weight.shape == (standard_model.config.vocab_size, standard_model.config.n_embd)


def test_parameter_counting_standard(standard_model: GPT) -> None:
    """Test parameter count for standard ~0.84M parameter configuration.

    wte: 65 * 128 = 8,320
    wpe: 256 * 128 = 32,768
    Block: ln_1(256) + c_attn(49,536) + c_proj(16,512) + ln_2(256) + c_fc(66,048) + c_proj(65,664) = 198,272
    4 Blocks: 793,088
    ln_f: 256
    lm_head tied to wte
    Total unique parameters: 8,320 + 32,768 + 793,088 + 256 = 834,432 (~0.84M)
    """
    total_params = standard_model.get_num_params(non_embedding=False)
    assert 800_000 < total_params < 900_000
    assert total_params == 834_432

    non_emb_params = standard_model.get_num_params(non_embedding=True)
    assert non_emb_params == 834_432 - 32_768


def test_mlp_forward_shape(small_config: GPTConfig) -> None:
    """Test MLP module forward pass preserves (B, T, n_embd) shape."""
    mlp = MLP(small_config)
    x = torch.randn(2, 10, small_config.n_embd)
    out = mlp(x)
    assert out.shape == x.shape


def test_gelu_approx_behavior(small_config: GPTConfig) -> None:
    """Test GELU tanh approximation is non-linear and zero at origin."""
    mlp = MLP(small_config)
    zero_in = torch.zeros(1, 1, small_config.n_embd * 4)
    zero_out = mlp.gelu(zero_in)
    assert torch.allclose(zero_out, torch.zeros_like(zero_out))


def test_block_forward_shape(small_config: GPTConfig) -> None:
    """Test Pre-LN Transformer Block forward pass preserves shape."""
    block = Block(small_config)
    x = torch.randn(2, 10, small_config.n_embd)
    out = block(x)
    assert out.shape == x.shape


def test_layernorm_eps(standard_model: GPT) -> None:
    """Test that all LayerNorm modules use eps=1e-5."""
    for module in standard_model.modules():
        if isinstance(module, torch.nn.LayerNorm):
            assert module.eps == 1e-5


def test_residual_projection_scaling() -> None:
    """Test scaled initialization std for residual projections: 0.02 / sqrt(2 * n_layer)."""
    n_layer = 4
    config = GPTConfig(n_layer=n_layer, n_embd=256, n_head=8)
    model = GPT(config)
    expected_std = 0.02 / math.sqrt(2 * n_layer)

    blocks = model.transformer["h"]
    assert isinstance(blocks, torch.nn.ModuleList)
    for block in blocks:
        assert isinstance(block, Block)
        c_proj_attn_std = block.attn.c_proj.weight.std().item()
        c_proj_mlp_std = block.mlp.c_proj.weight.std().item()
        assert abs(c_proj_attn_std - expected_std) < 0.01
        assert abs(c_proj_mlp_std - expected_std) < 0.01


def test_optimizer_parameter_groups_separation(standard_model: GPT) -> None:
    """Test configure_optimizers segregates 2D (decay) and 1D/LN (no decay) weights."""
    weight_decay = 0.1
    learning_rate = 6e-4
    optimizer = standard_model.configure_optimizers(
        weight_decay=weight_decay,
        learning_rate=learning_rate,
        device_type="cpu",
    )

    assert len(optimizer.param_groups) == 2
    decay_group = optimizer.param_groups[0]
    no_decay_group = optimizer.param_groups[1]

    assert decay_group["weight_decay"] == weight_decay
    assert no_decay_group["weight_decay"] == 0.0

    for p in decay_group["params"]:
        assert p.dim() >= 2

    for p in no_decay_group["params"]:
        assert p.dim() < 2
