"""Pytest fixtures and configuration for Nano-Transformer tests."""

from __future__ import annotations

import pytest

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    torch = None  # type: ignore[assignment]
    TORCH_AVAILABLE = False


def pytest_ignore_collect(collection_path, path, config):  # type: ignore[no-untyped-def]
    """Skip test collection if optional [torch] dependency is not installed."""
    if not TORCH_AVAILABLE:
        return True
    return False


if TORCH_AVAILABLE:
    from nano_transformer.data import CharTokenizer, get_tiny_shakespeare_data
    from nano_transformer.model import GPT, GPTConfig

    @pytest.fixture
    def small_config() -> GPTConfig:
        """Fixture for small GPT configuration used in fast testing."""
        return GPTConfig(
            vocab_size=65,
            block_size=64,
            n_layer=2,
            n_head=2,
            n_embd=32,
            dropout=0.0,
            bias=True,
        )

    @pytest.fixture
    def standard_config() -> GPTConfig:
        """Fixture for standard Nano-Transformer configuration (~0.84M params)."""
        return GPTConfig(
            vocab_size=65,
            block_size=256,
            n_layer=4,
            n_head=4,
            n_embd=128,
            dropout=0.0,
            bias=True,
        )

    @pytest.fixture
    def small_model(small_config: GPTConfig) -> GPT:
        """Fixture for a small GPT model in eval mode."""
        torch.manual_seed(42)
        model = GPT(small_config)
        model.eval()
        return model

    @pytest.fixture
    def standard_model(standard_config: GPTConfig) -> GPT:
        """Fixture for standard Nano-Transformer model in eval mode."""
        torch.manual_seed(42)
        model = GPT(standard_config)
        model.eval()
        return model

    @pytest.fixture
    def tokenizer() -> CharTokenizer:
        """Fixture for CharTokenizer with standard 65-char vocab."""
        return CharTokenizer()

    @pytest.fixture
    def sample_text() -> str:
        """Fixture for Tiny Shakespeare sample text."""
        return get_tiny_shakespeare_data()

