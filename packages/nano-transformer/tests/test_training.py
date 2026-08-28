"""Tests for training loop, micro-batch overfitting, gradient flow, and learning rate scheduler."""

import pytest
import torch

from nano_transformer.data import CharTokenizer, create_data_splits, get_batch
from nano_transformer.model import GPT, GPTConfig
from nano_transformer.train import Trainer, TrainerConfig


def test_overfitting_micro_batch() -> None:
    """Test that a small model can completely overfit a micro-batch (loss < 0.05 in <100 steps)."""
    torch.manual_seed(42)
    config = GPTConfig(
        vocab_size=65,
        block_size=16,
        n_layer=2,
        n_head=2,
        n_embd=64,
        dropout=0.0,
    )
    model = GPT(config)

    # Create fixed micro-batch of 2 sequences
    x = torch.randint(0, 65, (2, 16))
    # Targets are shifted next tokens
    y = torch.randint(0, 65, (2, 16))

    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=0.0)

    initial_loss: float = 0.0
    final_loss: float = 0.0

    for step in range(80):
        optimizer.zero_grad()
        logits, loss = model(x, targets=y)
        assert loss is not None
        if step == 0:
            initial_loss = loss.item()
        loss.backward()
        optimizer.step()
        final_loss = loss.item()

    assert initial_loss > 3.0
    assert final_loss < 0.05, f"Expected final loss < 0.05, got {final_loss}"


def test_gradient_flow_all_layers(small_model: GPT) -> None:
    """Test that backward pass computes valid non-zero gradients for all trainable parameters."""
    small_model.train()
    batch_size, seq_len = 2, 8
    x = torch.randint(0, small_model.config.vocab_size, (batch_size, seq_len))
    y = torch.randint(0, small_model.config.vocab_size, (batch_size, seq_len))

    _, loss = small_model(x, targets=y)
    assert loss is not None
    loss.backward()

    for name, param in small_model.named_parameters():
        if param.requires_grad:
            assert param.grad is not None, f"Missing gradient for {name}"
            assert not torch.isnan(param.grad).any(), f"NaN gradient in {name}"
            assert not torch.isinf(param.grad).any(), f"Inf gradient in {name}"
            assert param.grad.abs().sum().item() > 0.0, f"Dead gradient in {name}"


def test_cosine_annealing_lr_schedule(small_model: GPT) -> None:
    """Test cosine annealing learning rate scheduler with linear warmup."""
    train_data = torch.randint(0, 65, (500,))
    config = TrainerConfig(
        max_iters=100,
        warmup_iters=20,
        lr_decay_iters=100,
        learning_rate=1e-3,
        min_lr=1e-4,
        block_size=16,
    )
    trainer = Trainer(small_model, config, train_data)

    # 1) Start of warmup (iter 0)
    lr_0 = trainer.get_lr(0)
    assert abs(lr_0 - (1e-3 * 1 / 21)) < 1e-6

    # 2) End of warmup (iter 20)
    lr_20 = trainer.get_lr(20)
    assert abs(lr_20 - 1e-3) < 1e-6

    # 3) Mid-decay (iter 60)
    lr_60 = trainer.get_lr(60)
    assert 1e-4 < lr_60 < 1e-3

    # 4) End of decay (iter 100)
    lr_100 = trainer.get_lr(100)
    assert abs(lr_100 - 1e-4) < 1e-6

    # 5) Beyond decay (iter 150)
    lr_150 = trainer.get_lr(150)
    assert abs(lr_150 - 1e-4) < 1e-6


def test_gradient_clipping(small_model: GPT) -> None:
    """Test that gradient clipping bounds gradient norm to specified threshold."""
    small_model.train()
    x = torch.randint(0, small_model.config.vocab_size, (2, 8))
    y = torch.randint(0, small_model.config.vocab_size, (2, 8))

    _, loss = small_model(x, targets=y)
    assert loss is not None
    huge_loss = loss * 1000.0
    huge_loss.backward()

    max_norm = 1.0
    total_norm = torch.nn.utils.clip_grad_norm_(small_model.parameters(), max_norm)
    assert total_norm > max_norm

    clipped_norm = torch.sqrt(
        sum(p.grad.norm() ** 2 for p in small_model.parameters() if p.grad is not None)
    )
    assert clipped_norm.item() <= max_norm + 1e-5


def test_estimate_loss_eval_mode(small_model: GPT) -> None:
    """Test that loss estimation evaluates without tracking gradients."""
    train_data = torch.randint(0, 65, (500,))
    val_data = torch.randint(0, 65, (200,))
    config = TrainerConfig(
        batch_size=4,
        block_size=16,
        eval_iters=5,
        device="cpu",
    )
    trainer = Trainer(small_model, config, train_data, val_data)

    losses = trainer.estimate_loss()
    assert "train" in losses
    assert "val" in losses
    assert losses["train"] > 0.0
    assert losses["val"] > 0.0


def test_trainer_data_length_validation(small_model: GPT) -> None:
    """Test that Trainer raises ValueError if data length <= block_size."""
    short_data = torch.randint(0, 65, (10,))
    config = TrainerConfig(block_size=16)
    with pytest.raises(ValueError, match="must be > block_size"):
        Trainer(small_model, config, train_data=short_data)


def test_trainer_full_loop_execution(small_model: GPT) -> None:
    """Test full trainer execution runs specified max_iters and invokes callback."""
    train_data = torch.randint(0, 65, (200,))
    val_data = torch.randint(0, 65, (100,))
    config = TrainerConfig(
        max_iters=5,
        batch_size=2,
        block_size=8,
        eval_interval=2,
        eval_iters=2,
        device="cpu",
    )
    trainer = Trainer(small_model, config, train_data, val_data)
    callbacks_received: list[int] = []

    def cb(iter_num: int, loss: float, eval_losses: dict[str, float]) -> None:
        callbacks_received.append(iter_num)

    history = trainer.train(callback=cb)
    assert len(history) == 5
    assert len(callbacks_received) > 0


def test_data_batching() -> None:
    """Test dataset batch extraction and target alignment."""
    data = torch.arange(100, dtype=torch.long)
    batch_size = 4
    block_size = 10

    x, y = get_batch(data, batch_size=batch_size, block_size=block_size)
    assert x.shape == (batch_size, block_size)
    assert y.shape == (batch_size, block_size)

    for b in range(batch_size):
        assert torch.equal(y[b, :-1], x[b, 1:])


def test_data_batching_invalid_inputs() -> None:
    """Test that get_batch validates tensor dimensions and batch sizes."""
    with pytest.raises(ValueError, match="Dataset tensor must be 1D"):
        get_batch(torch.zeros(10, 10, dtype=torch.long), batch_size=2, block_size=4)

    with pytest.raises(ValueError, match="strictly greater than block_size"):
        get_batch(torch.zeros(5, dtype=torch.long), batch_size=2, block_size=10)

    with pytest.raises(ValueError, match="batch_size must be positive"):
        get_batch(torch.zeros(20, dtype=torch.long), batch_size=0, block_size=5)
