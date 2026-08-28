"""Training pipeline for Nano-Transformer with AdamW weight decay segregation and LR scheduling."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

import torch

from nano_transformer.data import get_batch
from nano_transformer.model import GPT


@dataclass
class TrainerConfig:
    """Hyperparameter configuration for training Nano-Transformer."""

    max_iters: int = 1000
    batch_size: int = 32
    block_size: int = 256
    learning_rate: float = 6e-4
    min_lr: float = 6e-5
    warmup_iters: int = 50
    lr_decay_iters: int = 1000
    weight_decay: float = 1e-1
    grad_clip: float = 1.0
    eval_interval: int = 100
    eval_iters: int = 20
    betas: Tuple[float, float] = (0.9, 0.95)
    device: str = "cpu"


class Trainer:
    """End-to-end Trainer with AdamW parameter segregation, cosine annealing, and grad clipping."""

    def __init__(
        self,
        model: GPT,
        config: TrainerConfig,
        train_data: torch.Tensor,
        val_data: Optional[torch.Tensor] = None,
    ) -> None:
        self.model = model
        self.config = config
        self.train_data = train_data
        self.val_data = val_data

        if len(train_data) <= config.block_size:
            raise ValueError(
                f"Training dataset length ({len(train_data)}) must be > block_size ({config.block_size})"
            )
        if val_data is not None and len(val_data) <= config.block_size:
            raise ValueError(
                f"Validation dataset length ({len(val_data)}) must be > block_size ({config.block_size})"
            )

        self.model.to(config.device)
        self.optimizer = model.configure_optimizers(
            weight_decay=config.weight_decay,
            learning_rate=config.learning_rate,
            betas=config.betas,
            device_type=config.device,
        )

    def get_lr(self, iter_num: int) -> float:
        """Compute the learning rate at a given iteration using cosine decay with warmup.

        Args:
            iter_num: Current training iteration step.

        Returns:
            Computed learning rate float.
        """
        # 1) Linear warmup for warmup_iters steps
        if iter_num < self.config.warmup_iters:
            return float(self.config.learning_rate * (iter_num + 1) / (self.config.warmup_iters + 1))
        # 2) If iter_num > lr_decay_iters, return min_lr
        if iter_num > self.config.lr_decay_iters:
            return float(self.config.min_lr)
        # 3) In between, use cosine decay down to min_lr
        decay_ratio = (iter_num - self.config.warmup_iters) / (
            self.config.lr_decay_iters - self.config.warmup_iters
        )
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
        return float(self.config.min_lr + coeff * (self.config.learning_rate - self.config.min_lr))

    @torch.no_grad()
    def estimate_loss(self) -> Dict[str, float]:
        """Estimate model cross-entropy loss over train and validation splits.

        Returns:
            Dictionary mapping split names ('train', 'val') to estimated average losses.
        """
        out: Dict[str, float] = {}
        was_training = self.model.training
        self.model.eval()

        splits: List[Tuple[str, torch.Tensor]] = [("train", self.train_data)]
        if self.val_data is not None:
            splits.append(("val", self.val_data))

        for split, dataset in splits:
            losses = torch.zeros(self.config.eval_iters)
            for k in range(self.config.eval_iters):
                x, y = get_batch(
                    dataset,
                    self.config.batch_size,
                    self.config.block_size,
                    self.config.device,
                )
                _, loss = self.model(x, y)
                if loss is not None:
                    losses[k] = float(loss.item())
            out[split] = float(losses.mean().item())

        self.model.train(was_training)
        return out

    def train_step(self, x: torch.Tensor, y: torch.Tensor, iter_num: int = 0) -> float:
        """Execute a single optimization step (forward, backward, clip, step).

        Args:
            x: Input token IDs batch, shape (B, T).
            y: Target token IDs batch, shape (B, T).
            iter_num: Current iteration counter.

        Returns:
            Loss value as a float.

        Raises:
            RuntimeError: If loss or gradients are NaN or Inf.
        """
        lr = self.get_lr(iter_num)
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr

        x = x.to(self.config.device)
        y = y.to(self.config.device)

        # Forward pass
        self.model.train()
        _, loss = self.model(x, y)
        if loss is None:
            raise RuntimeError("Model returned None loss during train_step")

        loss_val = float(loss.item())
        if math.isnan(loss_val) or math.isinf(loss_val):
            raise RuntimeError(f"Loss returned {loss_val} during train_step")

        # Backward pass
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()

        # Gradient sanity check
        for p in self.model.parameters():
            if p.grad is not None:
                if torch.isnan(p.grad).any() or torch.isinf(p.grad).any():
                    raise RuntimeError("NaN or Inf detected in gradients")

        # Gradient clipping for numerical stability
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip)

        # Optimizer step
        self.optimizer.step()
        return loss_val

    def train(
        self,
        callback: Optional[Callable[[int, float, Dict[str, float]], None]] = None,
    ) -> List[float]:
        """Run the full training loop for max_iters iterations.

        Args:
            callback: Optional callback receiving (iter_num, step_loss, eval_losses).

        Returns:
            List of loss values per iteration.
        """
        loss_history: List[float] = []

        for iter_num in range(self.config.max_iters):
            x, y = get_batch(
                self.train_data,
                self.config.batch_size,
                self.config.block_size,
                self.config.device,
            )
            step_loss = self.train_step(x, y, iter_num)
            loss_history.append(step_loss)

            if (
                iter_num % self.config.eval_interval == 0
                or iter_num == self.config.max_iters - 1
            ):
                eval_losses = self.estimate_loss()
                if callback is not None:
                    callback(iter_num, step_loss, eval_losses)

        return loss_history
