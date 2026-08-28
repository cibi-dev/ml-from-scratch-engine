"""Nano-Transformer: Small Character-Level Language Model (~1M parameters).

Pre-LN Transformer architecture (GPT-2 style) with scaled dot-product causal attention,
weight tying, residual projection scaling, and strict security guards.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple, cast

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class GPTConfig:
    """Hyperparameter configuration for the Nano-Transformer GPT model."""

    vocab_size: int = 65
    block_size: int = 256
    n_layer: int = 4
    n_head: int = 4
    n_embd: int = 128
    dropout: float = 0.0
    bias: bool = True


class CausalSelfAttention(nn.Module):
    """Multi-head scaled dot-product causal self-attention mechanism."""

    bias: torch.Tensor

    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        if config.n_embd % config.n_head != 0:
            raise ValueError(
                f"n_embd ({config.n_embd}) must be divisible by n_head ({config.n_head})"
            )

        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.head_dim = config.n_embd // config.n_head
        self.dropout = config.dropout

        # Key, Query, Value projections in a single combined linear layer
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        # Output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)

        # Regularization
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

        # Lower-triangular causal mask: buffer ensures future tokens receive zero attention
        self.register_buffer(
            "bias",
            torch.tril(torch.ones(config.block_size, config.block_size)).view(
                1, 1, config.block_size, config.block_size
            ),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for causal multi-head self-attention.

        Args:
            x: Input tensor of shape (batch_size, seq_len, n_embd)

        Returns:
            Output tensor of shape (batch_size, seq_len, n_embd)
        """
        B, T, C = x.size()

        # Calculate query, key, values for all heads in batch and split
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)

        # Reshape to (B, n_head, T, head_dim)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        # Scaled dot-product causal attention
        scale = 1.0 / math.sqrt(self.head_dim)
        att = (q @ k.transpose(-2, -1)) * scale

        # Mask out future positions (strictly lower-triangular causal mask)
        causal_mask = self.bias[:, :, :T, :T]
        att = att.masked_fill(causal_mask == 0, float("-inf"))
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)

        # Weighted sum of values
        y = att @ v  # (B, n_head, T, head_dim)
        y = y.transpose(1, 2).contiguous().view(B, T, C)  # Re-assemble all head outputs

        # Output projection with residual dropout
        out: torch.Tensor = self.resid_dropout(self.c_proj(y))
        return out


class MLP(nn.Module):
    """Multi-Layer Perceptron (Feed-Forward Network) with GELU (tanh approximation)."""

    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.gelu = nn.GELU(approximate="tanh")
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through MLP.

        Args:
            x: Input tensor of shape (batch_size, seq_len, n_embd)

        Returns:
            Output tensor of shape (batch_size, seq_len, n_embd)
        """
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x


class Block(nn.Module):
    """Pre-LayerNorm Transformer Block (GPT-2 style) with residual connections."""

    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd, eps=1e-5)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd, eps=1e-5)
        self.mlp = MLP(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with Pre-LN residual connections.

        Args:
            x: Input tensor of shape (batch_size, seq_len, n_embd)

        Returns:
            Output tensor of shape (batch_size, seq_len, n_embd)
        """
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class GPT(nn.Module):
    """Character-Level Nano-Transformer Language Model (~1M parameters)."""

    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.config = config

        self.wte = nn.Embedding(config.vocab_size, config.n_embd)
        self.wpe = nn.Embedding(config.block_size, config.n_embd)
        self.drop = nn.Dropout(config.dropout)
        self.h = nn.ModuleList([Block(config) for _ in range(config.n_layer)])
        self.ln_f = nn.LayerNorm(config.n_embd, eps=1e-5)

        self.transformer = nn.ModuleDict(
            dict(
                wte=self.wte,
                wpe=self.wpe,
                drop=self.drop,
                h=self.h,
                ln_f=self.ln_f,
            )
        )
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # Weight tying: tie token embedding weights to output projection weights
        self.wte.weight = self.lm_head.weight

        # Initialize all weights
        self.apply(self._init_weights)

        # Apply special scaled initialization to residual projections (per GPT-2 paper)
        for pn, p in self.named_parameters():
            if pn.endswith("c_proj.weight"):
                torch.nn.init.normal_(
                    p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer)
                )

    def _init_weights(self, module: nn.Module) -> None:
        """Initialize linear and embedding weights with N(0, 0.02) and biases with 0."""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def get_num_params(self, non_embedding: bool = False) -> int:
        """Return the number of parameters in the model.

        Args:
            non_embedding: If True, subtract position embeddings.

        Returns:
            Count of parameters.
        """
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.wpe.weight.numel()
        return n_params

    def forward(
        self, idx: torch.Tensor, targets: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Forward pass of the Nano-Transformer.

        Args:
            idx: Tensor of token IDs, shape (batch_size, seq_len)
            targets: Optional ground-truth token IDs for cross-entropy loss, shape (batch_size, seq_len)

        Returns:
            Tuple of (logits, loss). Loss is None if targets is not provided.

        Raises:
            ValueError: If token IDs are negative or >= vocab_size.
            RuntimeError: If computed loss is NaN or Inf.
        """
        if idx.dim() != 2:
            raise ValueError(f"Input tensor must be 2D (B, T), got {idx.dim()}D shape {idx.shape}")

        # Security check: Token ID bounds check
        if (idx < 0).any() or (idx >= self.config.vocab_size).any():
            min_val = idx.min().item()
            max_val = idx.max().item()
            raise ValueError(
                f"Token ID out of bounds: expected [0, {self.config.vocab_size}), "
                f"got min={min_val}, max={max_val}"
            )

        # Security check: Sequence length overflow truncation
        if idx.size(1) > self.config.block_size:
            idx = idx[:, -self.config.block_size :].contiguous()

        device = idx.device
        b, t = idx.size()

        if targets is not None:
            if targets.dim() != 2:
                raise ValueError(
                    f"Targets tensor must be 2D (B, T), got {targets.dim()}D shape {targets.shape}"
                )
            if targets.size(1) > self.config.block_size:
                targets = targets[:, -self.config.block_size :].contiguous()

            # Validate target token IDs (ignoring cross-entropy ignore_index=-1)
            valid_targets = targets[targets != -1]
            if valid_targets.numel() > 0:
                if (valid_targets < 0).any() or (valid_targets >= self.config.vocab_size).any():
                    raise ValueError(
                        f"Target token ID out of bounds: expected [0, {self.config.vocab_size}), "
                        f"got min={valid_targets.min().item()}, max={valid_targets.max().item()}"
                    )

        # Forward through transformer layers
        pos = torch.arange(0, t, dtype=torch.long, device=device)  # shape (t)
        tok_emb = self.wte(idx)  # token embeddings: (b, t, n_embd)
        pos_emb = self.wpe(pos)  # position embeddings: (t, n_embd)
        x = self.drop(tok_emb + pos_emb)

        for block in self.h:
            x = block(x)
        x = self.ln_f(x)

        logits = self.lm_head(x)  # (b, t, vocab_size)

        loss: Optional[torch.Tensor] = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), targets.reshape(-1), ignore_index=-1
            )
            # Security guard: NaN/Inf detection
            if torch.isnan(loss) or torch.isinf(loss):
                raise RuntimeError("NaN or Inf detected in loss calculation.")

        return logits, loss

    def configure_optimizers(
        self,
        weight_decay: float,
        learning_rate: float,
        betas: Tuple[float, float] = (0.9, 0.95),
        device_type: str = "cpu",
    ) -> torch.optim.AdamW:
        """Segregate parameters into 2D (weight decay) and 1D (no decay) groups for AdamW.

        2D parameters (weights of Linear layers and Embeddings) receive weight decay.
        1D parameters (biases of Linear layers, LayerNorm weights and biases) do not.

        Args:
            weight_decay: Weight decay coefficient for 2D parameters.
            learning_rate: Initial learning rate.
            betas: Adam beta parameters (beta1, beta2).
            device_type: Device type ('cpu' or 'cuda') for fused AdamW support.

        Returns:
            Configured AdamW optimizer.
        """
        decay_params = []
        no_decay_params = []
        seen_params = set()

        for p in self.parameters():
            if not p.requires_grad:
                continue
            if id(p) in seen_params:
                continue
            seen_params.add(id(p))

            if p.dim() >= 2:
                decay_params.append(p)
            else:
                no_decay_params.append(p)

        optim_groups = [
            {"params": decay_params, "weight_decay": weight_decay},
            {"params": no_decay_params, "weight_decay": 0.0},
        ]

        # Use fused AdamW if supported on CUDA
        use_fused = (device_type == "cuda") and ("fused" in torch.optim.AdamW.__init__.__code__.co_varnames)
        extra_args = {"fused": True} if use_fused else {}
        optimizer = torch.optim.AdamW(
            optim_groups, lr=learning_rate, betas=betas, **extra_args
        )
        return optimizer
