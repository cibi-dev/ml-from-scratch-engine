"""Autoregressive text generation with temperature, top-k filtering, and security guards."""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F

from nano_transformer.model import GPT


def generate(
    model: GPT,
    idx: torch.Tensor,
    max_new_tokens: int = 500,
    temperature: float = 1.0,
    top_k: Optional[int] = None,
    eos_token_id: Optional[int] = None,
) -> torch.Tensor:
    """Generate new tokens autoregressively given a conditioning sequence of token IDs.

    Args:
        model: GPT language model instance.
        idx: Conditioning token tensor of shape (B, T) of type torch.long.
        max_new_tokens: Maximum number of tokens to generate (hard cap default 500).
        temperature: Softmax temperature (1.0 = standard, <1.0 = more confident,
                     0.0 = greedy argmax, >1.0 = higher entropy).
        top_k: Optional top-k filtering (retains only the k highest probability tokens).
        eos_token_id: Optional token ID that halts generation early when encountered.

    Returns:
        Tensor of shape (B, T + generated_tokens) containing prompt and generated token IDs.

    Raises:
        ValueError: If max_new_tokens <= 0 or temperature < 0.
    """
    if max_new_tokens <= 0:
        raise ValueError(f"max_new_tokens must be positive, got {max_new_tokens}")
    if temperature < 0.0:
        raise ValueError(f"temperature must be non-negative, got {temperature}")
    if idx.dim() != 2:
        raise ValueError(f"Input idx must be 2D tensor (B, T), got {idx.dim()}D")

    was_training = model.training
    model.eval()

    try:
        with torch.no_grad():
            for _ in range(max_new_tokens):
                # If sequence context exceeds block_size, truncate to the most recent block_size tokens
                if idx.size(1) > model.config.block_size:
                    idx_cond = idx[:, -model.config.block_size :]
                else:
                    idx_cond = idx

                # Forward pass through model
                logits, _ = model(idx_cond)

                # Pluck the logits at the final step (B, vocab_size)
                logits = logits[:, -1, :]

                if temperature == 0.0:
                    # Deterministic greedy decoding
                    idx_next = torch.argmax(logits, dim=-1, keepdim=True)
                else:
                    # Scale logits by temperature
                    logits = logits / temperature

                    # Optional top-k filtering to truncate low-probability tail
                    if top_k is not None and top_k > 0:
                        k = min(top_k, logits.size(-1))
                        v, _ = torch.topk(logits, k)
                        logits[logits < v[:, [-1]]] = -float("Inf")

                    probs = F.softmax(logits, dim=-1)
                    idx_next = torch.multinomial(probs, num_samples=1)

                # Append sampled token to running sequence
                idx = torch.cat((idx, idx_next), dim=1)

                # Early stopping if EOS token is hit on all batch elements
                if eos_token_id is not None and (idx_next == eos_token_id).all():
                    break
    finally:
        model.train(was_training)

    return idx
