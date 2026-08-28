# Nano-Transformer

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Type Checked: mypy](https://img.shields.io/badge/mypy-strict-success.svg)](https://mypy-lang.org/)

A lightweight, clean, and robust character-level Transformer Language Model (~0.84M parameters) built from scratch in PyTorch following strict GPT-2 Pre-LN architecture, test-driven development (TDD), and enterprise security hardening.

---

## 🏛️ Transformer Architecture Diagram

```
                 Input Tokens [B, T]
                          │
            ┌─────────────┴─────────────┐
            ▼                           ▼
  Token Embedding (wte)      Position Embedding (wpe)
       [B, T, C]                   [T, C]
            └─────────────┬─────────────┘
                          ▼  (Sum + Dropout)
                     [B, T, C]
                          │
        ┌─────────────────┴─────────────────┐
        │  Transformer Block × N (Pre-LN)   │
        │                                   │
        │    ┌─────────────────────────┐    │
        │    │       LayerNorm 1       │    │
        │    └────────────┬────────────┘    │
        │                 ▼                 │
        │      Causal Self-Attention        │
        │     (Q, K, V Projections)         │
        │                 ▼                 │
        │      Lower-Triangular Mask        │
        │     + Softmax + Dropout           │
        │                 │                 │
        │                 ▼                 │
        │        (+) Residual Add ◄─────────┼── (Identity)
        │                 │                 │
        │    ┌────────────┴────────────┐    │
        │    │       LayerNorm 2       │    │
        │    └────────────┬────────────┘    │
        │                 ▼                 │
        │      Multi-Layer Perceptron       │
        │       Linear (C -> 4C)            │
        │       GELU (approx="tanh")        │
        │       Linear (4C -> C)            │
        │       + Dropout                   │
        │                 │                 │
        │                 ▼                 │
        │        (+) Residual Add ◄─────────┼── (Identity)
        └─────────────────┬─────────────────┘
                          │
                          ▼
                  Final LayerNorm (ln_f)
                          │
                          ▼
            Language Model Head (lm_head)
            [Weight-Tied to wte.weight]
                          │
                          ▼
                 Logits [B, T, vocab_size]
```

---

## 📊 Architecture & Parameter Breakdown

Configured with `vocab_size=65`, `block_size=256`, `n_layer=4`, `n_head=4`, `n_embd=128`:

| Component | Dimensions / Structure | Parameter Count |
| :--- | :--- | :--- |
| **Token Embeddings (`wte`)** | `65 × 128` | **8,320** |
| **Position Embeddings (`wpe`)** | `256 × 128` | **32,768** |
| **Transformer Blocks (`4 × Block`)** | 4 layers: | **793,088** |
| ├─ `ln_1` (LayerNorm) | `2 × 128` | 256 |
| ├─ `c_attn` (QKV Linear) | `128 × (3 × 128) + 384` | 49,536 |
| ├─ `c_proj` (Attn Projection) | `128 × 128 + 128` | 16,512 |
| ├─ `ln_2` (LayerNorm) | `2 × 128` | 256 |
| ├─ `mlp.c_fc` (FC Expansion) | `128 × 512 + 512` | 66,048 |
| └─ `mlp.c_proj` (FC Projection) | `512 × 128 + 128` | 65,664 |
| **Final LayerNorm (`ln_f`)** | `2 × 128` | **256** |
| **Language Model Head (`lm_head`)** | `128 × 65` (*Tied to `wte`*) | *0 (Tied)* |
| **Total Unique Parameters** | | **834,432 (~0.84M)** |

---

## 🔒 Security Hardening & Robustness

1. **Out-of-Bounds Token Rejection**:
   - Explicit range validation `0 <= t < vocab_size` before embedding lookups to prevent invalid memory indexing or silent corruption.
2. **Context Window Overflow Truncation**:
   - Sequences exceeding `block_size` are automatically sliced to `x[:, -block_size:]` preserving recent context while preventing shape mismatch exceptions.
3. **Loss & Gradient Anomaly Guards**:
   - Strict runtime assertions catch `NaN` or `Inf` in cross-entropy loss or gradient buffers before optimizer steps.
4. **Infinite Generation Prevention**:
   - Hard termination bounds (`max_new_tokens`) coupled with early `eos_token_id` breaking ensure generation loops always terminate.
5. **AdamW Parameter Segregation**:
   - 2D weight matrices (Linear & Embeddings) receive weight decay ($0.1$), while 1D tensors (biases and LayerNorm gain/bias) are exempt ($0.0$), maintaining parameter scale stability.
6. **Numerical Scaling**:
   - Scaled dot-product attention uses $1.0 / \sqrt{d_k}$ scaling, LayerNorm uses $\epsilon = 10^{-5}$, and gradients are clipped to $\le 1.0$.

---

## 🚀 Quickstart

### Installation

```bash
# Clone the repository
git clone https://github.com/cibi-dev/nano-transformer.git
cd nano-transformer

# Sync environment using uv
export PATH="$HOME/.local/bin:$PATH"
uv sync --extra dev
```

### Running Tests & Type Checking

```bash
# Run pytest suite
uv run pytest -v --tb=short

# Run strict type checking with mypy
uv run mypy nano_transformer/ --strict
```

### Running the Example

```bash
uv run python examples/train_and_generate.py
```

---

## 💻 API Reference

### 1. Model Configuration & Instantiation

```python
import torch
from nano_transformer import GPT, GPTConfig

config = GPTConfig(
    vocab_size=65,
    block_size=256,
    n_layer=4,
    n_head=4,
    n_embd=128,
    dropout=0.1,
)
model = GPT(config)
```

### 2. Tokenization & Data Preparation

```python
from nano_transformer import CharTokenizer, create_data_splits, get_tiny_shakespeare_data

raw_text = get_tiny_shakespeare_data()
tokenizer = CharTokenizer()
train_data, val_data = create_data_splits(raw_text, tokenizer=tokenizer, train_ratio=0.9)
```

### 3. Training Loop

```python
from nano_transformer import Trainer, TrainerConfig

trainer_cfg = TrainerConfig(
    max_iters=500,
    batch_size=32,
    block_size=128,
    learning_rate=6e-4,
    min_lr=6e-5,
    warmup_iters=50,
)
trainer = Trainer(model, trainer_cfg, train_data, val_data)
loss_history = trainer.train()
```

### 4. Autoregressive Generation

```python
from nano_transformer import generate

prompt = torch.tensor([tokenizer.encode("First Citizen:\n")], dtype=torch.long)
output_ids = generate(
    model=model,
    idx=prompt,
    max_new_tokens=100,
    temperature=0.8,
    top_k=10,
)
print(tokenizer.decode(output_ids[0].tolist()))
```

---

## 📄 License

MIT License (c) 2026 cibi-dev. See [LICENSE](LICENSE) for details.
