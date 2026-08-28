#!/usr/bin/env python3
"""Example script: Train Nano-Transformer on Tiny Shakespeare excerpt and generate text."""

import torch

from nano_transformer.data import (
    CharTokenizer,
    create_data_splits,
    get_tiny_shakespeare_data,
)
from nano_transformer.generate import generate
from nano_transformer.model import GPT, GPTConfig
from nano_transformer.train import Trainer, TrainerConfig


def main() -> None:
    print("=" * 70)
    print("Nano-Transformer: Training & Autoregressive Generation Demo")
    print("=" * 70)

    # 1. Prepare Data & Tokenizer
    raw_text = get_tiny_shakespeare_data()
    tokenizer = CharTokenizer()
    print(f"[1/4] Vocabulary size: {tokenizer.vocab_size} unique characters")
    print(f"      Dataset size: {len(raw_text)} characters")

    train_data, val_data = create_data_splits(raw_text, tokenizer=tokenizer, train_ratio=0.9)
    print(f"      Train tokens: {len(train_data)} | Val tokens: {len(val_data)}")

    # 2. Configure Model
    block_size = 64
    config = GPTConfig(
        vocab_size=tokenizer.vocab_size,
        block_size=block_size,
        n_layer=4,
        n_head=4,
        n_embd=128,
        dropout=0.1,
    )
    model = GPT(config)
    total_params = model.get_num_params()
    print(f"[2/4] Initialized Nano-Transformer: {total_params:,} parameters (~{total_params/1e6:.2f}M)")

    # 3. Setup Trainer & Train
    trainer_config = TrainerConfig(
        max_iters=250,
        batch_size=16,
        block_size=block_size,
        learning_rate=1e-3,
        min_lr=1e-4,
        warmup_iters=20,
        lr_decay_iters=250,
        eval_interval=50,
        eval_iters=10,
        device="cpu",
    )

    def log_eval(step: int, loss: float, eval_losses: dict[str, float]) -> None:
        train_l = eval_losses.get("train", 0.0)
        val_l = eval_losses.get("val", 0.0)
        print(f"      Step {step:4d} | Step Loss: {loss:.4f} | Train Loss: {train_l:.4f} | Val Loss: {val_l:.4f}")

    print("[3/4] Starting training loop...")
    trainer = Trainer(model, trainer_config, train_data, val_data)
    trainer.train(callback=log_eval)

    # 4. Generate Autoregressive Text
    print("\n[4/4] Generating text autoregressively...")
    prompt_text = "First Citizen:\n"
    prompt_tokens = torch.tensor([tokenizer.encode(prompt_text)], dtype=torch.long)

    generated_tokens = generate(
        model=model,
        idx=prompt_tokens,
        max_new_tokens=200,
        temperature=0.8,
        top_k=10,
    )

    generated_text = tokenizer.decode(generated_tokens[0].tolist())
    print("-" * 50)
    print("PROMPT:")
    print(prompt_text.strip())
    print("-" * 50)
    print("GENERATED CONTINUATION:")
    print(generated_text)
    print("=" * 70)


if __name__ == "__main__":
    main()
