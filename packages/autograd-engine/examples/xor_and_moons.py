#!/usr/bin/env python3
"""Example: Training an MLP on XOR and Synthetic 2D Classification Datasets.

Zero external dependencies - runs with pure standard library Python.
"""

from __future__ import annotations

import math
import random
from typing import List, Tuple

from autograd.engine import Value
from autograd.nn import MLP


def train_xor() -> None:
    print("=" * 60)
    print("1. Training MLP on Classic XOR Problem")
    print("=" * 60)

    random.seed(42)

    # Dataset: XOR Truth Table
    xs = [
        [0.0, 0.0],
        [0.0, 1.0],
        [1.0, 0.0],
        [1.0, 1.0],
    ]
    ys = [0.0, 1.0, 1.0, 0.0]

    # Model: 2 inputs -> 4 hidden -> 4 hidden -> 1 output
    model = MLP(nin=2, nouts=[4, 4, 1], nonlin=True, nonlin_type="tanh")
    print(f"Model Architecture: {model}")
    print(f"Total Parameters: {len(model.parameters())}")

    learning_rate = 0.1
    epochs = 300

    for epoch in range(1, epochs + 1):
        # Forward pass
        ypred = [model(x) for x in xs]

        # MSE Loss
        loss: Value = Value(0.0)
        for y_target, y_pred in zip(ys, ypred):
            assert isinstance(y_pred, Value)
            loss = loss + (y_pred - y_target) ** 2
        loss = loss * (1.0 / len(ys))

        # Backward pass
        model.zero_grad()
        loss.backward()

        # SGD parameter update
        for p in model.parameters():
            p.data -= learning_rate * p.grad

        if epoch % 50 == 0 or epoch == 1:
            print(f"Epoch {epoch:3d}/{epochs} | Loss: {loss.data:.6f}")

    print("\nXOR Evaluation Results:")
    print("-" * 35)
    print(" Input (x1, x2) | Target | Prediction | Class")
    print("-" * 35)
    all_correct = True
    for x, y in zip(xs, ys):
        pred = model(x)
        assert isinstance(pred, Value)
        pred_val = pred.data
        pred_cls = 1.0 if pred_val >= 0.5 else 0.0
        correct = "✓" if pred_cls == y else "✗"
        if pred_cls != y:
            all_correct = False
        print(f"   ({x[0]:.0f}, {x[1]:.0f})       |   {y:.0f}    |   {pred_val:+.4f}   |   {pred_cls:.0f} {correct}")
    print("-" * 35)
    print(f"XOR Status: {'CONVERGED (100% Accuracy)' if all_correct else 'FAILED'}\n")


def make_moons(n_samples: int = 60, noise: float = 0.08) -> Tuple[List[List[float]], List[float]]:
    """Generates two interleaving half circles (moons) without external libraries."""
    samples_per_moon = n_samples // 2
    xs: List[List[float]] = []
    ys: List[float] = []

    # Moon 1: upper semi-circle centered at (0, 0)
    for _ in range(samples_per_moon):
        theta = random.uniform(0.0, math.pi)
        r = 1.0 + random.gauss(0.0, noise)
        x1 = r * math.cos(theta)
        x2 = r * math.sin(theta)
        xs.append([x1, x2])
        ys.append(1.0)

    # Moon 2: lower semi-circle shifted to (1.0, -0.5)
    for _ in range(samples_per_moon):
        theta = random.uniform(0.0, math.pi)
        r = 1.0 + random.gauss(0.0, noise)
        x1 = 1.0 - r * math.cos(theta)
        x2 = 0.5 - r * math.sin(theta)
        xs.append([x1, x2])
        ys.append(0.0)

    return xs, ys


def train_moons() -> None:
    print("=" * 60)
    print("2. Training MLP on Synthetic Two-Moons 2D Dataset")
    print("=" * 60)

    random.seed(42)
    n_samples = 60
    xs, ys = make_moons(n_samples=n_samples, noise=0.06)

    # Architecture: 2 inputs -> 8 hidden -> 8 hidden -> 1 output
    model = MLP(nin=2, nouts=[8, 8, 1], nonlin=True, nonlin_type="tanh")
    print(f"Model Architecture: {model}")
    print(f"Total Parameters: {len(model.parameters())}")

    learning_rate = 0.05
    epochs = 150

    for epoch in range(1, epochs + 1):
        # Forward pass
        ypred = [model(x) for x in xs]

        # MSE Loss
        loss: Value = Value(0.0)
        for y_target, y_pred in zip(ys, ypred):
            assert isinstance(y_pred, Value)
            loss = loss + (y_pred - y_target) ** 2
        loss = loss * (1.0 / len(ys))

        # Backward pass
        model.zero_grad()
        loss.backward()

        # SGD parameter update
        for p in model.parameters():
            p.data -= learning_rate * p.grad

        if epoch % 25 == 0 or epoch == 1:
            # Calculate accuracy
            correct = sum(
                1 for x, y in zip(xs, ys) if ((1.0 if model(x).data >= 0.5 else 0.0) == y)  # type: ignore[union-attr]
            )
            acc = (correct / len(ys)) * 100.0
            print(f"Epoch {epoch:3d}/{epochs} | Loss: {loss.data:.6f} | Accuracy: {acc:.1f}%")

    # Final ASCII Decision Grid
    print("\nDecision Boundary ASCII Map (x1 in [-1.5, 2.5], x2 in [-1.0, 1.5]):")
    grid_w, grid_h = 30, 12
    for row in range(grid_h):
        line = ""
        x2 = 1.5 - (row / (grid_h - 1)) * 2.5
        for col in range(grid_w):
            x1 = -1.5 + (col / (grid_w - 1)) * 4.0
            val = model([x1, x2]).data  # type: ignore[union-attr]
            char = "#" if val >= 0.5 else "."
            line += char
        print(f"  {line}")
    print("  Legend: '#' = Class 1, '.' = Class 0\n")


if __name__ == "__main__":
    train_xor()
    train_moons()
