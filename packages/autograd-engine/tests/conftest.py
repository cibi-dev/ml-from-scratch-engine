"""Shared test fixtures and numerical gradient checking helpers."""

from __future__ import annotations

import random
from typing import Callable, List, Tuple
import pytest

from autograd.engine import Value


@pytest.fixture(autouse=True)
def set_seed() -> None:
    """Ensure determinism across tests."""
    random.seed(42)


def numerical_gradient(
    f: Callable[[Value], Value],
    x_val: float,
    eps: float = 1e-6,
) -> float:
    """Computes numerical derivative df/dx at x_val using central differences:
    (f(x + eps) - f(x - eps)) / (2 * eps)
    """
    x_plus = Value(x_val + eps)
    y_plus = f(x_plus)

    x_minus = Value(x_val - eps)
    y_minus = f(x_minus)

    return (y_plus.data - y_minus.data) / (2.0 * eps)


def numerical_gradient_multi(
    f: Callable[[List[Value]], Value],
    x_vals: List[float],
    eps: float = 1e-6,
) -> List[float]:
    """Computes numerical partial derivatives for multivariable function f: R^n -> R."""
    grads = []
    for i in range(len(x_vals)):
        x_plus_list = [Value(val + (eps if j == i else 0.0)) for j, val in enumerate(x_vals)]
        y_plus = f(x_plus_list)

        x_minus_list = [Value(val - (eps if j == i else 0.0)) for j, val in enumerate(x_vals)]
        y_minus = f(x_minus_list)

        grad_i = (y_plus.data - y_minus.data) / (2.0 * eps)
        grads.append(grad_i)
    return grads
