"""Autograd Engine: Reverse-mode automatic differentiation core.

Implements scalar-valued automatic differentiation with iterative topological sorting
and computational graph construction.
"""

from __future__ import annotations

import math
from typing import Callable, Iterable, Set, Tuple, Union

Number = Union[int, float]
ValueLike = Union["Value", Number]


class Value:
    """Represents a scalar value in a computational graph with gradient tracking."""

    __slots__ = ("data", "grad", "_prev", "_op", "_backward", "label")

    def __init__(
        self,
        data: ValueLike,
        _children: Iterable[Value] = (),
        _op: str = "",
        label: str = "",
    ) -> None:
        if isinstance(data, Value):
            self.data: float = float(data.data)
        elif isinstance(data, (int, float)):
            self.data = float(data)
        else:
            raise TypeError(f"Value data must be int, float, or Value, got {type(data).__name__}")

        self.grad: float = 0.0
        self._backward: Callable[[], None] = lambda: None
        self._prev: Set[Value] = set(_children)
        self._op: str = _op
        self.label: str = label

    def __repr__(self) -> str:
        return f"Value(data={self.data}, grad={self.grad})"

    def __add__(self, other: ValueLike) -> Value:
        other_val = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other_val.data, (self, other_val), "+")

        def _backward() -> None:
            self.grad += 1.0 * out.grad
            other_val.grad += 1.0 * out.grad

        out._backward = _backward
        return out

    def __radd__(self, other: ValueLike) -> Value:
        return self + other

    def __mul__(self, other: ValueLike) -> Value:
        other_val = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other_val.data, (self, other_val), "*")

        def _backward() -> None:
            self.grad += other_val.data * out.grad
            other_val.grad += self.data * out.grad

        out._backward = _backward
        return out

    def __rmul__(self, other: ValueLike) -> Value:
        return self * other

    def __pow__(self, other: Number) -> Value:
        if not isinstance(other, (int, float)):
            raise TypeError(f"Power exponent must be int or float, got {type(other).__name__}")

        if self.data == 0.0 and other < 0:
            raise ZeroDivisionError("0.0 cannot be raised to a negative power")

        out = Value(self.data**other, (self,), f"**{other}")

        def _backward() -> None:
            if self.data == 0.0 and other == 1:
                self.grad += 1.0 * out.grad
            elif self.data == 0.0 and other > 1:
                self.grad += 0.0
            else:
                self.grad += (other * (self.data ** (other - 1))) * out.grad

        out._backward = _backward
        return out

    def __neg__(self) -> Value:
        return self * -1.0

    def __sub__(self, other: ValueLike) -> Value:
        other_val = other if isinstance(other, Value) else Value(other)
        return self + (-other_val)

    def __rsub__(self, other: ValueLike) -> Value:
        other_val = other if isinstance(other, Value) else Value(other)
        return other_val + (-self)

    def __truediv__(self, other: ValueLike) -> Value:
        other_val = other if isinstance(other, Value) else Value(other)
        if other_val.data == 0.0:
            raise ZeroDivisionError("division by zero")
        return self * (other_val**-1.0)

    def __rtruediv__(self, other: ValueLike) -> Value:
        if self.data == 0.0:
            raise ZeroDivisionError("division by zero")
        other_val = other if isinstance(other, Value) else Value(other)
        return other_val * (self**-1.0)

    def relu(self) -> Value:
        """Rectified Linear Unit activation function."""
        out = Value(self.data if self.data > 0.0 else 0.0, (self,), "ReLU")

        def _backward() -> None:
            self.grad += (1.0 if self.data > 0.0 else 0.0) * out.grad

        out._backward = _backward
        return out

    def tanh(self) -> Value:
        """Hyperbolic tangent activation function."""
        t = math.tanh(self.data)
        out = Value(t, (self,), "tanh")

        def _backward() -> None:
            self.grad += (1.0 - t**2) * out.grad

        out._backward = _backward
        return out

    def exp(self) -> Value:
        """Exponential function e^x."""
        x = self.data
        out = Value(math.exp(x), (self,), "exp")

        def _backward() -> None:
            self.grad += out.data * out.grad

        out._backward = _backward
        return out

    def log(self) -> Value:
        """Natural logarithm ln(x)."""
        if self.data <= 0.0:
            raise ValueError(f"log domain error: input must be positive, got {self.data}")
        out = Value(math.log(self.data), (self,), "log")

        def _backward() -> None:
            self.grad += (1.0 / self.data) * out.grad

        out._backward = _backward
        return out

    def backward(self) -> None:
        """Executes reverse-mode autodiff using iterative topological sorting.

        Builds the topological ordering iteratively with an explicit stack to prevent
        RecursionError on very deep computational graphs (2000+ nodes).
        """
        topo: list[Value] = []
        visited: set[Value] = set()
        stack: list[tuple[Value, bool]] = [(self, False)]

        while stack:
            node, processed = stack.pop()
            if processed:
                topo.append(node)
            else:
                if node not in visited:
                    visited.add(node)
                    stack.append((node, True))
                    for child in node._prev:
                        if child not in visited:
                            stack.append((child, False))

        self.grad = 1.0
        for node in reversed(topo):
            node._backward()
