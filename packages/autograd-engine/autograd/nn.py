"""Autograd Neural Network: Modules, Neurons, Layers, and MLPs.

Provides PyTorch-like modular building blocks for constructing and training
neural networks from scratch on top of the Autograd engine.
"""

from __future__ import annotations

import random
from typing import Any, List, Sequence, Union

from autograd.engine import Value, ValueLike


class Module:
    """Base class for all neural network modules."""

    def zero_grad(self) -> None:
        """Reset gradients of all parameters to 0.0."""
        for p in self.parameters():
            p.grad = 0.0

    def parameters(self) -> list[Value]:
        """Return a list of all learnable Value parameters."""
        return []

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self.forward(*args, **kwargs)

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError


class Neuron(Module):
    """A single artificial neuron with linear combination and non-linear activation."""

    def __init__(
        self,
        nin: int,
        nonlin: bool = True,
        nonlin_type: str = "relu",
    ) -> None:
        if nin <= 0:
            raise ValueError(f"Number of inputs (nin) must be positive, got {nin}")
        if nonlin_type not in ("relu", "tanh"):
            raise ValueError(f"Unsupported nonlin_type: {nonlin_type}. Choose 'relu' or 'tanh'.")

        self.w: list[Value] = [Value(random.uniform(-1.0, 1.0)) for _ in range(nin)]
        self.b: Value = Value(0.0)
        self.nonlin: bool = nonlin
        self.nonlin_type: str = nonlin_type

    def forward(self, x: Sequence[ValueLike]) -> Value:
        if len(x) != len(self.w):
            raise ValueError(
                f"Neuron expected {len(self.w)} inputs, got {len(x)}"
            )
        # act = sum(w * x) + b
        act: Value = self.b
        for wi, xi in zip(self.w, x):
            act = act + (wi * xi)

        if not self.nonlin:
            return act
        if self.nonlin_type == "tanh":
            return act.tanh()
        return act.relu()

    def parameters(self) -> list[Value]:
        return self.w + [self.b]

    def __repr__(self) -> str:
        if not self.nonlin:
            act_name = "Linear"
        elif self.nonlin_type == "relu":
            act_name = "ReLU"
        else:
            act_name = "Tanh"
        return f"{act_name}Neuron({len(self.w)})"


class Layer(Module):
    """A fully connected neural network layer comprising multiple Neurons."""

    def __init__(
        self,
        nin: int,
        nout: int,
        nonlin: bool = True,
        nonlin_type: str = "relu",
    ) -> None:
        if nout <= 0:
            raise ValueError(f"Number of outputs (nout) must be positive, got {nout}")
        self.neurons: list[Neuron] = [
            Neuron(nin, nonlin=nonlin, nonlin_type=nonlin_type) for _ in range(nout)
        ]

    def forward(self, x: Sequence[ValueLike]) -> Union[list[Value], Value]:
        outs = [n(x) for n in self.neurons]
        return outs[0] if len(outs) == 1 else outs

    def parameters(self) -> list[Value]:
        return [p for n in self.neurons for p in n.parameters()]

    def __repr__(self) -> str:
        return f"Layer of [{', '.join(str(n) for n in self.neurons)}]"


class MLP(Module):
    """Multi-Layer Perceptron (feed-forward neural network)."""

    def __init__(
        self,
        nin: int,
        nouts: Sequence[int],
        nonlin: bool = True,
        nonlin_type: str = "relu",
    ) -> None:
        if not nouts:
            raise ValueError("MLP requires at least one output layer dimension")

        sz = [nin] + list(nouts)
        self.layers: list[Layer] = []
        for i in range(len(nouts)):
            # Last layer defaults to linear unless specifically required
            is_last = i == len(nouts) - 1
            layer_nonlin = False if is_last else nonlin
            self.layers.append(
                Layer(sz[i], sz[i + 1], nonlin=layer_nonlin, nonlin_type=nonlin_type)
            )

    def forward(self, x: Sequence[ValueLike]) -> Union[list[Value], Value]:
        out: Any = x
        for layer in self.layers:
            inp = out if isinstance(out, (list, tuple)) else [out]
            out = layer(inp)
        return out  # type: ignore[no-any-return]

    def parameters(self) -> list[Value]:
        return [p for layer in self.layers for p in layer.parameters()]

    def __repr__(self) -> str:
        return f"MLP of [{', '.join(str(layer) for layer in self.layers)}]"
