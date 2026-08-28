"""Unit tests for neural network components (Neuron, Layer, MLP, XOR training, parameter counts)."""

import random
import pytest
from autograd.engine import Value
from autograd.nn import Module, Neuron, Layer, MLP


def test_base_module() -> None:
    mod = Module()
    assert mod.parameters() == []
    mod.zero_grad()  # should not raise
    with pytest.raises(NotImplementedError):
        mod.forward()


def test_neuron_init_and_parameters() -> None:
    n = Neuron(3, nonlin=True, nonlin_type="relu")
    params = n.parameters()
    assert len(params) == 4  # 3 weights + 1 bias
    assert isinstance(n.b, Value)
    assert len(n.w) == 3
    for w in n.w:
        assert isinstance(w, Value)
    assert "ReLUNeuron(3)" in repr(n)


def test_neuron_linear_and_tanh_repr() -> None:
    n_lin = Neuron(2, nonlin=False)
    assert "LinearNeuron(2)" in repr(n_lin)

    n_tanh = Neuron(2, nonlin=True, nonlin_type="tanh")
    assert "TanhNeuron(2)" in repr(n_tanh)


def test_neuron_invalid_args() -> None:
    with pytest.raises(ValueError, match="nin.*must be positive"):
        Neuron(0)

    with pytest.raises(ValueError, match="Unsupported nonlin_type"):
        Neuron(2, nonlin_type="sigmoid")  # type: ignore[arg-type]


def test_neuron_input_dimension_mismatch() -> None:
    n = Neuron(3)
    with pytest.raises(ValueError, match="Neuron expected 3 inputs, got 2"):
        n([1.0, 2.0])


def test_neuron_forward_and_backward() -> None:
    n = Neuron(2, nonlin=False)
    n.w[0].data = 2.0
    n.w[1].data = -3.0
    n.b.data = 1.0

    x = [Value(0.5), Value(1.5)]
    out = n(x)
    # 2*0.5 + (-3)*1.5 + 1.0 = 1.0 - 4.5 + 1.0 = -2.5
    assert out.data == -2.5

    out.backward()
    assert n.w[0].grad == 0.5
    assert n.w[1].grad == 1.5
    assert n.b.grad == 1.0
    assert x[0].grad == 2.0
    assert x[1].grad == -3.0


def test_layer_init_and_parameters() -> None:
    layer = Layer(nin=3, nout=4)
    assert len(layer.neurons) == 4
    # 4 neurons * (3 weights + 1 bias) = 16 parameters
    assert len(layer.parameters()) == 16
    assert "Layer of" in repr(layer)


def test_layer_invalid_nout() -> None:
    with pytest.raises(ValueError, match="nout.*must be positive"):
        Layer(3, 0)


def test_layer_single_output_returns_value() -> None:
    layer = Layer(nin=2, nout=1)
    out = layer([1.0, 2.0])
    assert isinstance(out, Value)


def test_layer_multi_output_returns_list() -> None:
    layer = Layer(nin=2, nout=3)
    out = layer([1.0, 2.0])
    assert isinstance(out, list)
    assert len(out) == 3
    for v in out:
        assert isinstance(v, Value)


def test_mlp_parameter_counting() -> None:
    mlp = MLP(nin=2, nouts=[4, 4, 1])
    # Layer 1: 2 -> 4 = 4*(2+1) = 12
    # Layer 2: 4 -> 4 = 4*(4+1) = 20
    # Layer 3: 4 -> 1 = 1*(4+1) = 5
    # Total = 12 + 20 + 5 = 37
    assert len(mlp.parameters()) == 37
    assert len(mlp.layers) == 3
    assert "MLP of" in repr(mlp)


def test_mlp_single_layer_parameter_counting() -> None:
    mlp = MLP(nin=3, nouts=[2])
    # 2 * (3 + 1) = 8
    assert len(mlp.parameters()) == 8


def test_mlp_invalid_nouts() -> None:
    with pytest.raises(ValueError, match="requires at least one output layer"):
        MLP(2, [])


def test_mlp_zero_grad() -> None:
    mlp = MLP(2, [3, 1])
    out = mlp([1.0, 2.0])
    assert isinstance(out, Value)
    out.backward()

    # Verify gradients are non-zero after backward
    assert any(p.grad != 0.0 for p in mlp.parameters())

    mlp.zero_grad()
    assert all(p.grad == 0.0 for p in mlp.parameters())


def test_xor_classification_training_convergence() -> None:
    """Train MLP on XOR dataset and ensure loss < 0.1 and perfect separation."""
    random.seed(1337)
    mlp = MLP(2, [4, 4, 1], nonlin_type="tanh")

    xs = [
        [0.0, 0.0],
        [0.0, 1.0],
        [1.0, 0.0],
        [1.0, 1.0],
    ]
    # For tanh output, we can use target values -1.0 and 1.0 or 0.0 and 1.0 with MSE
    ys = [0.0, 1.0, 1.0, 0.0]

    learning_rate = 0.1
    final_loss = float("inf")

    for k in range(300):
        # Forward pass
        ypred = [mlp(x) for x in xs]
        # MSE loss
        loss: Value = Value(0.0)
        for ygt, yp in zip(ys, ypred):
            assert isinstance(yp, Value)
            loss = loss + (yp - ygt) ** 2
        loss = loss * (1.0 / len(ys))

        final_loss = loss.data

        # Backward pass
        mlp.zero_grad()
        loss.backward()

        # Gradient descent step
        for p in mlp.parameters():
            p.data -= learning_rate * p.grad

    # Verify convergence requirements: loss < 0.1
    assert final_loss < 0.1, f"Expected loss < 0.1, got {final_loss}"

    # Verify classification accuracy
    for x, y in zip(xs, ys):
        pred_val = mlp(x).data  # type: ignore[union-attr]
        pred_class = 1.0 if pred_val >= 0.5 else 0.0
        assert pred_class == y, f"Failed on input {x}: expected {y}, got {pred_val}"


def test_linear_regression_convergence() -> None:
    """Train a single neuron to learn y = 2*x1 - 3*x2 + 0.5."""
    random.seed(42)
    neuron = Neuron(2, nonlin=False)

    xs = [
        [1.0, 1.0],
        [1.0, -1.0],
        [-1.0, 1.0],
        [-1.0, -1.0],
        [2.0, 0.5],
    ]
    ys = [2.0 * x[0] - 3.0 * x[1] + 0.5 for x in xs]

    lr = 0.05
    for _ in range(100):
        ypred = [neuron(x) for x in xs]
        loss: Value = Value(0.0)
        for ygt, yp in zip(ys, ypred):
            loss = loss + (yp - ygt) ** 2
        loss = loss * (1.0 / len(ys))

        neuron.zero_grad()
        loss.backward()

        for p in neuron.parameters():
            p.data -= lr * p.grad

    assert loss.data < 1e-4
    assert pytest.approx(neuron.w[0].data, abs=0.05) == 2.0
    assert pytest.approx(neuron.w[1].data, abs=0.05) == -3.0
    assert pytest.approx(neuron.b.data, abs=0.05) == 0.5
