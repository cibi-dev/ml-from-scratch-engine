"""Security hardening, stress tests, and numerical edge-case verification."""

import sys
import pytest
from autograd.engine import Value
from autograd.nn import MLP, Neuron


def test_deep_linear_graph_stress_no_recursion_error() -> None:
    """Stress test with 2500+ sequential nodes to ensure iterative DFS prevents RecursionError."""
    # Temporarily set recursion limit low (e.g., 200) to strictly prove DFS is iterative
    orig_limit = sys.getrecursionlimit()
    sys.setrecursionlimit(200)
    try:
        x = Value(1.0)
        curr = x
        depth = 2500
        for _ in range(depth):
            curr = curr + 0.001

        assert pytest.approx(curr.data, rel=1e-5) == 1.0 + (depth * 0.001)
        curr.backward()
        assert x.grad == 1.0
    finally:
        sys.setrecursionlimit(orig_limit)


def test_deep_diamond_graph_stress() -> None:
    """Deep diamond computational graph with 2000+ operations."""
    orig_limit = sys.getrecursionlimit()
    sys.setrecursionlimit(200)
    try:
        x = Value(1.0001)
        curr = x
        depth = 1200
        for _ in range(depth):
            curr = (curr + curr) * 0.5

        assert pytest.approx(curr.data, rel=1e-3) == 1.0001
        curr.backward()
        # d/dx ((2*x)*0.5)^N = (1)^N = 1.0
        assert pytest.approx(x.grad, rel=1e-3) == 1.0
    finally:
        sys.setrecursionlimit(orig_limit)


def test_division_by_zero_exceptions() -> None:
    """Strictly verify division by zero raises ZeroDivisionError."""
    # Value / Value(0)
    with pytest.raises(ZeroDivisionError):
        _ = Value(1.0) / Value(0.0)

    # Value / 0.0
    with pytest.raises(ZeroDivisionError):
        _ = Value(5.0) / 0.0

    # Value / 0
    with pytest.raises(ZeroDivisionError):
        _ = Value(5.0) / 0

    # 1.0 / Value(0)
    with pytest.raises(ZeroDivisionError):
        _ = 10.0 / Value(0.0)

    # Value(0) / Value(0)
    with pytest.raises(ZeroDivisionError):
        _ = Value(0.0) / Value(0.0)

    # Value(0) ** -1
    with pytest.raises(ZeroDivisionError):
        _ = Value(0.0) ** -1

    # Value(0) ** -2.5
    with pytest.raises(ZeroDivisionError):
        _ = Value(0.0) ** -2.5


def test_zero_to_positive_powers() -> None:
    # 0.0 ** 1 => 0.0, grad => 1.0
    z1 = Value(0.0)
    out1 = z1**1
    assert out1.data == 0.0
    out1.backward()
    assert z1.grad == 1.0

    # 0.0 ** 2 => 0.0, grad => 0.0
    z2 = Value(0.0)
    out2 = z2**2
    assert out2.data == 0.0
    out2.backward()
    assert z2.grad == 0.0


def test_relu_subgradient_at_zero_strictly_zero() -> None:
    """Contract: ReLU subgradient at x = 0 is strictly 0.0."""
    z = Value(0.0)
    out = z.relu()
    assert out.data == 0.0
    out.backward()
    assert z.grad == 0.0


def test_tanh_extreme_values_numerical_stability() -> None:
    """Extreme values in tanh must not raise OverflowError and saturate cleanly."""
    large_pos = Value(1000.0)
    out_pos = large_pos.tanh()
    assert out_pos.data == 1.0
    out_pos.backward()
    assert large_pos.grad == 0.0

    large_neg = Value(-1000.0)
    out_neg = large_neg.tanh()
    assert out_neg.data == -1.0
    out_neg.backward()
    assert large_neg.grad == 0.0


def test_zero_grad_isolation_and_repeatability() -> None:
    """Verify zero_grad resets gradients without affecting subsequent iterations."""
    mlp = MLP(2, [4, 1])
    x = [0.5, -0.5]

    # Iteration 1
    out1 = mlp(x)
    assert isinstance(out1, Value)
    out1.backward()
    grads1 = [p.grad for p in mlp.parameters()]
    assert any(g != 0.0 for g in grads1)

    # Reset
    mlp.zero_grad()
    assert all(p.grad == 0.0 for p in mlp.parameters())

    # Iteration 2
    out2 = mlp(x)
    assert isinstance(out2, Value)
    out2.backward()
    grads2 = [p.grad for p in mlp.parameters()]

    # Gradients must be strictly identical across clean runs
    assert grads1 == grads2


def test_gradient_accumulation_without_zero_grad() -> None:
    """Calling backward multiple times without zero_grad accumulates gradients via +=."""
    x = Value(3.0)
    y = x * x  # dy/dx = 2*x = 6.0

    y.backward()
    assert x.grad == 6.0

    # Calling backward again without zeroing grad
    y.backward()
    assert x.grad == 12.0

    y.backward()
    assert x.grad == 18.0


def test_isolated_module_parameters() -> None:
    """Ensure modules have distinct parameter objects and no shared state leakage."""
    m1 = Neuron(2)
    m2 = Neuron(2)

    assert m1.parameters() is not m2.parameters()
    for p1, p2 in zip(m1.parameters(), m2.parameters()):
        assert p1 is not p2

    m1.w[0].grad = 5.0
    m2.zero_grad()
    assert m1.w[0].grad == 5.0
    assert m2.w[0].grad == 0.0
