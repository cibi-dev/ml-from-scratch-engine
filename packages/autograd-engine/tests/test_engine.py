"""Unit tests for the Autograd Engine core (Value, arithmetic, derivatives, diamond graphs)."""

import math
import pytest
from autograd.engine import Value
from tests.conftest import numerical_gradient, numerical_gradient_multi


def test_value_initialization_and_repr() -> None:
    v = Value(3.14, label="pi")
    assert v.data == 3.14
    assert v.grad == 0.0
    assert v.label == "pi"
    assert "Value(data=3.14, grad=0.0)" in repr(v)

    v_from_v = Value(v)
    assert v_from_v.data == 3.14

    v_int = Value(5)
    assert v_int.data == 5.0


def test_value_invalid_init() -> None:
    with pytest.raises(TypeError):
        Value("invalid_str")  # type: ignore[arg-type]


def test_add_values() -> None:
    a = Value(3.0)
    b = Value(4.0)
    c = a + b
    assert c.data == 7.0
    c.backward()
    assert a.grad == 1.0
    assert b.grad == 1.0


def test_radd() -> None:
    a = Value(2.5)
    c = 10.0 + a
    assert c.data == 12.5
    c.backward()
    assert a.grad == 1.0


def test_mul_values() -> None:
    a = Value(3.0)
    b = Value(-4.0)
    c = a * b
    assert c.data == -12.0
    c.backward()
    assert a.grad == -4.0
    assert b.grad == 3.0


def test_rmul() -> None:
    a = Value(4.0)
    c = 3.0 * a
    assert c.data == 12.0
    c.backward()
    assert a.grad == 3.0


def test_sub_values() -> None:
    a = Value(10.0)
    b = Value(3.0)
    c = a - b
    assert c.data == 7.0
    c.backward()
    assert a.grad == 1.0
    assert b.grad == -1.0


def test_rsub() -> None:
    a = Value(3.0)
    c = 10.0 - a
    assert c.data == 7.0
    c.backward()
    assert a.grad == -1.0


def test_neg() -> None:
    a = Value(5.0)
    b = -a
    assert b.data == -5.0
    b.backward()
    assert a.grad == -1.0


def test_truediv() -> None:
    a = Value(12.0)
    b = Value(4.0)
    c = a / b
    assert c.data == 3.0
    c.backward()
    # d/da (a/b) = 1/b = 1/4 = 0.25
    # d/db (a/b) = -a/b^2 = -12/16 = -0.75
    assert pytest.approx(a.grad, rel=1e-6) == 0.25
    assert pytest.approx(b.grad, rel=1e-6) == -0.75


def test_rtruediv() -> None:
    a = Value(2.0)
    c = 8.0 / a
    assert c.data == 4.0
    c.backward()
    # d/da (8/a) = -8 / a^2 = -8 / 4 = -2.0
    assert pytest.approx(a.grad, rel=1e-6) == -2.0


def test_pow_int() -> None:
    a = Value(3.0)
    c = a**3
    assert c.data == 27.0
    c.backward()
    # d/da (a^3) = 3 * a^2 = 27.0
    assert pytest.approx(a.grad, rel=1e-6) == 27.0


def test_pow_float() -> None:
    a = Value(4.0)
    c = a**0.5
    assert c.data == 2.0
    c.backward()
    # d/da (a^0.5) = 0.5 / sqrt(4) = 0.25
    assert pytest.approx(a.grad, rel=1e-6) == 0.25


def test_pow_negative() -> None:
    a = Value(2.0)
    c = a**-2
    assert c.data == 0.25
    c.backward()
    # d/da (a^-2) = -2 * a^-3 = -2 / 8 = -0.25
    assert pytest.approx(a.grad, rel=1e-6) == -0.25


def test_pow_invalid_type() -> None:
    a = Value(2.0)
    with pytest.raises(TypeError):
        _ = a ** "two"  # type: ignore[operator]


def test_relu() -> None:
    # Positive
    a = Value(3.0)
    c = a.relu()
    assert c.data == 3.0
    c.backward()
    assert a.grad == 1.0

    # Negative
    b = Value(-2.5)
    d = b.relu()
    assert d.data == 0.0
    d.backward()
    assert b.grad == 0.0

    # Zero (Strict Rule: subgradient at x=0 is strictly 0.0)
    z = Value(0.0)
    e = z.relu()
    assert e.data == 0.0
    e.backward()
    assert z.grad == 0.0


def test_tanh() -> None:
    a = Value(0.881373587019543)
    c = a.tanh()
    assert pytest.approx(c.data, rel=1e-5) == 0.70710678
    c.backward()
    # d/dx tanh(x) = 1 - tanh^2(x) = 1 - 0.5 = 0.5
    assert pytest.approx(a.grad, rel=1e-4) == 0.5


def test_exp_and_log() -> None:
    x = Value(2.0)
    e = x.exp()
    assert pytest.approx(e.data, rel=1e-6) == math.exp(2.0)
    e.backward()
    assert pytest.approx(x.grad, rel=1e-6) == math.exp(2.0)

    y = Value(5.0)
    ln_y = y.log()
    assert pytest.approx(ln_y.data, rel=1e-6) == math.log(5.0)
    ln_y.backward()
    assert pytest.approx(y.grad, rel=1e-6) == 0.2

    with pytest.raises(ValueError):
        Value(-1.0).log()


def test_diamond_graph_addition() -> None:
    # f(x) = x + x => df/dx = 2
    x = Value(3.0)
    y = x + x
    assert y.data == 6.0
    y.backward()
    assert x.grad == 2.0


def test_diamond_graph_multiplication() -> None:
    # f(x) = x * x => df/dx = 2x
    x = Value(5.0)
    y = x * x
    assert y.data == 25.0
    y.backward()
    assert x.grad == 10.0


def test_diamond_graph_quad_branch() -> None:
    # f(x) = (x * x) + (x * x) => df/dx = 4x
    x = Value(4.0)
    y = (x * x) + (x * x)
    assert y.data == 32.0
    y.backward()
    assert x.grad == 16.0


def test_diamond_graph_bivariate() -> None:
    # f(a, b) = a * b + a * b => df/da = 2b, df/db = 2a
    a = Value(3.0)
    b = Value(7.0)
    f = a * b + a * b
    assert f.data == 42.0
    f.backward()
    assert a.grad == 14.0
    assert b.grad == 6.0


def test_finite_differences_polynomial() -> None:
    def func(x: Value) -> Value:
        return 3.0 * (x**3) - 5.0 * (x**2) + 2.0 * x - 7.0

    x_val = 2.5
    x = Value(x_val)
    y = func(x)
    y.backward()

    num_grad = numerical_gradient(func, x_val)
    assert abs(x.grad - num_grad) < 1e-5


def test_finite_differences_rational() -> None:
    def func(x: Value) -> Value:
        return (2.0 * x + 1.0) / (x**2 + 3.0)

    x_val = 1.8
    x = Value(x_val)
    y = func(x)
    y.backward()

    num_grad = numerical_gradient(func, x_val)
    assert abs(x.grad - num_grad) < 1e-5


def test_finite_differences_tanh_composition() -> None:
    def func(x: Value) -> Value:
        return (2.0 * (x**2) + 1.0).tanh()

    x_val = 0.7
    x = Value(x_val)
    y = func(x)
    y.backward()

    num_grad = numerical_gradient(func, x_val)
    assert abs(x.grad - num_grad) < 1e-5


def test_finite_differences_multivariable() -> None:
    def func(vars: list[Value]) -> Value:
        x, y, z = vars[0], vars[1], vars[2]
        return (x * y + z.tanh()) / (x + y + 1.5)

    x_vals = [1.5, 2.0, -0.5]
    vars_val = [Value(v) for v in x_vals]
    out = func(vars_val)
    out.backward()

    num_grads = numerical_gradient_multi(func, x_vals)
    for i in range(3):
        assert abs(vars_val[i].grad - num_grads[i]) < 1e-5


def test_complex_dag_expression() -> None:
    # Karpathy's micrograd sanity test
    a = Value(-4.0)
    b = Value(2.0)
    c = a + b
    d = a * b + b**3
    c += c + 1
    c += 1 + c + (-a)
    d += d * 2 + (b + a).relu()
    d += 3 * d + (b - a).relu()
    e = c - d
    f = e**2
    g = f / 2.0
    g += 10.0 / f
    g.backward()

    # Numerical verification of a and b
    def f_eval(vars: list[Value]) -> Value:
        va, vb = vars[0], vars[1]
        vc = va + vb
        vd = va * vb + vb**3
        vc = vc + vc + 1
        vc = vc + 1 + vc + (-va)
        vd = vd + vd * 2 + (vb + va).relu()
        vd = vd + 3 * vd + (vb - va).relu()
        ve = vc - vd
        vf = ve**2
        vg = vf / 2.0
        vg = vg + 10.0 / vf
        return vg

    num_grads = numerical_gradient_multi(f_eval, [-4.0, 2.0])
    assert abs(a.grad - num_grads[0]) < 1e-4
    assert abs(b.grad - num_grads[1]) < 1e-4
