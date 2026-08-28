# Autograd Engine (Micrograd From Scratch)

A lightweight, pure Python reverse-mode automatic differentiation (autograd) engine and neural network library built strictly from scratch with zero runtime dependencies.

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Type Checked: mypy strict](https://img.shields.io/badge/mypy-strict-brightgreen.svg)](https://mypy-lang.org/)
[![Testing: pytest](https://img.shields.io/badge/pytest-8.0%2B-blueviolet.svg)](https://pytest.org/)

---

## 🌟 Key Features

- **Zero Runtime Dependencies**: Standard library Python only (`math`, `random`, `typing`).
- **Reverse-Mode Automatic Differentiation**: Scalar-level DAG tracking with exact multivariate chain rule gradient accumulation (`+=`).
- **Stack-Safe Iterative DFS**: Topological sorting uses an explicit iterative stack (post-order DFS), eliminating `RecursionError` on deep computational graphs (tested on 2000+ sequential DAG nodes).
- **PyTorch-like Neural Network API**: Modular abstractions (`Module`, `Neuron`, `Layer`, `MLP`) with `parameters()` and `zero_grad()`.
- **Numerical Verification**: Automated finite-difference gradient checking ($O(\epsilon^2)$ central differences) with relative tolerance $< 10^{-5}$.
- **Security Hardened**: Guarded against division by zero, negative zero-powers, extreme tanh inputs, and parameter leakage.

---

## 📐 Computational Graph & Reverse-Mode Autodiff

### ASCII DAG Computation Flow

```text
        [ a = 2.0 ]                [ b = -3.0 ]
             \                          /
              \                        /
               \                      /
                v                    v
              +------------------------+
              |      * (mul node)      |
              | c = a * b  = -6.0      |
              | dc/da = -3.0, dc/db = 2|
              +------------------------+
                          |
                          | [ c ]
                          v
              +------------------------+
              |      + (add node)      | <--- [ d = 10.0 ]
              | e = c + d  = 4.0       |
              | de/dc = 1.0, de/dd = 1 |
              +------------------------+
                          |
                          | [ e ]
                          v
              +------------------------+
              |     tanh (activation)  |
              | L = tanh(e) ≈ 0.9993   |
              | dL/de = 1 - tanh^2(e)  |
              +------------------------+
                          |
                   [ L.backward() ]
             (Gradients flow backwards)
```

---

## 🔬 Mathematical Foundations

### 1. Reverse-Mode Automatic Differentiation
For a scalar loss $L \in \mathbb{R}$ computed over intermediate computational nodes $v_i$, the chain rule computes the adjoint (gradient) $\bar{v}_i = \frac{\partial L}{\partial v_i}$.

When a node $v_i$ fans out to multiple downstream consumer nodes $\{v_j : v_i \in \text{Parents}(v_j)\}$, the multivariate chain rule dictates:

$$\frac{\partial L}{\partial v_i} = \sum_{j \in \text{Consumers}(v_i)} \frac{\partial L}{\partial v_j} \cdot \frac{\partial v_j}{\partial v_i}$$

In code, this necessitates **gradient accumulation** via `+=`:
```python
self.grad += local_derivative * out.grad
```

### 2. Primitive Operations & Local Derivatives

| Operation | Forward Evaluation | Local Derivative $\frac{\partial \text{out}}{\partial \text{self}}$ | Local Derivative $\frac{\partial \text{out}}{\partial \text{other}}$ |
| :--- | :--- | :--- | :--- |
| **Addition** ($a + b$) | $\text{out} = a + b$ | $1.0$ | $1.0$ |
| **Multiplication** ($a \cdot b$) | $\text{out} = a \cdot b$ | $b$ | $a$ |
| **Power** ($a^n$) | $\text{out} = a^n$ | $n \cdot a^{n-1}$ | N/A ($n \in \mathbb{R}$) |
| **Division** ($a / b$) | $\text{out} = a \cdot b^{-1}$ | $b^{-1}$ | $-a \cdot b^{-2}$ |
| **ReLU** ($\max(0, a)$) | $\text{out} = \begin{cases} a & a > 0 \\ 0 & a \le 0 \end{cases}$ | $\begin{cases} 1.0 & a > 0 \\ 0.0 & a \le 0 \end{cases}$ | N/A |
| **Tanh** ($\tanh(a)$) | $\text{out} = \frac{e^a - e^{-a}}{e^a + e^{-a}}$ | $1.0 - \text{out}^2$ | N/A |

### 3. Finite-Difference Numerical Gradient Checking
To guarantee the analytical autograd implementation is exact, central finite differences are used:

$$\frac{df}{dx} \approx \frac{f(x + \epsilon) - f(x - \epsilon)}{2\epsilon} + \mathcal{O}(\epsilon^2)$$

With $\epsilon = 10^{-6}$, the truncation error is on the order of $10^{-12}$, allowing verification within tolerance $|g_{\text{auto}} - g_{\text{num}}| < 10^{-5}$.

---

## 🔒 Security & Robustness Considerations

1. **Iterative Stack vs. Call Stack Overflow**:
   Standard recursive topological sorting fails with `RecursionError` on deep networks or unrolled recurrent/sequential computations when $N > \text{sys.getrecursionlimit()}$ (~1000). Our `backward()` builds the topological ordering iteratively with an explicit heap-allocated stack `list[tuple[Value, bool]]`.
2. **Division by Zero Protection**:
   `Value(0) ** -1`, `x / Value(0)`, and `x / 0` explicitly check operands and raise standard `ZeroDivisionError` cleanly.
3. **Subgradient Consistency**:
   At non-differentiable points such as ReLU at $x = 0$, the subgradient is explicitly defined as $0.0$.
4. **Gradient State Isolation**:
   `Module.zero_grad()` resets all leaf parameters' `.grad = 0.0`, preventing gradient accumulation leakage between training batches.

---

## 🚀 Quickstart

### Installation & Environment Setup

```bash
# Clone or navigate to target directory
git clone https://github.com/cibi-dev/autograd-engine.git
cd autograd-engine

# Ensure uv is in PATH
export PATH="$HOME/.local/bin:$PATH"

# Run tests
uv run pytest -v --tb=short

# Run strict type checking
uv run mypy autograd/ --strict
```

### Basic Autograd Example

```python
from autograd import Value

a = Value(2.0, label="a")
b = Value(-3.0, label="b")
c = a * b + 10.0
d = c.tanh()

d.backward()

print(f"Result: {d.data:.4f}")
print(f"da/dL: {a.grad:.4f}")
print(f"db/dL: {b.grad:.4f}")
```

### Training a Neural Network (MLP)

```python
from autograd import MLP, Value

# 2 inputs, two hidden layers of 4 neurons, 1 output
model = MLP(nin=2, nouts=[4, 4, 1], nonlin_type="tanh")

# XOR data
xs = [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]]
ys = [0.0, 1.0, 1.0, 0.0]

# Training loop
for epoch in range(200):
    # Forward pass
    ypred = [model(x) for x in xs]
    loss = sum(((yp - yg) ** 2 for yg, yp in zip(ys, ypred)), Value(0.0)) * (1.0 / len(ys))
    
    # Backward pass
    model.zero_grad()
    loss.backward()
    
    # Optimizer step (SGD)
    for p in model.parameters():
        p.data -= 0.1 * p.grad

print(f"Final Loss: {loss.data:.6f}")
```

---

## 📦 Running the Built-in Examples

A standalone demonstration script training on XOR and a synthetic 2D Two-Moons dataset:

```bash
uv run python examples/xor_and_moons.py
```

---

## 📂 Project Structure

```text
autograd-engine/
├── autograd/
│   ├── __init__.py      # Exports Value, Module, Neuron, Layer, MLP
│   ├── engine.py        # Core Value class, ops, iterative topological sort backward()
│   └── nn.py            # Neural network modules (Neuron, Layer, MLP)
├── tests/
│   ├── __init__.py
│   ├── conftest.py      # Fixtures & finite-difference gradient checkers
│   ├── test_engine.py   # Arithmetic, derivatives, diamond DAG accumulation
│   ├── test_nn.py       # Module, Neuron, Layer, MLP, XOR convergence
│   └── test_security.py # 2000+ deep graph stress, division by zero, stability
├── examples/
│   └── xor_and_moons.py # Runnable XOR and Two-Moons training script
├── pyproject.toml       # Hatchling configuration, strict mypy & pytest settings
├── LICENSE              # MIT License (c) 2026 cibi-dev
├── README.md            # Comprehensive architecture documentation
└── .gitignore
```

---

## 📜 License

Distributed under the [MIT License](LICENSE). Copyright &copy; 2026 `cibi-dev`.
