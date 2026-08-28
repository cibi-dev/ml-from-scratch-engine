# Hardened Pure NumPy Vector Database (`numpy-vectordb`)

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![NumPy 2.x](https://img.shields.io/badge/NumPy-2.x-013243?style=flat# Hardened Pure NumPy Vector Database (`numpy-vectordb`)logo=numpy# Hardened Pure NumPy Vector Database (`numpy-vectordb`)logoColor=white)](https://numpy.org/)
[![Type Checked: mypy strict](https://img.shields.io/badge/mypy-strict-brightgreen.svg)](https://mypy-lang.org/)
[![Tests: 91 Passed](https://img.shields.io/badge/tests-91%20passed-brightgreen.svg)](https://pytest.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

A high-performance, hardened in-memory vector database built from scratch in pure Python using contiguous **NumPy 2.x** buffers, **BLAS GEMV/GEMM** vectorization, $O(N + K \log K)$ selection via `np.argpartition`, and comprehensive security mitigations against deserialization (CWE-502) and path traversal (CWE-22) vulnerabilities.

---

## 🏛️ System Architecture

```text
                               +-------------------------------------------------------+
                               |                    VectorDB Client                    |
                               +---------------------------+---------------------------+
                                                           |
                                                [ Input Validation ]
                                                • Finite Check (np.isfinite)
                                                • Zero-Norm Guard (norm >= 1e-12)
                                                • Dim Check (1 <= dim <= MAX_DIM)
                                                • Doc ID Non-Empty String Check
                                                           |
                                                           v
+-------------------------------------------------------------------------------------------------------------------------+
|                                              VectorDB Memory Controller                                                 |
|                                                                                                                         |
|    +--------------------------+         threading.RLock         +--------------------------------------------------+    |
|    |      _id_to_idx Map      | <=============================> |                 _idx_to_id List                  |    |
|    |  {"doc_1": 0, "doc_2": 1}|                                 |               ["doc_1", "doc_2"]                 |    |
|    +--------------------------+                                 +--------------------------------------------------+    |
|                 |                                                                        |                              |
|                 v                                                                        v                              |
|    +---------------------------------------------------------------------------------------------------------------+    |
|    |                         Contiguous 2D Float32 Buffer (Geometric Capacity Doubling)                           |    |
|    |  Row 0: [  0.852,  0.124,  0.432,  ... ,  0.091 ]  <--- Active Document 1                                     |    |
|    |  Row 1: [  0.112,  0.941,  0.223,  ... ,  0.314 ]  <--- Active Document 2                                     |    |
|    |  Row 2: [  0.000,  0.000,  0.000,  ... ,  0.000 ]  <--- Unallocated Capacity Slot                             |    |
|    +---------------------------------------------------------------------------------------------------------------+    |
|                                                          |                                                              |
|                                                          | (O(1) Swap-and-Pop Defragmentation)                          |
+----------------------------------------------------------+--------------------------------------------------------------+
                                                           |
                                                           v
+-------------------------------------------------------------------------------------------------------------------------+
|                                           BLAS Similarity & Top-K Engine                                                |
|                                                                                                                         |
|    1. Normalization:      M_norm = M / ||M||_2 ,  q_norm = q / ||q||_2                                                  |
|    2. BLAS GEMV / GEMM:   scores = M_norm @ q_norm (Cosine Similarity / Dot Product)                                   |
|    3. Fast Top-K:         np.argpartition(-scores, K - 1)[:K] + np.argsort (O(N + K log K))                             |
+-------------------------------------------------------------------------------------------------------------------------+
                                                           |
                                                           v
+-------------------------------------------------------------------------------------------------------------------------+
|                                              Hardened Disk Persistence                                                  |
|                                                                                                                         |
|    • Vector Archive:     <collection>.npz        (np.savez_compressed, allow_pickle=False strictly enforced)           |
|    • Metadata Sidecar:   <collection>.meta.json  (JSON schema metadata, path traversal containment verified)            |
+-------------------------------------------------------------------------------------------------------------------------+
```

---

## ⚡ Mathematical Formulation & BLAS Acceleration

### 1. Cosine Similarity via BLAS Level 2 GEMV
Given a query vector $\mathbf{q} \in \mathbb{R}^d$ and a contiguous matrix of active vectors $\mathbf{M} \in \mathbb{R}^{N \times d}$, cosine similarity is computed in vectorized C/Fortran speed via BLAS matrix-vector multiplication:

$$\hat{\mathbf{q}} = \frac{\mathbf{q}}{\|\mathbf{q}\|_2}, \quad \hat{\mathbf{M}}_{i,:} = \frac{\mathbf{M}_{i,:}}{\|\mathbf{M}_{i,:}\|_2}$$

$$\mathbf{S} = \hat{\mathbf{M}} \hat{\mathbf{q}} \in [-1.0, 1.0]^N$$

### 2. Euclidean ($L_2$) Distance
$$\mathbf{D}_i = \|\mathbf{M}_{i,:} - \mathbf{q}\|_2 = \sqrt{\sum_{j=1}^d (\mathbf{M}_{i,j} - \mathbf{q}_j)^2}$$

### 3. $O(N + K \log K)$ Top-K Partition Selection
Traditional full sorting requires $O(N \log N)$ operations, which degrades significantly as the corpus size $N$ grows. `numpy-vectordb` utilizes `np.argpartition` (Introselect algorithm) to isolate the top-$K$ candidates in $O(N)$ linear time, followed by sorting only the top-$K$ subset in $O(K \log K)$:

```python
# O(N) candidate isolation
partition_indices = np.argpartition(-scores, k - 1)[:k]
# O(K log K) top-K sorting
sub_order = np.argsort(-scores[partition_indices])
top_k_indices = partition_indices[sub_order]
```

---

## 🛡️ Security Hardening & Vulnerability Mitigations

| Vulnerability / Threat | Mitigation Mechanism | CWE / Standard |
|---|---|---|
| **Insecure Deserialization** | Strict enforcement of `allow_pickle=False` in `np.load` and document IDs stored strictly as NumPy unicode string arrays (`np.str_`). | **CWE-502** |
| **Path Traversal** | Collection names validated against `^[a-zA-Z0-9_-]+$` pattern; storage paths validated with `os.path.commonpath` containment checks to prevent directory breakout (`../../../`). | **CWE-22** |
| **Non-Finite Vector Injection** | All vector inputs are inspected with `np.isfinite()`. Arrays containing `NaN`, `+Inf`, or `-Inf` are immediately rejected. | Robust Input Sanitization |
| **Zero-Norm & Underflow** | Queries and vectors with magnitude $\|\mathbf{v}\|_2 < 10^{-12}$ are rejected to prevent division-by-zero or numerical instability. | Numerical Guardrails |
| **Buffer Exhaustion (DoS)** | Hard-bounded resource caps: `MAX_DIM = 65,536` and configurable `MAX_VECTORS` limits prevent memory exhaustion attacks. | Resource Management |
| **Race Conditions** | Full thread safety on all read, write, and defragmentation operations via re-entrant locks (`threading.RLock`). | Thread Synchronization |

---

## 🚀 Quickstart

### Installation
```bash
# Using uv
uv pip install -e .
```

### Basic Usage
```python
from numpy_vectordb import VectorDB, SimilarityMetric

# 1. Initialize Vector Database
db = VectorDB(dim=4, metric=SimilarityMetric.COSINE)

# 2. Insert vectors with metadata
db.upsert("doc_1", [1.0, 0.0, 0.0, 0.0], {"title": "Linux Systems", "category": "tech"})
db.upsert("doc_2", [0.8, 0.2, 0.0, 0.0], {"title": "Operating Systems", "category": "tech"})
db.upsert("doc_3", [0.0, 1.0, 0.0, 0.0], {"title": "Quantum Physics", "category": "physics"})

# 3. Query Top-K most similar documents
results = db.query([0.9, 0.1, 0.0, 0.0], top_k=2)
for res in results:
    print(f"ID: {res.id}, Score: {res.score:.4f}, Title: {res.metadata['title']}")

# 4. Filtered Query
filtered = db.query(
    [0.9, 0.1, 0.0, 0.0],
    top_k=2,
    filter_fn=lambda meta: meta.get("category") == "physics",
)

# 5. Persist and Reload
db.save("my_knowledge_base")
reloaded_db = VectorDB.load_collection("my_knowledge_base")
```

---

## 📖 API Reference

### `VectorDB`
Main entry point for vector indexing and retrieval.

- `VectorDB(dim: int, metric: SimilarityMetric | str = "cosine", storage_dir: str | Path = "./data", max_vectors: int = 10_000_000, initial_capacity: int = 16)`
- `upsert(doc_id: str, vector: Sequence[float] | np.ndarray, metadata: dict[str, Any] | None = None) -> None`: Insert or update document.
- `upsert_batch(documents: Sequence[Document] | Sequence[tuple]) -> None`: Batch insertion.
- `delete(doc_id: str) -> bool`: Delete document in $O(1)$ time via swap-and-pop defragmentation.
- `get(doc_id: str) -> Document | None`: Retrieve document by ID.
- `query(vector: Sequence[float] | np.ndarray, top_k: int = 10, filter_fn: Callable[[dict[str, Any]], bool] | None = None, include_vector: bool = False) -> list[QueryResult]`: Perform similarity search.
- `count() -> int`: Return active vector count.
- `save(name: str) -> tuple[Path, Path]`: Persist `.npz` vector archive and `.meta.json` sidecar.
- `load(name: str) -> None`: Load collection into current instance.
- `VectorDB.load_collection(name: str, storage_dir: str | Path = "./data") -> VectorDB`: Classmethod to load and instantiate collection.

### `SimilarityMetric`
- `SimilarityMetric.COSINE` (`"cosine"`)
- `SimilarityMetric.EUCLIDEAN` (`"euclidean"`)
- `SimilarityMetric.DOT_PRODUCT` (`"dot_product"`)

### `Document` & `QueryResult`
- `Document(id: str, vector: np.ndarray, metadata: dict[str, Any])`
- `QueryResult(id: str, score: float, metadata: dict[str, Any], vector: np.ndarray | None)`

---

## 🧪 Testing & Verification

Run tests and strict type verification:

```bash
# Run test suite with pytest
uv run pytest -v --tb=short

# Run strict type checking with mypy
uv run mypy numpy_vectordb/ --strict
```

---

## 📄 License
MIT License — Copyright (c) 2026 cibi-dev.
