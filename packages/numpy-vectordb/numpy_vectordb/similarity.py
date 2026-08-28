"""Similarity metrics, vector normalization, and Top-K partition selection."""

from __future__ import annotations

from enum import Enum
from typing import Any, cast

import numpy as np
import numpy.typing as npt


class SimilarityMetric(str, Enum):
    """Supported distance and similarity metrics."""

    COSINE = "cosine"
    EUCLIDEAN = "euclidean"
    DOT_PRODUCT = "dot_product"

    @classmethod
    def from_str(cls, value: str | SimilarityMetric) -> SimilarityMetric:
        """Convert string to SimilarityMetric."""
        if isinstance(value, cls):
            return value
        if not isinstance(value, str):
            raise TypeError(f"Metric must be a string or SimilarityMetric, got {type(value).__name__}")
        val = value.strip().lower()
        if val in ("cosine", "cos"):
            return cls.COSINE
        if val in ("euclidean", "l2"):
            return cls.EUCLIDEAN
        if val in ("dot_product", "dot", "ip", "inner_product"):
            return cls.DOT_PRODUCT
        raise ValueError(
            f"Unsupported similarity metric '{value}'. Choose from: 'cosine', 'euclidean', 'dot_product'."
        )


def normalize_vector(v: npt.NDArray[np.float32], eps: float = 1e-12) -> npt.NDArray[np.float32]:
    """Normalize a 1D vector to unit Euclidean norm (L2)."""
    norm = float(np.linalg.norm(v))
    if norm < eps:
        raise ValueError(f"Zero-norm vector detected (norm {norm:.2e} < {eps:.2e}), cannot normalize.")
    res = np.asarray(v / norm, dtype=np.float32)
    return cast(npt.NDArray[np.float32], res)


def normalize_matrix(m: npt.NDArray[np.float32], eps: float = 1e-12) -> npt.NDArray[np.float32]:
    """Normalize rows of a 2D matrix to unit Euclidean norm (L2)."""
    if m.ndim != 2:
        raise ValueError(f"Expected 2D matrix, got ndim={m.ndim}")
    if m.shape[0] == 0:
        return m.copy()
    norms = np.linalg.norm(m, axis=1, keepdims=True)
    if (norms < eps).any():
        raise ValueError("Zero-norm vector detected in matrix rows during normalization.")
    res = np.asarray(m / norms, dtype=np.float32)
    return cast(npt.NDArray[np.float32], res)


def compute_cosine_similarity(
    query_vec: npt.NDArray[np.float32],
    matrix: npt.NDArray[np.float32],
) -> npt.NDArray[np.float32]:
    """Compute cosine similarity between 1D query and 2D matrix rows using BLAS GEMV/GEMM."""
    if matrix.shape[0] == 0:
        return np.empty(0, dtype=np.float32)

    q_norm = normalize_vector(query_vec)
    m_norm = normalize_matrix(matrix)

    # BLAS GEMV dot product
    scores = np.dot(m_norm, q_norm)
    # Numerical safety clamp to [-1.0, 1.0]
    res = np.asarray(np.clip(scores, -1.0, 1.0), dtype=np.float32)
    return cast(npt.NDArray[np.float32], res)


def compute_euclidean_distance(
    query_vec: npt.NDArray[np.float32],
    matrix: npt.NDArray[np.float32],
) -> npt.NDArray[np.float32]:
    """Compute Euclidean (L2) distance between 1D query and 2D matrix rows."""
    if matrix.shape[0] == 0:
        return np.empty(0, dtype=np.float32)

    diff = matrix - query_vec
    distances = np.linalg.norm(diff, axis=1)
    res = np.asarray(distances, dtype=np.float32)
    return cast(npt.NDArray[np.float32], res)


def compute_dot_product(
    query_vec: npt.NDArray[np.float32],
    matrix: npt.NDArray[np.float32],
) -> npt.NDArray[np.float32]:
    """Compute inner dot product between 1D query and 2D matrix rows using BLAS GEMV."""
    if matrix.shape[0] == 0:
        return np.empty(0, dtype=np.float32)

    scores = np.dot(matrix, query_vec)
    res = np.asarray(scores, dtype=np.float32)
    return cast(npt.NDArray[np.float32], res)


def top_k_indices_and_scores(
    scores: npt.NDArray[Any],
    top_k: int,
    largest: bool = True,
) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.float32]]:
    """Select top-k indices and scores in O(N + K log K) time using np.argpartition.
    
    Args:
        scores: 1D array of scores or distances.
        top_k: Number of top results to retrieve.
        largest: If True, select highest scores (e.g. Cosine, Dot Product).
                 If False, select lowest scores (e.g. Euclidean Distance).
                 
    Returns:
        tuple of (top_k_indices, top_k_scores) sorted in optimal order.
    """
    n = len(scores)
    if n == 0 or top_k <= 0:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.float32)

    scores_arr = np.asarray(scores, dtype=np.float32)
    k = min(top_k, n)

    if k == n:
        if largest:
            order = np.argsort(-scores_arr)
        else:
            order = np.argsort(scores_arr)
        indices = order.astype(np.int64)
        return indices, cast(npt.NDArray[np.float32], scores_arr[indices])

    # O(N) partitioning
    if largest:
        partition_indices = np.argpartition(-scores_arr, k - 1)[:k]
        sub_order = np.argsort(-scores_arr[partition_indices])
    else:
        partition_indices = np.argpartition(scores_arr, k - 1)[:k]
        sub_order = np.argsort(scores_arr[partition_indices])

    final_indices = partition_indices[sub_order].astype(np.int64)
    return final_indices, cast(npt.NDArray[np.float32], scores_arr[final_indices])
