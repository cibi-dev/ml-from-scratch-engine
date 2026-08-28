"""Tests for similarity metrics, vector normalization, and Top-K partition selection."""

import numpy as np
import pytest

from numpy_vectordb.similarity import (
    SimilarityMetric,
    compute_cosine_similarity,
    compute_dot_product,
    compute_euclidean_distance,
    normalize_matrix,
    normalize_vector,
    top_k_indices_and_scores,
)


class TestSimilarityMetricEnum:
    """Test SimilarityMetric parsing and conversions."""

    def test_from_str_valid(self) -> None:
        assert SimilarityMetric.from_str("cosine") == SimilarityMetric.COSINE
        assert SimilarityMetric.from_str("COS") == SimilarityMetric.COSINE
        assert SimilarityMetric.from_str("euclidean") == SimilarityMetric.EUCLIDEAN
        assert SimilarityMetric.from_str("l2") == SimilarityMetric.EUCLIDEAN
        assert SimilarityMetric.from_str("dot_product") == SimilarityMetric.DOT_PRODUCT
        assert SimilarityMetric.from_str("ip") == SimilarityMetric.DOT_PRODUCT
        assert SimilarityMetric.from_str(SimilarityMetric.COSINE) == SimilarityMetric.COSINE

    def test_from_str_invalid(self) -> None:
        with pytest.raises(ValueError, match="Unsupported similarity metric"):
            SimilarityMetric.from_str("manhattan")

    def test_from_str_wrong_type(self) -> None:
        with pytest.raises(TypeError, match="Metric must be a string"):
            SimilarityMetric.from_str(123)  # type: ignore[arg-type]


class TestNormalization:
    """Test L2 normalization for vectors and matrices."""

    def test_normalize_vector_unit_length(self) -> None:
        v = np.array([3.0, 4.0], dtype=np.float32)
        norm_v = normalize_vector(v)
        assert np.isclose(np.linalg.norm(norm_v), 1.0, atol=1e-6)
        assert np.allclose(norm_v, np.array([0.6, 0.8], dtype=np.float32))

    def test_normalize_zero_vector_raises(self) -> None:
        v = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        with pytest.raises(ValueError, match="Zero-norm vector detected"):
            normalize_vector(v)

    def test_normalize_matrix_unit_rows(self) -> None:
        m = np.array([[3.0, 4.0], [1.0, 0.0], [0.0, 5.0]], dtype=np.float32)
        norm_m = normalize_matrix(m)
        assert norm_m.shape == (3, 2)
        row_norms = np.linalg.norm(norm_m, axis=1)
        assert np.allclose(row_norms, np.ones(3, dtype=np.float32))

    def test_normalize_empty_matrix(self) -> None:
        m = np.empty((0, 4), dtype=np.float32)
        norm_m = normalize_matrix(m)
        assert norm_m.shape == (0, 4)

    def test_normalize_matrix_zero_row_raises(self) -> None:
        m = np.array([[1.0, 2.0], [0.0, 0.0]], dtype=np.float32)
        with pytest.raises(ValueError, match="Zero-norm vector detected in matrix"):
            normalize_matrix(m)


class TestCosineSimilarity:
    """Mathematical correctness of cosine similarity."""

    def test_cosine_identical_vectors(self) -> None:
        q = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        m = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)
        scores = compute_cosine_similarity(q, m)
        assert len(scores) == 1
        assert np.isclose(scores[0], 1.0, atol=1e-6)

    def test_cosine_opposite_vectors(self) -> None:
        q = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        m = np.array([[-1.0, -2.0, -3.0]], dtype=np.float32)
        scores = compute_cosine_similarity(q, m)
        assert len(scores) == 1
        assert np.isclose(scores[0], -1.0, atol=1e-6)

    def test_cosine_orthogonal_vectors(self) -> None:
        q = np.array([1.0, 0.0], dtype=np.float32)
        m = np.array([[0.0, 1.0], [0.0, -1.0]], dtype=np.float32)
        scores = compute_cosine_similarity(q, m)
        assert np.allclose(scores, [0.0, 0.0], atol=1e-6)

    def test_cosine_known_angle(self) -> None:
        # Angle of 45 deg -> cos(45 deg) = 1 / sqrt(2) ≈ 0.70710678
        q = np.array([1.0, 0.0], dtype=np.float32)
        m = np.array([[1.0, 1.0]], dtype=np.float32)
        scores = compute_cosine_similarity(q, m)
        expected = 1.0 / np.sqrt(2.0)
        assert np.isclose(scores[0], expected, atol=1e-6)

    def test_cosine_empty_matrix(self) -> None:
        q = np.array([1.0, 0.0], dtype=np.float32)
        m = np.empty((0, 2), dtype=np.float32)
        scores = compute_cosine_similarity(q, m)
        assert len(scores) == 0

    def test_cosine_clamping_range(self) -> None:
        # Scale by large factor to test numerical stability
        q = np.array([1e6, 1e6], dtype=np.float32)
        m = np.array([[1e6, 1e6]], dtype=np.float32)
        scores = compute_cosine_similarity(q, m)
        assert -1.0 <= scores[0] <= 1.0


class TestEuclideanDistance:
    """Mathematical correctness of Euclidean L2 distance."""

    def test_euclidean_identical_vectors(self) -> None:
        q = np.array([2.0, 3.0, 4.0], dtype=np.float32)
        m = np.array([[2.0, 3.0, 4.0]], dtype=np.float32)
        distances = compute_euclidean_distance(q, m)
        assert np.isclose(distances[0], 0.0, atol=1e-6)

    def test_euclidean_known_3_4_5(self) -> None:
        q = np.array([0.0, 0.0], dtype=np.float32)
        m = np.array([[3.0, 4.0]], dtype=np.float32)
        distances = compute_euclidean_distance(q, m)
        assert np.isclose(distances[0], 5.0, atol=1e-6)

    def test_euclidean_multiple_rows(self) -> None:
        q = np.array([0.0, 0.0], dtype=np.float32)
        m = np.array([[1.0, 0.0], [0.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        distances = compute_euclidean_distance(q, m)
        assert np.allclose(distances, [1.0, 2.0, 5.0], atol=1e-6)

    def test_euclidean_empty_matrix(self) -> None:
        q = np.array([1.0, 1.0], dtype=np.float32)
        m = np.empty((0, 2), dtype=np.float32)
        distances = compute_euclidean_distance(q, m)
        assert len(distances) == 0


class TestDotProduct:
    """Mathematical correctness of Inner Dot Product."""

    def test_dot_product_calculation(self) -> None:
        q = np.array([1.0, 2.0], dtype=np.float32)
        m = np.array([[3.0, 4.0], [5.0, 6.0]], dtype=np.float32)
        scores = compute_dot_product(q, m)
        # [1*3 + 2*4, 1*5 + 2*6] = [11, 17]
        assert np.allclose(scores, [11.0, 17.0], atol=1e-6)

    def test_dot_product_empty_matrix(self) -> None:
        q = np.array([1.0, 2.0], dtype=np.float32)
        m = np.empty((0, 2), dtype=np.float32)
        scores = compute_dot_product(q, m)
        assert len(scores) == 0


class TestTopKPartitionSelection:
    """Test O(N + K log K) argpartition selection."""

    def test_top_k_empty_scores(self) -> None:
        scores = np.empty(0, dtype=np.float32)
        idx, sc = top_k_indices_and_scores(scores, top_k=5)
        assert len(idx) == 0
        assert len(sc) == 0

    def test_top_k_zero_or_negative_k(self) -> None:
        scores = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        idx, sc = top_k_indices_and_scores(scores, top_k=0)
        assert len(idx) == 0
        idx_neg, sc_neg = top_k_indices_and_scores(scores, top_k=-2)
        assert len(idx_neg) == 0

    def test_top_k_exact_ordering_largest(self) -> None:
        scores = np.array([0.1, 0.9, 0.4, 0.8, 0.2, 0.7], dtype=np.float32)
        idx, sc = top_k_indices_and_scores(scores, top_k=3, largest=True)
        # Expected top 3 largest: 0.9 (idx 1), 0.8 (idx 3), 0.7 (idx 5)
        assert np.array_equal(idx, [1, 3, 5])
        assert np.allclose(sc, [0.9, 0.8, 0.7])

    def test_top_k_exact_ordering_smallest(self) -> None:
        distances = np.array([5.0, 1.2, 8.4, 0.3, 2.1], dtype=np.float32)
        idx, sc = top_k_indices_and_scores(distances, top_k=3, largest=False)
        # Expected top 3 smallest: 0.3 (idx 3), 1.2 (idx 1), 2.1 (idx 4)
        assert np.array_equal(idx, [3, 1, 4])
        assert np.allclose(sc, [0.3, 1.2, 2.1])

    def test_top_k_larger_than_n(self) -> None:
        scores = np.array([0.3, 0.9, 0.1], dtype=np.float32)
        idx, sc = top_k_indices_and_scores(scores, top_k=10, largest=True)
        assert len(idx) == 3
        assert np.array_equal(idx, [1, 0, 2])
        assert np.allclose(sc, [0.9, 0.3, 0.1])

    def test_top_k_equal_to_n(self) -> None:
        scores = np.array([0.2, 0.5, 0.1], dtype=np.float32)
        idx, sc = top_k_indices_and_scores(scores, top_k=3, largest=True)
        assert len(idx) == 3
        assert np.array_equal(idx, [1, 0, 2])
        assert np.allclose(sc, [0.5, 0.2, 0.1])

    def test_top_k_single_k(self) -> None:
        scores = np.array([0.2, 0.95, 0.1, 0.7], dtype=np.float32)
        idx, sc = top_k_indices_and_scores(scores, top_k=1, largest=True)
        assert len(idx) == 1
        assert idx[0] == 1
        assert np.isclose(sc[0], 0.95)

    def test_top_k_with_identical_scores(self) -> None:
        scores = np.array([0.5, 0.5, 0.5, 0.5], dtype=np.float32)
        idx, sc = top_k_indices_and_scores(scores, top_k=2, largest=True)
        assert len(idx) == 2
        assert np.allclose(sc, [0.5, 0.5])
