"""Tests for MinHasher vectorized 64-bit universal hashing and Jaccard estimation."""

from __future__ import annotations

import numpy as np
import pytest

from minhash_dedup.minhash import MERSENNE_PRIME, UINT64_MAX, MinHasher, estimate_jaccard
from tests.conftest import generate_synthetic_overlap_sets


class TestMinHasherCore:
    def test_signature_shape_and_dtype(self) -> None:
        hasher = MinHasher(num_perm=128, seed=42)
        shingles = ["apple", "banana", "cherry", "date"]
        sig = hasher.compute_signature(shingles)

        assert isinstance(sig, np.ndarray)
        assert sig.shape == (128,)
        assert sig.dtype == np.uint64
        # All values must be valid 64-bit integers below Mersenne prime
        assert np.all(sig < MERSENNE_PRIME)

    def test_deterministic_seed_reproducibility(self) -> None:
        hasher1 = MinHasher(num_perm=64, seed=123)
        hasher2 = MinHasher(num_perm=64, seed=123)
        hasher3 = MinHasher(num_perm=64, seed=999)

        shingles = ["alpha", "beta", "gamma", "delta"]
        sig1 = hasher1.compute_signature(shingles)
        sig2 = hasher2.compute_signature(shingles)
        sig3 = hasher3.compute_signature(shingles)

        np.testing.assert_array_equal(sig1, sig2)
        assert not np.array_equal(sig1, sig3)

    def test_coefficient_bounds(self) -> None:
        hasher = MinHasher(num_perm=500, seed=42)
        # a coefficients must be in [1, MERSENNE_PRIME - 1]
        assert np.all(hasher.a >= 1)
        assert np.all(hasher.a < MERSENNE_PRIME)
        # b coefficients must be in [0, MERSENNE_PRIME - 1]
        assert np.all(hasher.b >= 0)
        assert np.all(hasher.b < MERSENNE_PRIME)

    def test_empty_shingles_returns_max_uint64_signature(self) -> None:
        hasher = MinHasher(num_perm=32, seed=42)
        sig = hasher.compute_signature([])
        assert sig.shape == (32,)
        assert sig.dtype == np.uint64
        assert np.all(sig == UINT64_MAX)

    def test_duplicate_shingles_invariance(self) -> None:
        hasher = MinHasher(num_perm=64, seed=42)
        shingles_unique = ["cat", "dog", "bird"]
        shingles_repeated = ["cat", "cat", "dog", "bird", "dog", "cat"]

        sig1 = hasher.compute_signature(shingles_unique)
        sig2 = hasher.compute_signature(shingles_repeated)
        np.testing.assert_array_equal(sig1, sig2)

    def test_invalid_num_perm_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="num_perm must be a positive integer"):
            MinHasher(num_perm=0)
        with pytest.raises(ValueError, match="num_perm must be a positive integer"):
            MinHasher(num_perm=-10)


class TestJaccardEstimation:
    def test_identical_sets_give_similarity_one(self) -> None:
        hasher = MinHasher(num_perm=128, seed=42)
        shingles = [f"item_{i}" for i in range(50)]
        sig_a = hasher.compute_signature(shingles)
        sig_b = hasher.compute_signature(shingles)

        sim = hasher.estimate_jaccard(sig_a, sig_b)
        assert sim == 1.0

    def test_disjoint_sets_give_similarity_near_zero(self) -> None:
        hasher = MinHasher(num_perm=256, seed=42)
        set_a = [f"alpha_{i}" for i in range(100)]
        set_b = [f"beta_{i}" for i in range(100)]

        sig_a = hasher.compute_signature(set_a)
        sig_b = hasher.compute_signature(set_b)

        sim = hasher.estimate_jaccard(sig_a, sig_b)
        assert sim < 0.05

    @pytest.mark.parametrize("target_jaccard", [0.25, 0.50, 0.75])
    def test_unbiased_estimator_statistical_convergence(self, target_jaccard: float) -> None:
        """Verify that E[J_hat] converges to true Jaccard J with error |E[J_hat] - J| < 0.005."""
        set_a, set_b, true_jaccard = generate_synthetic_overlap_sets(
            size_a=200,
            size_b=200,
            target_jaccard=target_jaccard,
            seed=42,
        )

        num_trials = 60
        num_perm = 256
        estimates: list[float] = []

        for seed in range(num_trials):
            hasher = MinHasher(num_perm=num_perm, seed=seed)
            sig_a = hasher.compute_signature(set_a)
            sig_b = hasher.compute_signature(set_b)
            estimates.append(hasher.estimate_jaccard(sig_a, sig_b))

        expected_jaccard = float(np.mean(estimates))
        error = abs(expected_jaccard - true_jaccard)
        assert error < 0.005, f"Expected error < 0.005, got {error:.6f} (true={true_jaccard:.4f}, est={expected_jaccard:.4f})"

    def test_incompatible_signature_shapes_raise_value_error(self) -> None:
        hasher1 = MinHasher(num_perm=64, seed=42)
        hasher2 = MinHasher(num_perm=128, seed=42)

        sig1 = hasher1.compute_signature(["a", "b"])
        sig2 = hasher2.compute_signature(["a", "b"])

        with pytest.raises(ValueError, match="Signatures must have identical lengths"):
            estimate_jaccard(sig1, sig2)

        with pytest.raises(ValueError, match="Signatures must be 1D arrays"):
            estimate_jaccard(np.zeros((2, 2), dtype=np.uint64), np.zeros((2, 2), dtype=np.uint64))

        with pytest.raises(ValueError, match="Signatures must have non-zero length"):
            estimate_jaccard(np.array([], dtype=np.uint64), np.array([], dtype=np.uint64))
