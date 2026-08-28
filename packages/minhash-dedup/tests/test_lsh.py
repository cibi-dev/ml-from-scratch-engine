"""Tests for Locality-Sensitive Hashing (LSH) parameter optimization, S-curve fit, and querying."""

from __future__ import annotations

import numpy as np
import pytest

from minhash_dedup.lsh import MinHashLSH, optimize_lsh_parameters
from minhash_dedup.minhash import MinHasher
from tests.conftest import generate_synthetic_overlap_sets


class TestLSHParameterOptimization:
    def test_optimal_b_r_satisfies_constraints(self) -> None:
        b, r = optimize_lsh_parameters(threshold=0.75, num_perm=128)
        assert b >= 1
        assert r >= 1
        assert b * r <= 128
        # Inflection point (1/b)^(1/r) should be reasonably close to 0.75
        approx_threshold = (1.0 / b) ** (1.0 / r)
        assert 0.50 <= approx_threshold <= 0.90

    def test_weight_penalty_influences_b_r(self) -> None:
        # High false-negative penalty (w_fn >> w_fp) should increase b (more bands -> higher recall)
        b_high_recall, r_high_recall = optimize_lsh_parameters(
            threshold=0.75, num_perm=128, weights=(0.1, 0.9)
        )
        # High false-positive penalty (w_fp >> w_fn) should decrease b or increase r (higher precision)
        b_high_prec, r_high_prec = optimize_lsh_parameters(
            threshold=0.75, num_perm=128, weights=(0.9, 0.1)
        )

        assert b_high_recall >= b_high_prec

    def test_invalid_parameters_raise_value_error(self) -> None:
        with pytest.raises(ValueError, match="threshold must be in"):
            optimize_lsh_parameters(threshold=0.0, num_perm=128)
        with pytest.raises(ValueError, match="threshold must be in"):
            optimize_lsh_parameters(threshold=1.0, num_perm=128)
        with pytest.raises(ValueError, match="num_perm must be >= 1"):
            optimize_lsh_parameters(threshold=0.5, num_perm=0)
        with pytest.raises(ValueError, match="weights must be non-negative"):
            optimize_lsh_parameters(threshold=0.5, num_perm=128, weights=(-0.1, 0.5))


class TestMinHashLSHIndex:
    def test_explicit_b_and_r_parameters(self) -> None:
        lsh = MinHashLSH(threshold=0.75, num_perm=128, b=16, r=8)
        assert lsh.b == 16
        assert lsh.r == 8
        assert len(lsh.tables) == 16

    def test_mismatched_explicit_parameters_raise_value_error(self) -> None:
        with pytest.raises(ValueError, match="b and r must either both be specified"):
            MinHashLSH(threshold=0.75, num_perm=128, b=16, r=None)
        with pytest.raises(ValueError, match="b and r must either both be specified"):
            MinHashLSH(threshold=0.75, num_perm=128, b=None, r=8)
        with pytest.raises(ValueError, match="cannot exceed num_perm"):
            MinHashLSH(threshold=0.75, num_perm=128, b=20, r=10)  # 200 > 128

    def test_insert_and_query_candidates(self) -> None:
        hasher = MinHasher(num_perm=128, seed=42)
        lsh = MinHashLSH(threshold=0.7, num_perm=128)

        doc1_sig = hasher.compute_signature(["the", "quick", "brown", "fox", "jumps"])
        doc2_sig = hasher.compute_signature(["the", "quick", "brown", "fox", "jumps"])  # Identical
        doc3_sig = hasher.compute_signature(["unrelated", "astronomy", "galaxy", "telescope", "star"])

        lsh.insert("doc1", doc1_sig)
        lsh.insert("doc2", doc2_sig)
        lsh.insert("doc3", doc3_sig)

        candidates_1 = lsh.query_candidates(doc1_sig)
        assert "doc1" in candidates_1
        assert "doc2" in candidates_1
        assert "doc3" not in candidates_1

        candidates_3 = lsh.query_candidates(doc3_sig)
        assert "doc3" in candidates_3
        assert "doc1" not in candidates_3
        assert "doc2" not in candidates_3

    def test_query_short_signature_raises_value_error(self) -> None:
        lsh = MinHashLSH(threshold=0.75, num_perm=128, b=16, r=8)
        short_sig = np.zeros(64, dtype=np.uint64)
        with pytest.raises(ValueError, match="Signature length"):
            lsh.insert("doc1", short_sig)
        with pytest.raises(ValueError, match="Signature length"):
            lsh.query_candidates(short_sig)

    def test_s_curve_empirical_fit(self) -> None:
        """Verify empirical collision probability matches theoretical S-curve P(s) = 1 - (1 - s^r)^b."""
        b, r = 16, 8
        num_perm = b * r  # 128
        lsh = MinHashLSH(threshold=0.75, num_perm=num_perm, b=b, r=r)

        similarities_to_test = [0.2, 0.5, 0.7, 0.9]
        trials_per_similarity = 40

        for target_s in similarities_to_test:
            theoretical_prob = 1.0 - (1.0 - (target_s ** r)) ** b
            collisions = 0

            for trial_seed in range(trials_per_similarity):
                set_a, set_b, _ = generate_synthetic_overlap_sets(
                    size_a=150,
                    size_b=150,
                    target_jaccard=target_s,
                    seed=trial_seed + int(target_s * 1000),
                )
                hasher = MinHasher(num_perm=num_perm, seed=trial_seed)
                sig_a = hasher.compute_signature(set_a)
                sig_b = hasher.compute_signature(set_b)

                # Check if signatures share at least one band
                shared_band = False
                for band_idx in range(b):
                    band_a = sig_a[band_idx * r : (band_idx + 1) * r]
                    band_b = sig_b[band_idx * r : (band_idx + 1) * r]
                    if np.array_equal(band_a, band_b):
                        shared_band = True
                        break

                if shared_band:
                    collisions += 1

            empirical_prob = collisions / trials_per_similarity
            diff = abs(empirical_prob - theoretical_prob)
            assert diff < 0.20, (
                f"S-curve fit error too large for s={target_s}: "
                f"theoretical={theoretical_prob:.4f}, empirical={empirical_prob:.4f}"
            )
