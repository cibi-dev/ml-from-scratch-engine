"""Vectorized 64-bit MinHash signature generation and Jaccard similarity estimation."""

from __future__ import annotations

import hashlib
import struct
from typing import Iterable, Sequence

import numpy as np

# Mersenne prime 2^61 - 1
MERSENNE_PRIME: int = (1 << 61) - 1
UINT64_MAX: int = (1 << 64) - 1


class MinHasher:
    """MinHasher produces deterministic 64-bit MinHash signatures for shingled text.

    Uses universal hashing: h_i(x) = (a_i * x + b_i) mod (2^61 - 1)
    where a_i in [1, 2^61 - 2] and b_i in [0, 2^61 - 2].
    """

    def __init__(self, num_perm: int = 128, seed: int = 42) -> None:
        """Initialize MinHasher with a fixed number of permutations and RNG seed.

        Args:
            num_perm: Number of hash permutations (signature dimensionality). Must be >= 1.
            seed: Random seed for deterministic generation of hash parameters.

        Raises:
            ValueError: If num_perm < 1.
        """
        if num_perm < 1:
            raise ValueError(f"num_perm must be a positive integer >= 1, got {num_perm}")

        self.num_perm = num_perm
        self.seed = seed

        # Generate universal hash coefficients deterministically
        rng = np.random.default_rng(seed)
        # a_i must be in [1, MERSENNE_PRIME - 1]
        self.a: np.ndarray = rng.integers(1, MERSENNE_PRIME, size=num_perm, dtype=np.uint64)
        # b_i must be in [0, MERSENNE_PRIME - 1]
        self.b: np.ndarray = rng.integers(0, MERSENNE_PRIME, size=num_perm, dtype=np.uint64)

    @staticmethod
    def _hash_shingle(shingle: str) -> int:
        """Compute a 64-bit hash for a shingle string mapped into [0, MERSENNE_PRIME - 1]."""
        digest = hashlib.blake2b(shingle.encode("utf-8"), digest_size=8).digest()
        val: int = struct.unpack("<Q", digest)[0]
        return val % MERSENNE_PRIME

    def compute_signature(self, shingles: Iterable[str] | Sequence[str] | set[str]) -> np.ndarray:
        """Compute the MinHash signature vector for a collection of shingles.

        Args:
            shingles: Iterable collection of shingle strings.

        Returns:
            A 1D numpy array of shape (num_perm,) and dtype uint64 containing
            the minimum hashed values for each permutation.
        """
        unique_shingles = list(set(shingles))
        if not unique_shingles:
            # Fallback signature for empty shingles
            return np.full(self.num_perm, UINT64_MAX, dtype=np.uint64)

        # Hash shingles to 64-bit integers
        x_hashes = [self._hash_shingle(s) for s in unique_shingles]

        # Vectorized universal hashing modulo Mersenne prime using arbitrary-precision integers
        # to prevent 64-bit multiplication overflow: (a * x + b) % (2^61 - 1)
        a_obj = self.a[:, None].astype(object)
        b_obj = self.b[:, None].astype(object)
        x_obj = np.array(x_hashes, dtype=object)[None, :]

        # Shape: (num_perm, len(unique_shingles))
        hashed = (a_obj * x_obj + b_obj) % MERSENNE_PRIME
        signature = np.min(hashed, axis=1).astype(np.uint64)

        return signature

    def estimate_jaccard(self, sig_a: np.ndarray, sig_b: np.ndarray) -> float:
        """Estimate the Jaccard similarity between two MinHash signatures.

        Args:
            sig_a: 1D uint64 numpy array of shape (num_perm,).
            sig_b: 1D uint64 numpy array of shape (num_perm,).

        Returns:
            Estimated Jaccard similarity in [0.0, 1.0].

        Raises:
            ValueError: If signatures have incompatible shapes.
        """
        return estimate_jaccard(sig_a, sig_b)


def estimate_jaccard(sig_a: np.ndarray, sig_b: np.ndarray) -> float:
    """Estimate Jaccard similarity as the fraction of matching signature components.

    Args:
        sig_a: 1D uint64 numpy array of shape (num_perm,).
        sig_b: 1D uint64 numpy array of shape (num_perm,).

    Returns:
        Estimated Jaccard similarity in [0.0, 1.0].

    Raises:
        ValueError: If signatures have different dimensions or are not 1D.
    """
    if sig_a.ndim != 1 or sig_b.ndim != 1:
        raise ValueError(f"Signatures must be 1D arrays, got shapes {sig_a.shape} and {sig_b.shape}")
    if sig_a.shape[0] != sig_b.shape[0]:
        raise ValueError(f"Signatures must have identical lengths, got {sig_a.shape[0]} and {sig_b.shape[0]}")
    if sig_a.shape[0] == 0:
        raise ValueError("Signatures must have non-zero length")

    return float(np.mean(sig_a == sig_b))
