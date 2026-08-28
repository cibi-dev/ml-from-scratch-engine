"""Shared test fixtures and synthetic data generators for minhash-dedup tests."""

from __future__ import annotations

import random
from typing import Sequence

import pytest


@pytest.fixture
def sample_documents() -> list[str]:
    """Provide a standard list of sample documents with duplicates and variations."""
    return [
        "The quick brown fox jumps over the lazy dog in the sunny afternoon.",
        "The quick brown fox jumps over the lazy dog in the sunny afternoon.",  # Exact duplicate
        "The fast brown fox jumps over the lazy dog in the sunny afternoon.",  # Near duplicate
        "Deep learning models require massive curated datasets for pretraining.",
        "Deep learning architectures require massive curated datasets for pretraining.",  # Near duplicate
        "Quantum computing harnesses superposition and entanglement for computation.",
        "A completely unrelated document discussing cooking recipes and pastry dough.",
    ]


@pytest.fixture
def vocabulary() -> list[str]:
    """Provide a synthetic vocabulary for document generation."""
    return [
        f"token_{i}" for i in range(1000)
    ]


def generate_synthetic_overlap_sets(
    size_a: int,
    size_b: int,
    target_jaccard: float,
    seed: int = 42,
) -> tuple[set[str], set[str], float]:
    """Generate two sets of string tokens with exact known Jaccard similarity.

    J(A, B) = |A cap B| / |A cup B|
    Let intersection = I, |A| = |B| = N, then J = I / (2N - I) => I = 2N * J / (1 + J).
    """
    rng = random.Random(seed)
    n = size_a
    intersection_size = int(round((2 * n * target_jaccard) / (1.0 + target_jaccard)))

    common_tokens = [f"common_shingle_{i}" for i in range(intersection_size)]
    unique_a_size = n - intersection_size
    unique_b_size = n - intersection_size

    unique_a = [f"unique_a_shingle_{i}" for i in range(unique_a_size)]
    unique_b = [f"unique_b_shingle_{i}" for i in range(unique_b_size)]

    set_a = set(common_tokens + unique_a)
    set_b = set(common_tokens + unique_b)

    actual_jaccard = len(set_a & set_b) / len(set_a | set_b)
    return set_a, set_b, actual_jaccard
