"""Pytest fixtures for numpy-vectordb tests."""

from pathlib import Path
from typing import Generator
import pytest
import numpy as np

from numpy_vectordb.database import VectorDB
from numpy_vectordb.similarity import SimilarityMetric


@pytest.fixture
def temp_db_dir(tmp_path: Path) -> Path:
    """Provide a temporary directory for database persistence."""
    d = tmp_path / "vectordb_storage"
    d.mkdir(parents=True, exist_ok=True)
    return d


@pytest.fixture
def small_cosine_db(temp_db_dir: Path) -> VectorDB:
    """Provide a small 4-dimensional Cosine VectorDB."""
    return VectorDB(
        dim=4,
        metric=SimilarityMetric.COSINE,
        storage_dir=temp_db_dir,
        initial_capacity=4,
    )


@pytest.fixture
def small_euclidean_db(temp_db_dir: Path) -> VectorDB:
    """Provide a small 4-dimensional Euclidean VectorDB."""
    return VectorDB(
        dim=4,
        metric=SimilarityMetric.EUCLIDEAN,
        storage_dir=temp_db_dir,
        initial_capacity=4,
    )


@pytest.fixture
def small_dot_db(temp_db_dir: Path) -> VectorDB:
    """Provide a small 4-dimensional Dot Product VectorDB."""
    return VectorDB(
        dim=4,
        metric=SimilarityMetric.DOT_PRODUCT,
        storage_dir=temp_db_dir,
        initial_capacity=4,
    )
