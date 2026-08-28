"""Validation and security sanity checks for numpy-vectordb."""

from __future__ import annotations

import os
from pathlib import Path
import re
from typing import Any, Sequence

import numpy as np
import numpy.typing as npt

# Resource bounds
MAX_DIM: int = 65536
MAX_VECTORS: int = 10_000_000
MIN_VECTOR_NORM: float = 1e-12

# Collection name regex: alphanumeric, underscore, hyphen
COLLECTION_NAME_PATTERN: re.Pattern[str] = re.compile(r"^[a-zA-Z0-9_-]+$")


def validate_doc_id(doc_id: Any) -> str:
    """Validate document ID.
    
    Must be a non-empty string.
    """
    if not isinstance(doc_id, str):
        raise TypeError(f"Document ID must be a string, got {type(doc_id).__name__}")
    cleaned = doc_id.strip()
    if not cleaned:
        raise ValueError("Document ID must be a non-empty string")
    return doc_id


def validate_collection_name(name: str) -> str:
    """Validate collection name against safe regex pattern (CWE-22 mitigation)."""
    if not isinstance(name, str):
        raise TypeError(f"Collection name must be a string, got {type(name).__name__}")
    if not name or not COLLECTION_NAME_PATTERN.fullmatch(name):
        raise ValueError(
            f"Invalid collection name '{name}'. Must match ^[a-zA-Z0-9_-]+$ and contain no path separators."
        )
    return name


def validate_path_containment(base_dir: str | Path, target_path: str | Path) -> Path:
    """Ensure target_path resolves strictly within base_dir (CWE-22 path traversal prevention)."""
    base = Path(base_dir).resolve()
    target = Path(target_path).resolve()

    try:
        common = os.path.commonpath([str(base), str(target)])
    except ValueError as e:
        raise ValueError(f"Path containment check failed across drives or roots: {e}") from e

    if common != str(base):
        raise ValueError(
            f"Path traversal detected: target path '{target}' escapes base directory '{base}'"
        )
    return target


def validate_dimension(dim: int, max_dim: int = MAX_DIM) -> int:
    """Validate vector dimension."""
    if not isinstance(dim, int) or isinstance(dim, bool):
        raise TypeError(f"Dimension must be an integer, got {type(dim).__name__}")
    if dim <= 0:
        raise ValueError(f"Dimension must be positive, got {dim}")
    if dim > max_dim:
        raise ValueError(f"Dimension {dim} exceeds maximum allowed dimension {max_dim}")
    return dim


def validate_vector(
    vector: Sequence[float] | npt.NDArray[Any],
    expected_dim: int | None = None,
    dtype: type[np.float32] = np.float32,
    check_norm: bool = True,
    min_norm: float = MIN_VECTOR_NORM,
    max_dim: int = MAX_DIM,
) -> npt.NDArray[np.float32]:
    """Validate and sanitize vector input.
    
    - Ensures 1D shape
    - Converts to target dtype (float32)
    - Validates dimension matches expected_dim and max_dim
    - Rejects NaN, +Inf, -Inf values
    - Rejects zero-norm vectors (norm < min_norm)
    """
    if vector is None:
        raise ValueError("Vector cannot be None")

    try:
        arr = np.asarray(vector, dtype=dtype)
    except (ValueError, TypeError) as e:
        raise ValueError(f"Failed to convert input to numeric vector array: {e}") from e

    if arr.ndim != 1:
        raise ValueError(f"Vector must be 1-dimensional, got ndim={arr.ndim} with shape {arr.shape}")

    dim = arr.shape[0]
    if dim == 0:
        raise ValueError("Vector cannot be empty (dimension 0)")

    if dim > max_dim:
        raise ValueError(f"Vector dimension {dim} exceeds maximum allowed {max_dim}")

    if expected_dim is not None:
        if dim != expected_dim:
            raise ValueError(
                f"Vector dimension {dim} does not match expected database dimension {expected_dim}"
            )

    # Sanitize and check for finite values (reject NaN, Inf, -Inf)
    if not np.isfinite(arr).all():
        raise ValueError("Vector contains non-finite values (NaN, +Inf, or -Inf)")

    if check_norm:
        norm = float(np.linalg.norm(arr))
        if norm < min_norm:
            raise ValueError(
                f"Zero-norm vector detected (norm {norm:.2e} < {min_norm:.2e})"
            )

    return arr.astype(dtype, copy=False)
