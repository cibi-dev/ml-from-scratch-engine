"""Hardened storage persistence using np.savez_compressed and JSON metadata sidecars."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

from numpy_vectordb.validation import (
    validate_collection_name,
    validate_dimension,
    validate_path_containment,
)


def save_collection(
    base_dir: str | Path,
    name: str,
    vectors: npt.NDArray[np.float32],
    doc_ids: list[str],
    metadata: dict[str, dict[str, Any]],
    dim: int,
    metric: str,
) -> tuple[Path, Path]:
    """Safely save vector collection to disk.
    
    Mitigations:
    - CWE-22: Collection name validated with strict regex and path containment check.
    - CWE-502: doc_ids saved as raw NumPy unicode string array; allow_pickle=False enforced.
    
    Returns:
        tuple of (npz_path, json_path).
    """
    valid_name = validate_collection_name(name)
    base = Path(base_dir).resolve()
    base.mkdir(parents=True, exist_ok=True)

    npz_target = base / f"{valid_name}.npz"
    json_target = base / f"{valid_name}.meta.json"

    npz_path = validate_path_containment(base, npz_target)
    json_path = validate_path_containment(base, json_target)

    # Convert vectors to float32 array
    vec_arr = np.asarray(vectors, dtype=np.float32)
    # Save doc_ids as unicode string array (avoids object array / pickle)
    id_arr = np.asarray(doc_ids, dtype=np.str_)

    if vec_arr.ndim != 2:
        raise ValueError(f"Vectors array must be 2D, got ndim={vec_arr.ndim}")
    if vec_arr.shape[0] != len(doc_ids):
        raise ValueError(
            f"Row count mismatch: {vec_arr.shape[0]} vectors vs {len(doc_ids)} doc IDs"
        )
    if vec_arr.shape[0] > 0 and vec_arr.shape[1] != dim:
        raise ValueError(f"Vector dim {vec_arr.shape[1]} does not match collection dim {dim}")

    # Write compressed numpy archive with allow_pickle=False guarantee
    np.savez_compressed(
        npz_path,
        vectors=vec_arr,
        doc_ids=id_arr,
    )

    # Write sidecar JSON metadata
    meta_payload = {
        "version": 1,
        "name": valid_name,
        "dimension": validate_dimension(dim),
        "metric": metric,
        "count": len(doc_ids),
        "metadata": metadata,
    }

    temp_json = base / f"{valid_name}.meta.json.tmp"
    with open(temp_json, "w", encoding="utf-8") as f:
        json.dump(meta_payload, f, indent=2, ensure_ascii=False)
    temp_json.replace(json_path)

    return npz_path, json_path


def load_collection_data(
    base_dir: str | Path,
    name: str,
) -> tuple[npt.NDArray[np.float32], list[str], dict[str, dict[str, Any]], int, str]:
    """Safely load collection vectors and metadata from disk.
    
    Mitigations:
    - CWE-22: Path traversal checks on collection name.
    - CWE-502: np.load strictly enforces allow_pickle=False.
    
    Returns:
        tuple of (vectors, doc_ids, metadata, dimension, metric).
    """
    valid_name = validate_collection_name(name)
    base = Path(base_dir).resolve()

    npz_target = base / f"{valid_name}.npz"
    json_target = base / f"{valid_name}.meta.json"

    npz_path = validate_path_containment(base, npz_target)
    json_path = validate_path_containment(base, json_target)

    if not npz_path.exists():
        raise FileNotFoundError(f"Collection vector file '{npz_path}' not found.")
    if not json_path.exists():
        raise FileNotFoundError(f"Collection metadata file '{json_path}' not found.")

    # Read sidecar JSON metadata
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            meta_payload = json.load(f)
    except Exception as e:
        raise ValueError(f"Failed to parse collection metadata JSON: {e}") from e

    dimension: int = validate_dimension(int(meta_payload["dimension"]))
    metric: str = str(meta_payload["metric"])
    metadata: dict[str, dict[str, Any]] = meta_payload.get("metadata", {})

    # Load vectors and doc_ids strictly forbidding pickle
    try:
        with np.load(npz_path, allow_pickle=False) as data:
            if "vectors" not in data or "doc_ids" not in data:
                raise ValueError("NPZ archive is missing 'vectors' or 'doc_ids' arrays.")
            vectors_raw = data["vectors"]
            doc_ids_raw = data["doc_ids"]

            vectors = np.asarray(vectors_raw, dtype=np.float32)
            doc_ids = [str(x) for x in doc_ids_raw]
    except Exception as e:
        if isinstance(e, (FileNotFoundError, ValueError)):
            raise
        raise ValueError(f"Failed to safely load vector archive: {e}") from e

    if vectors.ndim != 2:
        raise ValueError(f"Corrupted vectors: expected 2D array, got ndim={vectors.ndim}")

    if vectors.shape[0] != len(doc_ids):
        raise ValueError(
            f"Corrupted archive: {vectors.shape[0]} vectors does not match {len(doc_ids)} IDs."
        )

    if vectors.shape[0] > 0 and vectors.shape[1] != dimension:
        raise ValueError(
            f"Corrupted archive: vector dimension {vectors.shape[1]} does not match collection dimension {dimension}."
        )

    if not np.isfinite(vectors).all():
        raise ValueError("Corrupted archive: vectors contain non-finite values (NaN/Inf).")

    return vectors, doc_ids, metadata, dimension, metric
