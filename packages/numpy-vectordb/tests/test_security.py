"""Comprehensive security hardening and vulnerability mitigation tests."""

import json
from pathlib import Path
from typing import Any
import numpy as np
import pytest

from numpy_vectordb.database import VectorDB
from numpy_vectordb.persistence import load_collection_data
from numpy_vectordb.validation import (
    MAX_DIM,
    validate_collection_name,
    validate_doc_id,
    validate_path_containment,
    validate_vector,
)


class TestPathTraversalMitigation:
    """CWE-22 Path Traversal prevention tests."""

    @pytest.mark.parametrize(
        "malicious_name",
        [
            "../../../etc/passwd",
            "/etc/shadow",
            "../../root",
            "sub/../../escape",
            "col/../../etc",
            "col\\..\\..\\windows",
            "../secret",
            "..\\config",
        ],
    )
    def test_path_traversal_collection_name_rejected(self, malicious_name: str) -> None:
        with pytest.raises(ValueError, match="Invalid collection name"):
            validate_collection_name(malicious_name)

    @pytest.mark.parametrize(
        "invalid_name",
        [
            "",
            "   ",
            "collection name",
            "col;rm -rf /",
            "col*name",
            "col$name",
            "col\nname",
            "col\tname",
            "col|name",
            "col`name`",
        ],
    )
    def test_invalid_characters_collection_name_rejected(self, invalid_name: str) -> None:
        with pytest.raises((ValueError, TypeError)):
            validate_collection_name(invalid_name)

    def test_path_containment_function(self, tmp_path: Path) -> None:
        base_dir = tmp_path / "sandbox"
        base_dir.mkdir()

        # Valid contained path
        valid_path = base_dir / "valid_file.npz"
        assert validate_path_containment(base_dir, valid_path) == valid_path.resolve()

        # Malicious traversal path
        malicious_path = base_dir / ".." / "escaped.txt"
        with pytest.raises(ValueError, match="Path traversal detected"):
            validate_path_containment(base_dir, malicious_path)


class TestVectorValidationAndSanitization:
    """Input sanitization, non-finite rejection, and zero-norm guards."""

    def test_nan_vector_injection_rejected(self, small_cosine_db: VectorDB) -> None:
        nan_vec = [1.0, np.nan, 0.0, 0.0]
        # Insert
        with pytest.raises(ValueError, match="non-finite values"):
            small_cosine_db.upsert("doc_nan", nan_vec)
        # Query
        with pytest.raises(ValueError, match="non-finite values"):
            small_cosine_db.query(nan_vec)

    def test_pos_inf_vector_injection_rejected(self, small_cosine_db: VectorDB) -> None:
        pos_inf_vec = [1.0, np.inf, 0.0, 0.0]
        with pytest.raises(ValueError, match="non-finite values"):
            small_cosine_db.upsert("doc_inf", pos_inf_vec)
        with pytest.raises(ValueError, match="non-finite values"):
            small_cosine_db.query(pos_inf_vec)

    def test_neg_inf_vector_injection_rejected(self, small_cosine_db: VectorDB) -> None:
        neg_inf_vec = [-np.inf, 1.0, 0.0, 0.0]
        with pytest.raises(ValueError, match="non-finite values"):
            small_cosine_db.upsert("doc_neginf", neg_inf_vec)
        with pytest.raises(ValueError, match="non-finite values"):
            small_cosine_db.query(neg_inf_vec)

    def test_zero_norm_vector_rejected(self, small_cosine_db: VectorDB) -> None:
        zero_vec = [0.0, 0.0, 0.0, 0.0]
        with pytest.raises(ValueError, match="Zero-norm vector detected"):
            small_cosine_db.upsert("doc_zero", zero_vec)
        with pytest.raises(ValueError, match="Zero-norm vector detected"):
            small_cosine_db.query(zero_vec)

    def test_subnormal_norm_vector_rejected(self, small_cosine_db: VectorDB) -> None:
        subnormal_vec = [1e-15, 1e-15, 1e-15, 1e-15]
        with pytest.raises(ValueError, match="Zero-norm vector detected"):
            small_cosine_db.upsert("doc_subnormal", subnormal_vec)

    def test_dimension_mismatch_rejected(self, small_cosine_db: VectorDB) -> None:
        # DB dimension is 4
        vec_3d = [1.0, 2.0, 3.0]
        with pytest.raises(ValueError, match="dimension 3 does not match"):
            small_cosine_db.upsert("doc_3d", vec_3d)

        with pytest.raises(ValueError, match="dimension 3 does not match"):
            small_cosine_db.query(vec_3d)

    def test_wrong_ndim_rejected(self) -> None:
        # 2D array passed to 1D validator
        with pytest.raises(ValueError, match="1-dimensional"):
            validate_vector([[1.0, 2.0], [3.0, 4.0]])  # type: ignore[list-item]

    def test_empty_vector_rejected(self) -> None:
        with pytest.raises(ValueError, match="cannot be empty"):
            validate_vector([])

    def test_none_vector_rejected(self) -> None:
        with pytest.raises(ValueError, match="cannot be None"):
            validate_vector(None)  # type: ignore[arg-type]

    def test_non_numeric_vector_rejected(self) -> None:
        with pytest.raises(ValueError, match="numeric"):
            validate_vector(["abc", "def"])  # type: ignore[list-item]


class TestDocIdValidation:
    """Document ID format and type checks."""

    def test_empty_doc_id_rejected(self, small_cosine_db: VectorDB) -> None:
        with pytest.raises(ValueError, match="non-empty string"):
            small_cosine_db.upsert("", [1.0, 0.0, 0.0, 0.0])

    def test_whitespace_doc_id_rejected(self, small_cosine_db: VectorDB) -> None:
        with pytest.raises(ValueError, match="non-empty string"):
            small_cosine_db.upsert("   ", [1.0, 0.0, 0.0, 0.0])

    def test_non_string_doc_id_rejected(self, small_cosine_db: VectorDB) -> None:
        with pytest.raises(TypeError, match="must be a string"):
            small_cosine_db.upsert(12345, [1.0, 0.0, 0.0, 0.0])  # type: ignore[arg-type]


class TestResourceCapsAndLimits:
    """Resource bounds enforcement (MAX_DIM, MAX_VECTORS)."""

    def test_max_dim_boundary(self) -> None:
        # Exceeds MAX_DIM
        with pytest.raises(ValueError, match="exceeds maximum allowed"):
            VectorDB(dim=MAX_DIM + 1)

        # Zero or negative dim
        with pytest.raises(ValueError, match="Dimension must be positive"):
            VectorDB(dim=0)

        with pytest.raises(ValueError, match="Dimension must be positive"):
            VectorDB(dim=-10)

    def test_max_vectors_limit_enforced(self, temp_db_dir: Path) -> None:
        db = VectorDB(dim=2, storage_dir=temp_db_dir, max_vectors=3, initial_capacity=2)
        db.upsert("d1", [1.0, 0.0])
        db.upsert("d2", [0.0, 1.0])
        db.upsert("d3", [1.0, 1.0])

        # 4th insertion must fail
        with pytest.raises(ValueError, match="Maximum vector capacity"):
            db.upsert("d4", [2.0, 2.0])

    def test_query_invalid_top_k_rejected(self, small_cosine_db: VectorDB) -> None:
        with pytest.raises(ValueError, match="top_k must be a positive integer"):
            small_cosine_db.query([1.0, 0.0, 0.0, 0.0], top_k=0)

        with pytest.raises(ValueError, match="top_k must be a positive integer"):
            small_cosine_db.query([1.0, 0.0, 0.0, 0.0], top_k=-5)

        with pytest.raises(ValueError, match="top_k must be a positive integer"):
            small_cosine_db.query([1.0, 0.0, 0.0, 0.0], top_k="5")  # type: ignore[arg-type]


class TestPickleExploitMitigation:
    """CWE-502 Deserialization of Untrusted Data mitigation tests."""

    def test_pickle_exploit_payload_rejected_by_allow_pickle_false(self, temp_db_dir: Path) -> None:
        """Simulate a malicious .npz file containing a pickled Python object payload."""
        collection_name = "malicious_exploit"
        npz_path = temp_db_dir / f"{collection_name}.npz"
        json_path = temp_db_dir / f"{collection_name}.meta.json"

        # Create a Python object that would execute arbitrary code if unpickled
        class ExploitPayload:
            def __reduce__(self) -> tuple[Any, tuple[str]]:
                import os
                return (os.system, ("echo VULNERABLE",))

        exploit_obj = ExploitPayload()
        # Create an object array containing the pickled exploit
        obj_array = np.array([exploit_obj], dtype=object)

        # Write directly to npz using pickle (mimicking attacker-crafted file)
        np.savez_compressed(
            npz_path,
            vectors=np.array([[1.0, 2.0]], dtype=np.float32),
            doc_ids=obj_array,
            allow_pickle=True,
        )

        json_path.write_text(
            json.dumps({"dimension": 2, "metric": "cosine", "count": 1, "metadata": {}}),
            encoding="utf-8",
        )

        # Loading with load_collection_data MUST reject the pickle payload and raise ValueError
        with pytest.raises(ValueError, match="allow_pickle=False|Object arrays cannot be loaded"):
            load_collection_data(temp_db_dir, collection_name)
