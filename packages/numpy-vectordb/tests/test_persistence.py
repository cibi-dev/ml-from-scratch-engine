"""Tests for hardened collection persistence and allow_pickle=False enforcement."""

import json
from pathlib import Path
import numpy as np
import pytest

from numpy_vectordb.database import VectorDB
from numpy_vectordb.persistence import load_collection_data, save_collection
from numpy_vectordb.similarity import SimilarityMetric


class TestPersistence:
    """Test save/load roundtrips, format integrity, and security."""

    def test_save_and_load_roundtrip_bit_level_equality(self, temp_db_dir: Path) -> None:
        dim = 8
        db = VectorDB(dim=dim, metric=SimilarityMetric.COSINE, storage_dir=temp_db_dir)

        # Generate sample vectors
        num_docs = 25
        vectors = np.random.randn(num_docs, dim).astype(np.float32)
        for i in range(num_docs):
            db.upsert(f"doc_{i}", vectors[i], {"index": i, "tag": f"group_{i % 3}"})

        # Save collection
        npz_path, json_path = db.save("test_collection")
        assert npz_path.exists()
        assert json_path.exists()

        # Load collection into a new VectorDB instance
        loaded_db = VectorDB.load_collection("test_collection", storage_dir=temp_db_dir)

        assert loaded_db.count() == num_docs
        assert loaded_db.dim == dim
        assert loaded_db.metric == SimilarityMetric.COSINE

        # Verify bit-level equality of vectors and metadata
        for i in range(num_docs):
            original_doc = db.get(f"doc_{i}")
            loaded_doc = loaded_db.get(f"doc_{i}")

            assert original_doc is not None
            assert loaded_doc is not None
            assert loaded_doc.id == original_doc.id
            assert np.array_equal(loaded_doc.vector, original_doc.vector)
            assert loaded_doc.metadata == original_doc.metadata

    def test_metadata_persistence_nested_structures(self, temp_db_dir: Path) -> None:
        db = VectorDB(dim=4, storage_dir=temp_db_dir)
        complex_meta = {
            "title": "Hardened VectorDB",
            "count": 42,
            "score": 9.875,
            "active": True,
            "tags": ["security", "blas", "numpy"],
            "nested": {
                "author": "cibi-dev",
                "versions": [1, 2, 3],
                "details": {"flag": None},
            },
        }
        db.upsert("doc_meta", [1.0, 0.0, 0.0, 0.0], complex_meta)
        db.save("meta_col")

        loaded = VectorDB.load_collection("meta_col", storage_dir=temp_db_dir)
        doc = loaded.get("doc_meta")
        assert doc is not None
        assert doc.metadata == complex_meta

    def test_load_into_existing_instance(self, temp_db_dir: Path) -> None:
        db1 = VectorDB(dim=4, storage_dir=temp_db_dir)
        db1.upsert("d1", [1.0, 0.0, 0.0, 0.0], {"meta": 1})
        db1.save("col1")

        # Existing instance with different initial state
        db2 = VectorDB(dim=4, storage_dir=temp_db_dir)
        db2.upsert("old_doc", [0.0, 1.0, 0.0, 0.0], {"old": True})
        assert db2.count() == 1

        db2.load("col1")
        assert db2.count() == 1
        assert db2.get("old_doc") is None
        assert db2.get("d1") is not None

    def test_load_dimension_mismatch_raises_error(self, temp_db_dir: Path) -> None:
        db = VectorDB(dim=4, storage_dir=temp_db_dir)
        db.upsert("d1", [1.0, 0.0, 0.0, 0.0])
        db.save("col4d")

        # Attempt to load 4D collection into 8D instance
        db8 = VectorDB(dim=8, storage_dir=temp_db_dir)
        with pytest.raises(ValueError, match="Collection dimension 4 does not match database dimension 8"):
            db8.load("col4d")

    def test_load_non_existent_collection_raises_filenotfound(self, temp_db_dir: Path) -> None:
        with pytest.raises(FileNotFoundError, match="not found"):
            VectorDB.load_collection("does_not_exist", storage_dir=temp_db_dir)

    def test_allow_pickle_false_enforcement(self, temp_db_dir: Path) -> None:
        """Verify that files are loaded with allow_pickle=False and cannot execute arbitrary objects."""
        db = VectorDB(dim=2, storage_dir=temp_db_dir)
        db.upsert("d1", [1.0, 2.0])
        db.save("pickle_test")

        # Direct call to load_collection_data
        vecs, ids, meta, dim, metric = load_collection_data(temp_db_dir, "pickle_test")
        assert len(vecs) == 1
        assert ids == ["d1"]

    def test_corrupted_json_metadata_raises_valueerror(self, temp_db_dir: Path) -> None:
        db = VectorDB(dim=2, storage_dir=temp_db_dir)
        db.upsert("d1", [1.0, 2.0])
        _, json_path = db.save("corrupt_meta_test")

        # Write invalid JSON content
        json_path.write_text("INVALID_JSON_CONTENT{{{", encoding="utf-8")

        with pytest.raises(ValueError, match="Failed to parse collection metadata JSON"):
            VectorDB.load_collection("corrupt_meta_test", storage_dir=temp_db_dir)

    def test_corrupted_npz_missing_arrays_raises_valueerror(self, temp_db_dir: Path) -> None:
        valid_name = "corrupt_npz_test"
        npz_path = temp_db_dir / f"{valid_name}.npz"
        json_path = temp_db_dir / f"{valid_name}.meta.json"

        # Save an npz missing 'vectors'
        np.savez_compressed(npz_path, wrong_key=np.array([1, 2, 3]))
        json_path.write_text(
            json.dumps({"dimension": 2, "metric": "cosine", "count": 1, "metadata": {}}),
            encoding="utf-8",
        )

        with pytest.raises(ValueError, match="NPZ archive is missing"):
            VectorDB.load_collection(valid_name, storage_dir=temp_db_dir)

    def test_save_empty_collection(self, temp_db_dir: Path) -> None:
        db = VectorDB(dim=4, storage_dir=temp_db_dir)
        db.save("empty_col")

        loaded = VectorDB.load_collection("empty_col", storage_dir=temp_db_dir)
        assert loaded.count() == 0
        assert loaded.dim == 4
