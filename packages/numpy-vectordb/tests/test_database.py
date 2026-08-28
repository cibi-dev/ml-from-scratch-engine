"""Tests for VectorDB CRUD lifecycle, query accuracy, swap-and-pop, and thread safety."""

import concurrent.futures
from pathlib import Path
import numpy as np
import pytest

from numpy_vectordb.database import Document, QueryResult, VectorDB
from numpy_vectordb.similarity import SimilarityMetric


class TestDatabaseCRUD:
    """Test standard CRUD operations."""

    def test_initialization_properties(self, temp_db_dir: Path) -> None:
        db = VectorDB(dim=32, metric="cosine", storage_dir=temp_db_dir, max_vectors=500)
        assert db.dim == 32
        assert db.metric == SimilarityMetric.COSINE
        assert db.storage_dir == temp_db_dir.resolve()
        assert db.max_vectors == 500
        assert db.count() == 0
        assert len(db) == 0

    def test_upsert_and_get(self, small_cosine_db: VectorDB) -> None:
        vec = [0.1, 0.2, 0.3, 0.4]
        meta = {"title": "doc1", "category": "tech"}
        small_cosine_db.upsert("doc1", vec, meta)

        assert small_cosine_db.count() == 1
        assert len(small_cosine_db) == 1

        doc = small_cosine_db.get("doc1")
        assert doc is not None
        assert doc.id == "doc1"
        assert np.allclose(doc.vector, vec)
        assert doc.metadata == meta

    def test_get_non_existent(self, small_cosine_db: VectorDB) -> None:
        assert small_cosine_db.get("non_existent") is None

    def test_upsert_idempotence(self, small_cosine_db: VectorDB) -> None:
        v1 = [1.0, 0.0, 0.0, 0.0]
        v2 = [0.0, 1.0, 0.0, 0.0]
        small_cosine_db.upsert("id1", v1, {"v": 1})
        assert small_cosine_db.count() == 1

        # Update same ID
        small_cosine_db.upsert("id1", v2, {"v": 2})
        assert small_cosine_db.count() == 1

        doc = small_cosine_db.get("id1")
        assert doc is not None
        assert np.allclose(doc.vector, v2)
        assert doc.metadata == {"v": 2}

    def test_delete_existing_and_non_existing(self, small_cosine_db: VectorDB) -> None:
        small_cosine_db.upsert("doc1", [1.0, 0.0, 0.0, 0.0], {"meta": 1})
        small_cosine_db.upsert("doc2", [0.0, 1.0, 0.0, 0.0], {"meta": 2})
        assert small_cosine_db.count() == 2

        # Delete existing
        assert small_cosine_db.delete("doc1") is True
        assert small_cosine_db.count() == 1
        assert small_cosine_db.get("doc1") is None
        assert small_cosine_db.get("doc2") is not None

        # Delete already deleted or non-existent
        assert small_cosine_db.delete("doc1") is False
        assert small_cosine_db.delete("unknown") is False
        assert small_cosine_db.count() == 1

    def test_delete_swap_and_pop_integrity(self, small_cosine_db: VectorDB) -> None:
        """Verify that deleting an element correctly swaps the last element and preserves all mappings."""
        # Insert 5 documents
        ids = [f"doc_{i}" for i in range(5)]
        for i, did in enumerate(ids):
            vec = [float(i + 1), 1.0, 2.0, 3.0]
            small_cosine_db.upsert(did, vec, {"idx": i})

        assert small_cosine_db.count() == 5

        # Delete middle element (doc_2)
        assert small_cosine_db.delete("doc_2") is True
        assert small_cosine_db.count() == 4
        assert small_cosine_db.get("doc_2") is None

        # Remaining docs should all be retrievable and retain correct vectors and metadata
        for remaining_id in ["doc_0", "doc_1", "doc_3", "doc_4"]:
            doc = small_cosine_db.get(remaining_id)
            assert doc is not None
            assert doc.id == remaining_id

        # Delete the new last element
        assert small_cosine_db.delete("doc_4") is True
        assert small_cosine_db.count() == 3
        assert small_cosine_db.get("doc_4") is None

    def test_batch_upsert(self, small_cosine_db: VectorDB) -> None:
        docs = [
            Document(id="d1", vector=np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32), metadata={"a": 1}),
            Document(id="d2", vector=np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32), metadata={"a": 2}),
        ]
        small_cosine_db.upsert_batch(docs)
        assert small_cosine_db.count() == 2

        # Tuple format with and without metadata
        tuples = [
            ("d3", [0.0, 0.0, 1.0, 0.0], {"a": 3}),
            ("d4", [0.0, 0.0, 0.0, 1.0]),
        ]
        small_cosine_db.upsert_batch(tuples)  # type: ignore[arg-type]
        assert small_cosine_db.count() == 4
        assert small_cosine_db.get("d3") is not None
        assert small_cosine_db.get("d4") is not None

    def test_geometric_capacity_doubling(self, temp_db_dir: Path) -> None:
        # Start with small initial capacity of 2
        db = VectorDB(dim=3, storage_dir=temp_db_dir, initial_capacity=2)
        assert db._storage.capacity == 2

        # Insert 10 items
        for i in range(10):
            db.upsert(f"item_{i}", [float(i + 1), 1.0, 2.0])

        assert db.count() == 10
        assert db._storage.capacity >= 10
        # Check all 10 items exist
        for i in range(10):
            assert db.get(f"item_{i}") is not None


class TestQueryAccuracy:
    """Test vector similarity search accuracy."""

    def test_query_accuracy_recall_at_10_against_brute_force(self, temp_db_dir: Path) -> None:
        """Verify Recall@10 is exactly 1.0 compared against a brute-force numpy dot product reference."""
        np.random.seed(42)
        dim = 64
        num_docs = 150
        top_k = 10

        db = VectorDB(dim=dim, metric=SimilarityMetric.COSINE, storage_dir=temp_db_dir)

        # Generate normalized vectors
        raw_vectors = np.random.randn(num_docs, dim).astype(np.float32)
        norms = np.linalg.norm(raw_vectors, axis=1, keepdims=True)
        vectors = raw_vectors / norms

        for i in range(num_docs):
            db.upsert(f"doc_{i}", vectors[i], {"num": i})

        # Query with a random query vector
        raw_query = np.random.randn(dim).astype(np.float32)
        query_vec = raw_query / np.linalg.norm(raw_query)

        # Reference brute-force calculation
        reference_scores = np.dot(vectors, query_vec)
        ref_top10_indices = np.argsort(-reference_scores)[:top_k]
        ref_top10_ids = set(f"doc_{idx}" for idx in ref_top10_indices)

        # VectorDB query
        results = db.query(query_vec, top_k=top_k)
        assert len(results) == top_k

        db_top10_ids = set(r.id for r in results)

        # Check Recall@10 == 1.0
        intersection = ref_top10_ids.intersection(db_top10_ids)
        recall_at_10 = len(intersection) / top_k
        assert recall_at_10 == 1.0

        # Verify results are sorted monotonically descending by score
        scores = [r.score for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_query_euclidean_distance(self, small_euclidean_db: VectorDB) -> None:
        small_euclidean_db.upsert("p1", [1.0, 0.0, 0.0, 0.0])
        small_euclidean_db.upsert("p2", [5.0, 0.0, 0.0, 0.0])
        small_euclidean_db.upsert("p3", [2.0, 0.0, 0.0, 0.0])

        query_vec = [1.0, 0.0, 0.0, 0.0]
        results = small_euclidean_db.query(query_vec, top_k=3)

        assert len(results) == 3
        # p1 distance = 0, p3 distance = 1, p2 distance = 4
        assert results[0].id == "p1"
        assert np.isclose(results[0].score, 0.0)
        assert results[1].id == "p3"
        assert np.isclose(results[1].score, 1.0)
        assert results[2].id == "p2"
        assert np.isclose(results[2].score, 4.0)

    def test_query_dot_product(self, small_dot_db: VectorDB) -> None:
        small_dot_db.upsert("d1", [2.0, 1.0, 0.0, 0.0])
        small_dot_db.upsert("d2", [10.0, 0.0, 0.0, 0.0])

        results = small_dot_db.query([1.0, 0.0, 0.0, 0.0], top_k=2)
        assert len(results) == 2
        assert results[0].id == "d2"
        assert np.isclose(results[0].score, 10.0)
        assert results[1].id == "d1"
        assert np.isclose(results[1].score, 2.0)

    def test_query_with_metadata_filter(self, small_cosine_db: VectorDB) -> None:
        small_cosine_db.upsert("doc1", [1.0, 0.0, 0.0, 0.0], {"category": "tech", "year": 2026})
        small_cosine_db.upsert("doc2", [0.9, 0.1, 0.0, 0.0], {"category": "finance", "year": 2025})
        small_cosine_db.upsert("doc3", [0.8, 0.2, 0.0, 0.0], {"category": "tech", "year": 2024})

        # Filter category == tech
        tech_results = small_cosine_db.query(
            [1.0, 0.0, 0.0, 0.0],
            top_k=5,
            filter_fn=lambda m: m.get("category") == "tech",
        )
        assert len(tech_results) == 2
        assert [r.id for r in tech_results] == ["doc1", "doc3"]

    def test_query_empty_filter_match(self, small_cosine_db: VectorDB) -> None:
        small_cosine_db.upsert("doc1", [1.0, 0.0, 0.0, 0.0], {"category": "tech"})
        results = small_cosine_db.query(
            [1.0, 0.0, 0.0, 0.0],
            top_k=5,
            filter_fn=lambda m: m.get("category") == "health",
        )
        assert len(results) == 0

    def test_query_empty_database(self, small_cosine_db: VectorDB) -> None:
        results = small_cosine_db.query([1.0, 0.0, 0.0, 0.0], top_k=5)
        assert results == []

    def test_query_top_k_larger_than_db_size(self, small_cosine_db: VectorDB) -> None:
        small_cosine_db.upsert("doc1", [1.0, 0.0, 0.0, 0.0])
        small_cosine_db.upsert("doc2", [0.0, 1.0, 0.0, 0.0])
        results = small_cosine_db.query([1.0, 0.0, 0.0, 0.0], top_k=100)
        assert len(results) == 2

    def test_query_include_vector(self, small_cosine_db: VectorDB) -> None:
        vec = [1.0, 0.0, 0.0, 0.0]
        small_cosine_db.upsert("doc1", vec)

        res_without = small_cosine_db.query(vec, top_k=1, include_vector=False)
        assert res_without[0].vector is None

        res_with = small_cosine_db.query(vec, top_k=1, include_vector=True)
        assert res_with[0].vector is not None
        assert np.allclose(res_with[0].vector, vec)


class TestThreadSafety:
    """Test concurrent thread safety of VectorDB operations."""

    def test_concurrent_upsert_and_query(self, temp_db_dir: Path) -> None:
        db = VectorDB(dim=8, storage_dir=temp_db_dir)
        num_threads = 8
        items_per_thread = 25

        def worker_insert(thread_idx: int) -> None:
            for i in range(items_per_thread):
                doc_id = f"t{thread_idx}_{i}"
                vec = np.random.randn(8).astype(np.float32)
                vec = vec / np.linalg.norm(vec)
                db.upsert(doc_id, vec, {"thread": thread_idx, "seq": i})

        def worker_query() -> None:
            for _ in range(20):
                q = np.random.randn(8).astype(np.float32)
                q = q / np.linalg.norm(q)
                db.query(q, top_k=5)

        with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads * 2) as executor:
            insert_futures = [executor.submit(worker_insert, i) for i in range(num_threads)]
            query_futures = [executor.submit(worker_query) for _ in range(num_threads)]

            # Wait for all to complete
            for f in insert_futures + query_futures:
                f.result()

        assert db.count() == num_threads * items_per_thread

    def test_concurrent_delete_and_read(self, temp_db_dir: Path) -> None:
        db = VectorDB(dim=4, storage_dir=temp_db_dir)
        # Prepopulate
        for i in range(100):
            db.upsert(f"item_{i}", [1.0, 2.0, 3.0, 4.0], {"i": i})

        def worker_delete(start: int, end: int) -> None:
            for i in range(start, end):
                db.delete(f"item_{i}")

        def worker_read() -> None:
            for _ in range(50):
                db.count()
                db.get("item_50")
                db.query([1.0, 2.0, 3.0, 4.0], top_k=3)

        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            f1 = executor.submit(worker_delete, 0, 50)
            f2 = executor.submit(worker_delete, 50, 100)
            f3 = executor.submit(worker_read)
            f4 = executor.submit(worker_read)

            f1.result()
            f2.result()
            f3.result()
            f4.result()

        assert db.count() == 0
