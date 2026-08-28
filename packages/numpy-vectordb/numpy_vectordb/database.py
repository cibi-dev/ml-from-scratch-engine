"""Thread-safe, hardened Vector Database using pure NumPy."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import threading
from typing import Any, Callable, Sequence

import numpy as np
import numpy.typing as npt

from numpy_vectordb.persistence import load_collection_data, save_collection
from numpy_vectordb.similarity import (
    SimilarityMetric,
    compute_cosine_similarity,
    compute_dot_product,
    compute_euclidean_distance,
    top_k_indices_and_scores,
)
from numpy_vectordb.storage import VectorStorage
from numpy_vectordb.validation import (
    MAX_DIM,
    MAX_VECTORS,
    validate_dimension,
    validate_doc_id,
    validate_vector,
)


@dataclass(frozen=True)
class Document:
    """Represents a document stored in the database."""

    id: str
    vector: npt.NDArray[np.float32]
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class QueryResult:
    """Result of a similarity search query."""

    id: str
    score: float
    metadata: dict[str, Any] = field(default_factory=dict)
    vector: npt.NDArray[np.float32] | None = None


class VectorDB:
    """Hardened, thread-safe in-memory vector database powered by contiguous NumPy buffers."""

    def __init__(
        self,
        dim: int,
        metric: SimilarityMetric | str = SimilarityMetric.COSINE,
        storage_dir: str | Path = "./data",
        max_vectors: int = MAX_VECTORS,
        initial_capacity: int = 16,
    ) -> None:
        self._dim = validate_dimension(dim, max_dim=MAX_DIM)
        self._metric = SimilarityMetric.from_str(metric)
        self._storage_dir = Path(storage_dir).resolve()
        self._max_vectors = max_vectors
        self._storage = VectorStorage(
            dim=self._dim,
            initial_capacity=initial_capacity,
            max_vectors=self._max_vectors,
        )
        self._metadata: dict[str, dict[str, Any]] = {}
        self._lock = threading.RLock()

    @property
    def dim(self) -> int:
        """Vector dimensionality."""
        return self._dim

    @property
    def metric(self) -> SimilarityMetric:
        """Similarity metric in use."""
        return self._metric

    @property
    def storage_dir(self) -> Path:
        """Directory used for persistence."""
        return self._storage_dir

    @property
    def max_vectors(self) -> int:
        """Configured upper limit on vector count."""
        return self._max_vectors

    def count(self) -> int:
        """Return the number of stored vectors."""
        with self._lock:
            return self._storage.size

    def __len__(self) -> int:
        return self.count()

    def upsert(
        self,
        doc_id: str,
        vector: Sequence[float] | npt.NDArray[Any],
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Insert or update a document vector and its metadata.
        
        Args:
            doc_id: Unique document identifier.
            vector: 1D numeric vector of dimension matching self.dim.
            metadata: Optional dictionary of metadata attributes.
        """
        valid_id = validate_doc_id(doc_id)
        valid_vec = validate_vector(vector, expected_dim=self._dim)
        meta_copy = dict(metadata) if metadata is not None else {}

        with self._lock:
            self._storage.add(valid_id, valid_vec)
            self._metadata[valid_id] = meta_copy

    def upsert_batch(
        self,
        documents: Sequence[Document]
        | Sequence[tuple[str, Sequence[float] | npt.NDArray[Any]]]
        | Sequence[tuple[str, Sequence[float] | npt.NDArray[Any], dict[str, Any] | None]],
    ) -> None:
        """Batch upsert documents."""
        with self._lock:
            for item in documents:
                if isinstance(item, Document):
                    self.upsert(item.id, item.vector, item.metadata)
                elif isinstance(item, tuple):
                    if len(item) == 2:
                        self.upsert(item[0], item[1], None)
                    elif len(item) == 3:
                        self.upsert(item[0], item[1], item[2])
                    else:
                        raise ValueError("Tuple must have 2 or 3 elements: (id, vector[, metadata])")
                else:
                    raise TypeError(f"Unsupported batch document type: {type(item).__name__}")

    def delete(self, doc_id: str) -> bool:
        """Delete document by ID using O(1) swap-and-pop defragmentation.
        
        Returns:
            True if removed, False if doc_id did not exist.
        """
        valid_id = validate_doc_id(doc_id)
        with self._lock:
            deleted = self._storage.delete(valid_id)
            if deleted:
                self._metadata.pop(valid_id, None)
            return deleted

    def get(self, doc_id: str) -> Document | None:
        """Retrieve a document by ID."""
        valid_id = validate_doc_id(doc_id)
        with self._lock:
            vec = self._storage.get_vector(valid_id)
            if vec is None:
                return None
            meta = dict(self._metadata.get(valid_id, {}))
            return Document(id=valid_id, vector=vec, metadata=meta)

    def query(
        self,
        vector: Sequence[float] | npt.NDArray[Any],
        top_k: int = 10,
        filter_fn: Callable[[dict[str, Any]], bool] | None = None,
        include_vector: bool = False,
    ) -> list[QueryResult]:
        """Perform top-K similarity search.
        
        Args:
            vector: Query vector matching database dimension.
            top_k: Maximum number of results to return (must be > 0).
            filter_fn: Optional predicate filtering documents based on metadata.
            include_vector: If True, attach vector arrays to QueryResult objects.
            
        Returns:
            Sorted list of QueryResult instances.
        """
        if not isinstance(top_k, int) or isinstance(top_k, bool) or top_k <= 0:
            raise ValueError(f"top_k must be a positive integer > 0, got {top_k}")

        valid_query = validate_vector(vector, expected_dim=self._dim)

        with self._lock:
            if self._storage.size == 0:
                return []

            active_matrix = self._storage.get_active_view()
            doc_ids = self._storage.get_doc_ids()

            # Metadata filtering
            if filter_fn is not None:
                matching_indices: list[int] = []
                filtered_ids: list[str] = []
                for idx, did in enumerate(doc_ids):
                    meta = self._metadata.get(did, {})
                    if filter_fn(meta):
                        matching_indices.append(idx)
                        filtered_ids.append(did)

                if not matching_indices:
                    return []

                matrix_to_query = active_matrix[matching_indices]
                doc_ids_to_query = filtered_ids
            else:
                matrix_to_query = active_matrix
                doc_ids_to_query = doc_ids

            # Compute metric scores
            if self._metric == SimilarityMetric.COSINE:
                scores = compute_cosine_similarity(valid_query, matrix_to_query)
                largest = True
            elif self._metric == SimilarityMetric.EUCLIDEAN:
                scores = compute_euclidean_distance(valid_query, matrix_to_query)
                largest = False
            elif self._metric == SimilarityMetric.DOT_PRODUCT:
                scores = compute_dot_product(valid_query, matrix_to_query)
                largest = True
            else:
                raise ValueError(f"Unsupported similarity metric {self._metric}")

            # Top-K partitioning
            top_indices, top_scores = top_k_indices_and_scores(
                scores, top_k=top_k, largest=largest
            )

            results: list[QueryResult] = []
            for rank_idx, score in zip(top_indices, top_scores):
                did = doc_ids_to_query[int(rank_idx)]
                meta = dict(self._metadata.get(did, {}))
                vec = self._storage.get_vector(did) if include_vector else None
                results.append(
                    QueryResult(
                        id=did,
                        score=float(score),
                        metadata=meta,
                        vector=vec,
                    )
                )

            return results

    def save(self, name: str) -> tuple[Path, Path]:
        """Save the database collection to the configured storage directory."""
        with self._lock:
            return save_collection(
                base_dir=self._storage_dir,
                name=name,
                vectors=self._storage.get_active_matrix(),
                doc_ids=self._storage.get_doc_ids(),
                metadata=self._metadata,
                dim=self._dim,
                metric=self._metric.value,
            )

    def load(self, name: str) -> None:
        """Load a collection from the configured storage directory into this instance."""
        with self._lock:
            vectors, doc_ids, metadata, dim, metric = load_collection_data(
                base_dir=self._storage_dir,
                name=name,
            )
            if dim != self._dim:
                raise ValueError(
                    f"Collection dimension {dim} does not match database dimension {self._dim}"
                )

            self._metric = SimilarityMetric.from_str(metric)
            self._storage = VectorStorage(
                dim=self._dim,
                initial_capacity=max(len(doc_ids), 16),
                max_vectors=self._max_vectors,
            )
            self._metadata.clear()

            for did, vec in zip(doc_ids, vectors):
                self._storage.add(did, vec)
                self._metadata[did] = dict(metadata.get(did, {}))

    @classmethod
    def load_collection(
        cls,
        name: str,
        storage_dir: str | Path = "./data",
    ) -> VectorDB:
        """Load an existing collection and return a new VectorDB instance."""
        vectors, doc_ids, metadata, dim, metric = load_collection_data(
            base_dir=storage_dir,
            name=name,
        )
        db = cls(
            dim=dim,
            metric=metric,
            storage_dir=storage_dir,
            initial_capacity=max(len(doc_ids), 16),
        )
        for did, vec in zip(doc_ids, vectors):
            db.upsert(did, vec, metadata=metadata.get(did, {}))
        return db
