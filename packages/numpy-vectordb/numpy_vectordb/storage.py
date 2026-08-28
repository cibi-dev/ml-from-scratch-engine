"""Contiguous NumPy vector storage with geometric doubling and swap-and-pop defragmentation."""

from __future__ import annotations

from typing import cast

import numpy as np
import numpy.typing as npt

from numpy_vectordb.validation import MAX_VECTORS, validate_dimension


class VectorStorage:
    """Contiguous 2D NumPy array storage with O(1) swap-and-pop deletions."""

    def __init__(
        self,
        dim: int,
        initial_capacity: int = 16,
        max_vectors: int = MAX_VECTORS,
    ) -> None:
        self._dim = validate_dimension(dim)
        self._max_vectors = max_vectors
        self._initial_capacity = max(1, initial_capacity)

        # Preallocate contiguous float32 buffer
        self._buffer: npt.NDArray[np.float32] = np.zeros(
            (self._initial_capacity, self._dim), dtype=np.float32
        )
        self._size: int = 0
        self._id_to_idx: dict[str, int] = {}
        self._idx_to_id: list[str] = []

    @property
    def dim(self) -> int:
        """Vector dimensionality."""
        return self._dim

    @property
    def size(self) -> int:
        """Number of active vectors currently stored."""
        return self._size

    @property
    def capacity(self) -> int:
        """Current buffer allocated row capacity."""
        return int(self._buffer.shape[0])

    @property
    def max_vectors(self) -> int:
        """Maximum allowed vectors in storage."""
        return self._max_vectors

    def _grow(self) -> None:
        """Geometrically double the underlying contiguous buffer capacity."""
        current_capacity = self.capacity
        new_capacity = current_capacity * 2
        if new_capacity > self._max_vectors:
            new_capacity = self._max_vectors

        if new_capacity <= current_capacity:
            raise ValueError(
                f"Storage capacity limit reached: cannot grow beyond {self._max_vectors} vectors."
            )

        new_buffer = np.zeros((new_capacity, self._dim), dtype=np.float32)
        if self._size > 0:
            new_buffer[: self._size] = self._buffer[: self._size]
        self._buffer = new_buffer

    def add(self, doc_id: str, vector: npt.NDArray[np.float32]) -> int:
        """Add or update a vector in storage (upsert).
        
        Returns:
            The row index in the buffer where the vector is stored.
        """
        # In-place update if doc_id already exists
        if doc_id in self._id_to_idx:
            idx = self._id_to_idx[doc_id]
            self._buffer[idx] = vector
            return idx

        # Capacity check
        if self._size >= self._max_vectors:
            raise ValueError(
                f"Maximum vector capacity ({self._max_vectors}) reached. Cannot insert new document."
            )

        if self._size >= self.capacity:
            self._grow()

        idx = self._size
        self._buffer[idx] = vector
        self._id_to_idx[doc_id] = idx
        self._idx_to_id.append(doc_id)
        self._size += 1
        return idx

    def delete(self, doc_id: str) -> bool:
        """Delete a vector using O(1) swap-and-pop defragmentation.
        
        Returns:
            True if deleted, False if doc_id was not found.
        """
        if doc_id not in self._id_to_idx:
            return False

        idx_to_remove = self._id_to_idx[doc_id]
        last_idx = self._size - 1

        if idx_to_remove != last_idx:
            last_doc_id = self._idx_to_id[last_idx]
            # Swap: copy last row into the removed slot
            self._buffer[idx_to_remove] = self._buffer[last_idx]
            # Update index mappings for the swapped element
            self._id_to_idx[last_doc_id] = idx_to_remove
            self._idx_to_id[idx_to_remove] = last_doc_id

        # Pop
        self._idx_to_id.pop()
        del self._id_to_idx[doc_id]
        # Hygiene: zero out old position
        self._buffer[last_idx].fill(0.0)
        self._size -= 1
        return True

    def get_vector(self, doc_id: str) -> npt.NDArray[np.float32] | None:
        """Retrieve a copy of a vector by document ID."""
        if doc_id in self._id_to_idx:
            idx = self._id_to_idx[doc_id]
            res = self._buffer[idx].copy()
            return cast(npt.NDArray[np.float32], res)
        return None

    def get_active_view(self) -> npt.NDArray[np.float32]:
        """Return a slice view of the active vectors buffer."""
        return self._buffer[: self._size]

    def get_active_matrix(self) -> npt.NDArray[np.float32]:
        """Return an independent copy of the active vectors buffer."""
        return self._buffer[: self._size].copy()

    def get_doc_ids(self) -> list[str]:
        """Return list of all active document IDs in buffer order."""
        return list(self._idx_to_id)

    def has(self, doc_id: str) -> bool:
        """Check if document ID exists in storage."""
        return doc_id in self._id_to_idx
