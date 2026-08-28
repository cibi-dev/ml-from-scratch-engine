"""Locality-Sensitive Hashing (LSH) indexing with DoS protection and parameter optimization."""

from __future__ import annotations

from typing import Union

import numpy as np

IdType = Union[int, str]


def _trapezoid_integration(y: np.ndarray, x: np.ndarray) -> float:
    """Calculate the definite integral using the composite trapezoidal rule."""
    if len(x) <= 1:
        return 0.0
    dx = x[1:] - x[:-1]
    avg_y = (y[:-1] + y[1:]) * 0.5
    return float(np.sum(dx * avg_y))


def optimize_lsh_parameters(
    threshold: float,
    num_perm: int,
    weights: tuple[float, float] = (0.5, 0.5),
) -> tuple[int, int]:
    """Find optimal (b, r) parameters minimizing weighted false positive and false negative area.

    The theoretical collision probability curve is:
        P(s) = 1 - (1 - s^r)^b

    Args:
        threshold: Jaccard similarity threshold in (0, 1).
        num_perm: Total number of MinHash permutations (K).
        weights: Tuple of (w_fp, w_fn) penalty weights for false positives and false negatives.

    Returns:
        Tuple of (b, r) where b * r <= num_perm.

    Raises:
        ValueError: If threshold or weights are out of valid bounds.
    """
    if not (0.0 < threshold < 1.0):
        raise ValueError(f"threshold must be in (0, 1), got {threshold}")
    if num_perm < 1:
        raise ValueError(f"num_perm must be >= 1, got {num_perm}")
    if weights[0] < 0.0 or weights[1] < 0.0 or (weights[0] == 0.0 and weights[1] == 0.0):
        raise ValueError(f"weights must be non-negative with non-zero sum, got {weights}")

    w_fp, w_fn = weights
    s = np.linspace(0.0, 1.0, 1001)
    mask_fp = s < threshold
    mask_fn = s >= threshold

    s_fp = s[mask_fp]
    s_fn = s[mask_fn]

    best_loss = float("inf")
    best_params = (1, num_perm)

    # Search all candidate (b, r) factorings where b * r <= num_perm
    for b in range(1, num_perm + 1):
        for r in range(1, (num_perm // b) + 1):
            # Compute S-curve probability for all s
            # Avoid overflow/underflow
            p_s = 1.0 - (1.0 - np.power(s, r)) ** b

            p_fp = p_s[mask_fp]
            p_fn = 1.0 - p_s[mask_fn]

            fp_area = _trapezoid_integration(p_fp, s_fp)
            fn_area = _trapezoid_integration(p_fn, s_fn)

            loss = w_fp * fp_area + w_fn * fn_area

            # Prefer pairs that utilize more permutations or have lower loss
            if loss < best_loss:
                best_loss = loss
                best_params = (b, r)
            elif abs(loss - best_loss) < 1e-9:
                # Tie breaker: prefer utilizing more permutations
                if (b * r) > (best_params[0] * best_params[1]):
                    best_params = (b, r)

    return best_params


class MinHashLSH:
    """MinHash Locality-Sensitive Hashing index with band slicing and Hash-DoS guards."""

    MAX_BUCKET_SIZE: int = 5000

    def __init__(
        self,
        threshold: float = 0.75,
        num_perm: int = 128,
        weights: tuple[float, float] = (0.5, 0.5),
        b: int | None = None,
        r: int | None = None,
    ) -> None:
        """Initialize MinHashLSH index.

        Args:
            threshold: Jaccard similarity threshold for candidate pairing.
            num_perm: Dimension of MinHash signatures.
            weights: (w_fp, w_fn) weights for (b, r) parameter optimization.
            b: Optional explicit number of bands. If provided, r must also be provided.
            r: Optional explicit number of rows per band. If provided, b must also be provided.

        Raises:
            ValueError: If parameters are invalid or b * r > num_perm.
        """
        if not (0.0 < threshold < 1.0):
            raise ValueError(f"threshold must be in (0, 1), got {threshold}")
        if num_perm < 1:
            raise ValueError(f"num_perm must be >= 1, got {num_perm}")

        self.threshold = threshold
        self.num_perm = num_perm

        if (b is not None and r is None) or (b is None and r is not None):
            raise ValueError("b and r must either both be specified or both be None")

        if b is not None and r is not None:
            if b < 1 or r < 1:
                raise ValueError(f"b and r must be >= 1, got b={b}, r={r}")
            if b * r > num_perm:
                raise ValueError(f"b * r ({b * r}) cannot exceed num_perm ({num_perm})")
            self.b = b
            self.r = r
        else:
            self.b, self.r = optimize_lsh_parameters(threshold, num_perm, weights)

        # Hash tables: one dictionary per band mapping band byte-hash to document IDs
        self.tables: list[dict[bytes, list[IdType]]] = [{} for _ in range(self.b)]
        # Track saturated buckets to guard against Hash DoS attacks
        self._saturated_buckets: list[set[bytes]] = [set() for _ in range(self.b)]

    @property
    def optimal_threshold(self) -> float:
        """Inflection point threshold where collision probability is approximately 0.5."""
        return float((1.0 / self.b) ** (1.0 / self.r))

    def insert(self, doc_id: IdType, signature: np.ndarray) -> None:
        """Insert a document ID and its MinHash signature into the LSH index.

        Args:
            doc_id: Unique identifier for the document.
            signature: MinHash signature array of shape (num_perm,) and dtype uint64.

        Raises:
            ValueError: If signature length is smaller than b * r.
        """
        if signature.ndim != 1 or len(signature) < self.b * self.r:
            raise ValueError(
                f"Signature length {len(signature)} must be >= b * r ({self.b * self.r})"
            )

        for band_idx in range(self.b):
            band_slice = signature[band_idx * self.r : (band_idx + 1) * self.r]
            band_key = band_slice.tobytes()

            bucket = self.tables[band_idx].setdefault(band_key, [])
            if len(bucket) < self.MAX_BUCKET_SIZE:
                bucket.append(doc_id)
            else:
                self._saturated_buckets[band_idx].add(band_key)

    def query_candidates(self, signature: np.ndarray) -> set[IdType]:
        """Query candidate document IDs sharing at least one band bucket.

        Args:
            signature: MinHash signature array.

        Returns:
            Set of candidate document IDs.

        Raises:
            ValueError: If signature length is smaller than b * r.
        """
        if signature.ndim != 1 or len(signature) < self.b * self.r:
            raise ValueError(
                f"Signature length {len(signature)} must be >= b * r ({self.b * self.r})"
            )

        candidates: set[IdType] = set()
        for band_idx in range(self.b):
            band_slice = signature[band_idx * self.r : (band_idx + 1) * self.r]
            band_key = band_slice.tobytes()

            if band_key in self.tables[band_idx]:
                bucket = self.tables[band_idx][band_key]
                # Enforce MAX_BUCKET_SIZE safety slice
                candidates.update(bucket[: self.MAX_BUCKET_SIZE])

        return candidates

    def get_bucket_count(self) -> int:
        """Return total number of unique buckets across all bands."""
        return sum(len(table) for table in self.tables)

    def get_saturated_bucket_count(self) -> int:
        """Return total number of saturated buckets exceeding MAX_BUCKET_SIZE."""
        return sum(len(sat) for sat in self._saturated_buckets)
