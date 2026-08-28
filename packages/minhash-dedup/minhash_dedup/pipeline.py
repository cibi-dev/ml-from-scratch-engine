"""End-to-end deduplication pipeline combining preprocessing, MinHash, LSH, and clustering."""

from __future__ import annotations

from typing import Any, Hashable, Mapping, Sequence, TypeVar, overload

import numpy as np

from minhash_dedup.clustering import UnionFind
from minhash_dedup.lsh import MinHashLSH
from minhash_dedup.minhash import MinHasher
from minhash_dedup.preprocessing import get_shingles, normalize_text

K = TypeVar("K", bound=Hashable)
MAX_CORPUS_SIZE: int = 1_000_000


@overload
def deduplicate_corpus(
    documents: Sequence[str],
    threshold: float = 0.75,
    k_shingle: int = 5,
    num_perm: int = 128,
    shingle_mode: str = "word",
    b: int | None = None,
    r: int | None = None,
    canonical_policy: str = "first",
    seed: int = 42,
) -> tuple[list[int], list[int]]: ...


@overload
def deduplicate_corpus(
    documents: Mapping[K, str],
    threshold: float = 0.75,
    k_shingle: int = 5,
    num_perm: int = 128,
    shingle_mode: str = "word",
    b: int | None = None,
    r: int | None = None,
    canonical_policy: str = "first",
    seed: int = 42,
) -> tuple[list[K], list[K]]: ...


def deduplicate_corpus(
    documents: Sequence[str] | Mapping[Any, str],
    threshold: float = 0.75,
    k_shingle: int = 5,
    num_perm: int = 128,
    shingle_mode: str = "word",
    b: int | None = None,
    r: int | None = None,
    canonical_policy: str = "first",
    seed: int = 42,
) -> tuple[list[Any], list[Any]]:
    """Deduplicate a corpus of documents using MinHash and Locality-Sensitive Hashing.

    Args:
        documents: List of raw document strings or dictionary mapping document IDs to strings.
        threshold: Jaccard similarity threshold for considering two documents as duplicates.
        k_shingle: Size of k-grams for shingling.
        num_perm: Number of permutations for MinHash signatures.
        shingle_mode: 'word' or 'char' shingling.
        b: Optional explicit number of LSH bands.
        r: Optional explicit number of rows per LSH band.
        canonical_policy: Policy to select the canonical document from a duplicate cluster:
            - 'first': Keeps the earliest document by original insertion order.
            - 'longest': Keeps the document with the longest raw text character length.
            - 'shortest': Keeps the document with the shortest raw text character length.
        seed: Random seed for deterministic MinHash universal hashing.

    Returns:
        Tuple of (kept_ids, removed_ids) preserving original sequence order.

    Raises:
        ValueError: If corpus size exceeds MAX_CORPUS_SIZE or invalid arguments are passed.
    """
    if len(documents) > MAX_CORPUS_SIZE:
        raise ValueError(
            f"Corpus size {len(documents)} exceeds maximum allowed limit of {MAX_CORPUS_SIZE}"
        )

    if canonical_policy not in ("first", "longest", "shortest"):
        raise ValueError(
            f"Invalid canonical_policy '{canonical_policy}'. Must be 'first', 'longest', or 'shortest'"
        )

    if not documents:
        return [], []

    # Standardize documents to list of (doc_id, text)
    if isinstance(documents, Mapping):
        doc_items: list[tuple[Any, str]] = list(documents.items())
    elif isinstance(documents, Sequence):
        doc_items = list(enumerate(documents))
    else:
        raise TypeError("documents must be a Sequence of strings or Mapping of {id: text}")

    # Map ID to original position for stable sorting
    id_to_order: dict[Any, int] = {doc_id: idx for idx, (doc_id, _) in enumerate(doc_items)}
    raw_texts: dict[Any, str] = {doc_id: text for doc_id, text in doc_items}

    # Step 1: Preprocessing & MinHash Signature Generation
    hasher = MinHasher(num_perm=num_perm, seed=seed)
    lsh = MinHashLSH(threshold=threshold, num_perm=num_perm, b=b, r=r)

    signatures: dict[Any, np.ndarray] = {}
    for doc_id, text in doc_items:
        norm_text = normalize_text(text)
        shingles = get_shingles(norm_text, k=k_shingle, mode=shingle_mode)
        sig = hasher.compute_signature(shingles)
        signatures[doc_id] = sig
        lsh.insert(doc_id, sig)

    # Step 2: Query LSH candidates & Cluster with Union-Find
    uf: UnionFind[Any] = UnionFind([doc_id for doc_id, _ in doc_items])
    checked_pairs: set[tuple[Any, Any]] = set()

    for doc_id, sig in signatures.items():
        candidates = lsh.query_candidates(sig)
        for cand_id in candidates:
            if cand_id == doc_id:
                continue

            # Create symmetric pair key using string representation for comparison ordering
            pair: tuple[Any, Any] = (
                doc_id,
                cand_id,
            ) if str(doc_id) < str(cand_id) else (
                cand_id,
                doc_id,
            )

            if pair in checked_pairs:
                continue
            checked_pairs.add(pair)

            # Compute estimated similarity between candidate pair
            similarity = hasher.estimate_jaccard(sig, signatures[cand_id])
            if similarity >= threshold:
                uf.union(doc_id, cand_id)

    # Step 3: Canonical Selection per Cluster
    clusters = uf.get_components()
    kept_ids: list[Any] = []
    removed_ids: list[Any] = []

    for _root, members in clusters.items():
        if len(members) == 1:
            kept_ids.append(members[0])
        else:
            # Sort members by original insertion order first
            sorted_members = sorted(members, key=lambda m: id_to_order[m])

            if canonical_policy == "first":
                canonical = sorted_members[0]
            elif canonical_policy == "longest":
                canonical = max(
                    sorted_members,
                    key=lambda m: (len(raw_texts[m]), -id_to_order[m]),
                )
            elif canonical_policy == "shortest":
                canonical = min(
                    sorted_members,
                    key=lambda m: (len(raw_texts[m]), id_to_order[m]),
                )
            else:
                canonical = sorted_members[0]

            kept_ids.append(canonical)
            for m in sorted_members:
                if m != canonical:
                    removed_ids.append(m)

    # Sort final kept and removed IDs by their original corpus index
    kept_ids.sort(key=lambda x: id_to_order[x])
    removed_ids.sort(key=lambda x: id_to_order[x])

    return kept_ids, removed_ids
