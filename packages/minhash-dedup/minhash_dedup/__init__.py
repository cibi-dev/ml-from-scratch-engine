"""MinHash & LSH Deduplication Package."""

from minhash_dedup.clustering import UnionFind
from minhash_dedup.lsh import MinHashLSH
from minhash_dedup.minhash import MinHasher, estimate_jaccard
from minhash_dedup.pipeline import deduplicate_corpus
from minhash_dedup.preprocessing import get_shingles, normalize_text

__version__ = "0.1.0"

__all__ = [
    "MinHasher",
    "MinHashLSH",
    "UnionFind",
    "deduplicate_corpus",
    "normalize_text",
    "get_shingles",
    "estimate_jaccard",
]
