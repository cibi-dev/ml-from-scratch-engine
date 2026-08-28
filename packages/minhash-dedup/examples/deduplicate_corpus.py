"""Standalone runnable script demonstrating MinHash & LSH deduplication on a synthetic corpus."""

from __future__ import annotations

import time

from minhash_dedup.pipeline import deduplicate_corpus
from minhash_dedup.preprocessing import normalize_text


def run_example() -> None:
    print("=" * 70)
    print("MinHash & LSH Data Curation & Deduplication Pipeline Demonstration")
    print("=" * 70)

    # Synthetic corpus with duplicates, near-duplicates, and unique articles
    documents = {
        "doc_001": (
            "Large language models such as GPT-4 and Gemini are trained on vast web corpora "
            "containing billions of tokens. Deduplicating this pretraining data is vital."
        ),
        "doc_002": (
            "Large language models such as GPT-4 and Gemini are trained on vast web corpora "
            "containing billions of tokens. Deduplicating this pretraining data is vital."  # Exact duplicate
        ),
        "doc_003": (
            "Large language models like GPT-4 and Gemini are trained on vast web datasets "
            "containing billions of tokens. Deduplicating this pretraining data is crucial."  # Near duplicate
        ),
        "doc_004": (
            "L\u200barge langu\u200cage models such as GPT-4 and Gemini are trained on \ufeffvast web corpora "
            "containing billions of tokens. Deduplicating this pretraining data is vital."  # Zero-width bypass
        ),
        "doc_005": (
            "Relational databases rely on ACID transactions, write-ahead logging (WAL), "
            "and B+ Trees to ensure data integrity and high transactional throughput."
        ),
        "doc_006": (
            "Relational databases rely on ACID transactions, write-ahead logging (WAL), "
            "and B+ Trees to ensure high data consistency and transactional durability."  # Near duplicate of doc_005
        ),
        "doc_007": (
            "Quantum error correction schemes such as the surface code are essential "
            "for building fault-tolerant quantum computers capable of running Shor's algorithm."
        ),
        "doc_008": (
            "French cuisine relies heavily on traditional mother sauces including "
            "Béchamel, Velouté, Espagnole, sauce Tomat, and Hollandaise."
        ),
    }

    print(f"\n[+] Input Corpus: {len(documents)} documents")
    for doc_id, text in documents.items():
        print(f"  - [{doc_id}] {text[:75]}...")

    print("\n[+] Running deduplicate_corpus (threshold=0.70, k_shingle=3, num_perm=128)...")
    start_time = time.perf_counter()

    kept_ids, removed_ids = deduplicate_corpus(
        documents=documents,
        threshold=0.70,
        k_shingle=3,
        num_perm=128,
        canonical_policy="first",
    )

    elapsed_ms = (time.perf_counter() - start_time) * 1000.0

    print(f"\n[+] Pipeline Completed in {elapsed_ms:.2f} ms")
    print(f"  - Total documents processed: {len(documents)}")
    print(f"  - Kept canonical documents: {len(kept_ids)} ({kept_ids})")
    print(f"  - Removed duplicate documents: {len(removed_ids)} ({removed_ids})")
    print(f"  - Deduplication reduction rate: {(len(removed_ids) / len(documents)) * 100.0:.1f}%")

    print("\n[+] Cluster Summary:")
    print("  - Kept:")
    for kid in kept_ids:
        print(f"    * [{kid}]: {documents[kid][:80]}...")
    print("  - Removed as duplicates:")
    for rid in removed_ids:
        print(f"    * [{rid}]: {documents[rid][:80]}...")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    run_example()
