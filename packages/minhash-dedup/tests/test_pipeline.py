"""Tests for end-to-end deduplicate_corpus pipeline and clustering fidelity."""

from __future__ import annotations

import pytest

from minhash_dedup.pipeline import deduplicate_corpus


class TestDeduplicateCorpus:
    def test_exact_duplicates_deduplication(self) -> None:
        docs = [
            "This is an introductory guide to deep learning and artificial neural networks.",
            "This is an introductory guide to deep learning and artificial neural networks.",  # duplicate of 0
            "A completely unrelated article about baking sourdough bread in a dutch oven.",
            "This is an introductory guide to deep learning and artificial neural networks.",  # duplicate of 0
        ]
        kept, removed = deduplicate_corpus(docs, threshold=0.8, k_shingle=3)

        assert 0 in kept
        assert 2 in kept
        assert 1 in removed
        assert 3 in removed
        assert len(kept) == 2
        assert len(removed) == 2

    def test_near_duplicates_deduplication(self) -> None:
        docs = [
            "The quick brown fox jumps gracefully over the very lazy sleeping dog in the warm sunshine.",
            "The quick brown fox jumps gracefully over the very lazy sleeping dog in the cold sunshine.",  # near-duplicate of 0
            "Quantum mechanics explores the fundamental properties of atoms and subatomic particles.",
        ]
        kept, removed = deduplicate_corpus(docs, threshold=0.7, k_shingle=2, num_perm=128)

        assert 0 in kept
        assert 2 in kept
        assert 1 in removed
        assert len(kept) == 2
        assert len(removed) == 1

    def test_canonical_policy_first(self) -> None:
        docs = [
            "the quick brown fox jumps over the lazy dog today",
            "the quick brown fox jumps over the lazy dog today and tomorrow",
            "the quick brown fox jumps over the lazy dog today indeed",
        ]
        kept, removed = deduplicate_corpus(docs, threshold=0.6, k_shingle=2, canonical_policy="first")

        assert kept == [0]
        assert set(removed) == {1, 2}

    def test_canonical_policy_longest(self) -> None:
        docs = [
            "the quick brown fox jumps over the lazy dog today",
            "the quick brown fox jumps over the lazy dog today and tomorrow with extra words",
            "the quick brown fox jumps over the lazy dog today indeed",
        ]
        kept, removed = deduplicate_corpus(docs, threshold=0.5, k_shingle=2, canonical_policy="longest")

        assert kept == [1]
        assert set(removed) == {0, 2}

    def test_canonical_policy_shortest(self) -> None:
        docs = [
            "the quick brown fox jumps over the lazy dog today indeed",
            "the quick brown fox jumps over the lazy dog today and tomorrow with extra words",
            "the quick brown fox jumps over the lazy dog today",
        ]
        kept, removed = deduplicate_corpus(docs, threshold=0.5, k_shingle=2, canonical_policy="shortest")

        assert kept == [2]
        assert set(removed) == {0, 1}

    def test_dictionary_input_with_custom_ids(self) -> None:
        docs_dict = {
            "doc_a": "MinHash provides efficient approximate similarity search for large text corpora.",
            "doc_b": "MinHash provides efficient approximate similarity search for large text corpora.",
            "doc_c": "Relational databases use B-trees and write-ahead logs for transactional safety.",
        }
        kept, removed = deduplicate_corpus(docs_dict, threshold=0.8, k_shingle=3)

        assert "doc_a" in kept
        assert "doc_c" in kept
        assert "doc_b" in removed
        assert len(kept) == 2
        assert len(removed) == 1

    def test_transitive_clustering_fidelity(self) -> None:
        """Verify that transitive similarities (A~B, B~C) form a unified cluster."""
        # A and B share high overlap, B and C share high overlap
        text_a = "alpha beta gamma delta epsilon zeta eta theta iota kappa"
        text_b = "alpha beta gamma delta epsilon zeta eta theta iota lambda"
        text_c = "alpha beta gamma delta epsilon zeta eta theta iota mu"

        docs = [text_a, text_b, text_c, "unrelated document about astrophysics"]
        kept, removed = deduplicate_corpus(docs, threshold=0.7, k_shingle=2, num_perm=256)

        assert 0 in kept
        assert 3 in kept
        assert 1 in removed
        assert 2 in removed

    def test_empty_corpus(self) -> None:
        kept, removed = deduplicate_corpus([])
        assert kept == []
        assert removed == []

        empty_dict: dict[str, str] = {}
        kept_dict, removed_dict = deduplicate_corpus(empty_dict)
        assert kept_dict == []
        assert removed_dict == []

    def test_single_document_corpus(self) -> None:
        docs = ["Single document in the dataset."]
        kept, removed = deduplicate_corpus(docs)
        assert kept == [0]
        assert removed == []

    def test_all_disjoint_corpus(self) -> None:
        docs = [
            "Quantum teleportation and quantum cryptography protocols.",
            "Deep convolutional neural networks for image segmentation.",
            "Culinary arts and traditional French pastry making techniques.",
            "Financial market microstructure and algorithmic high-frequency trading.",
        ]
        kept, removed = deduplicate_corpus(docs, threshold=0.5, k_shingle=3)
        assert len(kept) == 4
        assert len(removed) == 0

    def test_invalid_canonical_policy_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="Invalid canonical_policy"):
            deduplicate_corpus(["test doc"], canonical_policy="invalid_policy")
