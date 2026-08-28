"""Security and robustness tests: Hash DoS defense, corpus limits, ReDoS safety, and Unicode bypass."""

from __future__ import annotations

import time
import numpy as np
import pytest

from minhash_dedup.lsh import MinHashLSH
from minhash_dedup.pipeline import MAX_CORPUS_SIZE, deduplicate_corpus
from minhash_dedup.preprocessing import get_shingles, normalize_text


class TestHashDoSSecurity:
    def test_max_bucket_size_capped_on_insertion(self) -> None:
        """Verify that LSH buckets cap document collection at MAX_BUCKET_SIZE=5000."""
        lsh = MinHashLSH(threshold=0.75, num_perm=128, b=16, r=8)
        dummy_sig = np.full(128, 42, dtype=np.uint64)

        # Insert 5500 identical items (exceeding MAX_BUCKET_SIZE)
        num_items = 5500
        for i in range(num_items):
            lsh.insert(f"doc_{i}", dummy_sig)

        # Bucket sizes must be capped at MAX_BUCKET_SIZE
        for table in lsh.tables:
            for bucket in table.values():
                assert len(bucket) <= lsh.MAX_BUCKET_SIZE

        # Saturated buckets counter should reflect the saturation
        assert lsh.get_saturated_bucket_count() > 0

        # Querying candidates should return at most MAX_BUCKET_SIZE candidates
        candidates = lsh.query_candidates(dummy_sig)
        assert len(candidates) <= lsh.MAX_BUCKET_SIZE


class TestCorpusSizeLimits:
    def test_max_corpus_size_enforcement(self) -> None:
        """Verify that passing more than MAX_CORPUS_SIZE documents raises a ValueError."""
        # Create a mock collection with length exceeding limit without allocating RAM
        class MockOverloadedCorpus:
            def __len__(self) -> int:
                return MAX_CORPUS_SIZE + 1

            def __getitem__(self, item: int) -> str:
                return "mock document"

        with pytest.raises(ValueError, match="exceeds maximum allowed limit"):
            deduplicate_corpus(MockOverloadedCorpus())  # type: ignore[call-overload]


class TestUnicodeBypassDefense:
    def test_zero_width_adversarial_bypass_detected_as_duplicate(self) -> None:
        """Adversarial attackers often inject invisible characters to evade exact-hash matching.

        MinHash normalization must strip them and detect near/exact duplicate status.
        """
        original = "Antigravity is an advanced agentic coding assistant developed by Google."
        # Inject zero-width spaces (\u200b), non-joiners (\u200c), BOM (\ufeff), soft hyphens (\u00ad)
        adversarial = (
            "A\u200bnti\u200cgrav\u200dity is \ufeffan adv\u00adanced "
            "age\u200entic co\u200fding ass\u200bistant dev\u200celoped by Google."
        )

        docs = [original, adversarial]
        kept, removed = deduplicate_corpus(docs, threshold=0.8, k_shingle=3)

        assert kept == [0]
        assert removed == [1]

    def test_fullwidth_and_ligature_adversarial_bypass(self) -> None:
        original = "efficient machine learning workflows"
        adversarial = "ｅﬃｃｉｅｎｔ ｍａｃｈｉｎｅ ｌｅａｒｎｉｎｇ ｗｏｒｋﬂｏｗｓ"

        docs = [original, adversarial]
        kept, removed = deduplicate_corpus(docs, threshold=0.8, k_shingle=2)

        assert kept == [0]
        assert removed == [1]


class TestReDoSSafety:
    def test_linear_time_tokenization_on_pathological_inputs(self) -> None:
        """Verify tokenization executes in linear time without catastrophic backtracking."""
        # 100,000 repeated characters
        pathological_input = "a" * 100_000 + "!"
        t0 = time.perf_counter()
        normalized = normalize_text(pathological_input)
        shingles = get_shingles(normalized, k=5, mode="word")
        elapsed = time.perf_counter() - t0

        # Linear tokenization on 100k chars should complete in < 0.2s
        assert elapsed < 0.5
        assert len(shingles) == 1  # 1 word token fallback

    def test_pathological_whitespace_and_punctuation(self) -> None:
        pathological_whitespace = (" \t\n\r" * 25_000) + "end"
        t0 = time.perf_counter()
        normalized = normalize_text(pathological_whitespace)
        elapsed = time.perf_counter() - t0

        assert elapsed < 0.5
        assert normalized == "end"


class TestExtremeAndEdgeDocuments:
    def test_empty_and_whitespace_documents_in_pipeline(self) -> None:
        docs = [
            "",
            "   \t\n   ",
            "\u200b\u200c\ufeff",
            "Valid document discussing distributed databases.",
            "Valid document discussing distributed databases.",
        ]
        # Should not crash on empty documents
        kept, removed = deduplicate_corpus(docs, threshold=0.75, k_shingle=3)

        # Non-empty duplicate is deduplicated
        assert 3 in kept
        assert 4 in removed

    def test_multilingual_multi_script_documents(self) -> None:
        docs = [
            "人工智能是大语言模型的核心技术之一。",  # Chinese
            "Искусственный интеллект и машинное обучение.",  # Cyrillic
            "الذكاء الاصطناعي وتعلم الآلة.",  # Arabic
            "人工知能と機械学習の最新動向。",  # Japanese
            "人工智能是大语言模型的核心技术之一。",  # Exact Chinese duplicate
        ]
        kept, removed = deduplicate_corpus(docs, threshold=0.8, k_shingle=2)

        assert 0 in kept
        assert 4 in removed
        assert 1 in kept
        assert 2 in kept
        assert 3 in kept
        assert len(kept) == 4
        assert len(removed) == 1
