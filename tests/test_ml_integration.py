"""
End-to-end Integration Test Suite for ml-from-scratch-engine.

Validates the functional integrity of all 6 Machine Learning modules from scratch:
1. Reverse-Mode Autograd Engine & MLP
2. Byte Pair Encoding (BPE) Tokenizer
3. Contiguous NumPy Vector Database
4. MinHash & LSH Near-Duplicate Clustering
5. Self-Healing Guardrails JSON Extractor & Validator
6. Unified CLI `mlcore` Command Dispatcher
"""

from __future__ import annotations

import os
from pathlib import Path
import sys
import pytest

# Ensure packages are discoverable
_ROOT = Path(__file__).resolve().parent.parent
_PACKAGES_DIR = _ROOT / "packages"

for mod_path in [
    _ROOT,
    _PACKAGES_DIR / "autograd-engine",
    _PACKAGES_DIR / "bpe-tokenizer",
    _PACKAGES_DIR / "guardrails-engine",
    _PACKAGES_DIR / "minhash-dedup",
    _PACKAGES_DIR / "nano-transformer",
    _PACKAGES_DIR / "numpy-vectordb",
]:
    p_str = str(mod_path)
    if p_str not in sys.path:
        sys.path.insert(0, p_str)

import numpy as np
from pydantic import BaseModel, Field


# ==============================================================================
# 1. Autograd Engine & MLP Tests
# ==============================================================================

def test_autograd_scalar_derivatives() -> None:
    from autograd.engine import Value

    # f(x, y) = x^2 * y + y + 2
    x = Value(3.0)
    y = Value(4.0)
    z = (x ** 2) * y + y + 2.0

    z.backward()

    # z = (3^2)*4 + 4 + 2 = 36 + 4 + 2 = 42.0
    # dz/dx = 2 * x * y = 2 * 3 * 4 = 24.0
    # dz/dy = x^2 + 1 = 9 + 1 = 10.0
    assert abs(z.data - 42.0) < 1e-6
    assert abs(x.grad - 24.0) < 1e-6
    assert abs(y.grad - 10.0) < 1e-6


def test_autograd_mlp_forward_backward() -> None:
    from autograd.engine import Value
    from autograd.nn import MLP

    mlp = MLP(2, [4, 1])
    out = mlp([Value(1.0), Value(-2.0)])
    assert isinstance(out, Value)

    out.backward()
    params = mlp.parameters()
    assert len(params) > 0
    assert all(isinstance(p, Value) for p in params)
    assert any(p.grad != 0.0 for p in params)


# ==============================================================================
# 2. BPE Tokenizer Tests
# ==============================================================================

def test_bpe_tokenizer_train_encode_decode() -> None:
    from bpe_tokenizer.basic import BasicTokenizer

    corpus = "low low low low lower lower newest newest newest widest widest"
    tok = BasicTokenizer()
    tok.train(corpus, vocab_size=265, verbose=False)

    encoded = tok.encode(corpus)
    decoded = tok.decode(encoded)

    assert len(encoded) < len(corpus.encode("utf-8"))
    assert decoded == corpus


# ==============================================================================
# 3. NumPy VectorDB Tests
# ==============================================================================

def test_numpy_vectordb_indexing_and_search() -> None:
    from numpy_vectordb.database import VectorDB
    from numpy_vectordb.similarity import SimilarityMetric

    db = VectorDB(dim=4, metric=SimilarityMetric.COSINE)

    v1 = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    v2 = np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32)
    v3 = np.array([0.9, 0.1, 0.0, 0.0], dtype=np.float32)

    db.upsert("doc_1", v1, metadata={"category": "tech"})
    db.upsert("doc_2", v2, metadata={"category": "finance"})
    db.upsert("doc_3", v3, metadata={"category": "tech"})

    assert db.count() == 3

    # Query close to doc_1 and doc_3
    q = np.array([1.0, 0.05, 0.0, 0.0], dtype=np.float32)
    results = db.query(q, top_k=2)

    assert len(results) == 2
    assert results[0].id == "doc_1"
    assert results[1].id == "doc_3"
    assert results[0].score >= results[1].score


def test_numpy_vectordb_metrics() -> None:
    from numpy_vectordb.database import VectorDB
    from numpy_vectordb.similarity import SimilarityMetric

    for metric in [SimilarityMetric.COSINE, SimilarityMetric.EUCLIDEAN, SimilarityMetric.DOT_PRODUCT]:
        db = VectorDB(dim=3, metric=metric)
        db.upsert("a", np.array([1.0, 2.0, 3.0], dtype=np.float32))
        db.upsert("b", np.array([3.0, 2.0, 1.0], dtype=np.float32))
        res = db.query(np.array([1.0, 2.0, 3.0], dtype=np.float32), top_k=1)
        assert res[0].id == "a"


# ==============================================================================
# 4. MinHash & LSH Deduplication Tests
# ==============================================================================

def test_minhash_dedup_pipeline() -> None:
    from minhash_dedup.pipeline import deduplicate_corpus

    docs = [
        "Distributed consensus protocols like Raft and Paxos provide strong fault tolerance.",
        "Distributed consensus protocols like Raft and Paxos provide strong fault tolerance in clusters.",
        "A completely distinct document explaining neural network backpropagation.",
    ]

    unique_ids, duplicate_ids = deduplicate_corpus(docs, threshold=0.7, num_perm=128)
    assert 0 in unique_ids
    assert 2 in unique_ids
    assert 1 in duplicate_ids or len(unique_ids) == 2


# ==============================================================================
# 5. Guardrails Self-Healing Engine Tests
# ==============================================================================

class SampleIncident(BaseModel):
    incident_id: str
    severity: str
    confidence: float = Field(ge=0.0, le=1.0)
    tags: list[str]


def test_guardrails_json_repair_and_validation() -> None:
    from guardrails.extractor import extract_json_from_text
    from guardrails.validator import validate_schema

    raw_markdown_json = """
    Certainly! Here is the JSON payload:
    ```json
    {
        "incident_id": "INC-7721",
        "severity": "HIGH",
        "confidence": 0.95,
        "tags": ["sre", "outage"]
    }
    ```
    Hope this helps!
    """

    parsed = extract_json_from_text(raw_markdown_json)
    assert parsed is not None
    assert isinstance(parsed, dict)

    val = validate_schema(parsed, SampleIncident)
    assert val.success
    assert val.data is not None
    assert val.data.incident_id == "INC-7721"
    assert val.data.confidence == 0.95


# ==============================================================================
# 6. Nano-Transformer GPT Model Tests
# ==============================================================================

def test_nano_transformer_forward_and_generation() -> None:
    import torch
    from nano_transformer.model import GPT, GPTConfig
    from nano_transformer.generate import generate

    config = GPTConfig(
        vocab_size=32,
        block_size=16,
        n_layer=2,
        n_head=2,
        n_embd=32,
        dropout=0.0,
    )
    model = GPT(config)
    idx = torch.randint(0, 32, (2, 8), dtype=torch.long)
    logits, loss = model(idx)

    assert logits.shape == (2, 8, 32)
    assert loss is None

    gen_idx = generate(model, idx[:1], max_new_tokens=4, temperature=1.0)
    assert gen_idx.shape == (1, 12)


# ==============================================================================
# 7. Unified CLI Dispatcher Tests
# ==============================================================================

def test_cli_dispatch() -> None:
    import cli

    assert cli.main(["autograd", "--demo", "scalar"]) == 0
    assert cli.main(["tokenizer", "--vocab-size", "260"]) == 0
    assert cli.main(["vectordb", "--n-vectors", "50"]) == 0
    assert cli.main(["dedup", "--threshold", "0.7"]) == 0
    assert cli.main(["transformer", "--prompt", "test"]) == 0
    assert cli.main(["guardrails"]) == 0
    assert cli.main(["demo"]) == 0

