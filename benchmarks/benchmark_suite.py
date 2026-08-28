"""
Comparative Benchmark Suite for ml-from-scratch-engine.

Quantifies real-world performance metrics across all core modules:
1. BPE Tokenizer Throughput (Tokens/sec & MB/s)
2. NumPy VectorDB Search Latency & Insertion Throughput (Exact vs Batch Scan)
3. Reverse-Mode Autograd Backward Pass Speed (Evaluations/sec & Graph Traverse Time)
4. MinHash LSH Deduplication Processing Rate (Documents/sec)
5. Guardrails Self-Healing JSON Validation Latency (ms/op)
"""

from __future__ import annotations

import gc
import os
from pathlib import Path
import sys
import time
from typing import Any

# Add packages to sys.path
_ROOT = Path(__file__).resolve().parent.parent
_PACKAGES_DIR = _ROOT / "packages"

_MODULE_PATHS = [
    _ROOT,
    _PACKAGES_DIR / "autograd-engine",
    _PACKAGES_DIR / "bpe-tokenizer",
    _PACKAGES_DIR / "guardrails-engine",
    _PACKAGES_DIR / "minhash-dedup",
    _PACKAGES_DIR / "nano-transformer",
    _PACKAGES_DIR / "numpy-vectordb",
]

for p in _MODULE_PATHS:
    p_str = str(p)
    if p_str not in sys.path:
        sys.path.insert(0, p_str)

import numpy as np


def benchmark_autograd() -> dict[str, Any]:
    from autograd.engine import Value
    from autograd.nn import MLP

    # Deep computational graph benchmark (depth = 100 ops)
    n_nodes = 100
    n_runs = 100
    
    t0 = time.perf_counter()
    for _ in range(n_runs):
        curr = Value(1.05)
        for _ in range(n_nodes):
            curr = (curr * 1.001 + 0.002).relu()
        curr.backward()
    t_total = time.perf_counter() - t0
    
    total_evals = n_runs * n_nodes
    evals_per_sec = total_evals / t_total

    # Micro-MLP benchmark (2 -> 8 -> 8 -> 1)
    mlp = MLP(2, [8, 8, 1])
    xs = [[float(i % 5), float((i + 1) % 5)] for i in range(16)]
    ys = [1.0 if (x[0] + x[1]) > 4 else -1.0 for x in xs]

    mlp_steps = 25
    t_mlp0 = time.perf_counter()
    for _ in range(mlp_steps):
        ypred = [mlp(x) for x in xs]
        loss = sum([(yout - ygt) ** 2 for ygt, yout in zip(ys, ypred)], Value(0.0))  # type: ignore[assignment]
        mlp.zero_grad()
        loss.backward()
        for p in mlp.parameters():
            p.data -= 0.01 * p.grad
    t_mlp = time.perf_counter() - t_mlp0
    steps_per_sec = mlp_steps / t_mlp

    return {
        "graph_nodes": n_nodes,
        "evals_per_sec": evals_per_sec,
        "backward_latency_us": (t_total / n_runs) * 1_000_000,
        "mlp_steps_per_sec": steps_per_sec,
    }


def benchmark_bpe_tokenizer() -> dict[str, Any]:
    from bpe_tokenizer.basic import BasicTokenizer

    sample_paragraph = (
        "Machine learning from scratch requires a foundational understanding of calculus, "
        "linear algebra, probability, and computer systems. Fast tokenization with Byte Pair "
        "Encoding merges the most frequent adjacent byte pairs iteratively into high-level subwords. "
    ) * 20  # ~4KB text

    raw_bytes = len(sample_paragraph.encode("utf-8"))
    tok = BasicTokenizer()
    tok.train(sample_paragraph, vocab_size=300, verbose=False)

    # Encode benchmark
    n_iters = 50
    t0 = time.perf_counter()
    token_count = 0
    for _ in range(n_iters):
        enc = tok.encode(sample_paragraph)
        token_count += len(enc)
    t_encode = time.perf_counter() - t0

    throughput_tokens_sec = token_count / t_encode
    throughput_mb_sec = (raw_bytes * n_iters / (1024 * 1024)) / t_encode

    # Decode benchmark
    t1 = time.perf_counter()
    for _ in range(n_iters):
        _ = tok.decode(enc)
    t_decode = time.perf_counter() - t1
    decode_tokens_sec = token_count / t_decode

    return {
        "vocab_size": 300,
        "corpus_kb": raw_bytes / 1024,
        "encode_tokens_per_sec": throughput_tokens_sec,
        "encode_mb_per_sec": throughput_mb_sec,
        "decode_tokens_per_sec": decode_tokens_sec,
    }


def benchmark_vectordb() -> dict[str, Any]:
    from numpy_vectordb.database import VectorDB
    from numpy_vectordb.similarity import SimilarityMetric

    dim = 128
    n_vectors = 2000
    k = 10
    rng = np.random.default_rng(42)
    vectors = rng.standard_normal((n_vectors, dim)).astype(np.float32)

    db = VectorDB(dim=dim, metric=SimilarityMetric.COSINE)

    # Ingestion benchmark
    t0 = time.perf_counter()
    for i in range(n_vectors):
        db.upsert(f"doc_{i}", vectors[i])
    t_insert = time.perf_counter() - t0
    insert_vec_per_sec = n_vectors / t_insert

    # Query benchmark (Cosine similarity)
    queries = rng.standard_normal((100, dim)).astype(np.float32)
    t1 = time.perf_counter()
    for q in queries:
        _ = db.query(q, top_k=k)
    t_search = time.perf_counter() - t1
    search_latency_us = (t_search / len(queries)) * 1_000_000
    qps = len(queries) / t_search

    return {
        "dim": dim,
        "n_vectors": n_vectors,
        "insert_vec_per_sec": insert_vec_per_sec,
        "search_latency_us": search_latency_us,
        "queries_per_sec": qps,
    }


def benchmark_dedup() -> dict[str, Any]:
    from minhash_dedup.pipeline import deduplicate_corpus

    base_docs = [
        "Distributed database replication protocols ensure consensus across partitions.",
        "Zero-trust security architectures mandate continuous authentication and authorization.",
        "Forensic timeline reconstruction identifies deliberate timestamp manipulation and antiforensics.",
        "High-performance vectorized computing leverages contiguous SIMD cache lines in NumPy.",
        "Self-healing LLM guardrails repair invalid JSON payloads deterministically.",
    ]
    # Expand to 250 documents with variations
    docs = []
    for i in range(50):
        for doc in base_docs:
            if i % 3 == 0:
                docs.append(doc + f" Variation tag {i % 5}.")
            else:
                docs.append(doc)

    t0 = time.perf_counter()
    unique_ids, duplicate_ids = deduplicate_corpus(docs, threshold=0.75, num_perm=64)
    t_dedup = time.perf_counter() - t0

    docs_per_sec = len(docs) / t_dedup

    return {
        "total_docs": len(docs),
        "unique_docs": len(unique_ids),
        "duplicate_docs": len(duplicate_ids),
        "docs_per_sec": docs_per_sec,
        "total_time_ms": t_dedup * 1000,
    }


def benchmark_guardrails() -> dict[str, Any]:
    from pydantic import BaseModel, Field
    from guardrails.extractor import extract_json_from_text
    from guardrails.validator import validate_schema

    class BenchmarkModel(BaseModel):
        name: str
        score: float
        tags: list[str]

    raw_text = """
    ```json
    {
        "name": "benchmark_run",
        "score": 0.995,
        "tags": ["ml", "scratch", "perf"]
    }
    ```
    """

    n_runs = 250
    t0 = time.perf_counter()
    for _ in range(n_runs):
        parsed = extract_json_from_text(raw_text)
        _ = validate_schema(parsed, BenchmarkModel)
    t_total = time.perf_counter() - t0

    ops_per_sec = n_runs / t_total
    latency_us = (t_total / n_runs) * 1_000_000

    return {
        "ops_per_sec": ops_per_sec,
        "latency_us": latency_us,
    }


def main() -> None:
    print("=" * 80)
    print("🚀 ML FROM SCRATCH ENGINE - RIGOROUS PERFORMANCE BENCHMARK SUITE")
    print("=" * 80)

    print("\n[1/5] Benchmarking Reverse-Mode Scalar Autograd...")
    res_ag = benchmark_autograd()
    print(f"  • Backward Graph Traversal: {res_ag['evals_per_sec']:,.0f} evals/sec ({res_ag['backward_latency_us']:.2f} µs / graph)")
    print(f"  • Micro-MLP Optimization:  {res_ag['mlp_steps_per_sec']:.1f} steps/sec")

    print("\n[2/5] Benchmarking BPE Tokenizer...")
    res_bpe = benchmark_bpe_tokenizer()
    print(f"  • Encode Throughput:        {res_bpe['encode_tokens_per_sec']:,.0f} tokens/sec ({res_bpe['encode_mb_per_sec']:.2f} MB/s)")
    print(f"  • Decode Throughput:        {res_bpe['decode_tokens_per_sec']:,.0f} tokens/sec")

    print("\n[3/5] Benchmarking NumPy Contiguous VectorDB (2,000 vectors, dim=128)...")
    res_vdb = benchmark_vectordb()
    print(f"  • Ingestion Throughput:     {res_vdb['insert_vec_per_sec']:,.0f} vectors/sec")
    print(f"  • Cosine Search Latency:    {res_vdb['search_latency_us']:.2f} µs / query ({res_vdb['queries_per_sec']:,.0f} QPS)")

    print("\n[4/5] Benchmarking MinHash LSH Deduplication Pipeline (250 docs)...")
    res_dedup = benchmark_dedup()
    print(f"  • Processing Speed:         {res_dedup['docs_per_sec']:,.0f} docs/sec")
    print(f"  • Deduplication Time:       {res_dedup['total_time_ms']:.2f} ms ({res_dedup['duplicate_docs']} dupes filtered)")

    print("\n[5/5] Benchmarking Self-Healing Guardrails Extraction & Pydantic Validation...")
    res_gr = benchmark_guardrails()
    print(f"  • Extraction & Validation:  {res_gr['ops_per_sec']:,.0f} ops/sec ({res_gr['latency_us']:.2f} µs / op)")

    print("\n" + "=" * 80)
    print("📊 BENCHMARK SUMMARY (Markdown Exportable)")
    print("=" * 80)
    print(f"| Component | Metric | Value | Baseline Standard |")
    print(f"| :--- | :--- | :--- | :--- |")
    print(f"| **Autograd Engine** | Graph Traversal | **{res_ag['evals_per_sec']:,.0f} evals/sec** | Real-time gradient backprop |")
    print(f"| **BPE Tokenizer** | Encode Speed | **{res_bpe['encode_tokens_per_sec']:,.0f} tokens/s** ({res_bpe['encode_mb_per_sec']:.2f} MB/s) | Pure Python/Regex stream |")
    print(f"| **NumPy VectorDB** | Cosine Query Latency | **{res_vdb['search_latency_us']:.1f} µs / query** ({res_vdb['queries_per_sec']:,.0f} QPS) | Contiguous C-array vectorized |")
    print(f"| **MinHash LSH** | Deduplication Throughput | **{res_dedup['docs_per_sec']:,.0f} docs/sec** | Sub-linear candidate hashing |")
    print(f"| **Guardrails Engine** | Self-Healing Validation | **{res_gr['latency_us']:.1f} µs / op** ({res_gr['ops_per_sec']:,.0f} ops/s) | Strict Pydantic v2 validation |")
    print("=" * 80)


if __name__ == "__main__":
    main()
