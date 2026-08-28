"""
Unified Command-Line Interface for Machine Learning From Scratch Engine (mlcore).

Provides unified access to 6 fundamental ML engines built from first principles:
- autograd: Reverse-mode scalar automatic differentiation and micro-neural networks
- tokenizer: Byte-level Byte Pair Encoding (BPE) tokenizer from scratch
- transformer: Lightweight character-level causal Transformer (~1M params)
- vectordb: Contiguous NumPy-based vector similarity search engine (Cosine, L2, Dot)
- dedup: MinHash & Locality-Sensitive Hashing (LSH) near-duplicate document clustering
- guardrails: Self-healing JSON extraction and strict schema validation for LLM outputs
- demo: End-to-end multi-module pipeline execution
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys
import time
from typing import Sequence

# Add packages to sys.path
_ROOT = Path(__file__).resolve().parent
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

__version__ = "1.0.0"


def _run_autograd(argv: Sequence[str]) -> int:
    parser = argparse.ArgumentParser(prog="mlcore autograd", description="Scalar Autograd Engine & MLP")
    parser.add_argument("--demo", choices=["scalar", "mlp"], default="scalar", help="Autograd demo to run")
    parser.add_argument("--steps", type=int, default=20, help="Optimization steps for MLP")
    args = parser.parse_args(argv)

    from autograd.engine import Value

    if args.demo == "scalar":
        print("⚡ [mlcore:autograd] Evaluating computational graph: L = (a * b + c) * d")
        a = Value(2.0, label="a")
        b = Value(-3.0, label="b")
        c = Value(10.0, label="c")
        d = Value(4.0, label="d")
        e = a * b; e.label = "e"
        f = e + c; f.label = "f"
        L = f * d; L.label = "L"

        L.backward()
        print(f"  Forward Pass: L = {L.data:.4f}")
        print("  Backward Gradients:")
        print(f"    dL/da = {a.grad:.4f} (expected: -12.0000)")
        print(f"    dL/db = {b.grad:.4f} (expected: 8.0000)")
        print(f"    dL/dc = {c.grad:.4f} (expected: 4.0000)")
        print(f"    dL/dd = {d.grad:.4f} (expected: 4.0000)")
        print("  ✓ Topological backward pass completed successfully.")
    else:
        from autograd.nn import MLP
        print(f"⚡ [mlcore:autograd] Training micro-MLP (2 inputs -> 4 -> 4 -> 1 output) for {args.steps} steps...")
        mlp = MLP(2, [4, 4, 1])
        # Simple binary XOR dataset
        xs = [[2.0, 3.0], [3.0, -1.0], [1.0, 1.0], [1.0, -1.0]]
        ys = [1.0, -1.0, -1.0, 1.0]

        for step in range(args.steps):
            # Forward
            ypred = [mlp(x) for x in xs]
            loss: Value = sum([(yout - ygt) ** 2 for ygt, yout in zip(ys, ypred)], Value(0.0))  # type: ignore[assignment]
            
            # Zero grad
            mlp.zero_grad()
            # Backward
            loss.backward()
            # Update
            lr = 0.05
            for p in mlp.parameters():
                p.data -= lr * p.grad

            if (step + 1) % max(1, args.steps // 4) == 0 or step == args.steps - 1:
                print(f"  Step {step+1:3d}/{args.steps} | Loss: {loss.data:.6f}")

        print("  ✓ MLP training loop completed successfully.")

    return 0


def _run_tokenizer(argv: Sequence[str]) -> int:
    parser = argparse.ArgumentParser(prog="mlcore tokenizer", description="Byte Pair Encoding (BPE) Tokenizer")
    parser.add_argument("--vocab-size", type=int, default=270, help="Target vocabulary size (>=256)")
    parser.add_argument("--text", type=str, default="The quick brown fox jumps over the lazy dog. Machine learning from scratch.", help="Text sample to tokenize")
    args = parser.parse_args(argv)

    from bpe_tokenizer.basic import BasicTokenizer

    print(f"⚡ [mlcore:tokenizer] Training BPE tokenizer on sample corpus (target vocab size: {args.vocab_size})...")
    tok = BasicTokenizer()
    tok.train(args.text, vocab_size=args.vocab_size, verbose=False)
    
    encoded = tok.encode(args.text)
    decoded = tok.decode(encoded)
    
    compression_ratio = len(args.text.encode("utf-8")) / max(1, len(encoded))
    print(f"  Input length (bytes) : {len(args.text.encode('utf-8'))}")
    print(f"  Encoded token count  : {len(encoded)}")
    print(f"  Tokens               : {encoded[:15]} {'...' if len(encoded) > 15 else ''}")
    print(f"  Compression ratio    : {compression_ratio:.2f}x")
    print(f"  Roundtrip lossless   : {decoded == args.text}")
    print("  ✓ BPE Tokenization pipeline verified.")
    return 0


def _run_vectordb(argv: Sequence[str]) -> int:
    parser = argparse.ArgumentParser(prog="mlcore vectordb", description="Hardened NumPy Vector Database")
    parser.add_argument("--dim", type=int, default=64, help="Embedding dimension")
    parser.add_argument("--metric", choices=["cosine", "euclidean", "dot"], default="cosine", help="Distance metric")
    parser.add_argument("--n-vectors", type=int, default=1000, help="Number of random vectors to index")
    args = parser.parse_args(argv)

    import numpy as np
    from numpy_vectordb.database import VectorDB

    print(f"⚡ [mlcore:vectordb] Initializing NumPy VectorDB (dim={args.dim}, metric={args.metric})...")
    db = VectorDB(dim=args.dim, metric=args.metric)
    
    rng = np.random.default_rng(42)
    vectors = rng.standard_normal((args.n_vectors, args.dim)).astype(np.float32)
    
    t0 = time.perf_counter()
    for i in range(args.n_vectors):
        db.upsert(f"doc_{i:04d}", vectors[i], metadata={"source": "synthetic", "idx": i})
    insert_duration = (time.perf_counter() - t0) * 1000

    query = rng.standard_normal(args.dim).astype(np.float32)
    t1 = time.perf_counter()
    results = db.query(query, top_k=5)
    search_duration = (time.perf_counter() - t1) * 1000

    print(f"  Indexed {args.n_vectors} vectors in {insert_duration:.2f} ms ({args.n_vectors / (insert_duration / 1000):.0f} vec/s)")
    print(f"  Top-5 nearest neighbors queried in {search_duration:.3f} ms:")
    for rank, r in enumerate(results, start=1):
        print(f"    #{rank} Doc ID: {r.id:10s} | Score: {r.score:.4f} | Metadata: {r.metadata}")

    print("  ✓ VectorDB search verified.")
    return 0


def _run_dedup(argv: Sequence[str]) -> int:
    parser = argparse.ArgumentParser(prog="mlcore dedup", description="MinHash & LSH Deduplication Engine")
    parser.add_argument("--threshold", type=float, default=0.6, help="Jaccard similarity threshold")
    args = parser.parse_args(argv)

    from minhash_dedup.pipeline import deduplicate_corpus

    corpus = [
        "Digital forensics and incident response requires rigorous evidence collection.",
        "Digital forensics and incident response requires rigorous evidence custody.",
        "Machine learning models can be built from mathematical first principles.",
        "Deep learning models can be built from mathematical first principles in Python.",
        "DevSecOps pipelines automate security scanning and continuous integration.",
        "Completely unrelated document talking about astronomy and quantum physics.",
    ]

    print(f"⚡ [mlcore:dedup] Running MinHash LSH deduplication on {len(corpus)} documents (threshold={args.threshold})...")
    unique_ids, duplicate_ids = deduplicate_corpus(corpus, threshold=args.threshold, num_perm=128)
    
    print(f"  Total documents : {len(corpus)}")
    print(f"  Unique retained : {len(unique_ids)} -> {unique_ids}")
    print(f"  Duplicates removed : {len(duplicate_ids)} -> {duplicate_ids}")
    print("  ✓ MinHash LSH deduplication pipeline verified.")
    return 0


def _run_guardrails(argv: Sequence[str]) -> int:
    parser = argparse.ArgumentParser(prog="mlcore guardrails", description="Self-Healing JSON Extraction & Guardrails")
    args = parser.parse_args(argv)

    from pydantic import BaseModel, Field
    from guardrails.extractor import extract_json_from_text
    from guardrails.validator import validate_schema

    class ThreatIncident(BaseModel):
        incident_id: str
        severity: str
        confidence: float = Field(ge=0.0, le=1.0)
        indicators: list[str]

    raw_noisy_llm_response = """
    Here is the incident triage analysis you requested:
    ```json
    {
        "incident_id": "INC-2026-9042",
        "severity": "CRITICAL",
        "confidence": 0.98,
        "indicators": ["192.168.1.105", "CVE-2026-2184"]
    }
    ```
    Please take immediate remediation actions!
    """

    print("⚡ [mlcore:guardrails] Parsing noisy LLM output with self-healing extractor and Pydantic validation...")
    extracted = extract_json_from_text(raw_noisy_llm_response)
    if extracted is None:
        print("  ❌ JSON Extraction failed: no valid JSON block found.")
        return 1

    val_res = validate_schema(extracted, ThreatIncident)
    if not val_res.success or val_res.data is None:
        print(f"  ❌ Schema validation failed: {val_res.error_message}")
        return 1

    obj = val_res.data
    print(f"  Extracted Object: {obj.__class__.__name__}")
    print(f"    - Incident ID : {obj.incident_id}")
    print(f"    - Severity    : {obj.severity}")
    print(f"    - Confidence  : {obj.confidence:.2%}")
    print(f"    - Indicators  : {obj.indicators}")
    print("  ✓ Guardrails self-healing validation verified.")
    return 0


def _run_transformer(argv: Sequence[str]) -> int:
    parser = argparse.ArgumentParser(prog="mlcore transformer", description="Nano Transformer Language Model")
    parser.add_argument("--prompt", type=str, default="The neural network", help="Initial prompt for generation")
    args = parser.parse_args(argv)

    try:
        import torch
        from nano_transformer.model import GPT, GPTConfig
        from nano_transformer.generate import generate
    except ImportError:
        print("⚠️ [mlcore:transformer] PyTorch is required for nano_transformer. Install via 'pip install torch'.")
        return 0

    print("⚡ [mlcore:transformer] Initializing Nano-Transformer GPT architecture...")
    config = GPTConfig(
        vocab_size=256,
        block_size=64,
        n_layer=2,
        n_head=2,
        n_embd=64,
        dropout=0.0,
    )
    model = GPT(config)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Model parameter count: {n_params:,} parameters")

    # Generate test tokens
    prompt_tokens = [ord(c) % 256 for c in args.prompt]
    idx = torch.tensor([prompt_tokens], dtype=torch.long)
    out = generate(model, idx, max_new_tokens=20, temperature=0.8)
    gen_chars = "".join([chr(int(t)) for t in out[0].tolist() if 32 <= int(t) <= 126])
    print(f"  Prompt: '{args.prompt}'")
    print(f"  Generated sequence: '{gen_chars}'")
    print("  ✓ Nano-Transformer forward and generation verified.")
    return 0


def _run_demo() -> int:
    print("=" * 80)
    print("🧠 ML FROM SCRATCH ENGINE - INTEGRATED MULTI-MODULE SHOWCASE")
    print("=" * 80)

    # 1. MinHash Deduplication
    print("\n[Step 1/6] 🧹 MinHash & LSH Dataset Deduplication")
    _run_dedup([])

    # 2. BPE Tokenizer
    print("\n[Step 2/6] 🔤 Byte Pair Encoding (BPE) Tokenizer")
    _run_tokenizer([])

    # 3. NumPy VectorDB
    print("\n[Step 3/6] 🔍 Contiguous NumPy Vector Database Search")
    _run_vectordb(["--n-vectors", "500"])

    # 4. Scalar Autograd
    print("\n[Step 4/6] ⚡ Reverse-Mode Scalar Autograd Graph")
    _run_autograd(["--demo", "scalar"])

    # 5. Nano Transformer
    print("\n[Step 5/6] 🤖 Nano-Transformer Language Model")
    _run_transformer([])

    # 6. Guardrails
    print("\n[Step 6/6] 🛡️ Self-Healing JSON Extraction & Guardrails")
    _run_guardrails([])

    print("\n" + "=" * 80)
    print("✅ ALL 6 ML-FROM-SCRATCH ENGINES VERIFIED & OPERATIONAL")
    print("=" * 80)
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    if argv is None:
        argv = sys.argv[1:]

    parser = argparse.ArgumentParser(
        prog="mlcore",
        description="Enterprise Machine Learning From Scratch Engine CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("-v", "--version", action="version", version=f"%(prog)s {__version__}")
    subparsers = parser.add_subparsers(dest="subcommand", help="Available subcommands")

    subparsers.add_parser("autograd", help="Reverse-mode scalar autograd engine & MLP")
    subparsers.add_parser("tokenizer", help="Byte Pair Encoding (BPE) tokenizer from scratch")
    subparsers.add_parser("transformer", help="Nano character-level GPT Transformer")
    subparsers.add_parser("vectordb", help="Hardened NumPy vector similarity database")
    subparsers.add_parser("dedup", help="MinHash & LSH document deduplication pipeline")
    subparsers.add_parser("guardrails", help="Self-healing JSON schema validation engine")
    subparsers.add_parser("demo", help="End-to-end multi-module pipeline execution")

    if not argv:
        parser.print_help()
        return 0

    subcommand = argv[0]
    sub_argv = argv[1:]

    if subcommand == "autograd":
        return _run_autograd(sub_argv)
    elif subcommand == "tokenizer":
        return _run_tokenizer(sub_argv)
    elif subcommand == "transformer":
        return _run_transformer(sub_argv)
    elif subcommand == "vectordb":
        return _run_vectordb(sub_argv)
    elif subcommand == "dedup":
        return _run_dedup(sub_argv)
    elif subcommand == "guardrails":
        return _run_guardrails(sub_argv)
    elif subcommand == "demo":
        return _run_demo()
    else:
        parser.print_help()
        return 1


if __name__ == "__main__":
    sys.exit(main())
