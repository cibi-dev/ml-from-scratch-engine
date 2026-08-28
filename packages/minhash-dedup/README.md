# MinHash & LSH Data Curation & Deduplication Pipeline

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Type Checked](https://img.shields.io/badge/mypy-strict-green.svg)](https://mypy-lang.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A high-performance, mathematically rigorous, zero-external-dependency (NumPy only) **MinHash and Locality-Sensitive Hashing (LSH)** data curation and deduplication engine built from scratch in Python.

Designed specifically for **large-scale LLM pretraining dataset preparation** (matching the data filtering pipelines of **FineWeb**, **SlimPajama**, and **RefinedWeb**), this system eliminates exact and near-duplicate documents with sub-linear $O(N)$ query complexity, robust Hash-DoS defenses, and strict Unicode normalization.

---

## 🏛️ Architecture & Dataflow

```
   Raw Document Corpus (List or Dict)
                  │
                  ▼
   ┌──────────────────────────────────────────────┐
   │         1. TEXT NORMALIZATION & PREPROCESSING │
   │  - Unicode NFKC decomposition & composition  │
   │  - Lowercasing                               │
   │  - Strip Unicode Category 'C' (Format/Ctrl)  │
   │  - Collapse whitespace, Linear ReDoS-safe    │
   └──────────────────────┬───────────────────────┘
                          │ Clean Text
                          ▼
   ┌──────────────────────────────────────────────┐
   │            2. K-GRAM SHINGLING               │
   │  - Word-level or Character-level k-grams     │
   │  - Short document (< k) fallback shingle     │
   └──────────────────────┬───────────────────────┘
                          │ Shingle Set X = {s_1, s_2, ..., s_m}
                          ▼
   ┌──────────────────────────────────────────────┐
   │      3. VECTORIZED 64-BIT MINHASHING         │
   │  - 64-bit Blake2b shingle hashing            │
   │  - Universal hash: h_i(x)=(a_i*x+b_i)% (2^61-1)│
   │  - Signature vector S = [min h_1, ..., min h_K│
   └──────────────────────┬───────────────────────┘
                          │ Signature S in (num_perm,) uint64
                          ▼
   ┌──────────────────────────────────────────────┐
   │        4. LOCALITY-SENSITIVE HASHING (LSH)   │
   │  - Numerical parameter optimization (b, r)   │
   │  - Band slicing: b bands x r rows = K        │
   │  - Hash-DoS guard: MAX_BUCKET_SIZE = 5000    │
   └──────────────────────┬───────────────────────┘
                          │ Candidate Pairs (u, v)
                          ▼
   ┌──────────────────────────────────────────────┐
   │    5. DISJOINT-SET CLUSTERING & CANONICAL    │
   │  - Verify Jaccard estimate >= threshold      │
   │  - UnionFind with path compression & rank    │
   │  - Policy: 'first' | 'longest' | 'shortest'  │
   └──────────────────────┬───────────────────────┘
                          │
                          ▼
       (Kept Document IDs, Removed Document IDs)
```

---

## 📈 Locality-Sensitive Hashing: The S-Curve

LSH groups $K$ MinHash permutations into $b$ bands of $r$ rows ($b \times r \le K$). Two documents share a bucket if all $r$ hash values in any of the $b$ bands match.

The probability of two documents with Jaccard similarity $s \in [0, 1]$ becoming candidate duplicates is:

$$P(\text{candidate} \mid s) = 1 - (1 - s^r)^b$$

### ASCII S-Curve Illustration ($b=16, r=8, K=128, t=0.75$)

```
 Probability of Candidate Pairing P(s)
 1.0 ┤                                        *************
     │                                    ****
 0.8 ┤                                 ***
     │                               **
 0.6 ┤                             **
     │                           **
 0.5 ┼- - - - - - - - - - - - - * (Inflection Point s ≈ (1/b)^(1/r) ≈ 0.707)
 0.4 ┤                        *
     │                      **
 0.2 ┤                    **
     │               *****
 0.0 ┼***************─────────────────────────────────────
     0.0    0.2     0.4     0.6    0.75    0.8     1.0
                         Jaccard Similarity s
```

- **Pairs with $s < t$ (dissimilar):** $P(s) \approx 0 \implies$ filtered out in $O(1)$ without pair comparison.
- **Pairs with $s \ge t$ (near-duplicates):** $P(s) \approx 1 \implies$ captured in candidate buckets.

---

## 🧮 Mathematical Foundation

### 1. Jaccard Similarity & The MinHash Theorem
For two sets of shingles $A$ and $B$, their Jaccard similarity is:

$$J(A, B) = \frac{|A \cap B|}{|A \cup B|}$$

Under a truly random permutation $\pi$, the probability that the minimum hash of $A$ equals the minimum hash of $B$ is identically $J(A, B)$:

$$\Pr_{\pi}[\min(\pi(A)) = \min(\pi(B))] = J(A, B)$$

We estimate $J(A, B)$ across $K$ independent permutations as the unbiased estimator:

$$\hat{J}(A, B) = \frac{1}{K} \sum_{i=1}^K \mathbb{I}(\text{sig}_A[i] = \text{sig}_B[i])$$

### 2. Universal Hashing & Mersenne Prime $2^{61} - 1$
To compute $K$ permutations efficiently without storing full permutation tables, we use Carter-Wegman 2-universal hashing:

$$h_i(x) = (a_i \cdot x + b_i) \pmod p$$

where:
- $p = 2^{61} - 1 = 2{,}305{,}843{,}009{,}213{,}693{,}951$ is the 9th Mersenne prime ($M_{61}$).
- $a_i \in [1, p-1]$ and $b_i \in [0, p-1]$ are generated with deterministic seeds.
- $x$ is the 64-bit integer hash of a shingle.
- Calculations use arbitrary-precision intermediate operations to guarantee zero 64-bit overflow degradation before casting to `uint64`.

### 3. Parameter Optimization via Weighted Error Integration
To choose $(b, r)$ automatically for a given threshold $t \in (0, 1)$ and $K$ permutations, we minimize the weighted area of false positives ($s < t$) and false negatives ($s \ge t$):

$$\mathcal{L}(b, r) = w_{\text{FP}} \int_0^t \left(1 - (1 - s^r)^b\right) \, ds + w_{\text{FN}} \int_t^1 \left(1 - \left(1 - (1 - s^r)^b\right)\right) \, ds$$

Evaluated via numerical trapezoidal integration over all valid factorings $b \cdot r \le K$.

---

## 🌐 Real-World Context: FineWeb & SlimPajama

In modern LLM pretraining (such as **FineWeb** 15-Trillion tokens, **SlimPajama** 627B tokens, and **RefinedWeb**):
1. **Curriculum Efficiency:** Duplicate data causes memorization, reduces generalization, and wastes compute.
2. **Computational Feasibility:** Pairwise comparison of $10^9$ documents requires $\approx 5 \times 10^{17}$ comparisons ($O(N^2)$). MinHash + LSH reduces this to sub-linear time $O(N)$ candidate verification.
3. **Cluster Merging:** Large duplicate networks (e.g. syndicated news, spam templates) are connected via Union-Find, keeping exactly one canonical version.

---

## 🔒 Security Hardening & Robustness

1. **Hash-DoS Defense (`MAX_BUCKET_SIZE = 5000`):**
   Adversarial documents or repetitive boilerplate (e.g., copyright headers, cookie warnings) can flood a single LSH bucket. MinHashLSH caps bucket capacity at 5000 items, preventing $O(N^2)$ candidate explosions.
2. **Corpus Size Guard (`MAX_CORPUS_SIZE = 1_000_000`):**
   Prevents unconstrained RAM exhaustion on single-node pipeline invocations.
3. **Unicode Adversarial Bypass Defense:**
   Attackers and web scrapers inject invisible zero-width spaces (`\u200b`, `\u200c`, `\u200d`), BOM (`\ufeff`), and soft-hyphens (`\u00ad`) to bypass hash-matching. `normalize_text` strips all Unicode Category `C` (Other/Control/Format) characters after NFKC normalization.
4. **Linear ReDoS Protection:**
   Tokenization and normalization patterns avoid nested quantifiers, guaranteeing $O(N)$ execution time on pathological inputs.

---

## 📦 Installation

```bash
# Clone repository
git clone https://github.com/cibi-dev/minhash-dedup.git
cd minhash-dedup

# Install with uv
uv sync --all-extras
```

---

## 🚀 Quickstart & API Reference

### 1. End-to-End Pipeline

```python
from minhash_dedup import deduplicate_corpus

documents = {
    "doc_1": "The quick brown fox jumps over the lazy dog in the park.",
    "doc_2": "The quick brown fox jumps over the lazy dog in the park.",  # Exact duplicate
    "doc_3": "The quick brown fox jumps over the lazy dog in the green park.",  # Near duplicate
    "doc_4": "Quantum computing utilizes superposition and quantum entanglement.",
}

kept_ids, removed_ids = deduplicate_corpus(
    documents=documents,
    threshold=0.75,
    k_shingle=3,
    num_perm=128,
    canonical_policy="first",  # 'first', 'longest', or 'shortest'
)

print("Kept IDs:", kept_ids)        # ['doc_1', 'doc_4']
print("Removed IDs:", removed_ids)  # ['doc_2', 'doc_3']
```

### 2. Low-Level MinHasher & LSH Index

```python
from minhash_dedup import MinHasher, MinHashLSH, get_shingles, normalize_text

# 1. Normalize & Shingle
text_a = normalize_text("Machine learning for data science applications.")
text_b = normalize_text("Machine learning for large data science applications.")

shingles_a = get_shingles(text_a, k=3, mode="word")
shingles_b = get_shingles(text_b, k=3, mode="word")

# 2. MinHash Signatures
hasher = MinHasher(num_perm=128, seed=42)
sig_a = hasher.compute_signature(shingles_a)  # shape: (128,), dtype: uint64
sig_b = hasher.compute_signature(shingles_b)

sim = hasher.estimate_jaccard(sig_a, sig_b)
print(f"Estimated Jaccard: {sim:.4f}")

# 3. LSH Indexing
lsh = MinHashLSH(threshold=0.75, num_perm=128)
lsh.insert("doc_a", sig_a)
lsh.insert("doc_b", sig_b)

candidates = lsh.query_candidates(sig_a)
print("Candidates for doc_a:", candidates)
```

---

## 🧪 Testing & Verification

Run the full test suite with Pytest and strict MyPy type checking:

```bash
# Run test suite
uv run pytest -v --tb=short

# Run strict type checking
uv run mypy minhash_dedup/ --strict

# Run example script
uv run python examples/deduplicate_corpus.py
```

---

## 📄 License

MIT License - Copyright (c) 2026 cibi-dev. See [LICENSE](LICENSE) for details.
