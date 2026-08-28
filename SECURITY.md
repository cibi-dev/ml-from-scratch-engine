# Security Policy — `ml-from-scratch-engine`

## Standards Applied (SECURITY.md Canonical #1–17)

### Base Controls (#1–5)
- **#1 Secrets:** Zero credentials in repository. Continuous secret scanning.
- **#2 Input Validation:** Vector dimensions, token IDs, and hyperparameter configs validated with strict boundary checks.
- **#3 Safe Serialization:** NumPy persistence uses `.npz` with `allow_pickle=False` (CWE-502); zero `pickle` on untrusted inputs.
- **#4 Dependency Management:** Core algorithms implemented from scratch with minimal pinned dependencies.
- **#5 Error Masking:** Mathematical operations guard against NaN/Inf propagation with explicit validation.

### Phase 2 Controls (#6–13)
- **#7 Resource Limits:** Vector database enforces maximum vector count and dimension quotas to prevent memory exhaustion (CWE-400).
- **#8 Isolated Computation:** Tensor operations isolated within computational graphs; memory management with explicit gradient zeroing.
- **#9 Deterministic PRNG:** Seeded pseudorandom number generators for reproducible training and inference.
- **#10 Computational Guardrails:** Attention mechanisms enforce causal non-leakage invariant masks.
- **#12 Tokenizer Robustness:** BPE tokenizer handles out-of-vocabulary bytes, invalid UTF-8 sequences, and regex injection safely.
- **#13 LSH Hash Guarantees:** MinHash deduplication uses uniform pairwise independent hash functions with bounded buckets.

### AI Guardrail Controls (#14–17)
- **#14 Network Boundary:** Guardrails client enforces private network blocking on LLM API endpoints (CWE-918).
- **#15 Self-Healing Schema Guardrails:** LLM response validation with automated retry budget (max 3 self-healing attempts).
- **#16 Output Containment:** LLM response parsing rejects unvalidated fields and malicious prompt injections (OWASP LLM01).
- **#17 Generation Bounding:** Autoregressive generation strictly bounded by `max_new_tokens` and EOS token detection.

## Reporting Vulnerabilities
Open a private security advisory via GitHub Security Advisories or contact `cibi-dev@users.noreply.github.com`.
