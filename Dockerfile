# ==============================================================================
# Enterprise Multi-Stage Dockerfile for ML From Scratch Engine
# ==============================================================================

# Build Stage
FROM python:3.12-slim AS builder

WORKDIR /build

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml .
RUN pip install --no-cache-dir --prefix=/install .

# Production Stage
FROM python:3.12-slim AS runtime

LABEL org.opencontainers.image.title="ml-from-scratch-engine" \
      org.opencontainers.image.description="Enterprise Machine Learning From Scratch Suite: Autograd, Transformer, VectorDB, BPE, MinHash, Guardrails" \
      org.opencontainers.image.authors="cibi-dev" \
      org.opencontainers.image.vendor="Machine Learning Systems" \
      org.opencontainers.image.licenses="MIT"

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH="/app:/app/packages/autograd-engine:/app/packages/bpe-tokenizer:/app/packages/guardrails-engine:/app/packages/minhash-dedup:/app/packages/nano-transformer:/app/packages/numpy-vectordb"

# Copy installed dependencies from builder
COPY --from=builder /install /usr/local

# Create non-root user for DevSecOps compliance
RUN groupadd -r mle && useradd -r -g mle -u 1001 -m -d /app mle_user

# Copy application files
COPY --chown=mle_user:mle cli.py pyproject.toml README.md ./
COPY --chown=mle_user:mle packages/ ./packages/
COPY --chown=mle_user:mle benchmarks/ ./benchmarks/

USER mle_user

ENTRYPOINT ["python3", "/app/cli.py"]
CMD ["demo"]
