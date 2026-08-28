"""Semantic Search & Document Retrieval with Hardened Pure NumPy VectorDB.

This runnable example demonstrates:
1. Initializing VectorDB with Cosine Similarity.
2. Upserting documents with simulated dense embeddings and rich metadata.
3. Top-K semantic query execution.
4. Filtered search using custom metadata predicates.
5. Hardened persistence (np.savez_compressed + sidecar JSON).
"""

from pathlib import Path
import numpy as np

from numpy_vectordb import SimilarityMetric, VectorDB


def main() -> None:
    print("=" * 70)
    print("🚀 Hardened Pure NumPy Vector Database — Semantic Search Example")
    print("=" * 70)

    # 1. Initialize Vector Database
    storage_path = Path("./data/demo_storage")
    db = VectorDB(
        dim=4,
        metric=SimilarityMetric.COSINE,
        storage_dir=storage_path,
        initial_capacity=8,
    )
    print(f"📦 Initialized VectorDB (Dimension: {db.dim}, Metric: {db.metric.value})")

    # 2. Simulated dense semantic embeddings for corpus documents
    # In real applications, these embeddings would come from models like nomic-embed, text-embedding-3, etc.
    documents = [
        (
            "doc_linux_kernel",
            [0.90, 0.10, 0.05, 0.02],
            {"title": "Linux Kernel Internals", "category": "systems", "year": 2026},
        ),
        (
            "doc_numpy_blas",
            [0.85, 0.25, 0.10, 0.05],
            {"title": "High Performance Vector Math with NumPy & BLAS", "category": "systems", "year": 2026},
        ),
        (
            "doc_quantum_computing",
            [0.10, 0.95, 0.15, 0.05],
            {"title": "Introduction to Quantum Error Correction", "category": "physics", "year": 2025},
        ),
        (
            "doc_transformer_agents",
            [0.78, 0.30, 0.45, 0.10],
            {"title": "Agentic Workflows with Large Language Models", "category": "ai", "year": 2026},
        ),
        (
            "doc_astrophysics_dark_matter",
            [0.05, 0.88, 0.20, 0.10],
            {"title": "Observational Constraints on Dark Matter", "category": "physics", "year": 2024},
        ),
    ]

    print(f"\n📥 Inserting {len(documents)} documents with dense embeddings...")
    for doc_id, vec, meta in documents:
        db.upsert(doc_id, vec, metadata=meta)
    print(f"✅ Total stored vectors: {db.count()}")

    # 3. Top-K Semantic Similarity Search
    query_vector = [0.88, 0.15, 0.08, 0.03]  # Query related to systems/numpy/compilers
    print(f"\n🔍 Querying with vector: {query_vector} (Top 3):")
    results = db.query(query_vector, top_k=3, include_vector=False)

    for rank, res in enumerate(results, start=1):
        print(
            f"  [{rank}] Score: {res.score:.4f} | ID: {res.id:<25} | "
            f"Title: {res.metadata['title']} ({res.metadata['category']})"
        )

    # 4. Filtered Semantic Search (Category == 'physics')
    print("\n🔍 Querying with Metadata Filter (category == 'physics'):")
    physics_results = db.query(
        query_vector,
        top_k=3,
        filter_fn=lambda m: m.get("category") == "physics",
    )

    for rank, res in enumerate(physics_results, start=1):
        print(
            f"  [{rank}] Score: {res.score:.4f} | ID: {res.id:<30} | "
            f"Title: {res.metadata['title']} ({res.metadata['year']})"
        )

    # 5. Persistence: Safe Save and Reload
    collection_name = "knowledge_base"
    print(f"\n💾 Persisting collection '{collection_name}' to disk...")
    npz_path, json_path = db.save(collection_name)
    print(f"  • Vector Archive:   {npz_path}")
    print(f"  • Metadata Sidecar: {json_path}")

    print("\n🔄 Reloading collection into a new VectorDB instance...")
    reloaded_db = VectorDB.load_collection(collection_name, storage_dir=storage_path)
    print(f"✅ Reloaded successfully: {reloaded_db.count()} documents available.")

    # Clean up test artifacts
    if npz_path.exists():
        npz_path.unlink()
    if json_path.exists():
        json_path.unlink()
    if storage_path.exists():
        storage_path.rmdir()
        if storage_path.parent.exists() and not any(storage_path.parent.iterdir()):
            storage_path.parent.rmdir()

    print("\n🎉 Demo completed successfully!")


if __name__ == "__main__":
    main()
