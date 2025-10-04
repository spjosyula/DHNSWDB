"""Intent-aware vector database demonstration.

This example showcases DynHNSW's core innovation: intent-aware adaptive search
that learns optimal entry points for different query patterns.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from dynhnsw import VectorStore

def main():
    print("="*70)
    print("DynHNSW - Intent-Aware Adaptive Vector Database Demo")
    print("="*70)

    # Step 1: Create vector store with intent detection enabled
    print("\n1. Creating vector store with intent detection...")

    store = VectorStore(
        dimension=128,
        M=16,
        ef_search=50,
        enable_intent_detection=True,  # Enable adaptive learning
        k_intents=3,                    # 3 distinct query patterns
        learning_rate=0.1,              # Moderate learning speed
        min_queries_for_clustering=20   # Start learning after 20 queries
    )

    print(f"   [OK] Store created with {store.dimension}D vectors")
    print(f"   [OK] Intent detection: {store.enable_intent_detection}")

    # Step 2: Add vectors with metadata
    print("\n2. Adding vectors with metadata...")

    np.random.seed(42)

    # Create 3 clusters of vectors (representing different document types)
    cluster_1 = np.random.randn(30, 128).astype(np.float32) + 3.0  # Technical docs
    cluster_2 = np.random.randn(30, 128).astype(np.float32) - 3.0  # Marketing content
    cluster_3 = np.random.randn(30, 128).astype(np.float32)        # General content

    all_vectors = np.vstack([cluster_1, cluster_2, cluster_3])

    # Normalize for cosine similarity
    all_vectors = all_vectors / np.linalg.norm(all_vectors, axis=1, keepdims=True)

    # Add with custom IDs and metadata
    ids = [f"tech_{i}" for i in range(30)] + \
          [f"marketing_{i}" for i in range(30)] + \
          [f"general_{i}" for i in range(30)]

    metadata = [{"type": "technical", "category": "docs"} for _ in range(30)] + \
               [{"type": "marketing", "category": "content"} for _ in range(30)] + \
               [{"type": "general", "category": "misc"} for _ in range(30)]

    store.add(all_vectors, ids=ids, metadata=metadata)

    print(f"   [OK] Added {store.size()} vectors with IDs and metadata")

    # Step 3: Perform searches and provide feedback (learning phase)
    print("\n3. Learning phase: searching and providing feedback...")

    # Simulate 3 different query intents
    for intent_num in range(3):
        cluster_idx = intent_num * 30

        # Search with queries from each cluster
        for i in range(10):
            query = all_vectors[cluster_idx + i]
            results = store.search(query, k=5)

            # Simulate user marking relevant results
            # Higher relevance for same-cluster results
            relevant_ids = [
                r["id"] for r in results
                if r["metadata"].get("type") == metadata[cluster_idx]["type"]
            ]

            store.provide_feedback(relevant_ids=relevant_ids)

    print(f"   [OK] Completed 30 queries with feedback")

    # Step 4: Check statistics to see learning progress
    print("\n4. System statistics after learning:")

    stats = store.get_statistics()

    print(f"   Total vectors: {stats['total_vectors']}")
    print(f"   Active vectors: {stats['active_vectors']}")

    if stats.get('intent_detection'):
        intent_stats = stats['intent_detection']
        print(f"   Intent clustering active: {intent_stats['clustering_active']}")
        print(f"   Confident detections: {intent_stats['confident_detections']}")

    if stats.get('entry_selection'):
        entry_stats = stats['entry_selection']
        print(f"   Entry point candidates: {entry_stats['num_candidates']}")
        print(f"   Total entry point usage: {entry_stats['total_usage']}")

        print("\n   Per-intent entry points:")
        for intent_info in entry_stats['per_intent']:
            print(f"     Intent {intent_info['intent_id']}: "
                  f"Entry {intent_info['best_entry']} "
                  f"(score: {intent_info['best_score']:.3f}, "
                  f"usage: {intent_info['total_usage']})")

    # Step 5: Compare search before/after learning
    print("\n5. Demonstrating adaptive improvement...")

    query = all_vectors[0]  # Technical document query
    results = store.search(query, k=5)

    print(f"   Query type: {metadata[0]['type']}")
    print("   Top 5 results:")
    for i, r in enumerate(results[:5], 1):
        print(f"     {i}. {r['id']} (dist: {r['distance']:.4f}, "
              f"type: {r['metadata'].get('type')})")

    # Step 6: Test delete functionality
    print("\n6. Testing soft delete...")

    store.delete(["tech_0", "tech_1"])
    results_after_delete = store.search(query, k=5)

    print(f"   Deleted: tech_0, tech_1")
    print(f"   Results after delete (tech_0, tech_1 excluded):")
    for i, r in enumerate(results_after_delete[:5], 1):
        print(f"     {i}. {r['id']}")

    # Step 7: Persistence
    print("\n7. Testing persistence (save/load)...")

    import tempfile
    with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as tmp:
        tmp_path = tmp.name

    try:
        store.save(tmp_path)
        print(f"   [OK] Saved store to {tmp_path}")

        loaded_store = VectorStore.load(tmp_path)
        print(f"   [OK] Loaded store from disk")

        # Verify it works
        loaded_results = loaded_store.search(query, k=3)
        print(f"   [OK] Search works on loaded store: {len(loaded_results)} results")
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)

    # Final statistics
    print("\n" + "="*70)
    print("Demo Complete!")
    print("="*70)
    print("\nKey Takeaways:")
    print("  - Intent-aware search learns optimal entry points per query pattern")
    print("  - Metadata and custom IDs for practical applications")
    print("  - Soft delete for data management")
    print("  - Persistence for saving/loading indexes")
    print("  - Full feedback loop: search -> feedback -> improved results")

if __name__ == "__main__":
    main()
