"""
Demo: Using DynHNSW as a Library

This example shows how someone would use DynHNSW in their own project
after installing it with: pip install git+https://github.com/spjosyula/DHNSWDB.git
"""

import numpy as np
from dynhnsw import VectorStore


def main():
    print("=" * 70)
    print("DynHNSW Library Usage Demo")
    print("=" * 70)

    # Create a vector store
    print("\n1. Creating VectorStore...")
    store = VectorStore(
        dimension=128,
        M=16,
        ef_search=50,
        enable_intent_detection=True,
        k_intents=3,
        min_queries_for_clustering=10
    )
    print(f"   [OK] Created store for {store.dimension}D vectors")

    # Add some random vectors
    print("\n2. Adding vectors...")
    np.random.seed(42)

    # Create 3 clusters of vectors (simulating different document types)
    cluster_1 = np.random.randn(30, 128).astype(np.float32) + 3.0
    cluster_2 = np.random.randn(30, 128).astype(np.float32) - 3.0
    cluster_3 = np.random.randn(30, 128).astype(np.float32)

    all_vectors = np.vstack([cluster_1, cluster_2, cluster_3])

    # Add with custom IDs and metadata
    ids = []
    metadata = []
    for i in range(90):
        if i < 30:
            ids.append(f"tech_doc_{i}")
            metadata.append({"category": "technical", "cluster": 0})
        elif i < 60:
            ids.append(f"marketing_doc_{i}")
            metadata.append({"category": "marketing", "cluster": 1})
        else:
            ids.append(f"general_doc_{i}")
            metadata.append({"category": "general", "cluster": 2})

    added_ids = store.add(all_vectors, ids=ids, metadata=metadata)
    print(f"   [OK] Added {len(added_ids)} vectors")

    # Perform searches
    print("\n3. Searching...")
    query = cluster_1[0]  # Query from technical cluster
    results = store.search(query, k=5)

    print("   Top 5 results:")
    for i, result in enumerate(results, 1):
        print(f"     {i}. {result['id']} - "
              f"distance: {result['distance']:.4f}, "
              f"category: {result['metadata']['category']}")

    # Provide feedback and search again
    print("\n4. Learning from feedback...")
    for iteration in range(15):
        # Alternate queries from different clusters
        cluster_idx = iteration % 3
        if cluster_idx == 0:
            query = cluster_1[iteration // 3]
            expected_category = "technical"
        elif cluster_idx == 1:
            query = cluster_2[iteration // 3]
            expected_category = "marketing"
        else:
            query = cluster_3[iteration // 3]
            expected_category = "general"

        results = store.search(query, k=5)

        # Mark results from same category as relevant
        relevant_ids = [
            r["id"] for r in results
            if r["metadata"]["category"] == expected_category
        ]
        store.provide_feedback(relevant_ids=relevant_ids)

    print(f"   [OK] Completed {iteration + 1} queries with feedback")

    # Check statistics
    print("\n5. System Statistics:")
    stats = store.get_statistics()
    print(f"   Total vectors: {stats['total_vectors']}")
    print(f"   Active vectors: {stats['active_vectors']}")

    if stats.get('intent_detection'):
        intent = stats['intent_detection']
        print(f"   Intent clustering active: {intent['clustering_active']}")
        print(f"   Confident detections: {intent['confident_detections']}")

    if stats.get('entry_selection'):
        entry = stats['entry_selection']
        print(f"   Entry point candidates: {entry['num_candidates']}")

    # Final search to show improved results
    print("\n6. Search after learning:")
    query = cluster_1[5]  # Another technical query
    results = store.search(query, k=5)

    print("   Results (should be mostly technical docs):")
    for i, result in enumerate(results, 1):
        print(f"     {i}. {result['id']} - {result['metadata']['category']}")

    # Save and load
    print("\n7. Testing save/load...")
    import tempfile
    import os

    with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as tmp:
        tmp_path = tmp.name

    try:
        store.save(tmp_path)
        print(f"   [OK] Saved to {tmp_path}")

        loaded_store = VectorStore.load(tmp_path)
        print(f"   [OK] Loaded from disk")

        # Verify loaded store works
        test_results = loaded_store.search(query, k=3)
        print(f"   [OK] Loaded store works: {len(test_results)} results")
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)

    print("\n" + "=" * 70)
    print("Demo Complete!")
    print("=" * 70)
    print("\nThis is how you would use DynHNSW in your own project.")
    print("Simply install with:")
    print("  pip install git+https://github.com/spjosyula/DHNSWDB.git")
    print("\nThen import and use as shown above!")


if __name__ == "__main__":
    main()
