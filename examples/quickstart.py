"""Quick start guide for DynHNSW - Adaptive Vector Database.

This example shows the minimal code needed to:
1. Create an in-memory vector database
2. Search for similar vectors
3. Provide feedback to improve results
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from dynhnsw.hnsw.graph import HNSWGraph
from dynhnsw.hnsw.builder import HNSWBuilder
from dynhnsw.adaptive_hnsw import AdaptiveHNSWSearcher


def main():
    print("="*60)
    print("DynHNSW Quick Start")
    print("="*60)

    # Step 1: Create synthetic dataset
    print("\n1. Creating dataset...")
    np.random.seed(42)

    # 100 vectors, 128 dimensions
    vectors = np.random.randn(100, 128).astype(np.float32)

    # Normalize vectors for cosine similarity
    vectors = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)

    print(f"   Created {len(vectors)} vectors of dimension {vectors.shape[1]}")

    # Step 2: Build HNSW index
    print("\n2. Building HNSW index...")

    graph = HNSWGraph(dimension=128, M=16)
    builder = HNSWBuilder(graph)

    for i, vector in enumerate(vectors):
        # Assign random level using geometric distribution
        level = int(np.random.geometric(p=0.5)) - 1
        level = min(level, 5)  # Cap at level 5
        builder.insert(vector, node_id=i, level=level)

    print(f"   Indexed {graph.size()} vectors")
    print(f"   Graph has {graph.get_max_level() + 1} layers")

    # Step 3: Create adaptive searcher
    print("\n3. Creating adaptive searcher...")

    searcher = AdaptiveHNSWSearcher(
        graph,
        ef_search=50,           # Search quality (higher = better recall)
        learning_rate=0.1,      # Adaptation speed
        enable_adaptation=True  # Turn on learning
    )

    print("   Searcher ready with adaptive learning enabled")

    # Step 4: Perform search
    print("\n4. Searching...")

    query = vectors[0]  # Use first vector as query
    k = 10

    results = searcher.search(query, k=k)

    print(f"   Top {k} results:")
    for rank, (doc_id, distance) in enumerate(results[:5], 1):
        print(f"      {rank}. Document {doc_id} (distance: {distance:.4f})")

    # Step 5: Provide feedback
    print("\n5. Providing user feedback...")

    result_ids = [r[0] for r in results]

    # Simulate user marking first 5 results as relevant
    relevant_ids = set(result_ids[:5])

    searcher.provide_feedback(query, result_ids, relevant_ids)

    print(f"   Marked {len(relevant_ids)} results as relevant")
    print(f"   System learned from feedback")

    # Step 6: Track performance
    print("\n6. Recording performance metrics...")

    # In real app, compute these from your evaluation
    recall = len(relevant_ids) / k  # Simplified
    precision = len(relevant_ids) / k
    latency_ms = 5.0  # Example latency

    searcher.record_performance(recall, precision, latency_ms)

    print(f"   Recall@{k}: {recall:.2%}")
    print(f"   Precision@{k}: {precision:.2%}")

    # Step 7: Get system statistics
    print("\n7. System statistics:")

    stats = searcher.get_statistics()

    print(f"   Edges learned: {stats['weights']['count']}")
    print(f"   Mean weight: {stats['weights']['mean']:.2f}")
    print(f"   Stability score: {stats['stability']['stability_score']:.2%}")
    print(f"   Total queries: {stats['feedback']['total_queries']}")

    # Step 8: Demonstrate continuous learning
    print("\n8. Demonstrating continuous learning...")

    for i in range(10):
        query = vectors[i+1]
        results = searcher.search(query, k=10)
        result_ids = [r[0] for r in results]

        # Simulate varying feedback
        num_relevant = np.random.randint(3, 8)
        relevant_ids = set(result_ids[:num_relevant])

        searcher.provide_feedback(query, result_ids, relevant_ids)

    print(f"   Completed 10 more queries with feedback")

    final_stats = searcher.get_statistics()
    print(f"   Total queries processed: {final_stats['feedback']['total_queries']}")
    print(f"   Edges with learned weights: {final_stats['weights']['count']}")
    print(f"   Average user satisfaction: {final_stats['feedback']['avg_satisfaction']:.2%}")

    print("\n" + "="*60)
    print("Quick Start Complete!")
    print("="*60)
    print("\nKey Takeaways:")
    print("  - Build index once, search many times (in-memory)")
    print("  - Adaptation is optional (disable with enable_adaptation=False)")
    print("  - Provide feedback to improve results over time")
    print("  - Monitor system health with get_statistics()")


if __name__ == "__main__":
    main()
