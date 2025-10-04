"""Real-world demonstration of DynHNSW as an in-memory vector database.

This example shows:
1. Building a vector database from scratch
2. Using static HNSW for baseline performance
3. Using adaptive HNSW with feedback to improve results
4. Tracking performance improvements over time

Scenario: Semantic document search with user feedback
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import time
from typing import List, Set, Tuple
from dynhnsw.hnsw.graph import HNSWGraph
from dynhnsw.hnsw.builder import HNSWBuilder
from dynhnsw.hnsw.searcher import HNSWSearcher
from dynhnsw.adaptive_hnsw import AdaptiveHNSWSearcher


# Simulated document database with semantic clusters
DOCUMENTS = {
    # Cluster 1: Machine Learning Papers
    0: "Neural networks for image classification using deep learning",
    1: "Convolutional neural networks achieve state-of-the-art accuracy",
    2: "Transfer learning improves model performance on small datasets",
    3: "Attention mechanisms in transformer architectures",
    4: "Self-supervised learning reduces need for labeled data",
    5: "Gradient descent optimization for deep neural networks",
    6: "Batch normalization accelerates neural network training",
    7: "Dropout regularization prevents overfitting in deep models",
    8: "ResNet architecture enables very deep networks",
    9: "GANs generate realistic synthetic images",

    # Cluster 2: Database Systems Papers
    10: "B-tree indexing for efficient database queries",
    11: "ACID properties ensure transaction consistency",
    12: "Distributed databases use sharding for scalability",
    13: "NoSQL databases provide flexible schema design",
    14: "Query optimization reduces database execution time",
    15: "Database replication ensures high availability",
    16: "Indexing strategies improve query performance",
    17: "Transaction isolation levels prevent data anomalies",
    18: "Column-store databases excel at analytical queries",
    19: "Database partitioning enables horizontal scaling",

    # Cluster 3: Web Development Topics
    20: "React components enable reusable UI elements",
    21: "RESTful APIs provide standard HTTP interfaces",
    22: "JavaScript async/await simplifies asynchronous code",
    23: "CSS flexbox enables responsive web layouts",
    24: "GraphQL reduces API over-fetching problems",
    25: "Webpack bundles JavaScript modules for production",
    26: "Server-side rendering improves initial page load",
    27: "Progressive web apps work offline using service workers",
    28: "TypeScript adds static typing to JavaScript",
    29: "OAuth 2.0 provides secure API authorization",

    # Cluster 4: Cloud Computing Topics
    30: "Kubernetes orchestrates containerized applications",
    31: "Docker containers provide lightweight virtualization",
    32: "Serverless computing eliminates infrastructure management",
    33: "Auto-scaling adjusts resources based on demand",
    34: "Load balancers distribute traffic across servers",
    35: "CDN caching reduces latency for global users",
    36: "Microservices architecture enables independent deployment",
    37: "Service mesh manages inter-service communication",
    38: "Cloud storage offers unlimited scalable capacity",
    39: "Infrastructure as code automates cloud provisioning",
}


def create_semantic_vectors(seed: int = 42) -> np.ndarray:
    """Create synthetic vectors with semantic clustering.

    Vectors are positioned in embedding space to reflect semantic similarity:
    - Cluster 1 (ML): High values in dimensions 0-31
    - Cluster 2 (DB): High values in dimensions 32-63
    - Cluster 3 (Web): High values in dimensions 64-95
    - Cluster 4 (Cloud): High values in dimensions 96-127
    """
    np.random.seed(seed)
    vectors = []

    for doc_id in range(len(DOCUMENTS)):
        base_vector = np.random.randn(128).astype(np.float32) * 0.5

        if doc_id < 10:  # ML cluster
            base_vector[:32] += 5.0
        elif doc_id < 20:  # DB cluster
            base_vector[32:64] += 5.0
        elif doc_id < 30:  # Web cluster
            base_vector[64:96] += 5.0
        else:  # Cloud cluster
            base_vector[96:128] += 5.0

        # Normalize
        base_vector = base_vector / np.linalg.norm(base_vector)
        vectors.append(base_vector)

    return np.array(vectors)


def build_vector_database(vectors: np.ndarray) -> HNSWGraph:
    """Build HNSW graph from vectors.

    Args:
        vectors: Document vectors to index

    Returns:
        Built HNSW graph
    """
    print("Building HNSW index...")
    graph = HNSWGraph(dimension=128, M=16)
    builder = HNSWBuilder(graph)

    for i, vector in enumerate(vectors):
        level = int(np.random.geometric(p=0.5)) - 1
        level = min(level, 5)
        builder.insert(vector, node_id=i, level=level)

    print(f"  [OK] Indexed {graph.size()} documents")
    print(f"  [OK] Graph has {graph.get_max_level() + 1} layers")
    return graph


def simulate_user_query(query_type: str, vectors: np.ndarray) -> Tuple[np.ndarray, Set[int]]:
    """Simulate a user query with ground truth relevance.

    Args:
        query_type: One of "ml", "database", "web", "cloud"
        vectors: Document vectors

    Returns:
        (query_vector, set of relevant document IDs)
    """
    query_configs = {
        "ml": (vectors[0:10].mean(axis=0), set(range(0, 10))),
        "database": (vectors[10:20].mean(axis=0), set(range(10, 20))),
        "web": (vectors[20:30].mean(axis=0), set(range(20, 30))),
        "cloud": (vectors[30:40].mean(axis=0), set(range(30, 40))),
    }

    base_query, relevant_ids = query_configs[query_type]

    # Add some noise to query
    query_vector = base_query + np.random.randn(128).astype(np.float32) * 0.1
    query_vector = query_vector / np.linalg.norm(query_vector)

    return query_vector, relevant_ids


def compute_recall(retrieved_ids: List[int], relevant_ids: Set[int], k: int) -> float:
    """Compute recall@k metric."""
    retrieved_set = set(retrieved_ids[:k])
    relevant_retrieved = retrieved_set.intersection(relevant_ids)
    return len(relevant_retrieved) / len(relevant_ids) if relevant_ids else 0.0


def compute_precision(retrieved_ids: List[int], relevant_ids: Set[int], k: int) -> float:
    """Compute precision@k metric."""
    retrieved_set = set(retrieved_ids[:k])
    relevant_retrieved = retrieved_set.intersection(relevant_ids)
    return len(relevant_retrieved) / k if k > 0 else 0.0


def demo_static_hnsw(graph: HNSWGraph, vectors: np.ndarray):
    """Demonstrate static HNSW search (baseline)."""
    print("\n" + "="*70)
    print("DEMO 1: Static HNSW Search (Baseline)")
    print("="*70)

    searcher = HNSWSearcher(graph, ef_search=50)

    # Simulate 10 ML-related queries
    print("\nSearching for Machine Learning documents...")
    recalls = []
    precisions = []

    for i in range(10):
        query, relevant_ids = simulate_user_query("ml", vectors)

        start_time = time.time()
        results = searcher.search(query, k=10)
        latency_ms = (time.time() - start_time) * 1000

        result_ids = [r[0] for r in results]
        recall = compute_recall(result_ids, relevant_ids, k=10)
        precision = compute_precision(result_ids, relevant_ids, k=10)

        recalls.append(recall)
        precisions.append(precision)

    print(f"  Average Recall@10:    {np.mean(recalls):.2%}")
    print(f"  Average Precision@10: {np.mean(precisions):.2%}")
    print(f"  Average Latency:      {latency_ms:.2f}ms")

    # Show example results
    query, relevant_ids = simulate_user_query("ml", vectors)
    results = searcher.search(query, k=5)

    print("\n  Example Query: 'Machine Learning papers'")
    print("  Top 5 Results:")
    for rank, (doc_id, distance) in enumerate(results, 1):
        relevant = "[OK]" if doc_id in relevant_ids else "[X]"
        print(f"    {rank}. [{relevant}] Doc {doc_id}: {DOCUMENTS[doc_id][:60]}...")

    return np.mean(recalls), np.mean(precisions)


def demo_adaptive_hnsw(graph: HNSWGraph, vectors: np.ndarray):
    """Demonstrate adaptive HNSW with feedback loop."""
    print("\n" + "="*70)
    print("DEMO 2: Adaptive HNSW with User Feedback")
    print("="*70)

    searcher = AdaptiveHNSWSearcher(graph, ef_search=50, learning_rate=0.1)

    # Phase 1: Initial performance (no adaptation yet)
    print("\n--- Phase 1: Initial Performance (Cold Start) ---")
    query, relevant_ids = simulate_user_query("ml", vectors)
    results = searcher.search(query, k=10)
    result_ids = [r[0] for r in results]

    initial_recall = compute_recall(result_ids, relevant_ids, k=10)
    initial_precision = compute_precision(result_ids, relevant_ids, k=10)

    print(f"  Initial Recall@10:    {initial_recall:.2%}")
    print(f"  Initial Precision@10: {initial_precision:.2%}")

    # Phase 2: Learning from feedback
    print("\n--- Phase 2: Learning from User Feedback ---")
    print("  Simulating 30 ML queries with feedback...")

    recalls_over_time = []
    precisions_over_time = []

    for i in range(30):
        query, relevant_ids = simulate_user_query("ml", vectors)

        # Search
        start_time = time.time()
        results = searcher.search(query, k=10)
        latency_ms = (time.time() - start_time) * 1000

        result_ids = [r[0] for r in results]

        # Simulate user marking relevant results
        # User identifies which results are actually relevant
        user_marked_relevant = set(rid for rid in result_ids if rid in relevant_ids)

        # Provide feedback to system
        searcher.provide_feedback(query, result_ids, user_marked_relevant)

        # Track performance
        recall = compute_recall(result_ids, relevant_ids, k=10)
        precision = compute_precision(result_ids, relevant_ids, k=10)

        recalls_over_time.append(recall)
        precisions_over_time.append(precision)

        # Record performance metrics
        searcher.record_performance(recall, precision, latency_ms)

    print(f"  [OK] Completed 30 queries with feedback")
    print(f"  [OK] Edge weights learned: {searcher.weight_learner.get_total_edges()}")

    # Phase 3: Post-adaptation performance
    print("\n--- Phase 3: Performance After Adaptation ---")

    final_recalls = []
    final_precisions = []

    for i in range(10):
        query, relevant_ids = simulate_user_query("ml", vectors)
        results = searcher.search(query, k=10)
        result_ids = [r[0] for r in results]

        recall = compute_recall(result_ids, relevant_ids, k=10)
        precision = compute_precision(result_ids, relevant_ids, k=10)

        final_recalls.append(recall)
        final_precisions.append(precision)

    final_recall = np.mean(final_recalls)
    final_precision = np.mean(final_precisions)

    print(f"  Final Recall@10:      {final_recall:.2%} (Delta {final_recall - initial_recall:+.2%})")
    print(f"  Final Precision@10:   {final_precision:.2%} (Delta {final_precision - initial_precision:+.2%})")

    # Show system statistics
    print("\n--- System Statistics ---")
    stats = searcher.get_statistics()

    print(f"  Graph: {stats['graph']['nodes']} nodes, {stats['graph']['max_level']+1} layers")
    print(f"  Weights: {stats['weights']['count']} edges learned")
    print(f"    - Min weight: {stats['weights']['min']:.2f}")
    print(f"    - Max weight: {stats['weights']['max']:.2f}")
    print(f"    - Mean weight: {stats['weights']['mean']:.2f}")
    print(f"  Stability: {stats['stability']['stability_score']:.2%} score")
    print(f"    - Oscillating edges: {stats['stability']['oscillating_edges']}")
    print(f"  Feedback: {stats['feedback']['total_queries']} queries")
    print(f"    - Avg satisfaction: {stats['feedback']['avg_satisfaction']:.2%}")
    print(f"  Performance:")
    print(f"    - Baseline recall: {stats['performance']['baseline_recall']:.2%}")
    print(f"    - Current recall: {stats['performance']['current_recall']:.2%}")

    # Show example with adapted weights
    print("\n  Example Query After Adaptation: 'Machine Learning papers'")
    query, relevant_ids = simulate_user_query("ml", vectors)
    results = searcher.search(query, k=5)

    print("  Top 5 Results:")
    for rank, (doc_id, distance) in enumerate(results, 1):
        relevant = "[OK]" if doc_id in relevant_ids else "[X]"
        print(f"    {rank}. [{relevant}] Doc {doc_id}: {DOCUMENTS[doc_id][:60]}...")

    return recalls_over_time, precisions_over_time


def demo_multi_intent_adaptation(graph: HNSWGraph, vectors: np.ndarray):
    """Demonstrate adaptation with multiple query intents."""
    print("\n" + "="*70)
    print("DEMO 3: Multi-Intent Adaptation")
    print("="*70)

    searcher = AdaptiveHNSWSearcher(graph, ef_search=50, learning_rate=0.1)

    print("\nUser searches for different topics over time...")
    print("  - Queries 1-10:  Machine Learning")
    print("  - Queries 11-20: Database Systems")
    print("  - Queries 21-30: Web Development")

    query_sequence = (
        ["ml"] * 10 +
        ["database"] * 10 +
        ["web"] * 10
    )

    for i, query_type in enumerate(query_sequence, 1):
        query, relevant_ids = simulate_user_query(query_type, vectors)

        results = searcher.search(query, k=10)
        result_ids = [r[0] for r in results]

        # User feedback
        user_marked_relevant = set(rid for rid in result_ids if rid in relevant_ids)
        searcher.provide_feedback(query, result_ids, user_marked_relevant)

        # Track metrics
        recall = compute_recall(result_ids, relevant_ids, k=10)
        precision = compute_precision(result_ids, relevant_ids, k=10)
        searcher.record_performance(recall, precision, 10.0)

    print(f"\n  [OK] Completed 30 queries across 3 different topics")

    # Test performance on each topic
    print("\n  Performance by Topic (after adaptation):")

    for topic in ["ml", "database", "web"]:
        recalls = []
        for _ in range(5):
            query, relevant_ids = simulate_user_query(topic, vectors)
            results = searcher.search(query, k=10)
            result_ids = [r[0] for r in results]
            recalls.append(compute_recall(result_ids, relevant_ids, k=10))

        topic_name = topic.upper() if topic == "ml" else topic.capitalize()
        print(f"    {topic_name:15} Recall@10: {np.mean(recalls):.2%}")

    # Show weight distribution
    stats = searcher.get_statistics()
    print(f"\n  System adapted to {stats['feedback']['total_queries']} diverse queries")
    print(f"  Learned weights on {stats['weights']['count']} edges")


def main():
    """Run all demonstrations."""
    print("="*70)
    print("DynHNSW: Real-World In-Memory Vector Database Demo")
    print("="*70)
    print("\nScenario: Semantic Document Search")
    print(f"  - {len(DOCUMENTS)} technical documents")
    print("  - 4 semantic clusters (ML, Databases, Web, Cloud)")
    print("  - 128-dimensional embedding vectors")

    # Create dataset
    vectors = create_semantic_vectors()

    # Build index
    graph = build_vector_database(vectors)

    # Demo 1: Static HNSW (baseline)
    baseline_recall, baseline_precision = demo_static_hnsw(graph, vectors)

    # Demo 2: Adaptive HNSW (single intent)
    recalls, precisions = demo_adaptive_hnsw(graph, vectors)

    # Demo 3: Multi-intent adaptation
    demo_multi_intent_adaptation(graph, vectors)

    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print("\n[OK] Static HNSW provides fast, accurate baseline search")
    print("[OK] Adaptive HNSW learns from user feedback to improve results")
    print("[OK] System handles multiple query intents independently")
    print("[OK] Stability monitoring prevents degradation")
    print("[OK] All operations performed in-memory for low latency")

    print("\n" + "="*70)
    print("Library Usage Pattern:")
    print("="*70)
    print("""
1. Build Index:
   graph = HNSWGraph(dimension=D, M=16)
   builder = HNSWBuilder(graph)
   for vector in vectors:
       builder.insert(vector, node_id=i, level=random_level())

2. Create Adaptive Searcher:
   searcher = AdaptiveHNSWSearcher(graph, ef_search=50, learning_rate=0.05)

3. Search & Learn Loop:
   results = searcher.search(query, k=10)
   user_feedback = get_user_feedback(results)  # Your feedback mechanism
   searcher.provide_feedback(query, result_ids, relevant_ids)

4. Monitor Performance:
   searcher.record_performance(recall, precision, latency)
   stats = searcher.get_statistics()
    """)


if __name__ == "__main__":
    main()
