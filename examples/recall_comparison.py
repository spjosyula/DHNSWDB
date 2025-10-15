"""Recall-based comparison: Dynamic HNSW vs Static HNSW.

This example demonstrates the refactored intent-aware HNSW system that optimizes
for recall@k instead of latency. The system learns to allocate higher ef_search
to difficult queries and lower ef_search to easy queries.

Key Features:
1. Difficulty-based intent detection (5 tiers)
2. Recall-based Q-learning with phased cold start
3. Pre-computed ground truth for accurate evaluation
4. Comparison of dynamic vs static HNSW on recall metrics
"""

import numpy as np
import time
from typing import List, Tuple, Dict
from collections import defaultdict

from dynhnsw.hnsw.builder import HNSWBuilder
from dynhnsw.intent_aware_hnsw import IntentAwareHNSWSearcher
from dynhnsw.metrics import compute_ground_truth_brute_force, compute_recall_at_k


def generate_synthetic_dataset(n_vectors: int = 1000, dim: int = 128, seed: int = 42) -> np.ndarray:
    """Generate synthetic vector dataset.

    Args:
        n_vectors: Number of vectors
        dim: Vector dimensionality
        seed: Random seed

    Returns:
        Array of shape (n_vectors, dim)
    """
    np.random.seed(seed)
    return np.random.randn(n_vectors, dim).astype(np.float32)


def generate_test_queries(n_queries: int = 100, dim: int = 128, seed: int = 43) -> np.ndarray:
    """Generate test query vectors.

    Args:
        n_queries: Number of queries
        dim: Vector dimensionality
        seed: Random seed

    Returns:
        Array of shape (n_queries, dim)
    """
    np.random.seed(seed)
    return np.random.randn(n_queries, dim).astype(np.float32)


def precompute_ground_truth(
    queries: np.ndarray,
    database: np.ndarray,
    k: int = 10
) -> Dict[int, List[int]]:
    """Pre-compute ground truth k-NN for all queries.

    Args:
        queries: Query vectors (n_queries, dim)
        database: Database vectors (n_vectors, dim)
        k: Number of neighbors

    Returns:
        Dictionary mapping query index to ground truth neighbor IDs
    """
    print(f"Pre-computing ground truth for {len(queries)} queries...")
    ground_truth = {}

    for i, query in enumerate(queries):
        gt_ids, _ = compute_ground_truth_brute_force(query, database, k=k)
        ground_truth[i] = gt_ids

        if (i + 1) % 20 == 0:
            print(f"  Processed {i + 1}/{len(queries)} queries")

    print("Ground truth computation complete.")
    return ground_truth


def build_hnsw_index(vectors: np.ndarray, M: int = 16, ef_construction: int = 200) -> object:
    """Build HNSW index from vectors.

    Args:
        vectors: Database vectors (n_vectors, dim)
        M: Max connections per node
        ef_construction: Construction parameter (not used in current implementation)

    Returns:
        HNSW graph
    """
    from dynhnsw.hnsw.graph import HNSWGraph
    from dynhnsw.hnsw.utils import assign_layer

    print(f"Building HNSW index with {len(vectors)} vectors...")

    # Create graph with proper parameters
    dim = vectors.shape[1]
    graph = HNSWGraph(dimension=dim, M=M)

    # Create builder
    builder = HNSWBuilder(graph=graph)

    for i, vector in enumerate(vectors):
        # Assign layer using geometric distribution
        level = assign_layer(level_multiplier=graph.level_multiplier)
        # Insert vector into graph
        builder.insert(vector=vector, node_id=i, level=level)

        if (i + 1) % 200 == 0:
            print(f"  Indexed {i + 1}/{len(vectors)} vectors")

    print("HNSW index construction complete.")
    return graph


def run_static_hnsw_baseline(
    graph,
    queries: np.ndarray,
    ground_truth: Dict[int, List[int]],
    k: int = 10,
    ef_search: int = 100
) -> Dict[str, float]:
    """Run static HNSW baseline with fixed ef_search.

    Args:
        graph: HNSW graph
        queries: Query vectors
        ground_truth: Pre-computed ground truth
        k: Number of neighbors
        ef_search: Fixed ef_search value

    Returns:
        Results dictionary with recall metrics
    """
    print(f"\n=== Static HNSW Baseline (ef_search={ef_search}) ===")

    # Create searcher with adaptation disabled
    searcher = IntentAwareHNSWSearcher(
        graph=graph,
        ef_search=ef_search,
        enable_adaptation=False,
        enable_intent_detection=False
    )

    recalls = []
    latencies = []

    for i, query in enumerate(queries):
        start_time = time.perf_counter()
        results = searcher.search(query, k=k)
        latency_ms = (time.perf_counter() - start_time) * 1000

        result_ids = [node_id for node_id, _ in results]
        recall = compute_recall_at_k(result_ids, ground_truth[i], k=k)

        recalls.append(recall)
        latencies.append(latency_ms)

    avg_recall = np.mean(recalls)
    avg_latency = np.mean(latencies)

    print(f"Average Recall@{k}: {avg_recall:.4f} ({avg_recall*100:.2f}%)")
    print(f"Average Latency: {avg_latency:.2f} ms")

    return {
        "avg_recall": avg_recall,
        "avg_latency": avg_latency,
        "recalls": recalls,
        "latencies": latencies
    }


def run_dynamic_hnsw(
    graph,
    queries: np.ndarray,
    ground_truth: Dict[int, List[int]],
    k: int = 10,
    default_ef: int = 100,
    k_intents: int = 5,
    use_ucb1: bool = True
) -> Dict[str, float]:
    """Run dynamic HNSW with intent detection and adaptive ef_search.

    Args:
        graph: HNSW graph
        queries: Query vectors
        ground_truth: Pre-computed ground truth
        k: Number of neighbors
        default_ef: Default ef_search value
        k_intents: Number of intent tiers
        use_ucb1: Use UCB1 exploration strategy

    Returns:
        Results dictionary with recall metrics
    """
    print(f"\n=== Dynamic HNSW (k_intents={k_intents}, UCB1={use_ucb1}) ===")

    # Create searcher with adaptation enabled
    from dynhnsw.config import DynHNSWConfig
    config = DynHNSWConfig(
        enable_ucb1=use_ucb1,
        exploration_rate=0.15,
        enable_epsilon_decay=False,  # Fixed epsilon as recommended
        confidence_threshold=0.5
    )

    searcher = IntentAwareHNSWSearcher(
        graph=graph,
        ef_search=default_ef,
        k_intents=k_intents,
        enable_adaptation=True,
        enable_intent_detection=True,
        min_queries_for_clustering=10,
        config=config
    )

    recalls = []
    latencies = []
    intent_history = []
    ef_history = []

    for i, query in enumerate(queries):
        start_time = time.perf_counter()
        results = searcher.search(query, k=k)
        latency_ms = (time.perf_counter() - start_time) * 1000

        result_ids = [node_id for node_id, _ in results]
        recall = compute_recall_at_k(result_ids, ground_truth[i], k=k)

        recalls.append(recall)
        latencies.append(latency_ms)
        intent_history.append(searcher.last_intent_id)
        ef_history.append(searcher.last_ef_used)

        # Provide feedback for learning
        searcher.provide_feedback(
            query=query,
            result_ids=result_ids,
            ground_truth_ids=ground_truth[i],
            k=k
        )

    avg_recall = np.mean(recalls)
    avg_latency = np.mean(latencies)

    print(f"Average Recall@{k}: {avg_recall:.4f} ({avg_recall*100:.2f}%)")
    print(f"Average Latency: {avg_latency:.2f} ms")

    # Show intent detection statistics
    if searcher.enable_intent_detection:
        stats = searcher.get_statistics()
        print(f"\nIntent Detection Statistics:")
        print(f"  Clustering Active: {stats['intent_detection']['clustering_active']}")
        print(f"  Total Queries: {stats['intent_detection']['total_queries']}")

        if 'centroids' in stats['intent_detection']:
            centroids = stats['intent_detection']['centroids']
            print(f"  Difficulty Centroids:")
            for idx, centroid in enumerate(centroids):
                print(f"    Intent {idx}: {centroid:.4f}")

    # Show ef_search selection statistics
    if searcher.enable_adaptation:
        ef_stats = stats['ef_search_selection']
        print(f"\nef_search Selection Statistics:")
        print(f"  Current Phase: {ef_stats['phase_name']}")
        print(f"  Total Updates: {ef_stats['total_updates']}")
        print(f"  Learned ef_search per intent:")
        for intent_stat in ef_stats['per_intent']:
            intent_id = intent_stat['intent_id']
            learned_ef = intent_stat['learned_ef']
            num_queries = intent_stat['num_queries']
            print(f"    Intent {intent_id}: ef={learned_ef} (n={num_queries})")

    return {
        "avg_recall": avg_recall,
        "avg_latency": avg_latency,
        "recalls": recalls,
        "latencies": latencies,
        "intent_history": intent_history,
        "ef_history": ef_history
    }


def compare_results(static_results: Dict, dynamic_results: Dict, k: int = 10):
    """Compare static vs dynamic HNSW results.

    Args:
        static_results: Results from static HNSW
        dynamic_results: Results from dynamic HNSW
        k: Number of neighbors
    """
    print(f"\n{'='*60}")
    print("COMPARISON SUMMARY")
    print(f"{'='*60}")

    static_recall = static_results['avg_recall']
    dynamic_recall = dynamic_results['avg_recall']
    recall_improvement = (dynamic_recall - static_recall) / static_recall * 100

    static_latency = static_results['avg_latency']
    dynamic_latency = dynamic_results['avg_latency']
    latency_overhead = (dynamic_latency - static_latency) / static_latency * 100

    print(f"\nRecall@{k}:")
    print(f"  Static HNSW:  {static_recall:.4f} ({static_recall*100:.2f}%)")
    print(f"  Dynamic HNSW: {dynamic_recall:.4f} ({dynamic_recall*100:.2f}%)")
    print(f"  Improvement:  {recall_improvement:+.2f}%")

    print(f"\nLatency:")
    print(f"  Static HNSW:  {static_latency:.2f} ms")
    print(f"  Dynamic HNSW: {dynamic_latency:.2f} ms")
    print(f"  Overhead:     {latency_overhead:+.2f}%")

    print(f"\nVerdict:")
    if recall_improvement > 5.0:
        print(f"  EXCELLENT: Dynamic HNSW significantly outperforms static (+{recall_improvement:.1f}% recall)")
    elif recall_improvement > 0.0:
        print(f"  GOOD: Dynamic HNSW improves recall (+{recall_improvement:.1f}%)")
    else:
        print(f"  NEEDS WORK: Dynamic HNSW underperforms ({recall_improvement:.1f}%)")

    if latency_overhead > 100.0:
        print(f"  WARNING: High latency overhead (+{latency_overhead:.1f}%)")
    elif latency_overhead > 50.0:
        print(f"  ACCEPTABLE: Moderate latency overhead (+{latency_overhead:.1f}%)")
    else:
        print(f"  EXCELLENT: Low latency overhead (+{latency_overhead:.1f}%)")


def main():
    """Run recall-based comparison experiment."""
    print("="*60)
    print("Recall-Based Comparison: Dynamic HNSW vs Static HNSW")
    print("="*60)

    # Configuration
    n_vectors = 1000
    n_queries = 100
    dim = 128
    k = 10
    static_ef = 100

    # Step 1: Generate data
    print("\n[Step 1] Generating synthetic dataset...")
    database = generate_synthetic_dataset(n_vectors=n_vectors, dim=dim, seed=42)
    queries = generate_test_queries(n_queries=n_queries, dim=dim, seed=43)

    # Step 2: Pre-compute ground truth
    print("\n[Step 2] Pre-computing ground truth...")
    ground_truth = precompute_ground_truth(queries, database, k=k)

    # Step 3: Build HNSW index
    print("\n[Step 3] Building HNSW index...")
    graph = build_hnsw_index(database, M=16, ef_construction=200)

    # Step 4: Run static HNSW baseline
    print("\n[Step 4] Running static HNSW baseline...")
    static_results = run_static_hnsw_baseline(
        graph=graph,
        queries=queries,
        ground_truth=ground_truth,
        k=k,
        ef_search=static_ef
    )

    # Step 5: Run dynamic HNSW
    print("\n[Step 5] Running dynamic HNSW...")
    dynamic_results = run_dynamic_hnsw(
        graph=graph,
        queries=queries,
        ground_truth=ground_truth,
        k=k,
        default_ef=static_ef,
        k_intents=5,
        use_ucb1=True
    )

    # Step 6: Compare results
    compare_results(static_results, dynamic_results, k=k)

    print(f"\n{'='*60}")
    print("Experiment complete!")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
