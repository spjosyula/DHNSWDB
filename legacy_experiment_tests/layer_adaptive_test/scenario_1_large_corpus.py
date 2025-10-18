"""Scenario 1: Large Corpus Test (10K docs, 1K queries)

Compare Static HNSW vs Layer-Adaptive Multi-Path Search on large-scale data.

Objective:
    - Establish baseline performance of Static HNSW
    - Measure Layer-Adaptive recall improvement
    - Analyze latency overhead
    - Validate difficulty-based path selection

Configuration:
    - Corpus: 10,000 documents (diverse domains)
    - Queries: 1,000 queries (mixed types)
    - Embeddings: all-MiniLM-L6-v2 (384 dim)
    - ef_search: 100 (fixed for both)
    - Ground truth: Pre-computed exact k-NN

Expected:
    - Layer-Adaptive recall > Static HNSW recall
    - Latency overhead manageable (~2x)
    - Most queries use 3 paths (hard queries dominant)
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time
import numpy as np
from typing import List
from sentence_transformers import SentenceTransformer

from dynhnsw.hnsw.graph import HNSWGraph
from dynhnsw.hnsw.builder import HNSWBuilder
from dynhnsw.hnsw.utils import assign_layer
from dynhnsw.intent_aware_hnsw import IntentAwareHNSWSearcher

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from shared_utils import (
    RecallExperimentTracker,
    print_results_summary,
    generate_large_corpus,
    create_diverse_queries,
    compute_ground_truth_brute_force,
    compute_recall_at_k,
)


def build_hnsw_graph(vectors: np.ndarray, M: int = 16) -> HNSWGraph:
    """Build HNSW graph from vectors."""
    print(f"\n[Graph Construction] Building HNSW with {len(vectors)} vectors...")
    graph = HNSWGraph(dimension=vectors.shape[1], M=M)
    builder = HNSWBuilder(graph=graph)

    for i, vector in enumerate(vectors):
        level = assign_layer(M=graph.M)
        builder.insert(vector=vector, node_id=i, level=level)
        if (i + 1) % 1000 == 0:
            print(f"  {i+1}/{len(vectors)} indexed")

    print(f"[Complete] Max level: {graph.get_max_level()}")
    return graph


def run_static_hnsw(graph, queries, ground_truth, k=10, ef=100):
    """Run Static HNSW (no adaptation)."""
    print(f"\n{'='*80}")
    print(f"STATIC HNSW (ef={ef})")
    print(f"{'='*80}")

    searcher = IntentAwareHNSWSearcher(
        graph=graph, ef_search=ef,
        enable_adaptation=False,
        enable_intent_detection=False,
    )

    tracker = RecallExperimentTracker(f"Static_ef{ef}", compare_baseline=False)

    print(f"\nRunning {len(queries)} queries...")
    for i, query in enumerate(queries):
        start = time.perf_counter()
        results = searcher.search(query, k=k)
        latency_ms = (time.perf_counter() - start) * 1000

        result_ids = [nid for nid, _ in results]
        recall = compute_recall_at_k(result_ids, ground_truth[i], k)

        tracker.record_query(
            recall=recall, latency_ms=latency_ms, ef_used=ef,
            intent_id=-1, query_type="static", difficulty=0.0, difficulty_time_ms=0.0
        )

        if (i + 1) % 100 == 0:
            print(f"  {i+1}/{len(queries)} | Recall: {np.mean(tracker.recalls[-100:]):.1%} | "
                  f"Latency: {np.mean(tracker.latencies[-100:]):.2f}ms")

    print_results_summary(tracker)
    return tracker


def run_layer_adaptive(graph, queries, ground_truth, k=10, ef=100):
    """Run Layer-Adaptive Multi-Path Search."""
    print(f"\n{'='*80}")
    print(f"LAYER-ADAPTIVE MULTI-PATH (ef={ef})")
    print(f"{'='*80}")

    # Enable difficulty computation but not full adaptation (no UCB1/K-means)
    searcher = IntentAwareHNSWSearcher(
        graph=graph, ef_search=ef,
        enable_adaptation=False,
        enable_intent_detection=True,  # Only for difficulty computation
    )

    tracker = RecallExperimentTracker("LayerAdaptive", compare_baseline=False)
    path_counts = {1: 0, 2: 0, 3: 0}

    print(f"\nRunning {len(queries)} queries...")
    for i, query in enumerate(queries):
        # Compute difficulty
        entry_node = graph.get_node(graph.entry_point)
        from dynhnsw.hnsw.distance import cosine_distance
        difficulty = cosine_distance(query, entry_node.vector)

        # Determine num_paths
        if difficulty < 0.8:
            num_paths = 1
        elif difficulty < 0.9:
            num_paths = 2
        else:
            num_paths = 3
        path_counts[num_paths] += 1

        # Search (layer-adaptive happens inside)
        start = time.perf_counter()
        results = searcher.search(query, k=k)
        latency_ms = (time.perf_counter() - start) * 1000

        result_ids = [nid for nid, _ in results]
        recall = compute_recall_at_k(result_ids, ground_truth[i], k)

        tracker.record_query(
            recall=recall, latency_ms=latency_ms, ef_used=ef,
            intent_id=num_paths, query_type="adaptive",
            difficulty=difficulty, difficulty_time_ms=0.01
        )

        if (i + 1) % 100 == 0:
            print(f"  {i+1}/{len(queries)} | Recall: {np.mean(tracker.recalls[-100:]):.1%} | "
                  f"Latency: {np.mean(tracker.latencies[-100:]):.2f}ms")

    print(f"\n[Path Distribution]")
    print(f"  1-path: {path_counts[1]} ({100*path_counts[1]/len(queries):.1f}%)")
    print(f"  2-path: {path_counts[2]} ({100*path_counts[2]/len(queries):.1f}%)")
    print(f"  3-path: {path_counts[3]} ({100*path_counts[3]/len(queries):.1f}%)")

    print_results_summary(tracker)
    return tracker, path_counts


def main():
    print("="*80)
    print("SCENARIO 1: Large Corpus (10K docs, 1K queries)")
    print("="*80)

    # Config
    CORPUS_SIZE, NUM_QUERIES, K, EF = 10000, 1000, 10, 100

    # Step 1: Generate data
    print(f"\n[1/6] Generating corpus and queries")
    corpus = generate_large_corpus(size=CORPUS_SIZE, seed=42)
    queries, query_types = create_diverse_queries(
        exploratory_count=300, precise_count=500, mixed_count=200, seed=43
    )
    queries, query_types = queries[:NUM_QUERIES], query_types[:NUM_QUERIES]

    # Step 2: Embeddings
    print(f"\n[2/6] Generating embeddings")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    corpus_emb = model.encode(corpus, convert_to_numpy=True, show_progress_bar=True).astype(np.float32)
    query_emb = model.encode(queries, convert_to_numpy=True, show_progress_bar=True).astype(np.float32)

    # Step 3: Ground truth
    print(f"\n[3/6] Computing ground truth")
    ground_truth = compute_ground_truth_brute_force(query_emb, corpus_emb, k=K)

    # Step 4: Build graph
    print(f"\n[4/6] Building HNSW graph")
    graph = build_hnsw_graph(corpus_emb, M=16)

    # Step 5: Static HNSW
    print(f"\n[5/6] Testing Static HNSW")
    static_tracker = run_static_hnsw(graph, query_emb, ground_truth, k=K, ef=EF)

    # Step 6: Layer-Adaptive
    print(f"\n[6/6] Testing Layer-Adaptive")
    adaptive_tracker, path_dist = run_layer_adaptive(graph, query_emb, ground_truth, k=K, ef=EF)

    # Save results
    static_tracker.save_results("layer_adaptive_test/results/s1_static.json")
    adaptive_tracker.save_results("layer_adaptive_test/results/s1_adaptive.json")

    # Final comparison
    print(f"\n{'='*80}")
    print("RESULTS: Static vs Layer-Adaptive")
    print(f"{'='*80}")

    sm = static_tracker.get_metrics()
    am = adaptive_tracker.get_metrics()

    recall_diff = am['avg_recall'] - sm['avg_recall']
    recall_pct = 100 * recall_diff / sm['avg_recall']
    latency_pct = 100 * (am['avg_latency_ms'] / sm['avg_latency_ms'] - 1)

    print(f"\nMetric                  | Static      | Adaptive    | Delta")
    print(f"{'-'*60}")
    print(f"Recall@{K:<2}              | {sm['avg_recall']:>10.1%} | {am['avg_recall']:>10.1%} | +{recall_pct:>5.1f}%")
    print(f"Latency (ms)            | {sm['avg_latency_ms']:>10.2f} | {am['avg_latency_ms']:>10.2f} | +{latency_pct:>5.1f}%")
    print(f"P95 Latency (ms)        | {sm['p95_latency_ms']:>10.2f} | {am['p95_latency_ms']:>10.2f} |")

    print(f"\n[Analysis]")
    print(f"  Recall improvement: {recall_diff:+.1%} ({recall_pct:+.1f}%)")
    print(f"  Latency overhead: +{latency_pct:.1f}%")
    print(f"  Hard queries (3-path): {path_dist[3]} ({100*path_dist[3]/NUM_QUERIES:.1f}%)")

    print(f"\n{'='*80}")
    print("SCENARIO 1 COMPLETE")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
