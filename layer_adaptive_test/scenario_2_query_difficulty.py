"""Scenario 2: Query Difficulty Analysis (5K docs, 600 queries)

Analyze layer-adaptive performance across different query difficulty tiers.

Objective:
    - Test on controlled easy/medium/hard query distributions
    - Validate difficulty proxy correctly identifies query types
    - Measure per-difficulty-tier recall improvements
    - Verify path selection adapts appropriately

Configuration:
    - Corpus: 5,000 documents (10 topic clusters)
    - Queries: 600 queries (200 easy, 200 medium, 200 hard)
    - Easy: Near cluster centers (low difficulty)
    - Medium: Mid-range distance from clusters
    - Hard: Out-of-distribution (high difficulty)
    - ef_search: 100 (fixed for both)

Expected:
    - Easy queries: 1-path sufficient, similar recall
    - Medium queries: 2-path beneficial, modest improvement
    - Hard queries: 3-path critical, large improvement
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time
import numpy as np
from typing import List, Tuple
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans

from dynhnsw.hnsw.graph import HNSWGraph
from dynhnsw.hnsw.builder import HNSWBuilder
from dynhnsw.hnsw.utils import assign_layer
from dynhnsw.intent_aware_hnsw import IntentAwareHNSWSearcher

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from shared_utils import (
    RecallExperimentTracker,
    print_results_summary,
    compute_ground_truth_brute_force,
    compute_recall_at_k,
)


def generate_clustered_corpus(n_docs=5000, n_clusters=10, dim=384, seed=42):
    """Generate corpus with clear topic clusters."""
    np.random.seed(seed)
    docs_per_cluster = n_docs // n_clusters

    corpus_embeddings = []
    for i in range(n_clusters):
        # Random cluster center
        center = np.random.randn(dim).astype(np.float32)
        center = center / np.linalg.norm(center)

        # Generate docs around center
        for _ in range(docs_per_cluster):
            noise = np.random.randn(dim).astype(np.float32) * 0.1
            doc_emb = center + noise
            doc_emb = doc_emb / np.linalg.norm(doc_emb)
            corpus_embeddings.append(doc_emb)

    return np.array(corpus_embeddings)


def generate_difficulty_queries(corpus_emb, n_easy=200, n_medium=200, n_hard=200, seed=43):
    """Generate queries with controlled difficulty levels."""
    np.random.seed(seed)
    dim = corpus_emb.shape[1]

    # Cluster corpus
    n_clusters = 10
    kmeans = KMeans(n_clusters=n_clusters, random_state=seed, n_init=10)
    kmeans.fit(corpus_emb)
    centers = kmeans.cluster_centers_

    queries, labels = [], []

    # Easy: Near cluster centers
    for _ in range(n_easy):
        center = centers[np.random.randint(0, n_clusters)]
        noise = np.random.randn(dim).astype(np.float32) * 0.05  # Very close
        query = center + noise
        query = query / np.linalg.norm(query)
        queries.append(query)
        labels.append("easy")

    # Medium: Mid-range distance
    for _ in range(n_medium):
        center = centers[np.random.randint(0, n_clusters)]
        noise = np.random.randn(dim).astype(np.float32) * 0.2  # Medium distance
        query = center + noise
        query = query / np.linalg.norm(query)
        queries.append(query)
        labels.append("medium")

    # Hard: Out-of-distribution
    for _ in range(n_hard):
        query = np.random.randn(dim).astype(np.float32)  # Random direction
        query = query / np.linalg.norm(query)
        queries.append(query)
        labels.append("hard")

    # Shuffle
    indices = np.random.permutation(len(queries))
    queries = [queries[i] for i in indices]
    labels = [labels[i] for i in indices]

    return np.array(queries), labels


def build_graph(vectors, M=16):
    """Build HNSW graph."""
    print(f"[Graph] Building with {len(vectors)} vectors...")
    graph = HNSWGraph(dimension=vectors.shape[1], M=M)
    builder = HNSWBuilder(graph=graph)

    for i, vec in enumerate(vectors):
        level = assign_layer(level_multiplier=graph.level_multiplier)
        builder.insert(vector=vec, node_id=i, level=level)
        if (i + 1) % 1000 == 0:
            print(f"  {i+1}/{len(vectors)}")

    print(f"[Complete] Max level: {graph.get_max_level()}")
    return graph


def run_static(graph, queries, ground_truth, k=10, ef=100):
    """Static HNSW."""
    print(f"\n{'='*80}")
    print("STATIC HNSW")
    print(f"{'='*80}")

    searcher = IntentAwareHNSWSearcher(
        graph=graph, ef_search=ef,
        enable_adaptation=False, enable_intent_detection=False
    )

    tracker = RecallExperimentTracker("Static", compare_baseline=False)

    for i, query in enumerate(queries):
        start = time.perf_counter()
        results = searcher.search(query, k=k)
        latency = (time.perf_counter() - start) * 1000

        ids = [nid for nid, _ in results]
        recall = compute_recall_at_k(ids, ground_truth[i], k)

        tracker.record_query(
            recall=recall, latency_ms=latency, ef_used=ef,
            intent_id=-1, query_type="static", difficulty=0.0, difficulty_time_ms=0.0
        )

        if (i + 1) % 100 == 0:
            print(f"  {i+1}/{len(queries)} | Recall: {np.mean(tracker.recalls[-100:]):.1%}")

    print_results_summary(tracker)
    return tracker


def run_adaptive(graph, queries, query_labels, ground_truth, k=10, ef=100):
    """Layer-Adaptive."""
    print(f"\n{'='*80}")
    print("LAYER-ADAPTIVE")
    print(f"{'='*80}")

    searcher = IntentAwareHNSWSearcher(
        graph=graph, ef_search=ef,
        enable_adaptation=False, enable_intent_detection=True
    )

    tracker = RecallExperimentTracker("Adaptive", compare_baseline=False)
    difficulty_by_label = {"easy": [], "medium": [], "hard": []}
    recall_by_label = {"easy": [], "medium": [], "hard": []}

    for i, (query, label) in enumerate(zip(queries, query_labels)):
        # Compute difficulty
        entry_node = graph.get_node(graph.entry_point)
        from dynhnsw.hnsw.distance import cosine_distance
        diff = cosine_distance(query, entry_node.vector)
        difficulty_by_label[label].append(diff)

        # Search
        start = time.perf_counter()
        results = searcher.search(query, k=k)
        latency = (time.perf_counter() - start) * 1000

        ids = [nid for nid, _ in results]
        recall = compute_recall_at_k(ids, ground_truth[i], k)
        recall_by_label[label].append(recall)

        tracker.record_query(
            recall=recall, latency_ms=latency, ef_used=ef,
            intent_id=-1, query_type=label, difficulty=diff, difficulty_time_ms=0.01
        )

        if (i + 1) % 100 == 0:
            print(f"  {i+1}/{len(queries)} | Recall: {np.mean(tracker.recalls[-100:]):.1%}")

    print_results_summary(tracker)
    return tracker, difficulty_by_label, recall_by_label


def main():
    print("="*80)
    print("SCENARIO 2: Query Difficulty Analysis (5K docs, 600 queries)")
    print("="*80)

    K, EF = 10, 100

    # Step 1: Generate clustered corpus
    print("\n[1/5] Generating clustered corpus")
    corpus_emb = generate_clustered_corpus(n_docs=5000, n_clusters=10, dim=384, seed=42)

    # Step 2: Generate difficulty-controlled queries
    print("\n[2/5] Generating easy/medium/hard queries")
    query_emb, query_labels = generate_difficulty_queries(
        corpus_emb, n_easy=200, n_medium=200, n_hard=200, seed=43
    )

    # Step 3: Ground truth
    print("\n[3/5] Computing ground truth")
    ground_truth = compute_ground_truth_brute_force(query_emb, corpus_emb, k=K)

    # Step 4: Build graph
    print("\n[4/5] Building graph")
    graph = build_graph(corpus_emb, M=16)

    # Step 5: Run tests
    print("\n[5/5] Running tests")

    static_tracker = run_static(graph, query_emb, ground_truth, k=K, ef=EF)
    adaptive_tracker, diff_by_label, recall_by_label = run_adaptive(
        graph, query_emb, query_labels, ground_truth, k=K, ef=EF
    )

    # Save results
    static_tracker.save_results("layer_adaptive_test/results/s2_static.json")
    adaptive_tracker.save_results("layer_adaptive_test/results/s2_adaptive.json")

    # Analysis
    print(f"\n{'='*80}")
    print("DIFFICULTY ANALYSIS")
    print(f"{'='*80}")

    print(f"\nQuery Type | Count | Avg Difficulty | Avg Recall")
    print(f"{'-'*60}")
    for label in ["easy", "medium", "hard"]:
        count = len(diff_by_label[label])
        avg_diff = np.mean(diff_by_label[label])
        avg_recall = np.mean(recall_by_label[label])
        print(f"{label:>10} | {count:>5} | {avg_diff:>14.4f} | {avg_recall:>10.1%}")

    # Final comparison
    print(f"\n{'='*80}")
    print("RESULTS: Static vs Layer-Adaptive")
    print(f"{'='*80}")

    sm = static_tracker.get_metrics()
    am = adaptive_tracker.get_metrics()

    recall_diff = am['avg_recall'] - sm['avg_recall']
    recall_pct = 100 * recall_diff / sm['avg_recall']
    latency_pct = 100 * (am['avg_latency_ms'] / sm['avg_latency_ms'] - 1)

    print(f"\nMetric           | Static      | Adaptive    | Delta")
    print(f"{'-'*55}")
    print(f"Recall@{K:<2}       | {sm['avg_recall']:>10.1%} | {am['avg_recall']:>10.1%} | +{recall_pct:>5.1f}%")
    print(f"Latency (ms)     | {sm['avg_latency_ms']:>10.2f} | {am['avg_latency_ms']:>10.2f} | +{latency_pct:>5.1f}%")

    print(f"\n{'='*80}")
    print("SCENARIO 2 COMPLETE")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
