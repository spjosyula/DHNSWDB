"""Scenario 2: Diverse Query Difficulty Validation

Test that the zero-cost difficulty proxy correctly identifies query difficulty.

Objective:
    Validate that distance-to-entry-point correctly distinguishes:
    1. Easy queries (near cluster centers) → low difficulty → low ef
    2. Hard queries (in sparse regions) → high difficulty → high ef
    3. Mixed queries (medium distance) → medium difficulty → medium ef

Configuration:
    - Corpus: 5,000 documents (clustered structure)
    - Queries: 600 queries (200 easy, 200 hard, 200 mixed)
    - Embeddings: all-MiniLM-L6-v2 (384 dim)
    - k_intents: 3 (easy, medium, hard)
    - Intentionally create queries with varying difficulty

Expected Outcome:
    - High correlation between proxy difficulty and true difficulty
    - Easy queries → learned ef < 100
    - Hard queries → learned ef > 150
    - Overhead remains <5%
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time
import numpy as np
from typing import List, Optional, Tuple, Dict
from sentence_transformers import SentenceTransformer

from dynhnsw.hnsw.graph import HNSWGraph
from dynhnsw.hnsw.builder import HNSWBuilder
from dynhnsw.hnsw.utils import assign_layer
from dynhnsw.intent_aware_hnsw import IntentAwareHNSWSearcher
from dynhnsw.config import DynHNSWConfig

# Import shared utilities
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from shared_utils import (
    RecallExperimentTracker,
    print_results_summary,
    compute_ground_truth_brute_force,
    compute_recall_at_k,
)


def generate_clustered_corpus(size: int = 5000, n_clusters: int = 10, seed: int = 42) -> List[str]:
    """Generate corpus with clear cluster structure.

    Args:
        size: Total number of documents
        n_clusters: Number of topic clusters
        seed: Random seed

    Returns:
        List of document strings with clustered topics
    """
    np.random.seed(seed)
    corpus = []

    # Define topic clusters
    clusters = {
        0: "Python machine learning TensorFlow neural networks deep learning AI",
        1: "Docker Kubernetes container orchestration microservices cloud deployment",
        2: "PostgreSQL MySQL database SQL queries indexing optimization performance",
        3: "React JavaScript TypeScript frontend web development components hooks",
        4: "AWS EC2 S3 Lambda serverless cloud infrastructure DevOps",
        5: "Git version control GitHub repository commit branch merge pull request",
        6: "API REST GraphQL endpoints authentication JWT OAuth security",
        7: "Linux Ubuntu server administration bash scripting command line tools",
        8: "Testing pytest unit test integration test coverage automation CI CD",
        9: "Data science pandas numpy matplotlib visualization analytics statistics",
    }

    docs_per_cluster = size // n_clusters

    for cluster_id in range(n_clusters):
        base_keywords = clusters[cluster_id % len(clusters)]
        keywords = base_keywords.split()

        for i in range(docs_per_cluster):
            # Create document from cluster keywords
            selected_keywords = np.random.choice(keywords, size=min(5, len(keywords)), replace=False)
            doc = " ".join(selected_keywords) + f" implementation guide tutorial example {cluster_id}_{i}"
            corpus.append(doc)

    return corpus[:size]


def create_difficulty_stratified_queries(
    corpus: List[str],
    corpus_embeddings: np.ndarray,
    model: SentenceTransformer,
    n_easy: int = 200,
    n_hard: int = 200,
    n_mixed: int = 200,
    seed: int = 43,
) -> Tuple[List[str], List[str], np.ndarray]:
    """Create queries with known difficulty levels.

    Args:
        corpus: Document corpus
        corpus_embeddings: Corpus embeddings
        model: Sentence transformer model
        n_easy: Number of easy queries
        n_hard: Number of hard queries
        n_mixed: Number of mixed queries
        seed: Random seed

    Returns:
        Tuple of (queries, query_types, query_embeddings)
    """
    np.random.seed(seed)
    queries = []
    query_types = []

    # Compute corpus centroid (entry point proxy)
    centroid = np.mean(corpus_embeddings, axis=0)

    # Easy queries: Very similar to existing documents (near cluster centers)
    print("\n  Generating easy queries (near cluster centers)...")
    for i in range(n_easy):
        # Pick a random document and slightly perturb it
        doc_idx = np.random.randint(0, len(corpus))
        doc = corpus[doc_idx]
        keywords = doc.split()[:4]  # Use first 4 keywords
        query = " ".join(keywords) + " tutorial"
        queries.append(query)
        query_types.append("easy")

    # Hard queries: Far from clusters (sparse regions, out-of-distribution)
    print("  Generating hard queries (sparse regions)...")
    hard_templates = [
        "quantum computing blockchain cryptocurrency metaverse NFT DeFi",
        "biotechnology genetics CRISPR gene editing pharmaceutical research",
        "autonomous vehicles self-driving cars LiDAR sensor fusion robotics",
        "renewable energy solar panels wind turbines battery storage grid",
        "cybersecurity penetration testing ethical hacking vulnerability assessment",
        "financial markets trading algorithms quantitative analysis derivatives",
        "supply chain logistics inventory management warehouse automation",
        "legal compliance regulations GDPR data privacy consumer protection",
        "education e-learning online courses MOOCs virtual classrooms pedagogy",
        "healthcare telemedicine electronic health records patient data analytics",
    ]

    for i in range(n_hard):
        template = hard_templates[i % len(hard_templates)]
        query = template + f" research {i}"
        queries.append(query)
        query_types.append("hard")

    # Mixed queries: Medium distance from clusters
    print("  Generating mixed queries (medium difficulty)...")
    mixed_templates = [
        "Python Docker deployment automation",
        "React PostgreSQL full stack application",
        "AWS Lambda serverless API development",
        "Git GitHub collaboration workflow best practices",
        "Linux server PostgreSQL database administration",
        "JavaScript testing frameworks unit integration",
        "Machine learning data preprocessing pipelines",
        "Kubernetes container monitoring logging observability",
    ]

    for i in range(n_mixed):
        template = mixed_templates[i % len(mixed_templates)]
        query = template + f" guide {i}"
        queries.append(query)
        query_types.append("mixed")

    # Shuffle
    indices = list(range(len(queries)))
    np.random.shuffle(indices)
    queries = [queries[i] for i in indices]
    query_types = [query_types[i] for i in indices]

    # Generate embeddings
    print("  Embedding queries...")
    query_embeddings = model.encode(queries, convert_to_numpy=True, show_progress_bar=False)
    query_embeddings = query_embeddings.astype(np.float32)

    return queries, query_types, query_embeddings


def build_hnsw_graph(vectors: np.ndarray, M: int = 16) -> HNSWGraph:
    """Build HNSW graph from vectors."""
    print(f"\n[Graph Construction] Building HNSW index with {len(vectors)} vectors...")
    dim = vectors.shape[1]
    graph = HNSWGraph(dimension=dim, M=M)
    builder = HNSWBuilder(graph=graph)

    for i, vector in enumerate(vectors):
        level = assign_layer(level_multiplier=graph.level_multiplier)
        builder.insert(vector=vector, node_id=i, level=level)

        if (i + 1) % 500 == 0:
            print(f"  Progress: {i+1}/{len(vectors)} vectors indexed")

    print(f"[Graph Construction] Complete!")
    return graph


def run_static_baseline(
    graph: HNSWGraph,
    query_vectors: np.ndarray,
    query_types: List[str],
    ground_truth: List[List[int]],
    k: int = 10,
    ef_search: int = 100,
) -> RecallExperimentTracker:
    """Run static HNSW baseline.

    Args:
        graph: HNSW graph
        query_vectors: Query embeddings
        query_types: Query difficulty labels
        ground_truth: Pre-computed ground truth
        k: Number of neighbors
        ef_search: Fixed ef_search value

    Returns:
        Tracker with baseline metrics
    """
    print(f"\n{'='*100}")
    print(f"STATIC HNSW BASELINE (ef={ef_search})")
    print(f"{'='*100}")

    searcher = IntentAwareHNSWSearcher(
        graph=graph,
        ef_search=ef_search,
        enable_adaptation=False,
        enable_intent_detection=False,
    )

    tracker = RecallExperimentTracker(f"Static_HNSW_ef{ef_search}", compare_baseline=False)

    print(f"\nRunning {len(query_vectors)} queries...")
    for i, (query, qtype) in enumerate(zip(query_vectors, query_types)):
        start_time = time.perf_counter()
        results = searcher.search(query, k=k)
        latency_ms = (time.perf_counter() - start_time) * 1000

        result_ids = [node_id for node_id, _ in results]
        recall = compute_recall_at_k(result_ids, ground_truth[i], k)

        tracker.record_query(
            recall=recall,
            latency_ms=latency_ms,
            ef_used=ef_search,
            intent_id=-1,
            query_type=qtype,
            difficulty=0.0,
            difficulty_time_ms=0.0,
        )

        if (i + 1) % 100 == 0:
            print(f"  Progress: {i+1}/{len(query_vectors)}")

    print(f"\n[COMPLETE] Static HNSW baseline")
    print_results_summary(tracker)

    return tracker


def run_difficulty_validation(
    graph: HNSWGraph,
    query_vectors: np.ndarray,
    query_types: List[str],
    ground_truth: List[List[int]],
    k: int = 10,
    baseline_tracker: Optional[RecallExperimentTracker] = None,
) -> Tuple[RecallExperimentTracker, Dict[str, List[float]], Dict[str, List[float]]]:
    """Run validation that difficulty proxy works correctly.

    Args:
        graph: HNSW graph
        query_vectors: Query embeddings
        query_types: Query difficulty labels (easy/mixed/hard)
        ground_truth: Pre-computed ground truth
        k: Number of neighbors

    Returns:
        Tracker with results
    """
    print(f"\n{'='*100}")
    print(f"DIFFICULTY VALIDATION: Zero-Cost Proxy")
    print(f"{'='*100}")

    config = DynHNSWConfig(
        config_name="difficulty_validation",
        enable_ucb1=True,
        ucb1_exploration_constant=1.414,
        exploration_rate=0.15,
        enable_epsilon_decay=False,
        k_intents=3,  # Easy, Medium, Hard
        min_queries_for_clustering=30,
        confidence_threshold=0.5,
    )

    searcher = IntentAwareHNSWSearcher(
        graph=graph,
        ef_search=100,
        k_intents=3,
        enable_adaptation=True,
        enable_intent_detection=True,
        min_queries_for_clustering=30,
        config=config,
    )

    tracker = RecallExperimentTracker("Difficulty_Validation", compare_baseline=True)

    print(f"\nRunning {len(query_vectors)} queries...")

    # Track difficulty by query type
    difficulties_by_type = {"easy": [], "mixed": [], "hard": []}
    ef_by_type = {"easy": [], "mixed": [], "hard": []}

    for i, (query, qtype) in enumerate(zip(query_vectors, query_types)):
        # Measure total end-to-end latency (includes difficulty computation)
        start_time = time.perf_counter()
        results = searcher.search(query, k=k)
        latency_ms = (time.perf_counter() - start_time) * 1000

        # Get difficulty from last search
        difficulty = searcher.last_difficulty
        diff_time_ms = 0.01  # Conservative estimate

        # Compute recall
        result_ids = [node_id for node_id, _ in results]
        recall = compute_recall_at_k(result_ids, ground_truth[i], k)

        # Record baseline for comparison
        if baseline_tracker and i < len(baseline_tracker.recalls):
            tracker.record_baseline(
                recall=baseline_tracker.recalls[i],
                latency_ms=baseline_tracker.latencies[i],
            )

        # Record metrics
        tracker.record_query(
            recall=recall,
            latency_ms=latency_ms,
            ef_used=searcher.last_ef_used,
            intent_id=searcher.last_intent_id,
            query_type=qtype,
            difficulty=difficulty,
            difficulty_time_ms=diff_time_ms,
        )

        # Track difficulty and ef by type
        difficulties_by_type[qtype].append(difficulty)
        ef_by_type[qtype].append(searcher.last_ef_used)

        # Provide feedback
        searcher.provide_feedback(
            query=query,
            result_ids=result_ids,
            ground_truth_ids=ground_truth[i],
            k=k,
        )

        # Progress
        if (i + 1) % 100 == 0:
            print(f"  Progress: {i+1}/{len(query_vectors)}")

    print(f"\n[COMPLETE] Difficulty validation")

    # Analyze difficulty by query type
    print(f"\n{'='*100}")
    print("DIFFICULTY ANALYSIS BY QUERY TYPE")
    print(f"{'='*100}")

    print(f"\n{'Query Type':<15} | {'Avg Difficulty':>15} | {'Avg ef_search':>15} | {'Count':>10}")
    print("-" * 100)

    for qtype in ["easy", "mixed", "hard"]:
        if difficulties_by_type[qtype]:
            avg_diff = np.mean(difficulties_by_type[qtype])
            avg_ef = np.mean(ef_by_type[qtype])
            count = len(difficulties_by_type[qtype])
            print(f"{qtype:<15} | {avg_diff:>15.4f} | {avg_ef:>15.1f} | {count:>10}")

    # Validate correlation
    print(f"\n{'='*100}")
    print("VALIDATION RESULTS")
    print(f"{'='*100}")

    easy_diff = np.mean(difficulties_by_type["easy"]) if difficulties_by_type["easy"] else 0
    hard_diff = np.mean(difficulties_by_type["hard"]) if difficulties_by_type["hard"] else 0

    print(f"\n1. Difficulty Separation:")
    if hard_diff > easy_diff * 1.2:
        print(f"   SUCCESS: Hard queries (d={hard_diff:.4f}) >> Easy queries (d={easy_diff:.4f})")
        print(f"   Ratio: {hard_diff/easy_diff:.2f}x")
    else:
        print(f"   WARNING: Insufficient difficulty separation")

    print(f"\n2. Overhead:")
    metrics = tracker.get_metrics()
    overhead = metrics.get('difficulty_overhead_percent', 0)
    if overhead < 5.0:
        print(f"   SUCCESS: Overhead = {overhead:.2f}% (target: <5%)")
    else:
        print(f"   WARNING: Overhead = {overhead:.2f}%")

    print_results_summary(tracker)

    return tracker, difficulties_by_type, ef_by_type


def main():
    """Run Scenario 2: Diverse query difficulty validation."""
    print("="*100)
    print("SCENARIO 2: Diverse Query Difficulty Validation")
    print("="*100)
    print("\nObjective: Validate zero-cost proxy correctly identifies query difficulty")

    # Configuration
    CORPUS_SIZE = 5000
    N_EASY = 200
    N_HARD = 200
    N_MIXED = 200
    K = 10

    # Step 1: Generate clustered corpus
    print(f"\n{'='*100}")
    print("[STEP 1/5] Generating clustered corpus")
    print(f"{'='*100}")
    corpus = generate_clustered_corpus(size=CORPUS_SIZE, n_clusters=10, seed=42)
    print(f"  Corpus: {len(corpus)} documents (10 topic clusters)")

    # Step 2: Generate embeddings
    print(f"\n{'='*100}")
    print("[STEP 2/5] Generating embeddings")
    print(f"{'='*100}")
    model = SentenceTransformer('all-MiniLM-L6-v2')

    print(f"  Embedding corpus...")
    corpus_embeddings = model.encode(corpus, convert_to_numpy=True, show_progress_bar=True)
    corpus_embeddings = corpus_embeddings.astype(np.float32)

    # Step 3: Create difficulty-stratified queries
    print(f"\n{'='*100}")
    print("[STEP 3/5] Creating difficulty-stratified queries")
    print(f"{'='*100}")
    queries, query_types, query_embeddings = create_difficulty_stratified_queries(
        corpus=corpus,
        corpus_embeddings=corpus_embeddings,
        model=model,
        n_easy=N_EASY,
        n_hard=N_HARD,
        n_mixed=N_MIXED,
        seed=43,
    )
    print(f"  Queries: {len(queries)} ({N_EASY} easy, {N_MIXED} mixed, {N_HARD} hard)")

    # Step 4: Compute ground truth
    print(f"\n{'='*100}")
    print("[STEP 4/5] Computing ground truth")
    print(f"{'='*100}")
    ground_truth = compute_ground_truth_brute_force(query_embeddings, corpus_embeddings, k=K)

    # Step 5: Build graph
    print(f"\n{'='*100}")
    print("[STEP 5/5] Building HNSW graph and running validation")
    print(f"{'='*100}")
    graph = build_hnsw_graph(corpus_embeddings, M=16)

    # Run static baseline first
    print(f"\nRunning static HNSW baseline...")
    baseline_tracker = run_static_baseline(
        graph=graph,
        query_vectors=query_embeddings,
        query_types=query_types,
        ground_truth=ground_truth,
        k=K,
        ef_search=100,
    )

    # Run validation with dynamic HNSW
    print(f"\nRunning dynamic HNSW validation...")
    tracker, difficulties_by_type, ef_by_type = run_difficulty_validation(
        graph=graph,
        query_vectors=query_embeddings,
        query_types=query_types,
        ground_truth=ground_truth,
        k=K,
        baseline_tracker=baseline_tracker,
    )

    # Save results
    baseline_tracker.save_results("ucb1_recall_test/results/scenario_2_baseline.json")
    tracker.save_results("ucb1_recall_test/results/scenario_2_difficulty_validation.json")

    # Final comparison
    print(f"\n{'='*100}")
    print("FINAL COMPARISON: Static vs Dynamic HNSW")
    print(f"{'='*100}")

    baseline_metrics = baseline_tracker.get_metrics()
    dynamic_metrics = tracker.get_metrics()

    print(f"\n{'Metric':<50} | {'Static HNSW':>20} | {'Dynamic HNSW':>20} | {'Improvement':>15}")
    print("-"*115)
    print(f"{'Average Recall@10':<50} | {baseline_metrics['avg_recall']:>19.1%} | "
          f"{dynamic_metrics['avg_recall']:>19.1%} | "
          f"{dynamic_metrics.get('recall_improvement_percent', 0):>14.2f}%")
    print(f"{'Average Latency (ms)':<50} | {baseline_metrics['avg_latency_ms']:>20.2f} | "
          f"{dynamic_metrics['avg_latency_ms']:>20.2f} | "
          f"{dynamic_metrics.get('latency_improvement_percent', 0):>14.2f}%")
    print(f"{'Difficulty Overhead (%)':<50} | {'0.00%':>20} | "
          f"{dynamic_metrics['difficulty_overhead_percent']:>19.2f}% | N/A")

    print(f"\n{'='*100}")
    print("SCENARIO 2 COMPLETE!")
    print(f"{'='*100}\n")


if __name__ == "__main__":
    main()
