"""Scenario 1: Large Document Corpus Test

Test adaptive ef_search learning with large-scale realistic corpus.

Hypothesis:
    Large corpus (5000+ documents) with diverse query patterns should benefit from
    adaptive ef_search selection:
    1. Precise queries can use lower ef_search for speed
    2. Exploratory queries need higher ef_search for recall
    3. Adaptive learning identifies optimal ef per query type
    4. Should show 5-10% efficiency improvement over static ef_search

Configuration:
    - Corpus size: 5000 documents
    - Embedding: sentence-transformers all-MiniLM-L6-v2 (384 dims)
    - Queries: 1000 mixed (exploratory, precise, mixed intents)
    - k_intents: 5 (capture diverse query patterns)
    - ef_candidates: [20, 40, 60, 80, 100, 150]
    - Static baseline: ef_search = 100 (middle ground)
    - Adaptive: Q-learning with feedback

Expected Outcome:
    Adaptive should show 5-10% efficiency improvement by selecting appropriate
    ef_search values based on query intent patterns learned from feedback.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time
import numpy as np
from typing import List
from dynhnsw import VectorStore

# Import shared utilities
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from shared_utils import (
    ExperimentTracker,
    compare_results,
    print_comparison_table,
    generate_realistic_corpus,
    create_realistic_queries,
    simulate_feedback,
)


EF_CANDIDATES = [20, 40, 60, 80, 100, 150]


def run_static_baseline(
    corpus_embeddings: np.ndarray,
    corpus_ids: List[str],
    queries: List[str],
    query_types: List[str],
    embedding_model,
    ef_search: int = 100,
) -> ExperimentTracker:
    """Run baseline with static ef_search.

    Args:
        corpus_embeddings: Pre-computed corpus embeddings
        corpus_ids: Document IDs
        queries: Query strings
        query_types: Query type labels
        embedding_model: SentenceTransformer model for encoding queries
        ef_search: Static ef_search value

    Returns:
        ExperimentTracker with results
    """
    print(f"\nCreating static baseline VectorStore (ef_search={ef_search})...")

    store = VectorStore(
        dimension=384,
        M=16,
        ef_construction=200,
        ef_search=ef_search,
        enable_intent_detection=False,  # No adaptation
    )

    print(f"Adding {len(corpus_embeddings)} documents...")
    store.add(corpus_embeddings, ids=corpus_ids)

    tracker = ExperimentTracker(f"static_ef{ef_search}")

    print(f"Running {len(queries)} queries with static ef_search={ef_search}...")
    for i, (query_text, qtype) in enumerate(zip(queries, query_types)):
        if (i + 1) % 100 == 0:
            print(f"  Progress: {i+1}/{len(queries)} queries")

        # Embed query
        q_vec = embedding_model.encode([query_text], convert_to_numpy=True)[0].astype(np.float32)

        # Search
        start = time.perf_counter()
        results = store.search(q_vec, k=10)
        latency_ms = (time.perf_counter() - start) * 1000.0

        # Simulate feedback
        relevant_ids, satisfaction = simulate_feedback(qtype, results, k=10)

        # Record metrics
        tracker.record(latency_ms, satisfaction, ef_search, qtype)

    print(f"Static baseline complete!")
    return tracker


def run_adaptive_learning(
    corpus_embeddings: np.ndarray,
    corpus_ids: List[str],
    queries: List[str],
    query_types: List[str],
    embedding_model,
    k_intents: int = 5,
) -> ExperimentTracker:
    """Run experiment with adaptive ef_search learning.

    Args:
        corpus_embeddings: Pre-computed corpus embeddings
        corpus_ids: Document IDs
        queries: Query strings
        query_types: Query type labels
        embedding_model: SentenceTransformer model for encoding queries
        k_intents: Number of intent clusters

    Returns:
        ExperimentTracker with results
    """
    print(f"\nCreating adaptive VectorStore (k_intents={k_intents})...")

    store = VectorStore(
        dimension=384,
        M=16,
        ef_construction=200,
        ef_search=100,  # Default, will be adapted
        enable_intent_detection=True,
        k_intents=k_intents,
        min_queries_for_clustering=50,
    )

    # Override ef_candidates if needed
    from dynhnsw.ef_search_selector import EfSearchSelector
    store._searcher.ef_selector = EfSearchSelector(
        k_intents=k_intents,
        default_ef=100,
        exploration_rate=0.15,
        ef_candidates=EF_CANDIDATES,
    )

    # Lower confidence threshold for more aggressive adaptation
    store._searcher.confidence_threshold = 0.1

    print(f"Adding {len(corpus_embeddings)} documents...")
    store.add(corpus_embeddings, ids=corpus_ids)

    tracker = ExperimentTracker(f"adaptive_k{k_intents}")

    print(f"Running {len(queries)} queries with adaptive learning...")
    q_snapshot_interval = 100

    for i, (query_text, qtype) in enumerate(zip(queries, query_types)):
        if (i + 1) % 100 == 0:
            print(f"  Progress: {i+1}/{len(queries)} queries")

        # Embed query
        q_vec = embedding_model.encode([query_text], convert_to_numpy=True)[0].astype(np.float32)

        # Search
        start = time.perf_counter()
        results = store.search(q_vec, k=10)
        latency_ms = (time.perf_counter() - start) * 1000.0

        # Simulate feedback
        relevant_ids, satisfaction = simulate_feedback(qtype, results, k=10)
        store.provide_feedback(relevant_ids=relevant_ids)

        # Record metrics
        ef_used = store._searcher.last_ef_used
        intent_id = store._searcher.last_intent_id
        tracker.record(latency_ms, satisfaction, ef_used, qtype, intent_id)

        # Snapshot Q-values periodically
        if (i + 1) % q_snapshot_interval == 0:
            stats = store._searcher.ef_selector.get_statistics()
            q_values = {}
            for intent_info in stats["per_intent"]:
                intent_id = intent_info["intent_id"]
                q_values[intent_id] = intent_info["q_values"]
            tracker.record_q_values(q_values)

    print(f"Adaptive learning complete!")

    # Print final statistics
    stats = store._searcher.ef_selector.get_statistics()
    print(f"\nFinal Learning Statistics:")
    print(f"  Total updates: {stats['total_updates']}")
    print(f"  k_intents: {stats['k_intents']}")

    print(f"\nef_search distribution:")
    ef_counts = {}
    for ef in tracker.ef_values:
        ef_counts[ef] = ef_counts.get(ef, 0) + 1
    for ef in sorted(ef_counts.keys()):
        percentage = ef_counts[ef] / len(tracker.ef_values) * 100
        print(f"  ef={ef:>3}: {ef_counts[ef]:>4} queries ({percentage:>5.1f}%)")

    return tracker


def main():
    """Run Scenario 1: Large Document Corpus."""
    print("\n" + "=" * 100)
    print("SCENARIO 1: LARGE DOCUMENT CORPUS TEST")
    print("=" * 100)
    print("\nHypothesis:")
    print("  Large corpus (5000 docs) with diverse queries benefits from adaptive ef_search.")
    print("  Adaptive learning should show 5-10% efficiency improvement over static.")
    print("\nConfiguration:")
    print(f"  Corpus size: 5000 documents")
    print(f"  Embedding model: sentence-transformers all-MiniLM-L6-v2 (384 dims)")
    print(f"  Queries: 1000 (exploratory, precise, mixed)")
    print(f"  k_intents: 5")
    print(f"  ef_candidates: {EF_CANDIDATES}")
    print("\nComparison:")
    print("  Static: ef_search = 100 (fixed)")
    print("  Adaptive: Q-learning with intent clustering")

    # Check dependencies
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        print("\nERROR: sentence-transformers not installed!")
        print("Install with: pip install sentence-transformers")
        sys.exit(1)

    # Generate corpus
    print("\n" + "-" * 100)
    print("CORPUS GENERATION")
    print("-" * 100)
    print("Generating large realistic corpus...")
    corpus = generate_realistic_corpus(size=5000)
    print(f"  Generated {len(corpus)} documents")

    # Embed corpus once (reused for both experiments)
    print("\nEmbedding corpus with sentence-transformers...")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    print("  Model loaded: all-MiniLM-L6-v2")
    embeddings = model.encode(corpus, convert_to_numpy=True, show_progress_bar=True)
    embeddings = embeddings.astype(np.float32)
    corpus_ids = [f"doc_{i}" for i in range(len(embeddings))]
    print(f"  Embeddings shape: {embeddings.shape}")

    # Generate queries
    print("\nGenerating diverse query set...")
    queries, query_types = create_realistic_queries(
        exploratory_count=200,
        precise_count=400,
        mixed_count=400,
    )
    queries = queries[:1000]
    query_types = query_types[:1000]
    print(f"  Total queries: {len(queries)}")
    print(f"  Exploratory: {query_types.count('exploratory')}")
    print(f"  Precise: {query_types.count('precise')}")
    print(f"  Mixed: {query_types.count('mixed')}")

    # Run static baseline
    print("\n" + "-" * 100)
    print("STATIC BASELINE EXPERIMENT")
    print("-" * 100)
    static_tracker = run_static_baseline(
        corpus_embeddings=embeddings,
        corpus_ids=corpus_ids,
        queries=queries,
        query_types=query_types,
        embedding_model=model,
        ef_search=100,
    )

    # Run adaptive learning
    print("\n" + "-" * 100)
    print("ADAPTIVE LEARNING EXPERIMENT")
    print("-" * 100)
    adaptive_tracker = run_adaptive_learning(
        corpus_embeddings=embeddings,
        corpus_ids=corpus_ids,
        queries=queries,
        query_types=query_types,
        embedding_model=model,
        k_intents=5,
    )

    # Compare results
    comparison = compare_results(static_tracker, adaptive_tracker)
    print_comparison_table(comparison)

    # Convergence analysis
    print("\n" + "=" * 100)
    print("CONVERGENCE ANALYSIS")
    print("=" * 100)

    adaptive_conv = adaptive_tracker.get_convergence_metrics()

    print(f"\nAdaptive Learning Convergence:")
    print(f"  Average Q-value change: {adaptive_conv.get('avg_q_change', 0):.4f}")
    print(f"  Final Q-value change: {adaptive_conv.get('final_q_change', 0):.4f}")
    print(f"  Q-stability score: {adaptive_conv.get('q_stability', 0):.4f}")

    # Save results
    print("\n" + "=" * 100)
    print("Saving results...")
    static_tracker.save_results("adaptive_ef_tests/results/scenario_1_static.json")
    adaptive_tracker.save_results("adaptive_ef_tests/results/scenario_1_adaptive.json")
    print("  Saved to: adaptive_ef_tests/results/scenario_1_*.json")
    print("=" * 100)


if __name__ == "__main__":
    main()
