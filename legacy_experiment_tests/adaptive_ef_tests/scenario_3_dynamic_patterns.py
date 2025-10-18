"""Scenario 3: Dynamic Query Patterns Test

Test adaptive ef_search with shifting query patterns over time.

Hypothesis:
    Query patterns change during session (exploratory -> precise -> mixed):
    1. Adaptive learning should track pattern shifts
    2. Q-values should adapt to new query distributions
    3. Static ef_search is suboptimal for shifting patterns
    4. Should show 6-10% efficiency improvement over static

Configuration:
    - Corpus size: 2000 documents
    - Embedding: sentence-transformers all-MiniLM-L6-v2 (384 dims)
    - Queries: 1200 total in 3 phases:
      * Phase 1 (0-400): 90% exploratory, 10% precise
      * Phase 2 (400-800): 90% precise, 10% exploratory
      * Phase 3 (800-1200): 60% mixed, 20% exploratory, 20% precise
    - k_intents: 6 (more granular intent detection)
    - ef_candidates: [25, 40, 60, 80, 100, 120, 150]
    - Static baseline: ef_search = 80 (compromise)
    - Adaptive: Q-learning with feedback

Expected Outcome:
    Adaptive should show 6-10% efficiency improvement by adapting to each phase.
    Static ef_search cannot optimize for all phases simultaneously.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time
import numpy as np
from typing import List, Tuple
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


EF_CANDIDATES = [25, 40, 60, 80, 100, 120, 150]


def create_dynamic_query_set(
    phase1_count: int = 400,
    phase2_count: int = 400,
    phase3_count: int = 400,
) -> Tuple[List[str], List[str], List[int]]:
    """Create query set with shifting patterns across phases.

    Args:
        phase1_count: Queries in phase 1 (exploratory-heavy)
        phase2_count: Queries in phase 2 (precise-heavy)
        phase3_count: Queries in phase 3 (mixed)

    Returns:
        Tuple of (queries, query_types, phase_labels)
    """
    all_queries = []
    all_types = []
    all_phases = []

    # Phase 1: Exploratory-heavy (90% exploratory, 10% precise)
    exploratory_p1 = int(phase1_count * 0.9)
    precise_p1 = phase1_count - exploratory_p1

    queries_p1, types_p1 = create_realistic_queries(
        exploratory_count=exploratory_p1,
        precise_count=precise_p1,
        mixed_count=0,
    )
    all_queries.extend(queries_p1[:phase1_count])
    all_types.extend(types_p1[:phase1_count])
    all_phases.extend([1] * phase1_count)

    # Phase 2: Precise-heavy (10% exploratory, 90% precise)
    exploratory_p2 = int(phase2_count * 0.1)
    precise_p2 = phase2_count - exploratory_p2

    queries_p2, types_p2 = create_realistic_queries(
        exploratory_count=exploratory_p2,
        precise_count=precise_p2,
        mixed_count=0,
    )
    all_queries.extend(queries_p2[:phase2_count])
    all_types.extend(types_p2[:phase2_count])
    all_phases.extend([2] * phase2_count)

    # Phase 3: Mixed (20% exploratory, 20% precise, 60% mixed)
    exploratory_p3 = int(phase3_count * 0.2)
    precise_p3 = int(phase3_count * 0.2)
    mixed_p3 = phase3_count - exploratory_p3 - precise_p3

    queries_p3, types_p3 = create_realistic_queries(
        exploratory_count=exploratory_p3,
        precise_count=precise_p3,
        mixed_count=mixed_p3,
    )
    all_queries.extend(queries_p3[:phase3_count])
    all_types.extend(types_p3[:phase3_count])
    all_phases.extend([3] * phase3_count)

    return all_queries, all_types, all_phases


def run_static_baseline(
    corpus_embeddings: np.ndarray,
    corpus_ids: List[str],
    queries: List[str],
    query_types: List[str],
    phase_labels: List[int],
    embedding_model,
    ef_search: int = 80,
) -> ExperimentTracker:
    """Run baseline with static ef_search.

    Args:
        corpus_embeddings: Pre-computed corpus embeddings
        corpus_ids: Document IDs
        queries: Query strings
        query_types: Query type labels
        phase_labels: Phase labels for each query
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

    tracker = ExperimentTracker(f"static_ef{ef_search}_dynamic")

    print(f"Running {len(queries)} queries with static ef_search={ef_search}...")
    for i, (query_text, qtype, phase) in enumerate(zip(queries, query_types, phase_labels)):
        if (i + 1) % 100 == 0:
            print(f"  Progress: {i+1}/{len(queries)} queries (Phase {phase})")

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
    phase_labels: List[int],
    embedding_model,
    k_intents: int = 6,
) -> ExperimentTracker:
    """Run experiment with adaptive ef_search learning.

    Args:
        corpus_embeddings: Pre-computed corpus embeddings
        corpus_ids: Document IDs
        queries: Query strings
        query_types: Query type labels
        phase_labels: Phase labels for each query
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
        min_queries_for_clustering=60,
    )

    # Override ef_candidates
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

    tracker = ExperimentTracker(f"adaptive_k{k_intents}_dynamic")

    print(f"Running {len(queries)} queries with adaptive learning...")
    q_snapshot_interval = 100

    for i, (query_text, qtype, phase) in enumerate(zip(queries, query_types, phase_labels)):
        if (i + 1) % 100 == 0:
            print(f"  Progress: {i+1}/{len(queries)} queries (Phase {phase})")

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
    """Run Scenario 3: Dynamic Query Patterns."""
    print("\n" + "=" * 100)
    print("SCENARIO 3: DYNAMIC QUERY PATTERNS TEST")
    print("=" * 100)
    print("\nHypothesis:")
    print("  Query patterns shift over time (exploratory -> precise -> mixed).")
    print("  Adaptive learning tracks shifts, static ef_search is suboptimal.")
    print("  Should show 6-10% efficiency improvement.")
    print("\nConfiguration:")
    print(f"  Corpus size: 2000 documents")
    print(f"  Embedding model: sentence-transformers all-MiniLM-L6-v2 (384 dims)")
    print(f"  Queries: 1200 total in 3 phases:")
    print(f"    Phase 1 (0-400): 90% exploratory, 10% precise")
    print(f"    Phase 2 (400-800): 90% precise, 10% exploratory")
    print(f"    Phase 3 (800-1200): 60% mixed, 20% each")
    print(f"  k_intents: 6")
    print(f"  ef_candidates: {EF_CANDIDATES}")
    print("\nComparison:")
    print("  Static: ef_search = 80 (compromise for all phases)")
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
    print("Generating realistic corpus...")
    corpus = generate_realistic_corpus(size=2000)
    print(f"  Generated {len(corpus)} documents")

    # Embed corpus
    print("\nEmbedding corpus with sentence-transformers...")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    print("  Model loaded: all-MiniLM-L6-v2")
    embeddings = model.encode(corpus, convert_to_numpy=True, show_progress_bar=True)
    embeddings = embeddings.astype(np.float32)
    corpus_ids = [f"doc_{i}" for i in range(len(embeddings))]
    print(f"  Embeddings shape: {embeddings.shape}")

    # Generate dynamic query set
    print("\nGenerating dynamic query set with shifting patterns...")
    queries, query_types, phase_labels = create_dynamic_query_set(
        phase1_count=400,
        phase2_count=400,
        phase3_count=400,
    )
    print(f"  Total queries: {len(queries)}")

    # Count by phase
    phase1_types = [qt for qt, p in zip(query_types, phase_labels) if p == 1]
    phase2_types = [qt for qt, p in zip(query_types, phase_labels) if p == 2]
    phase3_types = [qt for qt, p in zip(query_types, phase_labels) if p == 3]

    print(f"\n  Phase 1 (exploratory-heavy):")
    print(f"    Exploratory: {phase1_types.count('exploratory')}")
    print(f"    Precise: {phase1_types.count('precise')}")
    print(f"    Mixed: {phase1_types.count('mixed')}")

    print(f"\n  Phase 2 (precise-heavy):")
    print(f"    Exploratory: {phase2_types.count('exploratory')}")
    print(f"    Precise: {phase2_types.count('precise')}")
    print(f"    Mixed: {phase2_types.count('mixed')}")

    print(f"\n  Phase 3 (mixed):")
    print(f"    Exploratory: {phase3_types.count('exploratory')}")
    print(f"    Precise: {phase3_types.count('precise')}")
    print(f"    Mixed: {phase3_types.count('mixed')}")

    # Run static baseline
    print("\n" + "-" * 100)
    print("STATIC BASELINE EXPERIMENT")
    print("-" * 100)
    static_tracker = run_static_baseline(
        corpus_embeddings=embeddings,
        corpus_ids=corpus_ids,
        queries=queries,
        query_types=query_types,
        phase_labels=phase_labels,
        embedding_model=model,
        ef_search=80,
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
        phase_labels=phase_labels,
        embedding_model=model,
        k_intents=6,
    )

    # Compare results overall and by phase
    phase_boundaries = [
        ("Phase 1 (Exploratory-heavy)", 0, 400),
        ("Phase 2 (Precise-heavy)", 400, 800),
        ("Phase 3 (Mixed)", 800, 1200),
        ("Overall", 0, 1200),
    ]

    comparison = compare_results(static_tracker, adaptive_tracker, phases=phase_boundaries)
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
    static_tracker.save_results("adaptive_ef_tests/results/scenario_3_static.json")
    adaptive_tracker.save_results("adaptive_ef_tests/results/scenario_3_adaptive.json")
    print("  Saved to: adaptive_ef_tests/results/scenario_3_*.json")
    print("=" * 100)


if __name__ == "__main__":
    main()
