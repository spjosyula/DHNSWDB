"""Scenario 3: Long Horizon Test with UCB1

Test UCB1 performance over extended query session (5000 queries).

Hypothesis:
    UCB1 should show strong long-term performance due to optimal exploration-exploitation.
    Expected to:
    1. Converge quickly to optimal ef_search values
    2. Maintain high efficiency throughout long session
    3. Show stable Q-values in late phase (low variance)

Configuration:
    - ef_candidates: [50, 75, 100, 150, 200, 250] (standard 6 actions)
    - k_intents: 3
    - num_queries: 5000 (long horizon)
    - ucb1_c: 1.414 (sqrt(2))

Expected Outcome:
    UCB1 should achieve optimal performance and maintain it over long horizon.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time
import numpy as np
from dynhnsw import VectorStore
from dynhnsw.config import DynHNSWConfig

# Import shared utilities
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from shared_utils import (
    UCB1ExperimentTracker,
    print_results_summary,
    generate_large_corpus,
    create_query_set,
    simulate_feedback,
)


def run_ucb1_long_horizon(num_queries: int = 5000) -> UCB1ExperimentTracker:
    """Run UCB1 experiment with long horizon.

    Args:
        num_queries: Number of queries to run

    Returns:
        UCB1ExperimentTracker with results
    """
    # Import sentence transformers
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        print("ERROR: sentence-transformers not installed!")
        print("Install with: pip install sentence-transformers")
        sys.exit(1)

    # Generate data
    print(f"\n[SCENARIO 3] Long Horizon (5000 queries) - UCB1 Exploration")
    print("=" * 100)
    print(f"\nGenerating corpus and queries...")
    corpus = generate_large_corpus(size=400)
    queries, query_types = create_query_set(
        exploratory_count=1500,
        precise_count=2500,
        mixed_count=1000,
    )
    queries = queries[:num_queries]
    query_types = query_types[:num_queries]

    print(f"  Corpus: {len(corpus)} documents")
    print(f"  Queries: {len(queries)} queries")
    print(f"  Exploratory: {query_types.count('exploratory')}")
    print(f"  Precise: {query_types.count('precise')}")
    print(f"  Mixed: {query_types.count('mixed')}")

    # Embed corpus
    print(f"\nEmbedding corpus with sentence transformers...")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = model.encode(corpus, convert_to_numpy=True, show_progress_bar=False)
    embeddings = embeddings.astype(np.float32)

    # Create store with UCB1
    print(f"\nCreating VectorStore with UCB1...")
    print(f"  Session length: {num_queries} queries (long horizon)")
    print(f"  Number of intents: 3")
    print(f"  Action space: 6 ef_search candidates")
    print(f"  UCB1 exploration constant: 1.414 (sqrt(2))")

    config = DynHNSWConfig(
        config_name="ucb1_long_horizon",
        enable_ucb1=True,
        ucb1_exploration_constant=1.414,
        k_intents=3,
        min_queries_for_clustering=30,
    )

    store = VectorStore(
        dimension=embeddings.shape[1],
        M=16,
        ef_construction=200,
        ef_search=100,
        enable_intent_detection=True,
        k_intents=3,
        config=config,
    )

    store.add(embeddings)
    print(f"  Added {len(embeddings)} vectors to store")

    # Initialize tracker
    tracker = UCB1ExperimentTracker("UCB1_Long_Horizon_5000q")

    # Run queries
    print(f"\nRunning {num_queries} queries...")
    query_embeddings = model.encode(queries, convert_to_numpy=True, show_progress_bar=False)

    for i, (query_vec, query_type) in enumerate(zip(query_embeddings, query_types)):
        # Determine k based on query type
        if query_type == "exploratory":
            k = 15
        elif query_type == "precise":
            k = 5
        else:
            k = 10

        # Search
        start_time = time.perf_counter()
        results = store.search(query_vec, k=k)
        latency_ms = (time.perf_counter() - start_time) * 1000.0

        # Simulate feedback
        relevant_ids, satisfaction = simulate_feedback(query_type, results, k)
        if relevant_ids:
            store.provide_feedback(relevant_ids=relevant_ids)

        # Get metadata from searcher
        ef_used = store._searcher.last_ef_used
        intent_id = store._searcher.last_intent_id
        ucb_value = ef_used

        # Record metrics
        tracker.record(
            latency_ms=latency_ms,
            satisfaction=satisfaction,
            ef_used=ef_used,
            query_type=query_type,
            ucb_value=ucb_value,
            intent_id=intent_id,
        )

        # Record Q-values and action counts periodically
        if (i + 1) % 100 == 0:
            stats = store.get_statistics()
            if "ef_search_selection" in stats:
                # Extract Q-values
                q_values_snapshot = {}
                for intent_data in stats["ef_search_selection"]["per_intent"]:
                    intent_id_snap = intent_data["intent_id"]
                    q_values_snapshot[intent_id_snap] = intent_data.get("q_values", {})
                tracker.record_q_values(q_values_snapshot)

                # Extract action counts
                action_counts_snapshot = {}
                for intent_data in stats["ef_search_selection"]["per_intent"]:
                    intent_id_snap = intent_data["intent_id"]
                    action_counts_snapshot[intent_id_snap] = intent_data.get("action_counts", {})
                tracker.record_action_counts(action_counts_snapshot)

        # Progress update
        if (i + 1) % 500 == 0:
            phase_metrics = tracker.get_phase_metrics(max(0, i - 499), i + 1)
            print(f"  Progress: {i+1}/{num_queries} | "
                  f"Efficiency: {phase_metrics['avg_efficiency']:.2f} | "
                  f"Avg ef: {phase_metrics['avg_ef']:.1f}")

    print(f"\n[COMPLETE] Processed {num_queries} queries")

    # Print results
    print_results_summary(tracker)

    # Save results
    tracker.save_results("ucb1_tests/results/scenario_3_long_horizon.json")

    # Analyze convergence over time
    print("\n" + "=" * 100)
    print("CONVERGENCE ANALYSIS")
    print("=" * 100)

    phases = [
        ("Phase 1 (0-1000)", 0, 1000),
        ("Phase 2 (1000-2000)", 1000, 2000),
        ("Phase 3 (2000-3000)", 2000, 3000),
        ("Phase 4 (3000-4000)", 3000, 4000),
        ("Phase 5 (4000-5000)", 4000, 5000),
    ]

    print(f"\n{'Phase':<25} | {'Efficiency':>15} | {'Avg ef':>10} | {'Satisfaction':>15}")
    print("-" * 100)

    for phase_name, start, end in phases:
        metrics = tracker.get_phase_metrics(start, end)
        if metrics:
            print(f"{phase_name:<25} | {metrics['avg_efficiency']:>15.2f} | "
                  f"{metrics['avg_ef']:>10.1f} | {metrics['avg_satisfaction']:>14.1%}")

    return tracker


if __name__ == "__main__":
    tracker = run_ucb1_long_horizon(num_queries=5000)

    print("\n" + "=" * 100)
    print("SCENARIO 3 ANALYSIS")
    print("=" * 100)

    print("\nKey Findings:")
    print(f"  1. Session Length: 5000 queries (long horizon)")

    convergence = tracker.get_convergence_metrics()
    if convergence:
        print(f"  2. Q-value Stability: {convergence.get('q_stability', 0):.1%}")
        print(f"  3. Final Q-value Change: {convergence.get('final_q_change', 0):.4f}")

    # Compare early vs late performance
    early = tracker.get_phase_metrics(0, 1000)
    late = tracker.get_phase_metrics(4000, 5000)

    if early and late:
        eff_improvement = ((late['avg_efficiency'] - early['avg_efficiency']) /
                          early['avg_efficiency']) * 100
        print(f"\n  4. Early Efficiency (0-1000): {early['avg_efficiency']:.2f} sat/sec")
        print(f"  5. Late Efficiency (4000-5000): {late['avg_efficiency']:.2f} sat/sec")
        print(f"  6. Efficiency Improvement: {eff_improvement:+.1f}%")

    exploration = tracker.get_exploration_metrics()
    if exploration:
        print(f"\n  7. Exploration Completeness: {exploration.get('exploration_completeness', 0):.1%}")

    overall = tracker.get_phase_metrics(0, len(tracker.latencies))
    print(f"  8. Overall Efficiency: {overall['avg_efficiency']:.2f} sat/sec")

    print("\n" + "=" * 100)
