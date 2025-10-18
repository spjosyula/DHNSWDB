"""Scenario 2: Many Intents Test with UCB1

Test UCB1 performance with increased intent complexity (7 intents instead of 3).

Hypothesis:
    UCB1's confidence bounds should handle increased intent complexity well.
    Expected to:
    1. Learn distinct ef_search values for each of 7 intents
    2. Maintain high action diversity across intents
    3. Show consistent performance despite increased state space

Configuration:
    - ef_candidates: [50, 75, 100, 150, 200, 250] (standard 6 actions)
    - k_intents: 7 (increased from 3)
    - num_queries: 800
    - ucb1_c: 1.414 (sqrt(2))

Expected Outcome:
    UCB1 should effectively learn intent-specific ef_search values even with many intents.
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


def run_ucb1_many_intents(num_queries: int = 800) -> UCB1ExperimentTracker:
    """Run UCB1 experiment with many intents.

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
    print(f"\n[SCENARIO 2] Many Intents (7) - UCB1 Exploration")
    print("=" * 100)
    print(f"\nGenerating corpus and queries...")
    corpus = generate_large_corpus(size=300)
    queries, query_types = create_query_set(
        exploratory_count=240,
        precise_count=400,
        mixed_count=160,
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

    # Create store with UCB1 and 7 intents
    print(f"\nCreating VectorStore with UCB1...")
    print(f"  Number of intents: 7 (increased complexity)")
    print(f"  Action space: 6 ef_search candidates")
    print(f"  UCB1 exploration constant: 1.414 (sqrt(2))")

    config = DynHNSWConfig(
        config_name="ucb1_many_intents",
        enable_ucb1=True,
        ucb1_exploration_constant=1.414,
        k_intents=7,  # Increased from 3 to 7
        min_queries_for_clustering=50,  # Need more queries for 7 clusters
    )

    store = VectorStore(
        dimension=embeddings.shape[1],
        M=16,
        ef_construction=200,
        ef_search=100,
        enable_intent_detection=True,
        k_intents=7,
        config=config,
    )

    store.add(embeddings)
    print(f"  Added {len(embeddings)} vectors to store")

    # Initialize tracker
    tracker = UCB1ExperimentTracker("UCB1_Many_Intents_k7")

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
        if (i + 1) % 50 == 0:
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
        if (i + 1) % 100 == 0:
            phase_metrics = tracker.get_phase_metrics(max(0, i - 99), i + 1)
            print(f"  Progress: {i+1}/{num_queries} | "
                  f"Efficiency: {phase_metrics['avg_efficiency']:.2f} | "
                  f"Avg ef: {phase_metrics['avg_ef']:.1f}")

    print(f"\n[COMPLETE] Processed {num_queries} queries")

    # Print results
    print_results_summary(tracker)

    # Save results
    tracker.save_results("ucb1_tests/results/scenario_2_many_intents.json")

    # Analyze per-intent learning
    print("\n" + "=" * 100)
    print("PER-INTENT ANALYSIS")
    print("=" * 100)

    stats = store.get_statistics()
    if "ef_search_selection" in stats:
        print(f"\n{'Intent ID':<12} | {'Learned ef':<12} | {'Queries':<12} | {'Top 3 ef_search':<30}")
        print("-" * 100)

        for intent_data in stats["ef_search_selection"]["per_intent"]:
            intent_id = intent_data["intent_id"]
            learned_ef = intent_data.get("learned_ef", 0)
            num_queries_intent = intent_data.get("num_queries", 0)

            # Get top 3 most-used ef_search values
            action_counts = intent_data.get("action_counts", {})
            if action_counts:
                top_actions = sorted(action_counts.items(), key=lambda x: x[1], reverse=True)[:3]
                top_str = ", ".join([f"{ef}({count})" for ef, count in top_actions])
            else:
                top_str = "N/A"

            print(f"{intent_id:<12} | {learned_ef:<12} | {num_queries_intent:<12} | {top_str:<30}")

    return tracker


if __name__ == "__main__":
    tracker = run_ucb1_many_intents(num_queries=800)

    print("\n" + "=" * 100)
    print("SCENARIO 2 ANALYSIS")
    print("=" * 100)

    print("\nKey Findings:")
    print(f"  1. Intent Complexity: 7 intents (state space = 7 × 6 = 42 actions)")

    exploration = tracker.get_exploration_metrics()
    if exploration:
        print(f"  2. Exploration Completeness: {exploration.get('exploration_completeness', 0):.1%}")
        print(f"  3. Final Explored Actions: {exploration.get('final_explored_actions', 0)}/42")

    convergence = tracker.get_convergence_metrics()
    if convergence:
        print(f"  4. Q-value Stability: {convergence.get('q_stability', 0):.1%}")

    overall = tracker.get_phase_metrics(0, len(tracker.latencies))
    print(f"  5. Overall Efficiency: {overall['avg_efficiency']:.2f} sat/sec")

    # Count intents with significant learning
    intent_counts = {}
    for intent_id in tracker.intent_ids:
        if intent_id >= 0:
            intent_counts[intent_id] = intent_counts.get(intent_id, 0) + 1

    active_intents = len([v for v in intent_counts.values() if v > 10])
    print(f"  6. Active Intents (>10 queries): {active_intents}/7")

    print("\n" + "=" * 100)
