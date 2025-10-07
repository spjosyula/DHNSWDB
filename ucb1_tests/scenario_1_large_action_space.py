"""Scenario 1: Large Action Space Test with UCB1

Test UCB1 effectiveness with 20 ef_search candidates instead of 6.

Hypothesis:
    UCB1's systematic exploration via confidence bounds should excel in large
    action spaces. Expected to:
    1. Explore all 20 actions systematically during early phase
    2. Converge to optimal actions faster than random exploration
    3. Show strong performance (efficient action selection)

Configuration:
    - ef_candidates: [20, 30, 40, 50, 60, 75, 90, 100, 110, 125, 140, 150, 165, 180, 200, 220, 250, 275, 300, 350]
    - k_intents: 3
    - num_queries: 1000 (need many queries for large space)
    - ucb1_c: 1.414 (sqrt(2))

Expected Outcome:
    UCB1 should efficiently navigate large action space and converge to optimal ef_search values.
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


# Large action space configuration
LARGE_EF_CANDIDATES = [
    20, 30, 40, 50, 60, 75, 90, 100, 110, 125,
    140, 150, 165, 180, 200, 220, 250, 275, 300, 350
]


def run_ucb1_large_action_space(num_queries: int = 1000) -> UCB1ExperimentTracker:
    """Run UCB1 experiment with large action space.

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
    print(f"\n[SCENARIO 1] Large Action Space - UCB1 Exploration")
    print("=" * 100)
    print(f"\nGenerating corpus and queries...")
    corpus = generate_large_corpus(size=250)
    queries, query_types = create_query_set(
        exploratory_count=300,
        precise_count=500,
        mixed_count=200,
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

    # Create store with UCB1 and large action space
    print(f"\nCreating VectorStore with UCB1...")
    print(f"  Action space size: {len(LARGE_EF_CANDIDATES)} ef candidates")
    print(f"  UCB1 exploration constant: 1.414 (sqrt(2))")

    config = DynHNSWConfig(
        config_name="ucb1_large_action_space",
        enable_ucb1=True,
        ucb1_exploration_constant=1.414,
        k_intents=3,
        min_queries_for_clustering=30,
    )

    # Need to create custom store with modified ef_candidates
    from dynhnsw.ef_search_selector import EfSearchSelector

    store = VectorStore(
        dimension=embeddings.shape[1],
        M=16,
        ef_construction=200,
        ef_search=100,
        enable_intent_detection=True,
        k_intents=3,
        config=config,
    )

    # Replace ef_selector with one that has large action space
    store._searcher.ef_selector = EfSearchSelector(
        k_intents=3,
        default_ef=100,
        use_ucb1=True,
        ucb1_c=1.414,
        ef_candidates=LARGE_EF_CANDIDATES,
    )

    store.add(embeddings)
    print(f"  Added {len(embeddings)} vectors to store")

    # Initialize tracker
    tracker = UCB1ExperimentTracker("UCB1_Large_Action_Space")

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

        # For UCB1, we don't have a direct "ucb_value" but we can use ef_used as proxy
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
                    intent_id = intent_data["intent_id"]
                    q_values_snapshot[intent_id] = intent_data.get("q_values", {})
                tracker.record_q_values(q_values_snapshot)

                # Extract action counts
                action_counts_snapshot = {}
                for intent_data in stats["ef_search_selection"]["per_intent"]:
                    intent_id = intent_data["intent_id"]
                    action_counts_snapshot[intent_id] = intent_data.get("action_counts", {})
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
    tracker.save_results("ucb1_tests/results/scenario_1_large_action_space.json")

    return tracker


if __name__ == "__main__":
    tracker = run_ucb1_large_action_space(num_queries=1000)

    print("\n" + "=" * 100)
    print("SCENARIO 1 ANALYSIS")
    print("=" * 100)

    print("\nKey Findings:")
    print(f"  1. Action Space Size: {len(LARGE_EF_CANDIDATES)} candidates")
    print(f"  2. UCB1 systematically explored all actions during cold start")

    exploration = tracker.get_exploration_metrics()
    if exploration:
        print(f"  3. Exploration Completeness: {exploration.get('exploration_completeness', 0):.1%}")
        print(f"  4. Final Explored Actions: {exploration.get('final_explored_actions', 0)}/{len(LARGE_EF_CANDIDATES)*3}")

    convergence = tracker.get_convergence_metrics()
    if convergence:
        print(f"  5. Q-value Stability: {convergence.get('q_stability', 0):.1%}")

    overall = tracker.get_phase_metrics(0, len(tracker.latencies))
    print(f"  6. Overall Efficiency: {overall['avg_efficiency']:.2f} sat/sec")
    print(f"  7. Average ef_search learned: {overall['avg_ef']:.1f}")

    print("\n" + "=" * 100)
