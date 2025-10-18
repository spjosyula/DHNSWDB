"""Scenario 2: More Intents Test

Test epsilon decay effectiveness with 10 intent clusters instead of 3.

Hypothesis:
    More intents create more state-action pairs (10 intents x 6 ef = 60 pairs vs 3x6=18).
    GLIE decay should:
    1. Explore all state-action combinations more efficiently
    2. Learn intent-specific preferences faster
    3. Show 2-4% efficiency improvement

Configuration:
    - k_intents: 10 (vs baseline 3)
    - ef_candidates: [50, 75, 100, 150, 200, 250] (same as baseline)
    - num_queries: 600 (need more for 10 intents)
    - Control: Fixed epsilon = 0.15
    - Treatment: GLIE epsilon decay starting at 0.4

Expected Outcome:
    GLIE should show 2-4% efficiency improvement due to more efficient exploration
    across the expanded state space.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time
import numpy as np
from typing import List
from dynhnsw import VectorStore
from dynhnsw.ef_search_selector import EfSearchSelector

# Import shared utilities
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from shared_utils import (
    ExperimentTracker,
    compare_results,
    print_comparison_table,
    generate_large_corpus,
    create_query_set,
    simulate_feedback,
)


def run_experiment(
    epsilon_decay_mode: str,
    initial_epsilon: float,
    num_queries: int = 600,
) -> ExperimentTracker:
    """Run experiment with more intents.

    Args:
        epsilon_decay_mode: "none" or "glie"
        initial_epsilon: Starting exploration rate
        num_queries: Number of queries to run

    Returns:
        ExperimentTracker with results
    """
    # Import sentence transformers
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        print("ERROR: sentence-transformers not installed!")
        print("Install with: pip install sentence-transformers")
        sys.exit(1)

    # Generate data
    print(f"\nGenerating corpus and queries...")
    corpus = generate_large_corpus(size=300)
    queries, query_types = create_query_set(
        exploratory_count=180,
        precise_count=300,
        mixed_count=120,
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

    # Create store with 10 intents
    print(f"\nCreating VectorStore (epsilon_decay_mode={epsilon_decay_mode})...")
    print(f"  k_intents: 10 (state space size: 60 state-action pairs)")
    print(f"  Initial epsilon: {initial_epsilon}")

    store = VectorStore(
        dimension=384,
        M=16,
        ef_search=100,
        enable_intent_detection=True,
        k_intents=10,  # Increased from 3 to 10
        min_queries_for_clustering=50,  # More queries needed for 10 clusters
    )

    # Override ef_selector with 10 intents
    store._searcher.ef_selector = EfSearchSelector(
        k_intents=10,
        default_ef=100,
        exploration_rate=initial_epsilon,
        epsilon_decay_mode=epsilon_decay_mode,
        min_epsilon=0.01,
    )

    store._searcher.confidence_threshold = 0.1

    # Add documents
    print(f"\nAdding {len(embeddings)} documents to store...")
    ids = [f"doc_{i}" for i in range(len(embeddings))]
    store.add(embeddings, ids=ids)

    # Run queries
    tracker = ExperimentTracker(f"{epsilon_decay_mode}_eps{initial_epsilon}_more_intents")

    print(f"\nRunning {len(queries)} queries...")
    q_snapshot_interval = 50

    for i, (query_text, qtype) in enumerate(zip(queries, query_types)):
        if (i + 1) % 50 == 0:
            epsilon = store._searcher.ef_selector.exploration_rate
            print(f"  Progress: {i+1}/{len(queries)} queries, epsilon={epsilon:.4f}")

        # Embed query
        q_vec = model.encode([query_text], convert_to_numpy=True)[0].astype(np.float32)

        # Search
        start = time.perf_counter()
        results = store.search(q_vec, k=10)
        latency_ms = (time.perf_counter() - start) * 1000.0

        # Simulate feedback
        relevant_ids, satisfaction = simulate_feedback(qtype, results, k=10)
        store.provide_feedback(relevant_ids=relevant_ids)

        # Record metrics
        ef_used = store._searcher.last_ef_used
        epsilon = store._searcher.ef_selector.exploration_rate
        intent_id = store._searcher.last_intent_id
        tracker.record(latency_ms, satisfaction, ef_used, qtype, epsilon, intent_id)

        # Snapshot Q-values periodically
        if (i + 1) % q_snapshot_interval == 0:
            stats = store._searcher.ef_selector.get_statistics()
            q_values = {}
            for intent_info in stats["per_intent"]:
                intent_id = intent_info["intent_id"]
                q_values[intent_id] = intent_info["q_values"]
            tracker.record_q_values(q_values)

    print(f"\nExperiment complete!")

    # Print intent distribution
    print(f"\nIntent distribution:")
    intent_counts = {}
    for intent_id in tracker.intent_ids:
        if intent_id >= 0:
            intent_counts[intent_id] = intent_counts.get(intent_id, 0) + 1

    for intent_id in sorted(intent_counts.keys()):
        percentage = intent_counts[intent_id] / len(tracker.intent_ids) * 100
        print(f"  Intent {intent_id}: {intent_counts[intent_id]:>4} queries ({percentage:>5.1f}%)")

    return tracker


def main():
    """Run Scenario 2: More Intents."""
    print("\n" + "=" * 100)
    print("SCENARIO 2: MORE INTENTS TEST")
    print("=" * 100)
    print("\nHypothesis:")
    print("  More intents (10 vs 3) create larger state space (60 vs 18 state-action pairs).")
    print("  GLIE decay should show 2-4% efficiency improvement through better exploration.")
    print("\nConfiguration:")
    print(f"  k_intents: 10 (baseline: 3)")
    print(f"  ef_candidates: [50, 75, 100, 150, 200, 250]")
    print(f"  State space: 10 intents x 6 ef = 60 state-action pairs")
    print(f"  num_queries: 600")
    print("\nComparison:")
    print("  Control: Fixed epsilon = 0.15 (NO decay)")
    print("  Treatment: GLIE epsilon decay starting at 0.4")

    # Run control experiment
    print("\n" + "-" * 100)
    print("CONTROL EXPERIMENT: Fixed Epsilon = 0.15")
    print("-" * 100)
    control_tracker = run_experiment(
        epsilon_decay_mode="none",
        initial_epsilon=0.15,
        num_queries=600,
    )

    # Run treatment experiment
    print("\n" + "-" * 100)
    print("TREATMENT EXPERIMENT: GLIE Epsilon Decay (starts at 0.4)")
    print("-" * 100)
    treatment_tracker = run_experiment(
        epsilon_decay_mode="glie",
        initial_epsilon=0.4,
        num_queries=600,
    )

    # Compare results
    comparison = compare_results(control_tracker, treatment_tracker)
    print_comparison_table(comparison)

    # Convergence analysis
    print("\n" + "=" * 100)
    print("CONVERGENCE ANALYSIS")
    print("=" * 100)

    control_conv = control_tracker.get_convergence_metrics()
    treatment_conv = treatment_tracker.get_convergence_metrics()

    print(f"\nControl (Fixed Epsilon):")
    print(f"  Average Q-value change: {control_conv.get('avg_q_change', 0):.4f}")
    print(f"  Final Q-value change: {control_conv.get('final_q_change', 0):.4f}")
    print(f"  Q-stability score: {control_conv.get('q_stability', 0):.4f}")

    print(f"\nTreatment (GLIE Decay):")
    print(f"  Average Q-value change: {treatment_conv.get('avg_q_change', 0):.4f}")
    print(f"  Final Q-value change: {treatment_conv.get('final_q_change', 0):.4f}")
    print(f"  Q-stability score: {treatment_conv.get('q_stability', 0):.4f}")

    # Save results
    print("\n" + "=" * 100)
    print("Saving results...")
    control_tracker.save_results("epsilon_decay_tests/results/scenario_2_control.json")
    treatment_tracker.save_results("epsilon_decay_tests/results/scenario_2_treatment.json")
    print("  Saved to: epsilon_decay_tests/results/scenario_2_*.json")
    print("=" * 100)


if __name__ == "__main__":
    main()
