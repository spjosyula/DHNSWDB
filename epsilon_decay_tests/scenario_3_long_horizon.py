"""Scenario 3: Long Horizon Test

Test epsilon decay effectiveness over 5000 queries instead of 300.

Hypothesis:
    GLIE benefits amortize over longer time horizons. With 5000 queries:
    1. GLIE has time to fully explore and converge to optimal policy
    2. Later queries benefit from highly optimized Q-values
    3. Cumulative efficiency should be 2-3% better than fixed epsilon

Configuration:
    - k_intents: 3 (same as baseline)
    - ef_candidates: [50, 75, 100, 150, 200, 250] (same as baseline)
    - num_queries: 5000 (vs baseline 300)
    - Control: Fixed epsilon = 0.15
    - Treatment: GLIE epsilon decay starting at 0.4

Expected Outcome:
    GLIE should show 2-3% cumulative efficiency improvement. The benefit should
    be most visible in the late phase (queries 3000-5000) where GLIE has fully
    converged while fixed epsilon still wastes effort on exploration.
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
    num_queries: int = 5000,
) -> ExperimentTracker:
    """Run experiment with long horizon.

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

    # Generate larger corpus for 5000 queries
    print(f"\nGenerating corpus and queries...")
    corpus = generate_large_corpus(size=500)

    # Create large query set
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

    # Create store
    print(f"\nCreating VectorStore (epsilon_decay_mode={epsilon_decay_mode})...")
    print(f"  Long horizon: {num_queries} queries")
    print(f"  Initial epsilon: {initial_epsilon}")

    store = VectorStore(
        dimension=384,
        M=16,
        ef_search=100,
        enable_intent_detection=True,
        k_intents=3,
        min_queries_for_clustering=30,
    )

    # Override ef_selector
    store._searcher.ef_selector = EfSearchSelector(
        k_intents=3,
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
    tracker = ExperimentTracker(f"{epsilon_decay_mode}_eps{initial_epsilon}_long_horizon")

    print(f"\nRunning {len(queries)} queries...")
    print("  This may take several minutes...")
    q_snapshot_interval = 100

    for i, (query_text, qtype) in enumerate(zip(queries, query_types)):
        if (i + 1) % 500 == 0:
            epsilon = store._searcher.ef_selector.exploration_rate
            avg_eff = np.mean(tracker.efficiencies[-100:]) if len(tracker.efficiencies) >= 100 else 0
            print(f"  Progress: {i+1}/{len(queries)} queries, epsilon={epsilon:.4f}, recent_eff={avg_eff:.2f}")

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
    return tracker


def main():
    """Run Scenario 3: Long Horizon."""
    print("\n" + "=" * 100)
    print("SCENARIO 3: LONG HORIZON TEST")
    print("=" * 100)
    print("\nHypothesis:")
    print("  GLIE benefits amortize over long horizons (5000 queries vs 300).")
    print("  Late-phase queries benefit from converged Q-values.")
    print("  Expected: 2-3% cumulative efficiency improvement.")
    print("\nConfiguration:")
    print(f"  k_intents: 3")
    print(f"  ef_candidates: [50, 75, 100, 150, 200, 250]")
    print(f"  num_queries: 5000 (baseline: 300)")
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
        num_queries=5000,
    )

    # Run treatment experiment
    print("\n" + "-" * 100)
    print("TREATMENT EXPERIMENT: GLIE Epsilon Decay (starts at 0.4)")
    print("-" * 100)
    treatment_tracker = run_experiment(
        epsilon_decay_mode="glie",
        initial_epsilon=0.4,
        num_queries=5000,
    )

    # Compare results with more detailed phases
    total = len(control_tracker.latencies)
    phases = [
        ("Early (0-1000)", 0, 1000),
        ("Mid-Early (1000-2000)", 1000, 2000),
        ("Mid-Late (2000-3000)", 2000, 3000),
        ("Late (3000-4000)", 3000, 4000),
        ("Very Late (4000-5000)", 4000, 5000),
        ("Overall", 0, total),
    ]

    comparison = compare_results(control_tracker, treatment_tracker, phases=phases)
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

    # Epsilon trajectory
    print("\n" + "=" * 100)
    print("EPSILON TRAJECTORY")
    print("=" * 100)

    print(f"\nTreatment (GLIE Decay) epsilon values:")
    sample_indices = [0, 500, 1000, 2000, 3000, 4000, 4999]
    for idx in sample_indices:
        if idx < len(treatment_tracker.epsilon_values):
            print(f"  Query {idx:>4}: epsilon = {treatment_tracker.epsilon_values[idx]:.4f}")

    # Save results
    print("\n" + "=" * 100)
    print("Saving results...")
    control_tracker.save_results("epsilon_decay_tests/results/scenario_3_control.json")
    treatment_tracker.save_results("epsilon_decay_tests/results/scenario_3_treatment.json")
    print("  Saved to: epsilon_decay_tests/results/scenario_3_*.json")
    print("=" * 100)


if __name__ == "__main__":
    main()
