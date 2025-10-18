"""Scenario 4: Non-Stationary Environment Test

Test epsilon decay in a non-stationary environment where query patterns shift mid-experiment.

Hypothesis:
    In non-stationary environments, fixed epsilon may OUTPERFORM GLIE decay because:
    1. GLIE reduces exploration over time, making it slow to adapt to shifts
    2. Fixed epsilon maintains constant exploration, detecting shifts faster
    3. This is a scenario where decay HURTS performance

Configuration:
    - k_intents: 3 (same as baseline)
    - ef_candidates: [50, 75, 100, 150, 200, 250] (same as baseline)
    - num_queries: 1000
    - Pattern shift at query 500:
      * Queries 0-499: 70% exploratory, 30% precise
      * Queries 500-999: 30% exploratory, 70% precise
    - Control: Fixed epsilon = 0.15
    - Treatment: GLIE epsilon decay starting at 0.4

Expected Outcome:
    Fixed epsilon should show BETTER performance post-shift (queries 500-999) because
    it maintains exploration capability to adapt to new patterns. GLIE may be stuck
    with low epsilon and fail to re-explore optimally.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time
import numpy as np
from typing import List, Tuple
from dynhnsw import VectorStore
from dynhnsw.ef_search_selector import EfSearchSelector

# Import shared utilities
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from shared_utils import (
    ExperimentTracker,
    compare_results,
    print_comparison_table,
    generate_large_corpus,
    simulate_feedback,
)


def create_shifting_query_set(shift_point: int = 500, total: int = 1000) -> Tuple[List[str], List[str]]:
    """Create query set with distribution shift at midpoint.

    Args:
        shift_point: Query index where distribution shifts
        total: Total number of queries

    Returns:
        Tuple of (queries, query_types) lists
    """
    queries = []
    query_types = []

    # Phase 1 (0 to shift_point): 70% exploratory, 30% precise
    phase1_exploratory = [
        "tell me about programming",
        "what is cloud computing",
        "explain machine learning",
        "overview of web development",
        "introduction to databases",
        "what are microservices",
        "what products do you sell",
        "show me office equipment",
        "help with my account",
        "general questions support",
    ]

    phase1_precise = [
        "Python list comprehension syntax",
        "PostgreSQL index optimization",
        "React hooks useState example",
        "Docker compose networking",
    ]

    for i in range(shift_point):
        if i % 10 < 7:  # 70% exploratory
            q = phase1_exploratory[i % len(phase1_exploratory)]
            queries.append(q)
            query_types.append("exploratory")
        else:  # 30% precise
            q = phase1_precise[i % len(phase1_precise)]
            queries.append(q)
            query_types.append("precise")

    # Phase 2 (shift_point to total): 30% exploratory, 70% precise
    phase2_exploratory = [
        "software architecture patterns",
        "testing strategies guide",
        "productivity tools overview",
        "shipping information guide",
    ]

    phase2_precise = [
        "wireless bluetooth headphones model 1000",
        "ergonomic office chair model 1001",
        "mechanical keyboard RGB model 1004",
        "how to reset password account",
        "track my order tracking number",
        "cancel subscription immediately",
        "apply discount code checkout",
        "photosynthesis chemical equation",
        "DNA double helix structure",
        "Newton second law formula",
    ]

    for i in range(shift_point, total):
        if i % 10 < 3:  # 30% exploratory
            q = phase2_exploratory[i % len(phase2_exploratory)]
            queries.append(q)
            query_types.append("exploratory")
        else:  # 70% precise
            q = phase2_precise[i % len(phase2_precise)]
            queries.append(q)
            query_types.append("precise")

    return queries, query_types


def run_experiment(
    epsilon_decay_mode: str,
    initial_epsilon: float,
    num_queries: int = 1000,
    shift_point: int = 500,
) -> ExperimentTracker:
    """Run experiment with non-stationary environment.

    Args:
        epsilon_decay_mode: "none" or "glie"
        initial_epsilon: Starting exploration rate
        num_queries: Number of queries to run
        shift_point: Query index where distribution shifts

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
    queries, query_types = create_shifting_query_set(shift_point=shift_point, total=num_queries)

    # Verify distribution shift
    phase1_exp = sum(1 for qt in query_types[:shift_point] if qt == "exploratory")
    phase2_exp = sum(1 for qt in query_types[shift_point:] if qt == "exploratory")
    print(f"  Corpus: {len(corpus)} documents")
    print(f"  Queries: {len(queries)} queries")
    print(f"  Phase 1 (0-{shift_point}): {phase1_exp}/{shift_point} exploratory ({phase1_exp/shift_point*100:.0f}%)")
    print(f"  Phase 2 ({shift_point}-{num_queries}): {phase2_exp}/{num_queries-shift_point} exploratory ({phase2_exp/(num_queries-shift_point)*100:.0f}%)")

    # Embed corpus
    print(f"\nEmbedding corpus with sentence transformers...")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = model.encode(corpus, convert_to_numpy=True, show_progress_bar=False)
    embeddings = embeddings.astype(np.float32)

    # Create store
    print(f"\nCreating VectorStore (epsilon_decay_mode={epsilon_decay_mode})...")
    print(f"  Non-stationary: distribution shifts at query {shift_point}")
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
    tracker = ExperimentTracker(f"{epsilon_decay_mode}_eps{initial_epsilon}_non_stationary")

    print(f"\nRunning {len(queries)} queries...")
    q_snapshot_interval = 50

    for i, (query_text, qtype) in enumerate(zip(queries, query_types)):
        if (i + 1) % 100 == 0:
            epsilon = store._searcher.ef_selector.exploration_rate
            avg_eff = np.mean(tracker.efficiencies[-50:]) if len(tracker.efficiencies) >= 50 else 0
            print(f"  Progress: {i+1}/{len(queries)} queries, epsilon={epsilon:.4f}, recent_eff={avg_eff:.2f}")

        # Mark shift point
        if i == shift_point:
            print(f"  [SHIFT POINT] Distribution changing from 70% exploratory to 70% precise")

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
    """Run Scenario 4: Non-Stationary Environment."""
    print("\n" + "=" * 100)
    print("SCENARIO 4: NON-STATIONARY ENVIRONMENT TEST")
    print("=" * 100)
    print("\nHypothesis:")
    print("  In non-stationary environments, fixed epsilon may OUTPERFORM GLIE decay.")
    print("  GLIE reduces exploration over time, making it slow to adapt to shifts.")
    print("  Fixed epsilon maintains exploration, detecting shifts faster.")
    print("\nConfiguration:")
    print(f"  k_intents: 3")
    print(f"  ef_candidates: [50, 75, 100, 150, 200, 250]")
    print(f"  num_queries: 1000")
    print(f"  Distribution shift at query 500:")
    print(f"    - Phase 1 (0-499): 70% exploratory, 30% precise")
    print(f"    - Phase 2 (500-999): 30% exploratory, 70% precise")
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
        num_queries=1000,
        shift_point=500,
    )

    # Run treatment experiment
    print("\n" + "-" * 100)
    print("TREATMENT EXPERIMENT: GLIE Epsilon Decay (starts at 0.4)")
    print("-" * 100)
    treatment_tracker = run_experiment(
        epsilon_decay_mode="glie",
        initial_epsilon=0.4,
        num_queries=1000,
        shift_point=500,
    )

    # Compare results with focus on pre/post shift
    phases = [
        ("Pre-Shift (0-500)", 0, 500),
        ("Post-Shift (500-1000)", 500, 1000),
        ("Early Post-Shift (500-700)", 500, 700),
        ("Late Post-Shift (700-1000)", 700, 1000),
        ("Overall", 0, 1000),
    ]

    comparison = compare_results(control_tracker, treatment_tracker, phases=phases)
    print_comparison_table(comparison)

    # Adaptation speed analysis
    print("\n" + "=" * 100)
    print("ADAPTATION SPEED ANALYSIS")
    print("=" * 100)
    print("\nHow quickly did each method adapt to the distribution shift?")
    print("(Measuring efficiency in 100-query windows post-shift)")

    for window_start in [500, 600, 700, 800, 900]:
        window_end = window_start + 100
        control_window = control_tracker.get_phase_metrics(window_start, window_end)
        treatment_window = treatment_tracker.get_phase_metrics(window_start, window_end)

        control_eff = control_window.get("avg_efficiency", 0)
        treatment_eff = treatment_window.get("avg_efficiency", 0)
        control_eps = control_window.get("avg_epsilon", 0)
        treatment_eps = treatment_window.get("avg_epsilon", 0)

        print(f"\nQueries {window_start}-{window_end}:")
        print(f"  Control:   eff={control_eff:.2f}, epsilon={control_eps:.4f}")
        print(f"  Treatment: eff={treatment_eff:.2f}, epsilon={treatment_eps:.4f}")

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
    control_tracker.save_results("epsilon_decay_tests/results/scenario_4_control.json")
    treatment_tracker.save_results("epsilon_decay_tests/results/scenario_4_treatment.json")
    print("  Saved to: epsilon_decay_tests/results/scenario_4_*.json")
    print("=" * 100)


if __name__ == "__main__":
    main()
