"""Scenario 4: Non-Stationary Environment Test with UCB1

Test UCB1 robustness when query patterns shift mid-session.

Hypothesis:
    UCB1 assumes stationary rewards, so performance may degrade when patterns shift.
    Expected to:
    1. Perform well in initial stable phase
    2. Show adaptation lag when patterns change
    3. Eventually re-converge in new environment

Configuration:
    - ef_candidates: [50, 75, 100, 150, 200, 250] (standard 6 actions)
    - k_intents: 3
    - num_queries: 1500
    - ucb1_c: 1.414 (sqrt(2))
    - Pattern shift at query 750

Expected Outcome:
    UCB1 should show temporary performance drop during shift, then recover.
    Standard UCB1 is not designed for non-stationarity, so adaptation may be slow.
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


def run_ucb1_non_stationary(num_queries: int = 1500) -> UCB1ExperimentTracker:
    """Run UCB1 experiment with non-stationary environment.

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
    print(f"\n[SCENARIO 4] Non-Stationary Environment - UCB1 Robustness")
    print("=" * 100)
    print(f"\nGenerating corpus and queries...")
    corpus = generate_large_corpus(size=350)

    # Phase 1 queries (0-750): Mostly exploratory
    queries_phase1, types_phase1 = create_query_set(
        exploratory_count=525,  # 70% exploratory
        precise_count=150,      # 20% precise
        mixed_count=75,         # 10% mixed
    )
    queries_phase1 = queries_phase1[:750]
    types_phase1 = types_phase1[:750]

    # Phase 2 queries (750-1500): Mostly precise
    queries_phase2, types_phase2 = create_query_set(
        exploratory_count=150,  # 20% exploratory
        precise_count=525,      # 70% precise
        mixed_count=75,         # 10% mixed
    )
    queries_phase2 = queries_phase2[:750]
    types_phase2 = types_phase2[:750]

    # Combine phases
    queries = queries_phase1 + queries_phase2
    query_types = types_phase1 + types_phase2

    print(f"  Corpus: {len(corpus)} documents")
    print(f"  Total Queries: {len(queries)}")
    print(f"\n  Phase 1 (0-750): Exploratory-heavy")
    print(f"    Exploratory: {types_phase1.count('exploratory')}")
    print(f"    Precise: {types_phase1.count('precise')}")
    print(f"    Mixed: {types_phase1.count('mixed')}")
    print(f"\n  Phase 2 (750-1500): Precise-heavy (PATTERN SHIFT)")
    print(f"    Exploratory: {types_phase2.count('exploratory')}")
    print(f"    Precise: {types_phase2.count('precise')}")
    print(f"    Mixed: {types_phase2.count('mixed')}")

    # Embed corpus
    print(f"\nEmbedding corpus with sentence transformers...")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = model.encode(corpus, convert_to_numpy=True, show_progress_bar=False)
    embeddings = embeddings.astype(np.float32)

    # Create store with UCB1
    print(f"\nCreating VectorStore with UCB1...")
    print(f"  Number of intents: 3")
    print(f"  Action space: 6 ef_search candidates")
    print(f"  UCB1 exploration constant: 1.414 (sqrt(2))")
    print(f"  Note: Standard UCB1 (not designed for non-stationarity)")

    config = DynHNSWConfig(
        config_name="ucb1_non_stationary",
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
    tracker = UCB1ExperimentTracker("UCB1_Non_Stationary")

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
        if (i + 1) % 150 == 0:
            phase_metrics = tracker.get_phase_metrics(max(0, i - 149), i + 1)
            phase_label = "Phase 1" if i < 750 else "Phase 2"
            print(f"  Progress [{phase_label}]: {i+1}/{num_queries} | "
                  f"Efficiency: {phase_metrics['avg_efficiency']:.2f} | "
                  f"Avg ef: {phase_metrics['avg_ef']:.1f}")

    print(f"\n[COMPLETE] Processed {num_queries} queries")

    # Print results
    print_results_summary(tracker)

    # Save results
    tracker.save_results("ucb1_tests/results/scenario_4_non_stationary.json")

    # Analyze phase transition
    print("\n" + "=" * 100)
    print("NON-STATIONARITY ANALYSIS")
    print("=" * 100)

    phases = [
        ("Phase 1: Pre-shift (0-750)", 0, 750),
        ("Transition (700-800)", 700, 800),
        ("Phase 2: Post-shift (750-1500)", 750, 1500),
        ("Recovery (1200-1500)", 1200, 1500),
    ]

    print(f"\n{'Phase':<30} | {'Efficiency':>15} | {'Avg ef':>10} | {'Satisfaction':>15}")
    print("-" * 100)

    for phase_name, start, end in phases:
        metrics = tracker.get_phase_metrics(start, end)
        if metrics:
            print(f"{phase_name:<30} | {metrics['avg_efficiency']:>15.2f} | "
                  f"{metrics['avg_ef']:>10.1f} | {metrics['avg_satisfaction']:>14.1%}")

    return tracker


if __name__ == "__main__":
    tracker = run_ucb1_non_stationary(num_queries=1500)

    print("\n" + "=" * 100)
    print("SCENARIO 4 ANALYSIS")
    print("=" * 100)

    print("\nKey Findings:")
    print(f"  1. Environment: Non-stationary (pattern shift at query 750)")

    # Compare pre-shift vs post-shift performance
    phase1 = tracker.get_phase_metrics(0, 750)
    phase2 = tracker.get_phase_metrics(750, 1500)
    transition = tracker.get_phase_metrics(700, 850)

    if phase1 and phase2:
        eff_change = ((phase2['avg_efficiency'] - phase1['avg_efficiency']) /
                     phase1['avg_efficiency']) * 100
        print(f"\n  2. Phase 1 Efficiency (0-750): {phase1['avg_efficiency']:.2f} sat/sec")
        print(f"  3. Phase 2 Efficiency (750-1500): {phase2['avg_efficiency']:.2f} sat/sec")
        print(f"  4. Efficiency Change: {eff_change:+.1f}%")

    if transition:
        print(f"\n  5. Transition Period (700-850): {transition['avg_efficiency']:.2f} sat/sec")

    convergence = tracker.get_convergence_metrics()
    if convergence:
        print(f"\n  6. Final Q-value Change: {convergence.get('final_q_change', 0):.4f}")
        print(f"  7. Q-value Stability: {convergence.get('q_stability', 0):.1%}")

    # Analyze if UCB1 adapted
    recovery = tracker.get_phase_metrics(1200, 1500)
    if recovery and phase1:
        recovery_pct = (recovery['avg_efficiency'] / phase1['avg_efficiency']) * 100
        print(f"\n  8. Recovery Efficiency (1200-1500): {recovery['avg_efficiency']:.2f} sat/sec")
        print(f"  9. Recovery vs Pre-shift: {recovery_pct:.1f}%")

    print("\n  Interpretation:")
    if eff_change > -5:
        print("    [ROBUST] UCB1 maintained performance despite pattern shift")
    elif eff_change > -15:
        print("    [MODERATE] UCB1 showed some degradation but remained functional")
    else:
        print("    [POOR] UCB1 struggled with non-stationarity (expected for standard UCB1)")

    print("\n" + "=" * 100)
