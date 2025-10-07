"""Run all UCB1 scenarios with optimized parameters.

Tests both baseline (c=1.414, no warm start) and optimized (c=0.5, warm start)
configurations to compare performance.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time
import json
import numpy as np
from dynhnsw import VectorStore
from dynhnsw.config import DynHNSWConfig
from dynhnsw.ef_search_selector import EfSearchSelector

# Import shared utilities
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from shared_utils import (
    UCB1ExperimentTracker,
    print_results_summary,
    generate_large_corpus,
    create_query_set,
    simulate_feedback,
)


def run_scenario_1_optimized(use_optimized: bool = True):
    """Scenario 1: Large Action Space (20 ef candidates)"""

    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        print("ERROR: sentence-transformers not installed!")
        return None

    variant = "optimized" if use_optimized else "baseline"
    print(f"\n{'='*100}")
    print(f"[SCENARIO 1 - {variant.upper()}] Large Action Space (20 candidates)")
    print(f"{'='*100}")

    # Configuration
    large_ef_candidates = [
        20, 30, 40, 50, 60, 75, 90, 100, 110, 125,
        140, 150, 165, 180, 200, 220, 250, 275, 300, 350
    ]

    c_value = 0.5 if use_optimized else 1.414
    use_warm_start = use_optimized

    print(f"  ef_candidates: {len(large_ef_candidates)} actions")
    print(f"  UCB1 c: {c_value}")
    print(f"  Warm start: {use_warm_start}")

    # Generate data
    corpus = generate_large_corpus(size=250)
    queries, query_types = create_query_set(300, 500, 200)
    queries = queries[:1000]
    query_types = query_types[:1000]

    # Embed
    model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = model.encode(corpus, convert_to_numpy=True, show_progress_bar=False)
    embeddings = embeddings.astype(np.float32)

    # Create store
    config = DynHNSWConfig(
        config_name=f"ucb1_large_space_{variant}",
        enable_ucb1=True,
        enable_ucb1_warm_start=use_warm_start,
        ucb1_exploration_constant=c_value,
        k_intents=3,
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

    # Override ef_candidates
    store._searcher.ef_selector = EfSearchSelector(
        k_intents=3,
        default_ef=100,
        use_ucb1=True,
        ucb1_c=c_value,
        use_warm_start=use_warm_start,
        ef_candidates=large_ef_candidates,
    )

    store.add(embeddings)

    # Run queries
    tracker = UCB1ExperimentTracker(f"Scenario1_{variant}")
    query_embeddings = model.encode(queries, convert_to_numpy=True, show_progress_bar=False)

    for i, (query_vec, query_type) in enumerate(zip(query_embeddings, query_types)):
        k = 15 if query_type == "exploratory" else (5 if query_type == "precise" else 10)

        start_time = time.perf_counter()
        results = store.search(query_vec, k=k)
        latency_ms = (time.perf_counter() - start_time) * 1000.0

        relevant_ids, satisfaction = simulate_feedback(query_type, results, k)
        if relevant_ids:
            store.provide_feedback(relevant_ids=relevant_ids)

        ef_used = store._searcher.last_ef_used
        intent_id = store._searcher.last_intent_id

        tracker.record(latency_ms, satisfaction, ef_used, query_type, ef_used, intent_id)

        if (i + 1) % 200 == 0:
            print(f"  Progress: {i+1}/1000")

    print_results_summary(tracker)
    tracker.save_results(f"ucb1_tests/results/scenario_1_{variant}.json")

    return tracker


def run_scenario_2_optimized(use_optimized: bool = True):
    """Scenario 2: Many Intents (7 clusters)"""

    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        return None

    variant = "optimized" if use_optimized else "baseline"
    print(f"\n{'='*100}")
    print(f"[SCENARIO 2 - {variant.upper()}] Many Intents (7 clusters)")
    print(f"{'='*100}")

    c_value = 0.5 if use_optimized else 1.414
    use_warm_start = use_optimized

    print(f"  k_intents: 7")
    print(f"  UCB1 c: {c_value}")
    print(f"  Warm start: {use_warm_start}")

    # Generate data
    corpus = generate_large_corpus(size=300)
    queries, query_types = create_query_set(250, 500, 250)
    queries = queries[:1000]
    query_types = query_types[:1000]

    # Embed
    model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = model.encode(corpus, convert_to_numpy=True, show_progress_bar=False)
    embeddings = embeddings.astype(np.float32)

    # Create store
    config = DynHNSWConfig(
        config_name=f"ucb1_many_intents_{variant}",
        enable_ucb1=True,
        enable_ucb1_warm_start=use_warm_start,
        ucb1_exploration_constant=c_value,
        k_intents=7,
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

    # Run queries
    tracker = UCB1ExperimentTracker(f"Scenario2_{variant}")
    query_embeddings = model.encode(queries, convert_to_numpy=True, show_progress_bar=False)

    for i, (query_vec, query_type) in enumerate(zip(query_embeddings, query_types)):
        k = 15 if query_type == "exploratory" else (5 if query_type == "precise" else 10)

        start_time = time.perf_counter()
        results = store.search(query_vec, k=k)
        latency_ms = (time.perf_counter() - start_time) * 1000.0

        relevant_ids, satisfaction = simulate_feedback(query_type, results, k)
        if relevant_ids:
            store.provide_feedback(relevant_ids=relevant_ids)

        ef_used = store._searcher.last_ef_used
        intent_id = store._searcher.last_intent_id

        tracker.record(latency_ms, satisfaction, ef_used, query_type, ef_used, intent_id)

        if (i + 1) % 200 == 0:
            print(f"  Progress: {i+1}/1000")

    print_results_summary(tracker)
    tracker.save_results(f"ucb1_tests/results/scenario_2_{variant}.json")

    return tracker


def run_scenario_3_optimized(use_optimized: bool = True):
    """Scenario 3: Long Horizon (5000 queries)"""

    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        return None

    variant = "optimized" if use_optimized else "baseline"
    print(f"\n{'='*100}")
    print(f"[SCENARIO 3 - {variant.upper()}] Long Horizon (5000 queries)")
    print(f"{'='*100}")

    c_value = 0.5 if use_optimized else 1.414
    use_warm_start = use_optimized

    print(f"  Queries: 5000")
    print(f"  UCB1 c: {c_value}")
    print(f"  Warm start: {use_warm_start}")

    # Generate data
    corpus = generate_large_corpus(size=400)
    queries, query_types = create_query_set(1500, 2500, 1000)
    queries = queries[:5000]
    query_types = query_types[:5000]

    # Embed
    model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = model.encode(corpus, convert_to_numpy=True, show_progress_bar=False)
    embeddings = embeddings.astype(np.float32)

    # Create store
    config = DynHNSWConfig(
        config_name=f"ucb1_long_horizon_{variant}",
        enable_ucb1=True,
        enable_ucb1_warm_start=use_warm_start,
        ucb1_exploration_constant=c_value,
        k_intents=3,
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

    # Run queries
    tracker = UCB1ExperimentTracker(f"Scenario3_{variant}")
    query_embeddings = model.encode(queries, convert_to_numpy=True, show_progress_bar=False)

    for i, (query_vec, query_type) in enumerate(zip(query_embeddings, query_types)):
        k = 15 if query_type == "exploratory" else (5 if query_type == "precise" else 10)

        start_time = time.perf_counter()
        results = store.search(query_vec, k=k)
        latency_ms = (time.perf_counter() - start_time) * 1000.0

        relevant_ids, satisfaction = simulate_feedback(query_type, results, k)
        if relevant_ids:
            store.provide_feedback(relevant_ids=relevant_ids)

        ef_used = store._searcher.last_ef_used
        intent_id = store._searcher.last_intent_id

        tracker.record(latency_ms, satisfaction, ef_used, query_type, ef_used, intent_id)

        if (i + 1) % 500 == 0:
            print(f"  Progress: {i+1}/5000")

    print_results_summary(tracker)
    tracker.save_results(f"ucb1_tests/results/scenario_3_{variant}.json")

    return tracker


def run_scenario_4_optimized(use_optimized: bool = True):
    """Scenario 4: Non-Stationary (pattern shift at query 750)"""

    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        return None

    variant = "optimized" if use_optimized else "baseline"
    print(f"\n{'='*100}")
    print(f"[SCENARIO 4 - {variant.upper()}] Non-Stationary (pattern shift)")
    print(f"{'='*100}")

    c_value = 0.5 if use_optimized else 1.414
    use_warm_start = use_optimized

    print(f"  Pattern shift at query 750")
    print(f"  UCB1 c: {c_value}")
    print(f"  Warm start: {use_warm_start}")

    # Generate data with phase shift
    corpus = generate_large_corpus(size=350)

    # Phase 1: Exploratory-heavy
    queries_p1, types_p1 = create_query_set(525, 150, 75)
    queries_p1 = queries_p1[:750]
    types_p1 = types_p1[:750]

    # Phase 2: Precise-heavy
    queries_p2, types_p2 = create_query_set(150, 525, 75)
    queries_p2 = queries_p2[:750]
    types_p2 = types_p2[:750]

    queries = queries_p1 + queries_p2
    query_types = types_p1 + types_p2

    # Embed
    model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = model.encode(corpus, convert_to_numpy=True, show_progress_bar=False)
    embeddings = embeddings.astype(np.float32)

    # Create store
    config = DynHNSWConfig(
        config_name=f"ucb1_non_stationary_{variant}",
        enable_ucb1=True,
        enable_ucb1_warm_start=use_warm_start,
        ucb1_exploration_constant=c_value,
        k_intents=3,
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

    # Run queries
    tracker = UCB1ExperimentTracker(f"Scenario4_{variant}")
    query_embeddings = model.encode(queries, convert_to_numpy=True, show_progress_bar=False)

    for i, (query_vec, query_type) in enumerate(zip(query_embeddings, query_types)):
        k = 15 if query_type == "exploratory" else (5 if query_type == "precise" else 10)

        start_time = time.perf_counter()
        results = store.search(query_vec, k=k)
        latency_ms = (time.perf_counter() - start_time) * 1000.0

        relevant_ids, satisfaction = simulate_feedback(query_type, results, k)
        if relevant_ids:
            store.provide_feedback(relevant_ids=relevant_ids)

        ef_used = store._searcher.last_ef_used
        intent_id = store._searcher.last_intent_id

        tracker.record(latency_ms, satisfaction, ef_used, query_type, ef_used, intent_id)

        if (i + 1) % 150 == 0:
            phase = "Phase 1" if i < 750 else "Phase 2"
            print(f"  Progress [{phase}]: {i+1}/1500")

    print_results_summary(tracker)
    tracker.save_results(f"ucb1_tests/results/scenario_4_{variant}.json")

    # Phase analysis
    print(f"\n{'='*80}")
    print("PHASE ANALYSIS")
    print(f"{'='*80}")

    phases = [
        ("Pre-shift (0-750)", 0, 750),
        ("Post-shift (750-1500)", 750, 1500),
    ]

    print(f"\n{'Phase':<25} | {'Efficiency':>15}")
    print("-" * 45)

    for phase_name, start, end in phases:
        metrics = tracker.get_phase_metrics(start, end)
        if metrics:
            print(f"{phase_name:<25} | {metrics['avg_efficiency']:>15.2f}")

    return tracker


if __name__ == "__main__":
    print("\n" + "="*100)
    print("UCB1 OPTIMIZATION TEST SUITE")
    print("="*100)
    print("\nTesting two configurations:")
    print("  BASELINE:  c=1.414 (sqrt(2)), no warm start")
    print("  OPTIMIZED: c=0.5, warm start with HNSW priors")
    print("\n" + "="*100)

    results = {}

    # Run all scenarios with both configurations
    for scenario_num, scenario_func in enumerate([
        run_scenario_1_optimized,
        run_scenario_2_optimized,
        run_scenario_3_optimized,
        run_scenario_4_optimized,
    ], 1):
        print(f"\n\n{'#'*100}")
        print(f"SCENARIO {scenario_num}")
        print(f"{'#'*100}")

        # Baseline
        baseline_tracker = scenario_func(use_optimized=False)
        if baseline_tracker:
            results[f"scenario_{scenario_num}_baseline"] = baseline_tracker.get_metrics()

        # Optimized
        optimized_tracker = scenario_func(use_optimized=True)
        if optimized_tracker:
            results[f"scenario_{scenario_num}_optimized"] = optimized_tracker.get_metrics()

        # Compare
        if baseline_tracker and optimized_tracker:
            baseline_eff = baseline_tracker.get_metrics()["avg_efficiency"]
            optimized_eff = optimized_tracker.get_metrics()["avg_efficiency"]
            improvement = ((optimized_eff - baseline_eff) / baseline_eff) * 100

            print(f"\n{'='*80}")
            print(f"SCENARIO {scenario_num} COMPARISON")
            print(f"{'='*80}")
            print(f"  Baseline:  {baseline_eff:.2f} sat/sec")
            print(f"  Optimized: {optimized_eff:.2f} sat/sec")
            print(f"  Change:    {improvement:+.1f}%")

            if improvement > 5:
                print(f"  ✅ SIGNIFICANT IMPROVEMENT")
            elif improvement > 0:
                print(f"  ⚠️  MINOR IMPROVEMENT")
            else:
                print(f"  ❌ NO IMPROVEMENT")

    # Save comparison results
    with open("ucb1_tests/results/optimization_comparison.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n\n{'='*100}")
    print("TEST SUITE COMPLETE")
    print(f"{'='*100}")
    print("\nResults saved to: ucb1_tests/results/optimization_comparison.json")
