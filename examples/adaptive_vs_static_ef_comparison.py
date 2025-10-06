"""Direct A/B comparison: Adaptive vs Static ef_search.

This script tests the SAME queries on both adaptive and static modes
to directly measure the performance difference.

Hypothesis:
- Adaptive mode should learn different ef_search values for different query types
- This should lead to better efficiency (satisfaction/latency ratio)
- Precise queries should be faster with adaptive mode
- Exploratory queries may have similar or slightly different performance
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time
import numpy as np
from dynhnsw import VectorStore


def create_dataset(n_vectors=5000, dim=128):
    """Create clustered dataset."""
    np.random.seed(42)
    vectors = []

    vectors_per_cluster = n_vectors // 3

    # Cluster 0
    cluster_0 = np.random.randn(vectors_per_cluster, dim).astype(np.float32) * 3 + 15.0
    vectors.extend(cluster_0)

    # Cluster 1
    cluster_1 = np.random.randn(vectors_per_cluster, dim).astype(np.float32) * 3 - 15.0
    vectors.extend(cluster_1)

    # Cluster 2
    cluster_2 = np.random.randn(vectors_per_cluster, dim).astype(np.float32) * 3
    cluster_2[:, 1] += 30.0
    vectors.extend(cluster_2)

    return np.array(vectors)


def train_adaptive_store(vectors):
    """Train adaptive store with feedback."""
    store = VectorStore(
        dimension=128,
        M=16,
        ef_construction=200,
        ef_search=100,
        enable_intent_detection=True,
        k_intents=3,
        learning_rate=0.2,
        min_queries_for_clustering=30
    )

    store.add(vectors)

    # Cold start
    for i in range(35):
        cluster_id = i % 3
        if cluster_id == 0:
            query = np.random.randn(128).astype(np.float32) * 3 + 15.0
        elif cluster_id == 1:
            query = np.random.randn(128).astype(np.float32) * 3 - 15.0
        else:
            query = np.random.randn(128).astype(np.float32) * 3
            query[1] += 30.0

        store.search(query, k=10)

    # Train with exploratory feedback (Cluster 0)
    for _ in range(30):
        query = np.random.randn(128).astype(np.float32) * 3 + 15.0
        results = store.search(query, k=20)
        relevant_ids = [r["id"] for r in results]  # All relevant
        store.provide_feedback(relevant_ids=relevant_ids)

    # Train with precise feedback (Cluster 1)
    for _ in range(30):
        query = np.random.randn(128).astype(np.float32) * 3 - 15.0
        results = store.search(query, k=10)
        relevant_ids = [r["id"] for r in results[:5]]  # Only top-5 relevant
        store.provide_feedback(relevant_ids=relevant_ids)

    return store


def create_static_store(vectors):
    """Create static store (no adaptation)."""
    store = VectorStore(
        dimension=128,
        M=16,
        ef_construction=200,
        ef_search=100,  # Fixed
        enable_intent_detection=False  # Static mode
    )

    store.add(vectors)
    return store


def run_test_queries(store, query_type, n_queries=50):
    """Run test queries and measure performance."""
    latencies = []
    satisfactions = []
    ef_values = []

    for _ in range(n_queries):
        if query_type == "exploratory":
            # Exploratory: from Cluster 0
            query = np.random.randn(128).astype(np.float32) * 3 + 15.0
            k = 20
            expected_relevant = 20  # Want all results
        else:
            # Precise: from Cluster 1
            query = np.random.randn(128).astype(np.float32) * 3 - 15.0
            k = 10
            expected_relevant = 5  # Want only top-5

        # Measure latency
        start = time.perf_counter()
        results = store.search(query, k=k)
        latency_ms = (time.perf_counter() - start) * 1000.0

        # Calculate satisfaction
        relevant_count = min(expected_relevant, len(results))
        satisfaction = relevant_count / len(results) if results else 0

        latencies.append(latency_ms)
        satisfactions.append(satisfaction)

        # Track ef_search used (adaptive only)
        if hasattr(store._searcher, 'last_ef_used'):
            ef_values.append(store._searcher.last_ef_used)
        else:
            ef_values.append(100)  # Static uses default

    return {
        "latencies": latencies,
        "satisfactions": satisfactions,
        "ef_values": ef_values,
        "avg_latency": np.mean(latencies),
        "avg_satisfaction": np.mean(satisfactions),
        "avg_efficiency": np.mean([s / (l/1000) for s, l in zip(satisfactions, latencies)]),
        "avg_ef": np.mean(ef_values)
    }


def main():
    print("="*80)
    print("A/B TEST: Adaptive vs Static ef_search")
    print("="*80)

    # Create dataset
    print("\n[1] Creating dataset...")
    vectors = create_dataset(n_vectors=5000, dim=128)
    print(f"    Created {len(vectors)} vectors")

    # Create and train adaptive store
    print("\n[2] Training adaptive store...")
    print("    - Running 35 cold-start queries")
    print("    - Training with 30 exploratory queries (Cluster 0)")
    print("    - Training with 30 precise queries (Cluster 1)")

    adaptive_store = train_adaptive_store(vectors)

    # Show learned ef_search values
    adaptive_stats = adaptive_store.get_statistics()
    if "ef_search_selection" in adaptive_stats:
        print("\n    Learned ef_search values:")
        for intent_data in adaptive_stats["ef_search_selection"]["per_intent"]:
            if intent_data["num_queries"] > 0:
                print(f"      Intent {intent_data['intent_id']}: ef={intent_data['learned_ef']}, queries={intent_data['num_queries']}")

    # Create static store
    print("\n[3] Creating static store (ef_search=100, no adaptation)...")
    static_store = create_static_store(vectors)

    # Test 1: Exploratory queries
    print("\n" + "="*80)
    print("TEST 1: Exploratory Queries (want all 20 results)")
    print("="*80)

    print("\n  Running 50 test queries on ADAPTIVE store...")
    adaptive_exp = run_test_queries(adaptive_store, "exploratory", n_queries=50)

    print(f"\n  Running 50 test queries on STATIC store...")
    static_exp = run_test_queries(static_store, "exploratory", n_queries=50)

    print(f"\n  Results:")
    print(f"    {'Metric':<20} | {'Adaptive':>15} | {'Static':>15} | {'Difference':>15}")
    print(f"    {'-'*20} | {'-'*15} | {'-'*15} | {'-'*15}")
    print(f"    {'Avg Latency (ms)':<20} | {adaptive_exp['avg_latency']:>15.2f} | {static_exp['avg_latency']:>15.2f} | {(adaptive_exp['avg_latency'] - static_exp['avg_latency']):>15.2f}")
    print(f"    {'Avg Satisfaction':<20} | {adaptive_exp['avg_satisfaction']:>15.2%} | {static_exp['avg_satisfaction']:>15.2%} | {(adaptive_exp['avg_satisfaction'] - static_exp['avg_satisfaction']):>15.2%}")
    print(f"    {'Avg Efficiency':<20} | {adaptive_exp['avg_efficiency']:>15.2f} | {static_exp['avg_efficiency']:>15.2f} | {(adaptive_exp['avg_efficiency'] - static_exp['avg_efficiency']):>15.2f}")
    print(f"    {'Avg ef_search':<20} | {adaptive_exp['avg_ef']:>15.0f} | {static_exp['avg_ef']:>15.0f} | {(adaptive_exp['avg_ef'] - static_exp['avg_ef']):>15.0f}")

    # Test 2: Precise queries
    print("\n" + "="*80)
    print("TEST 2: Precise Queries (want only top-5 results)")
    print("="*80)

    print("\n  Running 50 test queries on ADAPTIVE store...")
    adaptive_prec = run_test_queries(adaptive_store, "precise", n_queries=50)

    print(f"\n  Running 50 test queries on STATIC store...")
    static_prec = run_test_queries(static_store, "precise", n_queries=50)

    print(f"\n  Results:")
    print(f"    {'Metric':<20} | {'Adaptive':>15} | {'Static':>15} | {'Difference':>15}")
    print(f"    {'-'*20} | {'-'*15} | {'-'*15} | {'-'*15}")
    print(f"    {'Avg Latency (ms)':<20} | {adaptive_prec['avg_latency']:>15.2f} | {static_prec['avg_latency']:>15.2f} | {(adaptive_prec['avg_latency'] - static_prec['avg_latency']):>15.2f}")
    print(f"    {'Avg Satisfaction':<20} | {adaptive_prec['avg_satisfaction']:>15.2%} | {static_prec['avg_satisfaction']:>15.2%} | {(adaptive_prec['avg_satisfaction'] - static_prec['avg_satisfaction']):>15.2%}")
    print(f"    {'Avg Efficiency':<20} | {adaptive_prec['avg_efficiency']:>15.2f} | {static_prec['avg_efficiency']:>15.2f} | {(adaptive_prec['avg_efficiency'] - static_prec['avg_efficiency']):>15.2f}")
    print(f"    {'Avg ef_search':<20} | {adaptive_prec['avg_ef']:>15.0f} | {static_prec['avg_ef']:>15.0f} | {(adaptive_prec['avg_ef'] - static_prec['avg_ef']):>15.0f}")

    # Overall summary
    print("\n" + "="*80)
    print("OVERALL SUMMARY")
    print("="*80)

    # Calculate overall improvement
    total_adaptive_latency = np.mean([adaptive_exp['avg_latency'], adaptive_prec['avg_latency']])
    total_static_latency = np.mean([static_exp['avg_latency'], static_prec['avg_latency']])
    latency_improvement = ((total_static_latency - total_adaptive_latency) / total_static_latency) * 100

    total_adaptive_efficiency = np.mean([adaptive_exp['avg_efficiency'], adaptive_prec['avg_efficiency']])
    total_static_efficiency = np.mean([static_exp['avg_efficiency'], static_prec['avg_efficiency']])
    efficiency_improvement = ((total_adaptive_efficiency - total_static_efficiency) / total_static_efficiency) * 100

    print(f"\nOverall Performance (averaged across both query types):")
    print(f"  Latency:")
    print(f"    Adaptive: {total_adaptive_latency:.2f} ms")
    print(f"    Static:   {total_static_latency:.2f} ms")
    if latency_improvement > 0:
        print(f"    --> Adaptive is {latency_improvement:.1f}% FASTER")
    else:
        print(f"    --> Adaptive is {abs(latency_improvement):.1f}% SLOWER")

    print(f"\n  Efficiency (satisfaction/latency):")
    print(f"    Adaptive: {total_adaptive_efficiency:.2f} sat/sec")
    print(f"    Static:   {total_static_efficiency:.2f} sat/sec")
    if efficiency_improvement > 0:
        print(f"    --> Adaptive has {efficiency_improvement:.1f}% HIGHER efficiency")
    else:
        print(f"    --> Adaptive has {abs(efficiency_improvement):.1f}% LOWER efficiency")

    # Interpretation
    print("\n" + "="*80)
    print("INTERPRETATION")
    print("="*80)

    print("\nDid adaptive mode learn different ef_search values?")
    exp_ef_diff = abs(adaptive_exp['avg_ef'] - static_exp['avg_ef'])
    prec_ef_diff = abs(adaptive_prec['avg_ef'] - static_prec['avg_ef'])

    if exp_ef_diff > 5 or prec_ef_diff > 5:
        print("  [YES] Adaptive mode learned different ef_search values")
        print(f"    Exploratory: ef={adaptive_exp['avg_ef']:.0f} (static: {static_exp['avg_ef']:.0f})")
        print(f"    Precise:     ef={adaptive_prec['avg_ef']:.0f} (static: {static_prec['avg_ef']:.0f})")
    else:
        print("  [NO] ef_search values are similar to static")
        print("  Possible reasons:")
        print("    - Learning rate too low")
        print("    - Not enough training queries")
        print("    - Both query types have similar optimal ef_search")

    print("\nDid adaptive mode improve efficiency?")
    if efficiency_improvement > 2:
        print(f"  [YES] {efficiency_improvement:.1f}% improvement in efficiency")
    elif efficiency_improvement < -2:
        print(f"  [NO] {abs(efficiency_improvement):.1f}% decrease in efficiency")
    else:
        print(f"  [NEUTRAL] {abs(efficiency_improvement):.1f}% difference (negligible)")

    print("\nConclusion:")
    if efficiency_improvement > 2 and (exp_ef_diff > 5 or prec_ef_diff > 5):
        print("  [SUCCESS] Adaptive ef_search learning IS EFFECTIVE")
        print("    - Learned different values for different query types")
        print("    - Improved overall efficiency")
    elif exp_ef_diff > 5 or prec_ef_diff > 5:
        print("  [POTENTIAL] Adaptive ef_search learning shows promise")
        print("    - Successfully learned different values")
        print("    - But efficiency improvement is minimal")
        print("    - May need more training or tuning")
    else:
        print("  [NEEDS_WORK] Adaptive ef_search learning NOT YET EFFECTIVE")
        print("    - Did not learn significantly different values")
        print("    - Consider: higher learning rate, more training queries, or different feedback")

    print("\n" + "="*80)
    print("A/B Test Complete!")
    print("="*80)


if __name__ == "__main__":
    main()
