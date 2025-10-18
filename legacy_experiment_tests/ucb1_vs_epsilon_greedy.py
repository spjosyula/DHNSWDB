"""UCB1 vs Epsilon-Greedy Comparison

Direct A/B test comparing UCB1 and epsilon-greedy exploration strategies
on the same dataset and queries.

Hypothesis:
    UCB1 should outperform epsilon-greedy due to:
    1. Systematic exploration (confidence bounds) vs random exploration
    2. Optimal exploration-exploitation tradeoff (theoretically proven)
    3. Better handling of large action spaces

Expected Results:
    - UCB1 converges faster (fewer queries to optimal ef_search)
    - UCB1 achieves 3-5% higher efficiency
    - UCB1 explores more systematically (tries all actions early)
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time
import numpy as np
from dynhnsw import VectorStore
from dynhnsw.config import DynHNSWConfig


def create_dataset(n_vectors=3000, dim=128):
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


def train_store(store, vectors, n_training_queries=60):
    """Train store with cold start and feedback."""
    # Cold start: activate clustering
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

    # Training with feedback
    for i in range(n_training_queries):
        if i < n_training_queries // 2:
            # Exploratory queries (Cluster 0)
            query = np.random.randn(128).astype(np.float32) * 3 + 15.0
            results = store.search(query, k=20)
            relevant_ids = [r["id"] for r in results]  # All relevant
        else:
            # Precise queries (Cluster 1)
            query = np.random.randn(128).astype(np.float32) * 3 - 15.0
            results = store.search(query, k=10)
            relevant_ids = [r["id"] for r in results[:5]]  # Top-5 relevant

        store.provide_feedback(relevant_ids=relevant_ids)


def evaluate_store(store, n_test_queries=50):
    """Evaluate store performance."""
    latencies = []
    satisfactions = []
    ef_values = []

    # Test exploratory queries
    for _ in range(n_test_queries // 2):
        query = np.random.randn(128).astype(np.float32) * 3 + 15.0
        k = 20
        expected_relevant = 20

        start = time.perf_counter()
        results = store.search(query, k=k)
        latency_ms = (time.perf_counter() - start) * 1000.0

        satisfaction = min(expected_relevant, len(results)) / len(results) if results else 0
        latencies.append(latency_ms)
        satisfactions.append(satisfaction)

        if hasattr(store._searcher, 'last_ef_used'):
            ef_values.append(store._searcher.last_ef_used)

    # Test precise queries
    for _ in range(n_test_queries // 2):
        query = np.random.randn(128).astype(np.float32) * 3 - 15.0
        k = 10
        expected_relevant = 5

        start = time.perf_counter()
        results = store.search(query, k=k)
        latency_ms = (time.perf_counter() - start) * 1000.0

        satisfaction = min(expected_relevant, len(results)) / len(results) if results else 0
        latencies.append(latency_ms)
        satisfactions.append(satisfaction)

        if hasattr(store._searcher, 'last_ef_used'):
            ef_values.append(store._searcher.last_ef_used)

    return {
        "avg_latency": np.mean(latencies),
        "avg_satisfaction": np.mean(satisfactions),
        "avg_efficiency": np.mean([s / (l/1000) for s, l in zip(satisfactions, latencies)]),
        "avg_ef": np.mean(ef_values) if ef_values else 0,
        "latencies": latencies,
        "satisfactions": satisfactions,
    }


def main():
    print("="*80)
    print("UCB1 vs EPSILON-GREEDY COMPARISON")
    print("="*80)

    # Create dataset
    print("\n[1] Creating dataset...")
    vectors = create_dataset(n_vectors=3000, dim=128)
    print(f"    Created {len(vectors)} vectors")

    # Create stores
    print("\n[2] Creating vector stores...")

    # Control: Epsilon-greedy (fixed)
    epsilon_config = DynHNSWConfig(
        config_name="epsilon_greedy",
        enable_ucb1=False,
        exploration_rate=0.15,
        enable_epsilon_decay=False
    )

    # Treatment: UCB1
    ucb1_config = DynHNSWConfig(
        config_name="ucb1",
        enable_ucb1=True,
        ucb1_exploration_constant=1.414
    )

    epsilon_store = VectorStore(
        dimension=128,
        config=epsilon_config,
        enable_intent_detection=True,
        k_intents=3
    )

    ucb1_store = VectorStore(
        dimension=128,
        config=ucb1_config,
        enable_intent_detection=True,
        k_intents=3
    )

    epsilon_store.add(vectors)
    ucb1_store.add(vectors)

    # Training phase
    print("\n[3] Training stores (60 queries with feedback)...")
    print("    Control: Epsilon-greedy (ε=0.15)")
    train_store(epsilon_store, vectors, n_training_queries=60)

    print("    Treatment: UCB1 (c=1.414)")
    train_store(ucb1_store, vectors, n_training_queries=60)

    # Show learned parameters
    print("\n[4] Learned ef_search values:")

    epsilon_stats = epsilon_store.get_statistics()
    if "ef_search_selection" in epsilon_stats:
        print("\n    Epsilon-greedy:")
        for intent_data in epsilon_stats["ef_search_selection"]["per_intent"]:
            if intent_data["num_queries"] > 0:
                print(f"      Intent {intent_data['intent_id']}: ef={intent_data['learned_ef']}, queries={intent_data['num_queries']}")

    ucb1_stats = ucb1_store.get_statistics()
    if "ef_search_selection" in ucb1_stats:
        print("\n    UCB1:")
        for intent_data in ucb1_stats["ef_search_selection"]["per_intent"]:
            if intent_data["num_queries"] > 0:
                print(f"      Intent {intent_data['intent_id']}: ef={intent_data['learned_ef']}, queries={intent_data['num_queries']}")

    # Evaluation phase
    print("\n" + "="*80)
    print("EVALUATION (50 test queries)")
    print("="*80)

    print("\n  Evaluating epsilon-greedy store...")
    epsilon_results = evaluate_store(epsilon_store, n_test_queries=50)

    print("  Evaluating UCB1 store...")
    ucb1_results = evaluate_store(ucb1_store, n_test_queries=50)

    # Results table
    print("\n  Results:")
    print(f"    {'Metric':<25} | {'Epsilon-Greedy':>20} | {'UCB1':>20} | {'Difference':>15}")
    print(f"    {'-'*25} | {'-'*20} | {'-'*20} | {'-'*15}")
    print(f"    {'Avg Latency (ms)':<25} | {epsilon_results['avg_latency']:>20.2f} | {ucb1_results['avg_latency']:>20.2f} | {(ucb1_results['avg_latency'] - epsilon_results['avg_latency']):>15.2f}")
    print(f"    {'Avg Satisfaction':<25} | {epsilon_results['avg_satisfaction']:>20.2%} | {ucb1_results['avg_satisfaction']:>20.2%} | {(ucb1_results['avg_satisfaction'] - epsilon_results['avg_satisfaction']):>15.2%}")
    print(f"    {'Avg Efficiency':<25} | {epsilon_results['avg_efficiency']:>20.2f} | {ucb1_results['avg_efficiency']:>20.2f} | {(ucb1_results['avg_efficiency'] - epsilon_results['avg_efficiency']):>15.2f}")
    print(f"    {'Avg ef_search':<25} | {epsilon_results['avg_ef']:>20.0f} | {ucb1_results['avg_ef']:>20.0f} | {(ucb1_results['avg_ef'] - epsilon_results['avg_ef']):>15.0f}")

    # Calculate improvement
    efficiency_improvement = ((ucb1_results['avg_efficiency'] - epsilon_results['avg_efficiency']) /
                              epsilon_results['avg_efficiency']) * 100
    latency_improvement = ((epsilon_results['avg_latency'] - ucb1_results['avg_latency']) /
                           epsilon_results['avg_latency']) * 100

    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)

    print(f"\nEfficiency Improvement:")
    if efficiency_improvement > 2:
        print(f"  [SUCCESS] UCB1 is {efficiency_improvement:.1f}% MORE EFFICIENT")
    elif efficiency_improvement < -2:
        print(f"  [WORSE] UCB1 is {abs(efficiency_improvement):.1f}% LESS EFFICIENT")
    else:
        print(f"  [NEUTRAL] {abs(efficiency_improvement):.1f}% difference (negligible)")

    print(f"\nLatency Improvement:")
    if latency_improvement > 2:
        print(f"  [SUCCESS] UCB1 is {latency_improvement:.1f}% FASTER")
    elif latency_improvement < -2:
        print(f"  [WORSE] UCB1 is {abs(latency_improvement):.1f}% SLOWER")
    else:
        print(f"  [NEUTRAL] {abs(latency_improvement):.1f}% difference (negligible)")

    # Exploration analysis
    print("\n" + "="*80)
    print("EXPLORATION ANALYSIS")
    print("="*80)

    if "ef_search_selection" in epsilon_stats and "ef_search_selection" in ucb1_stats:
        epsilon_per_intent = epsilon_stats["ef_search_selection"]["per_intent"]
        ucb1_per_intent = ucb1_stats["ef_search_selection"]["per_intent"]

        print("\nAction Distribution (per intent):")
        for i in range(3):
            if i < len(epsilon_per_intent) and i < len(ucb1_per_intent):
                print(f"\n  Intent {i}:")

                epsilon_actions = epsilon_per_intent[i]["action_counts"]
                ucb1_actions = ucb1_per_intent[i]["action_counts"]

                epsilon_unique = sum(1 for count in epsilon_actions.values() if count > 0)
                ucb1_unique = sum(1 for count in ucb1_actions.values() if count > 0)

                print(f"    Epsilon-greedy: Tried {epsilon_unique}/6 actions")
                print(f"    UCB1:           Tried {ucb1_unique}/6 actions")

    # Conclusion
    print("\n" + "="*80)
    print("CONCLUSION")
    print("="*80)

    if efficiency_improvement > 3:
        print("\n  [SUCCESS] UCB1 significantly outperforms epsilon-greedy")
        print("  Recommendation: Use UCB1 for production workloads")
    elif efficiency_improvement > 0:
        print("\n  [POSITIVE] UCB1 shows slight improvement over epsilon-greedy")
        print("  Recommendation: Consider UCB1 for long-running sessions")
    else:
        print("\n  [NEGATIVE] UCB1 does not outperform epsilon-greedy")
        print("  Possible reasons:")
        print("    - Action space too small (6 candidates)")
        print("    - Training queries insufficient")
        print("    - UCB1 exploration constant needs tuning")

    print("\n" + "="*80)


if __name__ == "__main__":
    main()
