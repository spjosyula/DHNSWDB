"""Track and visualize ef_search convergence over time.

This script shows how ef_search values converge to optimal values
for different query intents through learning.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from dynhnsw import VectorStore


def create_dataset(n_vectors=3000, dim=128):
    """Create simple 3-cluster dataset."""
    np.random.seed(42)
    vectors = []

    for i in range(3):
        if i == 0:
            cluster = np.random.randn(1000, dim).astype(np.float32) * 3 + 15.0
        elif i == 1:
            cluster = np.random.randn(1000, dim).astype(np.float32) * 3 - 15.0
        else:
            cluster = np.random.randn(1000, dim).astype(np.float32) * 3
            cluster[:, 1] += 30.0
        vectors.extend(cluster)

    return np.array(vectors)


def plot_text_chart(values, width=60, title=""):
    """Simple text-based chart."""
    if not values:
        return

    print(f"\n{title}")
    print("="*80)

    min_val = min(values)
    max_val = max(values)

    # Add some padding
    range_val = max_val - min_val
    if range_val == 0:
        range_val = 1

    min_val -= range_val * 0.1
    max_val += range_val * 0.1

    # Scale values
    scaled = [(v - min_val) / (max_val - min_val) * width for v in values]

    # Print chart
    print(f"Value range: {min_val:.1f} to {max_val:.1f}")
    print()

    for i, (val, scaled_val) in enumerate(zip(values, scaled)):
        bar = "#" * int(scaled_val)
        print(f"  {i+1:3d} | {bar} {val:.1f}")

    print()


def main():
    print("="*80)
    print("ef_search Convergence Tracker")
    print("="*80)

    # Create dataset
    print("\n[1] Creating dataset...")
    vectors = create_dataset(n_vectors=3000, dim=128)
    print(f"    {len(vectors)} vectors in 3 clusters")

    # Create store
    print("\n[2] Initializing adaptive store...")
    store = VectorStore(
        dimension=128,
        M=16,
        ef_construction=200,
        ef_search=100,
        enable_intent_detection=True,
        k_intents=3,
        learning_rate=0.25,  # Higher for faster convergence
        min_queries_for_clustering=30
    )

    store.add(vectors)

    # Track ef_search values over time
    exploratory_ef_history = []
    precise_ef_history = []
    query_numbers = []

    # Cold start
    print("\n[3] Cold start (30 queries)...")
    for i in range(30):
        cluster_id = i % 3
        if cluster_id == 0:
            query = np.random.randn(128).astype(np.float32) * 3 + 15.0
        elif cluster_id == 1:
            query = np.random.randn(128).astype(np.float32) * 3 - 15.0
        else:
            query = np.random.randn(128).astype(np.float32) * 3
            query[1] += 30.0

        store.search(query, k=10)

    print("    Intent detection activated")

    # Training phase - alternating query types
    print("\n[4] Training phase (100 queries with feedback)...")
    print("    Alternating between exploratory and precise queries...")

    for i in range(100):
        if i % 2 == 0:
            # Exploratory query (Cluster 0)
            query = np.random.randn(128).astype(np.float32) * 3 + 15.0
            results = store.search(query, k=20)
            relevant_ids = [r["id"] for r in results]  # All relevant
            store.provide_feedback(relevant_ids=relevant_ids)

            # Track ef_search
            ef_used = store._searcher.last_ef_used
            exploratory_ef_history.append(ef_used)
            query_numbers.append(i + 1)

        else:
            # Precise query (Cluster 1)
            query = np.random.randn(128).astype(np.float32) * 3 - 15.0
            results = store.search(query, k=10)
            relevant_ids = [r["id"] for r in results[:5]]  # Only top-5
            store.provide_feedback(relevant_ids=relevant_ids)

            # Track ef_search
            ef_used = store._searcher.last_ef_used
            precise_ef_history.append(ef_used)

    # Show convergence
    print("\n" + "="*80)
    print("CONVERGENCE RESULTS")
    print("="*80)

    # Show learned values
    stats = store.get_statistics()
    if "ef_search_selection" in stats:
        print("\nFinal learned ef_search values:")
        for intent_data in stats["ef_search_selection"]["per_intent"]:
            if intent_data["num_queries"] > 0:
                print(f"  Intent {intent_data['intent_id']}: ef={intent_data['learned_ef']}, queries={intent_data['num_queries']}")

    # Show history
    print(f"\nExploratory queries (first 10, last 10):")
    print(f"  First 10 ef_search values: {[int(v) for v in exploratory_ef_history[:10]]}")
    print(f"  Last  10 ef_search values: {[int(v) for v in exploratory_ef_history[-10:]]}")

    print(f"\nPrecise queries (first 10, last 10):")
    print(f"  First 10 ef_search values: {[int(v) for v in precise_ef_history[:10]]}")
    print(f"  Last  10 ef_search values: {[int(v) for v in precise_ef_history[-10:]]}")

    # Show convergence visually
    if exploratory_ef_history:
        plot_text_chart(exploratory_ef_history, title="Exploratory ef_search Over Time")

    if precise_ef_history:
        plot_text_chart(precise_ef_history, title="Precise ef_search Over Time")

    # Analysis
    print("\n" + "="*80)
    print("CONVERGENCE ANALYSIS")
    print("="*80)

    if exploratory_ef_history and precise_ef_history:
        # Check convergence
        exp_early = np.mean(exploratory_ef_history[:10])
        exp_late = np.mean(exploratory_ef_history[-10:])
        exp_variance = np.var(exploratory_ef_history[-10:])

        prec_early = np.mean(precise_ef_history[:10])
        prec_late = np.mean(precise_ef_history[-10:])
        prec_variance = np.var(precise_ef_history[-10:])

        print(f"\nExploratory queries:")
        print(f"  Early avg (first 10): {exp_early:.1f}")
        print(f"  Late avg (last 10):   {exp_late:.1f}")
        print(f"  Change:               {exp_late - exp_early:+.1f}")
        print(f"  Late variance:        {exp_variance:.2f}")

        if exp_variance < 10:
            print(f"  --> CONVERGED (low variance)")
        else:
            print(f"  --> STILL LEARNING (high variance)")

        print(f"\nPrecise queries:")
        print(f"  Early avg (first 10): {prec_early:.1f}")
        print(f"  Late avg (last 10):   {prec_late:.1f}")
        print(f"  Change:               {prec_late - prec_early:+.1f}")
        print(f"  Late variance:        {prec_variance:.2f}")

        if prec_variance < 10:
            print(f"  --> CONVERGED (low variance)")
        else:
            print(f"  --> STILL LEARNING (high variance)")

        # Check differentiation
        print(f"\nDifferentiation between query types:")
        diff = abs(exp_late - prec_late)
        print(f"  Exploratory ef (late): {exp_late:.1f}")
        print(f"  Precise ef (late):     {prec_late:.1f}")
        print(f"  Difference:            {diff:.1f}")

        if diff > 10:
            print(f"  --> SIGNIFICANT differentiation")
        elif diff > 5:
            print(f"  --> MODERATE differentiation")
        else:
            print(f"  --> NO differentiation")

    print("\n" + "="*80)
    print("Tracking Complete!")
    print("="*80)


if __name__ == "__main__":
    main()
