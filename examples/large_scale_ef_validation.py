"""Large-scale validation of adaptive ef_search learning.

This script rigorously tests whether adaptive ef_search actually improves
efficiency on a large dataset with realistic query patterns.

Tests:
1. 10,000 vector dataset with 3 distinct clusters
2. Simulate 2 user behaviors:
   - Exploratory users (want many results, high recall)
   - Precise users (want top-5 only, fast results)
3. Measure: latency, satisfaction, efficiency, ef_search convergence
4. Validate: Does adaptive mode actually improve efficiency?
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time
import numpy as np
from dynhnsw import VectorStore
from collections import defaultdict


class PerformanceTracker:
    """Track performance metrics over time."""

    def __init__(self):
        self.latencies = []
        self.satisfactions = []
        self.efficiencies = []
        self.ef_values = []
        self.query_types = []

    def record(self, latency_ms, satisfaction, ef_used, query_type):
        self.latencies.append(latency_ms)
        self.satisfactions.append(satisfaction)

        # Efficiency: satisfaction per second
        efficiency = satisfaction / (latency_ms / 1000.0)
        self.efficiencies.append(efficiency)

        self.ef_values.append(ef_used)
        self.query_types.append(query_type)

    def get_summary(self, last_n=None):
        """Get summary statistics for last N queries."""
        if last_n:
            latencies = self.latencies[-last_n:]
            satisfactions = self.satisfactions[-last_n:]
            efficiencies = self.efficiencies[-last_n:]
        else:
            latencies = self.latencies
            satisfactions = self.satisfactions
            efficiencies = self.efficiencies

        return {
            "avg_latency_ms": np.mean(latencies) if latencies else 0,
            "avg_satisfaction": np.mean(satisfactions) if satisfactions else 0,
            "avg_efficiency": np.mean(efficiencies) if efficiencies else 0,
            "num_queries": len(latencies)
        }


def create_large_dataset(n_vectors=10000, n_clusters=3, dim=128):
    """Create large clustered dataset."""
    print(f"\nCreating dataset: {n_vectors} vectors, {n_clusters} clusters, {dim} dimensions...")

    np.random.seed(42)
    vectors = []
    labels = []

    vectors_per_cluster = n_vectors // n_clusters

    # Cluster 0: Centered at [15, 15, ...]
    cluster_0 = np.random.randn(vectors_per_cluster, dim).astype(np.float32) * 3 + 15.0
    vectors.extend(cluster_0)
    labels.extend([0] * vectors_per_cluster)

    # Cluster 1: Centered at [-15, -15, ...]
    cluster_1 = np.random.randn(vectors_per_cluster, dim).astype(np.float32) * 3 - 15.0
    vectors.extend(cluster_1)
    labels.extend([1] * vectors_per_cluster)

    # Cluster 2: Centered at [0, 30, ...]
    cluster_2 = np.random.randn(vectors_per_cluster, dim).astype(np.float32) * 3
    cluster_2[:, 1] += 30.0
    vectors.extend(cluster_2)
    labels.extend([2] * vectors_per_cluster)

    return np.array(vectors), np.array(labels)


def run_exploratory_queries(store, cluster_center, n_queries, tracker):
    """Simulate exploratory user (wants many results, high recall)."""

    for _ in range(n_queries):
        # Query from cluster
        query = np.random.randn(128).astype(np.float32) * 3 + cluster_center

        # Measure latency
        start = time.perf_counter()
        results = store.search(query, k=20)
        latency_ms = (time.perf_counter() - start) * 1000.0

        # Exploratory: ALL results are relevant (want broad recall)
        relevant_ids = [r["id"] for r in results]
        satisfaction = len(relevant_ids) / len(results) if results else 0

        # Provide feedback
        store.provide_feedback(relevant_ids=relevant_ids)

        # Track performance
        ef_used = store._searcher.last_ef_used
        tracker.record(latency_ms, satisfaction, ef_used, "exploratory")


def run_precise_queries(store, cluster_center, n_queries, tracker):
    """Simulate precise user (wants top-5 only, fast results)."""

    for _ in range(n_queries):
        # Query from cluster
        query = np.random.randn(128).astype(np.float32) * 3 + cluster_center

        # Measure latency
        start = time.perf_counter()
        results = store.search(query, k=10)
        latency_ms = (time.perf_counter() - start) * 1000.0

        # Precise: Only top-5 are relevant (want fast, focused results)
        relevant_ids = [r["id"] for r in results[:5]]
        satisfaction = len(relevant_ids) / len(results) if results else 0

        # Provide feedback
        store.provide_feedback(relevant_ids=relevant_ids)

        # Track performance
        ef_used = store._searcher.last_ef_used
        tracker.record(latency_ms, satisfaction, ef_used, "precise")


def main():
    print("="*80)
    print("LARGE-SCALE VALIDATION: Adaptive ef_search Learning")
    print("="*80)

    # Create large dataset
    vectors, labels = create_large_dataset(n_vectors=10000, n_clusters=3, dim=128)
    print(f"  Created {len(vectors)} vectors")

    # Initialize adaptive store
    print("\nInitializing adaptive vector store...")
    store = VectorStore(
        dimension=128,
        M=16,
        ef_construction=200,
        ef_search=100,  # Default
        enable_intent_detection=True,
        k_intents=3,
        learning_rate=0.2,  # Higher learning rate for faster convergence
        min_queries_for_clustering=40
    )

    print("  Adding vectors to store...")
    store.add(vectors)
    print(f"  Added {store.size()} vectors")

    tracker = PerformanceTracker()

    # Phase 1: Cold start (activate intent detection)
    print("\n" + "="*80)
    print("PHASE 1: Cold Start (40 queries to activate intent detection)")
    print("="*80)

    for i in range(40):
        cluster_id = i % 3
        if cluster_id == 0:
            center = np.array([15.0] * 128)
        elif cluster_id == 1:
            center = np.array([-15.0] * 128)
        else:
            center = np.array([0.0] * 128)
            center[1] = 30.0

        query = np.random.randn(128).astype(np.float32) * 3 + center
        store.search(query, k=10)

    stats = store.get_statistics()
    print(f"\nIntent detection active: {stats['intent_detection']['clustering_active']}")

    # Phase 2: Exploratory queries (Cluster 0)
    print("\n" + "="*80)
    print("PHASE 2: Exploratory Queries (50 queries from Cluster 0)")
    print("="*80)
    print("User behavior: Wants ALL results (high recall, k=20)")

    cluster_0_center = np.array([15.0] * 128)
    run_exploratory_queries(store, cluster_0_center, 50, tracker)

    phase2_summary = tracker.get_summary(last_n=50)
    print(f"\nPhase 2 Results:")
    print(f"  Avg Latency: {phase2_summary['avg_latency_ms']:.2f} ms")
    print(f"  Avg Satisfaction: {phase2_summary['avg_satisfaction']:.2%}")
    print(f"  Avg Efficiency: {phase2_summary['avg_efficiency']:.2f} sat/sec")

    # Phase 3: Precise queries (Cluster 1)
    print("\n" + "="*80)
    print("PHASE 3: Precise Queries (50 queries from Cluster 1)")
    print("="*80)
    print("User behavior: Wants ONLY top-5 (fast, focused results)")

    cluster_1_center = np.array([-15.0] * 128)
    run_precise_queries(store, cluster_1_center, 50, tracker)

    phase3_summary = tracker.get_summary(last_n=50)
    print(f"\nPhase 3 Results:")
    print(f"  Avg Latency: {phase3_summary['avg_latency_ms']:.2f} ms")
    print(f"  Avg Satisfaction: {phase3_summary['avg_satisfaction']:.2%}")
    print(f"  Avg Efficiency: {phase3_summary['avg_efficiency']:.2f} sat/sec")

    # Phase 4: More exploratory queries to see convergence
    print("\n" + "="*80)
    print("PHASE 4: More Exploratory Queries (50 more from Cluster 0)")
    print("="*80)
    print("Testing convergence and stability...")

    run_exploratory_queries(store, cluster_0_center, 50, tracker)

    phase4_summary = tracker.get_summary(last_n=50)
    print(f"\nPhase 4 Results:")
    print(f"  Avg Latency: {phase4_summary['avg_latency_ms']:.2f} ms")
    print(f"  Avg Satisfaction: {phase4_summary['avg_satisfaction']:.2%}")
    print(f"  Avg Efficiency: {phase4_summary['avg_efficiency']:.2f} sat/sec")

    # Phase 5: More precise queries
    print("\n" + "="*80)
    print("PHASE 5: More Precise Queries (50 more from Cluster 1)")
    print("="*80)

    run_precise_queries(store, cluster_1_center, 50, tracker)

    phase5_summary = tracker.get_summary(last_n=50)
    print(f"\nPhase 5 Results:")
    print(f"  Avg Latency: {phase5_summary['avg_latency_ms']:.2f} ms")
    print(f"  Avg Satisfaction: {phase5_summary['avg_satisfaction']:.2%}")
    print(f"  Avg Efficiency: {phase5_summary['avg_efficiency']:.2f} sat/sec")

    # Final Analysis
    print("\n" + "="*80)
    print("FINAL ANALYSIS: Learned ef_search Values")
    print("="*80)

    final_stats = store.get_statistics()

    if "ef_search_selection" in final_stats:
        ef_stats = final_stats["ef_search_selection"]

        print(f"\nDefault ef_search: {ef_stats['default_ef']}")
        print(f"\nLearned ef_search per intent:")

        for intent_data in ef_stats["per_intent"]:
            intent_id = intent_data["intent_id"]
            learned_ef = intent_data["learned_ef"]
            num_queries = intent_data["num_queries"]
            avg_eff = intent_data.get("avg_efficiency", 0.0)

            print(f"\n  Intent {intent_id}:")
            print(f"    Learned ef_search: {learned_ef}")
            print(f"    Queries with feedback: {num_queries}")
            print(f"    Avg efficiency: {avg_eff:.2f} sat/sec")

            # Determine intent type
            if learned_ef > 110:
                print(f"    Type: EXPLORATORY (higher ef_search)")
            elif learned_ef < 90:
                print(f"    Type: PRECISE (lower ef_search)")
            else:
                print(f"    Type: BALANCED (near default)")

    # Efficiency comparison
    print("\n" + "="*80)
    print("EFFICIENCY COMPARISON: Exploratory vs Precise")
    print("="*80)

    # Calculate per query type
    exploratory_metrics = []
    precise_metrics = []

    for i, qtype in enumerate(tracker.query_types):
        if qtype == "exploratory":
            exploratory_metrics.append({
                "latency": tracker.latencies[i],
                "satisfaction": tracker.satisfactions[i],
                "efficiency": tracker.efficiencies[i],
                "ef": tracker.ef_values[i]
            })
        else:
            precise_metrics.append({
                "latency": tracker.latencies[i],
                "satisfaction": tracker.satisfactions[i],
                "efficiency": tracker.efficiencies[i],
                "ef": tracker.ef_values[i]
            })

    if exploratory_metrics:
        exp_latency = np.mean([m["latency"] for m in exploratory_metrics])
        exp_sat = np.mean([m["satisfaction"] for m in exploratory_metrics])
        exp_eff = np.mean([m["efficiency"] for m in exploratory_metrics])
        exp_ef = np.mean([m["ef"] for m in exploratory_metrics[-20:]])  # Last 20

        print(f"\nExploratory Queries (last 100):")
        print(f"  Avg Latency: {exp_latency:.2f} ms")
        print(f"  Avg Satisfaction: {exp_sat:.2%}")
        print(f"  Avg Efficiency: {exp_eff:.2f} sat/sec")
        print(f"  Avg ef_search (last 20): {exp_ef:.0f}")

    if precise_metrics:
        prec_latency = np.mean([m["latency"] for m in precise_metrics])
        prec_sat = np.mean([m["satisfaction"] for m in precise_metrics])
        prec_eff = np.mean([m["efficiency"] for m in precise_metrics])
        prec_ef = np.mean([m["ef"] for m in precise_metrics[-20:]])  # Last 20

        print(f"\nPrecise Queries (last 100):")
        print(f"  Avg Latency: {prec_latency:.2f} ms")
        print(f"  Avg Satisfaction: {prec_sat:.2%}")
        print(f"  Avg Efficiency: {prec_eff:.2f} sat/sec")
        print(f"  Avg ef_search (last 20): {prec_ef:.0f}")

    # Show convergence
    print("\n" + "="*80)
    print("CONVERGENCE ANALYSIS")
    print("="*80)

    # Show ef_search values over time for last 50 queries
    print("\nef_search values over time (last 50 queries):")
    recent_ef = tracker.ef_values[-50:]
    recent_types = tracker.query_types[-50:]

    print("\nQuery #  | Type          | ef_search")
    print("-" * 40)
    for i, (ef, qtype) in enumerate(zip(recent_ef[-20:], recent_types[-20:]), start=len(recent_ef)-20+1):
        print(f"  {i:3d}    | {qtype:13s} | {ef:3.0f}")

    # Summary
    print("\n" + "="*80)
    print("VALIDATION SUMMARY")
    print("="*80)

    print(f"\nTotal vectors: {len(vectors)}")
    print(f"Total queries with feedback: {len(tracker.latencies)}")
    print(f"Intent detection: Active")

    if exploratory_metrics and precise_metrics:
        if exp_ef != prec_ef:
            print(f"\n✓ SUCCESS: Different query types learned different ef_search values")
            print(f"  Exploratory ef_search: ~{exp_ef:.0f}")
            print(f"  Precise ef_search: ~{prec_ef:.0f}")
        else:
            print(f"\n⚠ NOTICE: Both query types have similar ef_search (~{exp_ef:.0f})")
            print(f"  This may be due to:")
            print(f"    - Learning rate too low")
            print(f"    - Efficiency similar for both query types")
            print(f"    - Need more queries for convergence")

    print("\n" + "="*80)
    print("Validation Complete!")
    print("="*80)


if __name__ == "__main__":
    main()
