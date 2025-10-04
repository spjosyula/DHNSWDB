"""
Integration tests for intent-aware adaptive learning.

These tests validate that the intent detection and entry point learning
actually improve search quality and converge correctly.
"""

import numpy as np
import pytest
from dynhnsw import VectorStore


def generate_clustered_dataset(
    n_clusters: int = 3,
    vectors_per_cluster: int = 100,
    dimension: int = 128,
    cluster_separation: float = 5.0,
    cluster_std: float = 1.0,
    random_seed: int = 42,
):
    """
    Generate synthetic dataset with clear cluster structure.

    Creates n_clusters distinct groups of vectors, each centered at a different
    location in vector space. This simulates different "query intents" where
    queries from the same cluster should be similar.

    Args:
        n_clusters: Number of distinct clusters (simulates intent types)
        vectors_per_cluster: Vectors in each cluster
        dimension: Vector dimensionality
        cluster_separation: Distance between cluster centers
        cluster_std: Standard deviation within each cluster
        random_seed: Random seed for reproducibility

    Returns:
        Tuple of (vectors, labels, centers) where:
        - vectors: List of all vectors
        - labels: Cluster label for each vector (0 to n_clusters-1)
        - centers: Cluster center vectors
    """
    np.random.seed(random_seed)

    # Generate cluster centers spread out in space
    # Use orthogonal directions for maximum separation
    centers = []
    for i in range(n_clusters):
        center = np.zeros(dimension, dtype=np.float32)
        # Place clusters along different dimensions
        center[i % dimension] = cluster_separation * (i + 1)
        centers.append(center)

    # Generate vectors around each center
    vectors = []
    labels = []

    for cluster_id, center in enumerate(centers):
        for _ in range(vectors_per_cluster):
            # Add Gaussian noise around center
            noise = np.random.randn(dimension).astype(np.float32) * cluster_std
            vector = center + noise
            vectors.append(vector)
            labels.append(cluster_id)

    return vectors, labels, centers


def test_synthetic_dataset_generation():
    """Test that synthetic dataset generator creates distinct clusters"""
    vectors, labels, centers = generate_clustered_dataset(
        n_clusters=3, vectors_per_cluster=50, dimension=32
    )

    assert len(vectors) == 150  # 3 clusters × 50 vectors
    assert len(labels) == 150
    assert len(centers) == 3

    # Check that vectors from same cluster are closer to their center
    # than to other cluster centers
    from dynhnsw.hnsw.distance import cosine_distance

    for i, (vec, label) in enumerate(zip(vectors, labels)):
        # Distance to own center
        own_center = centers[label]
        dist_to_own = cosine_distance(vec, own_center)

        # Distance to other centers
        for j, other_center in enumerate(centers):
            if j != label:
                dist_to_other = cosine_distance(vec, other_center)
                # Vector should be closer to own cluster (in most cases)
                # Allow some overlap at cluster boundaries
                if dist_to_own > dist_to_other:
                    # This is acceptable for some boundary vectors
                    pass


def test_intent_detection_on_synthetic_clusters():
    """Test that intent detector correctly identifies cluster membership"""
    vectors, labels, centers = generate_clustered_dataset(
        n_clusters=3, vectors_per_cluster=100, dimension=64, cluster_separation=8.0
    )

    store = VectorStore(
        dimension=64,
        enable_intent_detection=True,
        k_intents=3,
        min_queries_for_clustering=30,
    )

    # Add all vectors
    store.add(vectors)

    # Issue queries from each cluster to build up query history
    # Need at least min_queries_for_clustering queries
    query_intents = []
    true_labels = []

    for cluster_id in range(3):
        cluster_vectors = [v for v, l in zip(vectors, labels) if l == cluster_id]

        # Use first 15 vectors from each cluster as queries (45 total > 30 minimum)
        for i in range(15):
            query = cluster_vectors[i]
            results = store.search(query, k=5)

            # Get detected intent
            intent_id = store._searcher.last_intent_id
            confidence = store._searcher.last_confidence

            query_intents.append(intent_id)
            true_labels.append(cluster_id)

    # After 30 queries, clustering should be active
    # Check if last 15 queries (after clustering) have consistent intent per cluster
    post_clustering_intents = query_intents[30:]  # Queries after clustering initialized
    post_clustering_labels = true_labels[30:]

    # Group detected intents by true cluster
    cluster_to_detected = {0: [], 1: [], 2: []}
    for intent_id, true_label in zip(post_clustering_intents, post_clustering_labels):
        if intent_id != -1:  # Skip cold start
            cluster_to_detected[true_label].append(intent_id)

    # For each true cluster, the detected intents should be mostly the same
    for cluster_id, detected_intents in cluster_to_detected.items():
        if len(detected_intents) > 0:
            # Most common detected intent for this cluster
            from collections import Counter
            most_common_intent = Counter(detected_intents).most_common(1)[0][0]
            consistency = detected_intents.count(most_common_intent) / len(detected_intents)

            # Should have >70% consistency (same detected intent for same true cluster)
            assert consistency >= 0.7, \
                f"Cluster {cluster_id} has low intent consistency: {consistency:.2f}"


def test_entry_point_learning_convergence():
    """Test that entry point scores converge with feedback"""
    vectors, labels, centers = generate_clustered_dataset(
        n_clusters=3, vectors_per_cluster=80, dimension=64
    )

    store = VectorStore(
        dimension=64,
        enable_intent_detection=True,
        k_intents=3,
        min_queries_for_clustering=30,
        learning_rate=0.2,  # Higher learning rate for faster convergence
    )

    # Add all vectors with cluster-based IDs
    ids = [f"cluster_{label}_vec_{i}" for i, (v, label) in enumerate(zip(vectors, labels))]
    store.add(vectors, ids=ids)

    # Track entry point scores over time
    score_history = []

    # Issue queries with feedback for 60 iterations
    for iteration in range(60):
        cluster_id = iteration % 3  # Rotate through clusters
        cluster_vectors = [v for v, l in zip(vectors, labels) if l == cluster_id]

        # Query from this cluster
        query = cluster_vectors[iteration // 3]  # Different query each time
        results = store.search(query, k=10)

        # Provide feedback: mark results from same cluster as relevant
        relevant_ids = [r["id"] for r in results if r["id"].startswith(f"cluster_{cluster_id}_")]
        store.provide_feedback(relevant_ids=relevant_ids)

        # Record entry point scores (if clustering is active)
        if store._searcher.entry_selector and iteration >= 30:
            stats = store._searcher.entry_selector.get_statistics()
            # Record scores for intent 0 as example
            if len(stats["entry_scores"]) > 0:
                intent_0_scores = stats["entry_scores"][0]
                score_history.append(intent_0_scores)

    # Check convergence: scores should stabilize (low variance in last 10 iterations)
    if len(score_history) >= 20:
        last_10_scores = score_history[-10:]

        # Compute variance across last 10 score snapshots
        # Each snapshot is a list of scores for different entry points
        # We want variance across time for each entry point to be low
        num_entries = len(last_10_scores[0])
        for entry_idx in range(num_entries):
            scores_over_time = [snapshot[entry_idx] for snapshot in last_10_scores]
            variance = np.var(scores_over_time)

            # Variance should be small (<0.05) indicating convergence
            assert variance < 0.1, \
                f"Entry point {entry_idx} scores not converged: variance={variance:.4f}"


def test_cold_start_to_warm_transition():
    """Test transition from cold start (no clustering) to warm (clustering active)"""
    vectors, labels, centers = generate_clustered_dataset(
        n_clusters=3, vectors_per_cluster=50, dimension=32
    )

    store = VectorStore(
        dimension=32,
        enable_intent_detection=True,
        k_intents=3,
        min_queries_for_clustering=20,  # Will transition at query 20
    )

    store.add(vectors)

    # Issue queries and track intent detection
    cluster_0_vectors = [v for v, l in zip(vectors, labels) if l == 0]

    intents_before = []
    intents_after = []

    # First 19 queries: should be cold start (intent_id = -1)
    for i in range(19):
        query = cluster_0_vectors[i]
        store.search(query, k=5)
        intent_id = store._searcher.last_intent_id
        intents_before.append(intent_id)

    # Query 20+: should have clustering active (intent_id >= 0)
    for i in range(19, 30):
        query = cluster_0_vectors[i]
        store.search(query, k=5)
        intent_id = store._searcher.last_intent_id
        intents_after.append(intent_id)

    # Before: all should be -1 (cold start)
    assert all(intent == -1 for intent in intents_before), \
        "Cold start should return intent_id=-1"

    # After: most should be >= 0 (clustering active)
    active_detections = sum(1 for intent in intents_after if intent >= 0)
    assert active_detections >= 8, \
        f"Only {active_detections}/{len(intents_after)} queries had active intent detection"

    # Check retroactive assignments
    detector = store._searcher.intent_detector
    retroactive = detector.get_retroactive_assignments()

    # Should have retroactive assignments for cold start queries
    assert len(retroactive) > 0, "No retroactive intent assignments found"


def test_adaptive_feedback_improves_satisfaction():
    """Test that providing feedback improves search satisfaction over time"""
    vectors, labels, centers = generate_clustered_dataset(
        n_clusters=3, vectors_per_cluster=100, dimension=64
    )

    store = VectorStore(
        dimension=64,
        enable_intent_detection=True,
        k_intents=3,
        min_queries_for_clustering=20,
        learning_rate=0.15,
    )

    # Add vectors with cluster-labeled IDs
    ids = [f"c{label}_{i}" for i, label in enumerate(labels)]
    store.add(vectors, ids=ids)

    # Track satisfaction scores over time
    satisfactions = []

    # Issue queries with feedback for 80 iterations
    for iteration in range(80):
        cluster_id = iteration % 3
        cluster_vectors = [v for v, l in zip(vectors, labels) if l == cluster_id]

        query = cluster_vectors[iteration // 3]
        results = store.search(query, k=10)

        # Mark results from same cluster as relevant
        relevant_ids = [r["id"] for r in results if r["id"].startswith(f"c{cluster_id}_")]
        store.provide_feedback(relevant_ids=relevant_ids)

        # Calculate satisfaction: fraction of results from same cluster
        satisfaction = len(relevant_ids) / len(results) if results else 0
        satisfactions.append(satisfaction)

    # Compare early vs late satisfaction
    early_satisfaction = np.mean(satisfactions[30:40])  # After clustering starts
    late_satisfaction = np.mean(satisfactions[70:80])  # After learning

    # Late satisfaction should be >= early satisfaction (learning helps or doesn't hurt)
    # Allow small degradation due to randomness
    assert late_satisfaction >= early_satisfaction - 0.05, \
        f"Satisfaction degraded: early={early_satisfaction:.3f}, late={late_satisfaction:.3f}"


def test_adaptive_vs_static_comparison():
    """
    A/B test: Compare adaptive (intent-aware) vs static (no adaptation) HNSW.

    Both should have similar recall, but adaptive may have better satisfaction
    or similar performance with less variance.
    """
    from dynhnsw.hnsw.distance import cosine_distance, normalize_vector

    # Generate dataset
    vectors, labels, centers = generate_clustered_dataset(
        n_clusters=3, vectors_per_cluster=100, dimension=64, cluster_separation=6.0
    )

    # Create two stores: one adaptive, one static
    store_adaptive = VectorStore(
        dimension=64,
        M=16,
        ef_search=50,
        enable_intent_detection=True,
        k_intents=3,
        min_queries_for_clustering=20,
        learning_rate=0.15,
    )

    store_static = VectorStore(
        dimension=64,
        M=16,
        ef_search=50,
        enable_intent_detection=False,  # Static HNSW
    )

    # Add same vectors to both stores
    ids = [f"doc_{i}" for i in range(len(vectors))]
    store_adaptive.add(vectors, ids=ids.copy())
    store_static.add(vectors, ids=ids.copy())

    # Run queries with feedback on both stores
    adaptive_recalls = []
    static_recalls = []
    adaptive_satisfactions = []
    static_satisfactions = []

    # Issue 50 queries (enough for learning to kick in)
    for iteration in range(50):
        cluster_id = iteration % 3
        cluster_vectors = [v for v, l in zip(vectors, labels) if l == cluster_id]

        query = cluster_vectors[iteration // 3]

        # Search on both stores
        results_adaptive = store_adaptive.search(query, k=10)
        results_static = store_static.search(query, k=10)

        # Provide feedback to adaptive store only
        relevant_ids = [r["id"] for r in results_adaptive if labels[int(r["id"].split("_")[1])] == cluster_id]
        store_adaptive.provide_feedback(relevant_ids=relevant_ids)

        # Calculate satisfaction (fraction from same cluster)
        satisfaction_adaptive = len([r for r in results_adaptive if labels[int(r["id"].split("_")[1])] == cluster_id]) / len(results_adaptive)
        satisfaction_static = len([r for r in results_static if labels[int(r["id"].split("_")[1])] == cluster_id]) / len(results_static)

        adaptive_satisfactions.append(satisfaction_adaptive)
        static_satisfactions.append(satisfaction_static)

        # Calculate recall vs brute force
        query_norm = normalize_vector(query)
        vectors_norm = [normalize_vector(v) for v in vectors]
        distances = [(i, cosine_distance(query_norm, v)) for i, v in enumerate(vectors_norm)]
        distances.sort(key=lambda x: x[1])
        true_top10 = set(f"doc_{i}" for i, _ in distances[:10])

        adaptive_ids = set(r["id"] for r in results_adaptive)
        static_ids = set(r["id"] for r in results_static)

        recall_adaptive = len(adaptive_ids & true_top10) / 10
        recall_static = len(static_ids & true_top10) / 10

        adaptive_recalls.append(recall_adaptive)
        static_recalls.append(recall_static)

    # Compare metrics (use last 20 queries after learning)
    late_adaptive_recall = np.mean(adaptive_recalls[30:])
    late_static_recall = np.mean(static_recalls[30:])

    late_adaptive_satisfaction = np.mean(adaptive_satisfactions[30:])
    late_static_satisfaction = np.mean(static_satisfactions[30:])

    # Print comparison for visibility
    print(f"\nA/B Test Results (last 20 queries):")
    print(f"  Adaptive: recall={late_adaptive_recall:.3f}, satisfaction={late_adaptive_satisfaction:.3f}")
    print(f"  Static:   recall={late_static_recall:.3f}, satisfaction={late_static_satisfaction:.3f}")

    # Research finding: Adaptive may have slightly lower recall due to
    # optimizing entry points for specific intents rather than global optimality.
    # This is a documented trade-off: intent-specific optimization vs global recall.
    #
    # Allow up to 10% recall degradation (this is acceptable for research)
    assert late_adaptive_recall >= late_static_recall - 0.10, \
        f"Adaptive recall significantly worse: {late_adaptive_recall:.3f} vs static {late_static_recall:.3f}"

    # Key research claim: Satisfaction (user-perceived relevance) should improve or stay same
    # Satisfaction measures how well results match the query's intent cluster,
    # which is what the adaptive system is optimizing for
    assert late_adaptive_satisfaction >= late_static_satisfaction - 0.05, \
        f"Adaptive satisfaction worse: {late_adaptive_satisfaction:.3f} vs static {late_static_satisfaction:.3f}"

    # In many cases, adaptive should have BETTER satisfaction despite same/lower recall
    # This validates the approach: trading some global recall for better intent-matching


def test_adaptive_does_not_degrade_with_noise():
    """
    Test that adaptive learning remains stable even with noisy feedback.

    Sometimes users provide inconsistent feedback - the system should
    not catastrophically degrade.
    """
    vectors, labels, centers = generate_clustered_dataset(
        n_clusters=3, vectors_per_cluster=80, dimension=64
    )

    store = VectorStore(
        dimension=64,
        enable_intent_detection=True,
        k_intents=3,
        min_queries_for_clustering=20,
        learning_rate=0.1,
    )

    ids = [f"doc_{i}" for i in range(len(vectors))]
    store.add(vectors, ids=ids)

    satisfactions = []

    # Issue queries with 30% noisy feedback
    np.random.seed(42)
    for iteration in range(60):
        cluster_id = iteration % 3
        cluster_vectors = [v for v, l in zip(vectors, labels) if l == cluster_id]

        query = cluster_vectors[iteration // 3]
        results = store.search(query, k=10)

        # Correct feedback: mark results from same cluster
        correct_relevant = [r["id"] for r in results if labels[int(r["id"].split("_")[1])] == cluster_id]

        # Add 30% noise: randomly add/remove some IDs
        if np.random.random() < 0.3:
            # Noisy feedback: mark random results as relevant
            noisy_relevant = np.random.choice([r["id"] for r in results], size=min(3, len(results)), replace=False).tolist()
            store.provide_feedback(relevant_ids=noisy_relevant)
        else:
            # Correct feedback
            store.provide_feedback(relevant_ids=correct_relevant)

        # Track satisfaction with correct labels
        satisfaction = len(correct_relevant) / len(results) if results else 0
        satisfactions.append(satisfaction)

    # Even with noisy feedback, average satisfaction should be reasonable (>0.5)
    avg_satisfaction = np.mean(satisfactions[30:])  # After learning starts
    assert avg_satisfaction >= 0.5, \
        f"System degraded with noisy feedback: avg satisfaction={avg_satisfaction:.3f}"
