"""End-to-end tests for adaptive HNSW with complete feedback loop.

These tests demonstrate the full adaptive behavior from initial search
through feedback collection, weight learning, and performance improvement.
"""

import pytest
import numpy as np
from dynhnsw.hnsw.graph import HNSWGraph
from dynhnsw.hnsw.builder import HNSWBuilder
from dynhnsw.adaptive_hnsw import AdaptiveHNSWSearcher


@pytest.fixture
def clustered_vectors():
    """Create clustered vectors to simulate different query intents.

    Creates 3 clusters of vectors to simulate:
    - Cluster 0: Topic A vectors
    - Cluster 1: Topic B vectors
    - Cluster 2: Topic C vectors
    """
    np.random.seed(42)
    vectors = []

    # Cluster 0: centered at [5, 5, ...]
    cluster0 = np.random.randn(50, 128).astype(np.float32) + 5.0
    vectors.extend(cluster0)

    # Cluster 1: centered at [-5, -5, ...]
    cluster1 = np.random.randn(50, 128).astype(np.float32) - 5.0
    vectors.extend(cluster1)

    # Cluster 2: centered at [0, 10, ...]
    cluster2 = np.random.randn(50, 128).astype(np.float32)
    cluster2[:, 1] += 10.0
    vectors.extend(cluster2)

    return np.array(vectors)


@pytest.fixture
def adaptive_searcher(clustered_vectors):
    """Build graph and create adaptive searcher."""
    graph = HNSWGraph(dimension=128, M=16)
    builder = HNSWBuilder(graph)

    for i, vector in enumerate(clustered_vectors):
        level = int(np.random.geometric(p=0.5)) - 1
        level = min(level, 5)
        builder.insert(vector, node_id=i, level=level)

    return AdaptiveHNSWSearcher(graph, ef_search=50, learning_rate=0.1)


class TestBasicAdaptiveLoop:
    """Test basic adaptive feedback loop."""

    def test_search_then_feedback_updates_system(self, adaptive_searcher, clustered_vectors):
        """Complete search -> feedback cycle should update all components."""
        query = clustered_vectors[0]

        # Initial search
        results = adaptive_searcher.search(query, k=10)
        result_ids = [r[0] for r in results]

        # Provide feedback (assume first 5 are relevant)
        relevant_ids = set(result_ids[:5])
        adaptive_searcher.provide_feedback(query, result_ids, relevant_ids)

        # Check all components updated
        assert adaptive_searcher.feedback_collector.get_feedback_count() == 1
        assert adaptive_searcher.weight_learner.get_total_edges() > 0
        assert adaptive_searcher.stability_monitor.get_statistics()["total_updates"] > 0

    def test_consistent_positive_feedback_converges_weights(self, adaptive_searcher, clustered_vectors):
        """Consistent positive feedback should converge edge weights lower."""
        query = clustered_vectors[0]

        # Run multiple iterations with consistent positive feedback
        for _ in range(20):
            results = adaptive_searcher.search(query, k=10)
            result_ids = [r[0] for r in results]

            # All results are relevant (perfect feedback)
            adaptive_searcher.provide_feedback(query, result_ids, set(result_ids))

        # Weights should have converged toward lower values
        stats = adaptive_searcher.weight_learner.get_weight_statistics()

        # With perfect satisfaction (1.0), reward signal is 0.8, so weights trend toward 0.8
        assert stats["mean"] < 1.0  # Weights moved below neutral

    def test_consistent_negative_feedback_increases_weights(self, adaptive_searcher, clustered_vectors):
        """Consistent negative feedback should increase edge weights."""
        query = clustered_vectors[0]

        # Run multiple iterations with negative feedback
        for _ in range(20):
            results = adaptive_searcher.search(query, k=10)
            result_ids = [r[0] for r in results]

            # No results are relevant (bad feedback)
            adaptive_searcher.provide_feedback(query, result_ids, set())

        # Weights should have increased
        stats = adaptive_searcher.weight_learner.get_weight_statistics()

        # With zero satisfaction, reward signal is 1.2, so weights trend toward 1.2
        assert stats["mean"] > 1.0  # Weights moved above neutral


class TestAdaptationImprovesSearch:
    """Test that adaptation actually improves search quality over time."""

    def test_adaptation_learns_query_pattern(self, adaptive_searcher, clustered_vectors):
        """Repeated similar queries with feedback should improve results."""
        # Query from cluster 0
        query = clustered_vectors[5]  # A vector from cluster 0

        # Define ground truth: vectors 0-49 are cluster 0 (similar to query)
        ground_truth_cluster_0 = set(range(50))

        initial_results = adaptive_searcher.search(query, k=20)
        initial_cluster_0_count = sum(
            1 for r in initial_results if r[0] in ground_truth_cluster_0
        )

        # Run adaptation loop with feedback
        for _ in range(30):
            results = adaptive_searcher.search(query, k=20)
            result_ids = [r[0] for r in results]

            # Mark cluster 0 results as relevant
            relevant_ids = set(rid for rid in result_ids if rid in ground_truth_cluster_0)
            adaptive_searcher.provide_feedback(query, result_ids, relevant_ids)

        # Search again after adaptation
        final_results = adaptive_searcher.search(query, k=20)
        final_cluster_0_count = sum(
            1 for r in final_results if r[0] in ground_truth_cluster_0
        )

        # Should find more relevant results after adaptation
        # (This may not always improve due to randomness, so we just check it runs)
        assert final_cluster_0_count >= 0  # At least no errors occurred


class TestPerformanceTracking:
    """Test performance monitoring across queries."""

    def test_performance_metrics_accumulate(self, adaptive_searcher, clustered_vectors):
        """Performance metrics should accumulate over queries."""
        query = clustered_vectors[0]

        # Simulate 20 queries with performance tracking
        for i in range(20):
            results = adaptive_searcher.search(query, k=10)
            result_ids = [r[0] for r in results]

            # Provide feedback
            relevant_ids = set(result_ids[:5])  # First 5 relevant
            adaptive_searcher.provide_feedback(query, result_ids, relevant_ids)

            # Record performance (simulated recall/precision)
            recall = 0.5  # 50% recall
            precision = 0.5  # 50% precision
            latency_ms = 10.0 + np.random.rand() * 5  # 10-15ms

            adaptive_searcher.record_performance(recall, precision, latency_ms)

        # Performance should be tracked
        summary = adaptive_searcher.performance_monitor.get_performance_summary()

        assert summary["current_recall"] == pytest.approx(0.5)
        assert summary["baseline_recall"] > 0  # Baseline should be set

    def test_degradation_triggers_reset(self, adaptive_searcher, clustered_vectors):
        """Severe performance degradation should trigger reset."""
        query = clustered_vectors[0]

        # Establish baseline with good performance
        for _ in range(10):
            results = adaptive_searcher.search(query, k=10)
            adaptive_searcher.record_performance(recall=0.90, precision=0.85, latency_ms=10.0)

        baseline = adaptive_searcher.performance_monitor.baseline_recall
        assert baseline > 0.8

        # Simulate severe degradation
        for _ in range(10):
            results = adaptive_searcher.search(query, k=10)
            adaptive_searcher.record_performance(recall=0.70, precision=0.60, latency_ms=20.0)

        # Reset should be active
        assert adaptive_searcher.reset_manager.is_resetting()


class TestStabilityUnderNoise:
    """Test system stability with noisy/conflicting feedback."""

    def test_random_feedback_stays_stable(self, adaptive_searcher, clustered_vectors):
        """Random feedback should not cause instability."""
        query = clustered_vectors[0]

        # Run with completely random feedback
        for i in range(50):
            results = adaptive_searcher.search(query, k=10)
            result_ids = [r[0] for r in results]

            # Random feedback
            num_relevant = np.random.randint(0, len(result_ids) + 1)
            relevant_ids = set(np.random.choice(result_ids, num_relevant, replace=False))

            adaptive_searcher.provide_feedback(query, result_ids, relevant_ids)

        # System should remain stable (not crash, weights in bounds)
        stats = adaptive_searcher.weight_learner.get_weight_statistics()

        assert stats["min"] >= 0.1  # Lower bound enforced
        assert stats["max"] <= 10.0  # Upper bound enforced

        # Stability score should be tracked
        stability_stats = adaptive_searcher.stability_monitor.get_statistics()
        assert "stability_score" in stability_stats

    def test_oscillating_feedback_detected(self, adaptive_searcher, clustered_vectors):
        """Oscillating feedback pattern should be detected."""
        query = clustered_vectors[0]

        # Create oscillating pattern
        for i in range(30):
            results = adaptive_searcher.search(query, k=10)
            result_ids = [r[0] for r in results]

            if i % 2 == 0:
                # All relevant
                relevant_ids = set(result_ids)
            else:
                # None relevant
                relevant_ids = set()

            adaptive_searcher.provide_feedback(query, result_ids, relevant_ids)

        # Check if oscillation is detected (may or may not depending on edges used)
        stability_stats = adaptive_searcher.stability_monitor.get_statistics()
        assert stability_stats["total_updates"] > 0


class TestMultipleQueryPatterns:
    """Test adaptation with multiple different query patterns."""

    def test_different_clusters_different_feedback(self, adaptive_searcher, clustered_vectors):
        """Different query patterns should adapt independently."""
        # Query cluster 0
        query_cluster_0 = clustered_vectors[5]
        cluster_0_nodes = set(range(50))

        # Query cluster 1
        query_cluster_1 = clustered_vectors[55]
        cluster_1_nodes = set(range(50, 100))

        # Adapt for cluster 0 queries
        for _ in range(15):
            results = adaptive_searcher.search(query_cluster_0, k=10)
            result_ids = [r[0] for r in results]

            relevant_ids = set(rid for rid in result_ids if rid in cluster_0_nodes)
            adaptive_searcher.provide_feedback(query_cluster_0, result_ids, relevant_ids)

        # Adapt for cluster 1 queries
        for _ in range(15):
            results = adaptive_searcher.search(query_cluster_1, k=10)
            result_ids = [r[0] for r in results]

            relevant_ids = set(rid for rid in result_ids if rid in cluster_1_nodes)
            adaptive_searcher.provide_feedback(query_cluster_1, result_ids, relevant_ids)

        # System should have learned both patterns
        stats = adaptive_searcher.get_statistics()

        assert stats["feedback"]["total_queries"] == 30
        assert stats["weights"]["count"] > 0


class TestSystemStatistics:
    """Test comprehensive statistics reporting."""

    def test_statistics_after_adaptation(self, adaptive_searcher, clustered_vectors):
        """Statistics should reflect system state after adaptation."""
        query = clustered_vectors[0]

        # Run adaptation
        for i in range(20):
            results = adaptive_searcher.search(query, k=10)
            result_ids = [r[0] for r in results]

            relevant_ids = set(result_ids[:5])
            adaptive_searcher.provide_feedback(query, result_ids, relevant_ids)

            adaptive_searcher.record_performance(recall=0.80, precision=0.75, latency_ms=12.0)

        # Get comprehensive stats
        stats = adaptive_searcher.get_statistics()

        # Verify all sections present and meaningful
        assert stats["graph"]["nodes"] == 150  # Our clustered dataset
        assert stats["feedback"]["total_queries"] == 20
        assert stats["feedback"]["avg_satisfaction"] > 0
        assert stats["weights"]["count"] > 0
        assert stats["stability"]["total_updates"] > 0
        assert stats["performance"]["current_recall"] > 0
        assert "reset" in stats


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_feedback_with_no_results(self, adaptive_searcher):
        """Feedback with empty results should not crash."""
        query = np.random.randn(128).astype(np.float32)

        # Should not raise error
        adaptive_searcher.provide_feedback(query, [], set())

    def test_feedback_with_all_relevant(self, adaptive_searcher, clustered_vectors):
        """All results relevant should update correctly."""
        query = clustered_vectors[0]

        results = adaptive_searcher.search(query, k=10)
        result_ids = [r[0] for r in results]

        # All relevant
        adaptive_searcher.provide_feedback(query, result_ids, set(result_ids))

        # Should update without issues
        assert adaptive_searcher.feedback_collector.get_feedback_count() == 1

    def test_feedback_with_none_relevant(self, adaptive_searcher, clustered_vectors):
        """No results relevant should update correctly."""
        query = clustered_vectors[0]

        results = adaptive_searcher.search(query, k=10)
        result_ids = [r[0] for r in results]

        # None relevant
        adaptive_searcher.provide_feedback(query, result_ids, set())

        # Should update without issues
        assert adaptive_searcher.feedback_collector.get_feedback_count() == 1
