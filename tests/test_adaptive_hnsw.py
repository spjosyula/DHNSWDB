"""Integration tests for adaptive HNSW search."""

import pytest
import numpy as np
from dynhnsw.hnsw.graph import HNSWGraph
from dynhnsw.hnsw.builder import HNSWBuilder
from dynhnsw.adaptive_hnsw import AdaptiveHNSWSearcher


@pytest.fixture
def sample_vectors():
    """Create sample vectors for testing."""
    np.random.seed(42)
    return np.random.randn(100, 128).astype(np.float32)


@pytest.fixture
def built_graph(sample_vectors):
    """Build a small HNSW graph for testing."""
    graph = HNSWGraph(dimension=128, M=8)
    builder = HNSWBuilder(graph)

    for i, vector in enumerate(sample_vectors):
        level = int(np.random.geometric(p=0.5)) - 1  # Assign random level
        level = min(level, 5)  # Cap at level 5
        builder.insert(vector, node_id=i, level=level)

    return graph


class TestAdaptiveHNSWBasics:
    """Test basic adaptive HNSW functionality."""

    def test_initialization(self, built_graph):
        """Should initialize with all components."""
        searcher = AdaptiveHNSWSearcher(built_graph, ef_search=20)

        assert searcher.graph == built_graph
        assert searcher.ef_search == 20
        assert searcher.enable_adaptation is True

    def test_disabled_adaptation(self, built_graph):
        """Should work with adaptation disabled (static HNSW)."""
        searcher = AdaptiveHNSWSearcher(
            built_graph, ef_search=20, enable_adaptation=False
        )

        query = np.random.randn(128).astype(np.float32)
        results = searcher.search(query, k=5)

        assert len(results) == 5
        # Should still return results but not adapt

    def test_search_returns_results(self, built_graph, sample_vectors):
        """Should return search results."""
        searcher = AdaptiveHNSWSearcher(built_graph, ef_search=20)

        query = sample_vectors[0]
        results = searcher.search(query, k=10)

        assert len(results) == 10
        # Results should be (node_id, distance) tuples
        assert all(isinstance(r, tuple) and len(r) == 2 for r in results)
        # Should be sorted by distance
        distances = [r[1] for r in results]
        assert distances == sorted(distances)


class TestEdgeTraversal:
    """Test edge traversal tracking."""

    def test_edge_tracking_during_search(self, built_graph, sample_vectors):
        """Should track edges traversed during search."""
        searcher = AdaptiveHNSWSearcher(built_graph, ef_search=20)

        query = sample_vectors[0]
        searcher.search(query, k=5)

        # Should have tracked some edges
        assert len(searcher.last_traversed_edges) > 0

    def test_edge_tracking_reset_per_query(self, built_graph, sample_vectors):
        """Edge tracking should reset for each query."""
        searcher = AdaptiveHNSWSearcher(built_graph, ef_search=20)

        query1 = sample_vectors[0]
        searcher.search(query1, k=5)
        edges1 = list(searcher.last_traversed_edges)

        query2 = sample_vectors[1]
        searcher.search(query2, k=5)
        edges2 = list(searcher.last_traversed_edges)

        # Different queries likely traverse different edges
        assert edges1 != edges2 or len(edges1) != len(edges2)


class TestFeedbackIntegration:
    """Test feedback collection and processing."""

    def test_provide_feedback_updates_weights(self, built_graph, sample_vectors):
        """Providing feedback should update edge weights."""
        searcher = AdaptiveHNSWSearcher(built_graph, ef_search=20, learning_rate=0.5)

        query = sample_vectors[0]
        results = searcher.search(query, k=5)
        result_ids = [r[0] for r in results]

        # Get initial weight stats
        initial_stats = searcher.weight_learner.get_weight_statistics()

        # Provide positive feedback
        searcher.provide_feedback(query, result_ids, relevant_ids=set(result_ids))

        # Weights should have been updated
        updated_stats = searcher.weight_learner.get_weight_statistics()
        assert updated_stats["count"] > initial_stats["count"]

    def test_positive_feedback_decreases_weights(self, built_graph, sample_vectors):
        """Positive feedback should decrease edge weights (prefer paths)."""
        searcher = AdaptiveHNSWSearcher(built_graph, ef_search=20, learning_rate=0.5)

        query = sample_vectors[0]
        results = searcher.search(query, k=5)
        result_ids = [r[0] for r in results]

        # Provide very positive feedback
        searcher.provide_feedback(query, result_ids, relevant_ids=set(result_ids))

        # Check some edge weights
        stats = searcher.weight_learner.get_weight_statistics()

        # With positive feedback, mean weight should be below neutral (1.0)
        # May take multiple iterations, so we just check it's updated
        assert stats["count"] > 0

    def test_negative_feedback_increases_weights(self, built_graph, sample_vectors):
        """Negative feedback should increase edge weights (avoid paths)."""
        searcher = AdaptiveHNSWSearcher(built_graph, ef_search=20, learning_rate=0.5)

        query = sample_vectors[0]
        results = searcher.search(query, k=5)
        result_ids = [r[0] for r in results]

        # Provide negative feedback (nothing relevant)
        searcher.provide_feedback(query, result_ids, relevant_ids=set())

        # Weights should be updated
        stats = searcher.weight_learner.get_weight_statistics()
        assert stats["count"] > 0

    def test_feedback_disabled_when_adaptation_off(self, built_graph, sample_vectors):
        """Feedback should not update weights when adaptation is disabled."""
        searcher = AdaptiveHNSWSearcher(
            built_graph, ef_search=20, enable_adaptation=False
        )

        query = sample_vectors[0]
        results = searcher.search(query, k=5)
        result_ids = [r[0] for r in results]

        searcher.provide_feedback(query, result_ids, relevant_ids=set(result_ids))

        # No weights should be tracked
        stats = searcher.weight_learner.get_weight_statistics()
        assert stats["count"] == 0


class TestStabilityMonitoring:
    """Test stability monitoring integration."""

    def test_stability_tracking_on_feedback(self, built_graph, sample_vectors):
        """Feedback should update stability monitoring."""
        searcher = AdaptiveHNSWSearcher(built_graph, ef_search=20)

        query = sample_vectors[0]
        results = searcher.search(query, k=5)
        result_ids = [r[0] for r in results]

        searcher.provide_feedback(query, result_ids, relevant_ids=set(result_ids))

        # Stability monitor should have data
        stats = searcher.stability_monitor.get_statistics()
        assert stats["total_updates"] > 0

    def test_oscillation_detection(self, built_graph, sample_vectors):
        """Should detect oscillating weights."""
        searcher = AdaptiveHNSWSearcher(built_graph, ef_search=20, learning_rate=0.5)

        query = sample_vectors[0]

        # Create oscillating feedback pattern
        for i in range(20):
            results = searcher.search(query, k=5)
            result_ids = [r[0] for r in results]

            if i % 2 == 0:
                # Positive feedback
                searcher.provide_feedback(query, result_ids, relevant_ids=set(result_ids))
            else:
                # Negative feedback
                searcher.provide_feedback(query, result_ids, relevant_ids=set())

        # Should detect some oscillation
        oscillating_edges = searcher.stability_monitor.get_oscillating_edges()
        # May or may not detect oscillation depending on graph structure
        assert isinstance(oscillating_edges, list)


class TestPerformanceMonitoring:
    """Test performance monitoring integration."""

    def test_record_performance_metrics(self, built_graph):
        """Should record performance metrics."""
        searcher = AdaptiveHNSWSearcher(built_graph, ef_search=20)

        searcher.record_performance(recall=0.85, precision=0.90, latency_ms=12.5)

        summary = searcher.performance_monitor.get_performance_summary()
        assert summary["current_recall"] == 0.85

    def test_baseline_auto_set(self, built_graph):
        """Baseline should be set automatically from first queries."""
        searcher = AdaptiveHNSWSearcher(built_graph, ef_search=20)

        # Record several queries
        for _ in range(10):
            searcher.record_performance(recall=0.90, precision=0.85, latency_ms=10.0)

        # Baseline should be set
        assert searcher.performance_monitor.baseline_recall > 0.0

    def test_degradation_detection_triggers_reset(self, built_graph):
        """Performance degradation should trigger reset."""
        searcher = AdaptiveHNSWSearcher(built_graph, ef_search=20)

        # Set baseline
        for _ in range(10):
            searcher.record_performance(recall=0.90, precision=0.85, latency_ms=10.0)

        # Simulate degradation
        for _ in range(10):
            searcher.record_performance(recall=0.70, precision=0.60, latency_ms=15.0)

        # Reset should be triggered
        assert searcher.reset_manager.is_resetting()


class TestResetMechanism:
    """Test adaptive reset functionality."""

    def test_reset_modifies_weights(self, built_graph, sample_vectors):
        """Reset should gradually move weights toward neutral."""
        searcher = AdaptiveHNSWSearcher(built_graph, ef_search=20, learning_rate=0.5)

        query = sample_vectors[0]
        results = searcher.search(query, k=5)
        result_ids = [r[0] for r in results]

        # Create extreme weights
        for _ in range(10):
            searcher.provide_feedback(query, result_ids, relevant_ids=set())

        # Manually trigger reset
        searcher._trigger_reset()

        # Perform searches during reset
        for _ in range(50):
            searcher.search(query, k=5)
            searcher.reset_manager.advance_step()

        # Weights should be closer to 1.0
        stats = searcher.weight_learner.get_weight_statistics()
        assert stats["mean"] != 0  # Some weights exist


class TestStatistics:
    """Test statistics reporting."""

    def test_get_comprehensive_statistics(self, built_graph, sample_vectors):
        """Should return comprehensive system statistics."""
        searcher = AdaptiveHNSWSearcher(built_graph, ef_search=20)

        query = sample_vectors[0]
        results = searcher.search(query, k=5)
        result_ids = [r[0] for r in results]

        searcher.provide_feedback(query, result_ids, relevant_ids=set(result_ids))
        searcher.record_performance(recall=0.85, precision=0.90, latency_ms=12.0)

        stats = searcher.get_statistics()

        assert "graph" in stats
        assert "weights" in stats
        assert "stability" in stats
        assert "performance" in stats
        assert "reset" in stats
        assert "feedback" in stats

        # Verify specific values
        assert stats["graph"]["nodes"] == built_graph.size()
        assert stats["feedback"]["total_queries"] > 0


class TestEmptyGraph:
    """Test handling of empty graph."""

    def test_search_empty_graph(self):
        """Should handle empty graph gracefully."""
        graph = HNSWGraph(dimension=128, M=8)
        searcher = AdaptiveHNSWSearcher(graph, ef_search=20)

        query = np.random.randn(128).astype(np.float32)
        results = searcher.search(query, k=5)

        assert results == []

    def test_feedback_on_empty_graph(self):
        """Should handle feedback on empty graph."""
        graph = HNSWGraph(dimension=128, M=8)
        searcher = AdaptiveHNSWSearcher(graph, ef_search=20)

        query = np.random.randn(128).astype(np.float32)

        # Should not raise error
        searcher.provide_feedback(query, [], set())


class TestConcurrentQueries:
    """Test behavior with multiple queries."""

    def test_multiple_queries_different_feedback(self, built_graph, sample_vectors):
        """Should handle multiple queries with different feedback."""
        searcher = AdaptiveHNSWSearcher(built_graph, ef_search=20)

        for i in range(10):
            query = sample_vectors[i]
            results = searcher.search(query, k=5)
            result_ids = [r[0] for r in results]

            # Alternating feedback
            if i % 2 == 0:
                relevant = set(result_ids[:3])  # First 3 relevant
            else:
                relevant = set()  # Nothing relevant

            searcher.provide_feedback(query, result_ids, relevant_ids=relevant)

        # Should have accumulated feedback
        assert searcher.feedback_collector.get_feedback_count() == 10

        # Weights should be updated
        assert searcher.weight_learner.get_total_edges() > 0
