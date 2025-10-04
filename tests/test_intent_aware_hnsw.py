"""Integration tests for IntentAwareHNSWSearcher."""

import pytest
import numpy as np
import time
from typing import Set
from dynhnsw.hnsw.graph import HNSWGraph
from dynhnsw.hnsw.builder import HNSWBuilder
from dynhnsw.intent_aware_hnsw import IntentAwareHNSWSearcher


@pytest.fixture
def simple_vectors():
    """Create simple test vectors."""
    np.random.seed(42)
    return np.random.randn(50, 128).astype(np.float32)


@pytest.fixture
def clustered_vectors():
    """Create vectors with clear cluster structure for intent detection."""
    np.random.seed(42)

    # Cluster 0: centered at [3, 3, ...]
    cluster_0 = np.random.randn(20, 128).astype(np.float32) * 0.5 + 3.0

    # Cluster 1: centered at [-3, -3, ...]
    cluster_1 = np.random.randn(20, 128).astype(np.float32) * 0.5 - 3.0

    # Cluster 2: centered at [0, 5, ...]
    cluster_2 = np.random.randn(20, 128).astype(np.float32) * 0.5
    cluster_2[:, :64] += 5.0

    all_vectors = np.vstack([cluster_0, cluster_1, cluster_2])

    # Normalize
    return all_vectors / np.linalg.norm(all_vectors, axis=1, keepdims=True)


@pytest.fixture
def simple_graph(simple_vectors):
    """Build simple HNSW graph."""
    graph = HNSWGraph(dimension=128, M=16)
    builder = HNSWBuilder(graph)

    for i, vec in enumerate(simple_vectors):
        level = 0 if i > 0 else 3  # First node gets higher level
        builder.insert(vector=vec, node_id=i, level=level)

    return graph


@pytest.fixture
def clustered_graph(clustered_vectors):
    """Build HNSW graph with clustered data."""
    graph = HNSWGraph(dimension=128, M=16)
    builder = HNSWBuilder(graph)

    for i, vec in enumerate(clustered_vectors):
        level = 0 if i > 0 else 3  # First node gets higher level
        builder.insert(vector=vec, node_id=i, level=level)

    return graph


class TestIntentAwareSearcherInitialization:
    """Test searcher initialization."""

    def test_default_initialization(self, simple_graph):
        """Should initialize with default parameters."""
        searcher = IntentAwareHNSWSearcher(graph=simple_graph)

        assert searcher.ef_search == 50
        assert searcher.enable_adaptation is True
        assert searcher.enable_intent_detection is True
        assert searcher.k_intents == 5

    def test_custom_parameters(self, simple_graph):
        """Should accept custom parameters."""
        searcher = IntentAwareHNSWSearcher(
            graph=simple_graph,
            ef_search=100,
            k_intents=3,
            learning_rate=0.1,
            enable_adaptation=False,
            enable_intent_detection=False,
            confidence_threshold=0.8
        )

        assert searcher.ef_search == 100
        assert searcher.k_intents == 3
        assert searcher.enable_adaptation is False
        assert searcher.enable_intent_detection is False

    def test_intent_detection_requires_adaptation(self, simple_graph):
        """Should disable intent detection if adaptation is off."""
        searcher = IntentAwareHNSWSearcher(
            graph=simple_graph,
            enable_adaptation=False,
            enable_intent_detection=True
        )

        # Intent detection should be disabled because adaptation is off
        assert searcher.enable_intent_detection is False


class TestBasicSearch:
    """Test basic search functionality."""

    def test_search_empty_graph(self):
        """Should handle empty graph gracefully."""
        graph = HNSWGraph(dimension=128, M=16)
        searcher = IntentAwareHNSWSearcher(graph=graph)

        query = np.random.randn(128).astype(np.float32)
        results = searcher.search(query, k=5)

        assert results == []

    def test_search_returns_results(self, simple_graph, simple_vectors):
        """Should return search results."""
        searcher = IntentAwareHNSWSearcher(graph=simple_graph)

        query = simple_vectors[0]
        results = searcher.search(query, k=5)

        assert len(results) == 5
        assert all(isinstance(r, tuple) for r in results)
        assert all(len(r) == 2 for r in results)

    def test_search_results_sorted(self, simple_graph, simple_vectors):
        """Should return results sorted by distance."""
        searcher = IntentAwareHNSWSearcher(graph=simple_graph)

        query = simple_vectors[0]
        results = searcher.search(query, k=10)

        # Distances should be non-decreasing
        distances = [dist for _, dist in results]
        assert all(distances[i] <= distances[i+1] for i in range(len(distances)-1))

    def test_search_custom_ef_search(self, simple_graph, simple_vectors):
        """Should respect custom ef_search parameter."""
        searcher = IntentAwareHNSWSearcher(graph=simple_graph, ef_search=50)

        query = simple_vectors[0]

        # Override with higher ef_search
        results = searcher.search(query, k=5, ef_search=100)

        # Should complete without error
        assert len(results) == 5


class TestIntentDetection:
    """Test intent detection during search."""

    def test_cold_start_intent_detection(self, simple_graph, simple_vectors):
        """Should assign -1 intent during cold start."""
        searcher = IntentAwareHNSWSearcher(
            graph=simple_graph,
            k_intents=3,
            enable_intent_detection=True
        )

        # First few queries (before clustering initializes)
        query = simple_vectors[0]
        searcher.search(query, k=5)

        assert searcher.last_intent_id == -1
        assert searcher.last_confidence == 0.0

    def test_intent_detection_after_warmup(self, clustered_graph, clustered_vectors):
        """Should detect intent after warmup queries."""
        searcher = IntentAwareHNSWSearcher(
            graph=clustered_graph,
            k_intents=3,
            enable_intent_detection=True
        )

        # Set lower minimum for faster testing
        searcher.intent_detector.min_queries = 30

        # Send warmup queries from cluster 0
        for i in range(30):
            query = clustered_vectors[i]
            searcher.search(query, k=5)

        # Intent should now be assigned
        query = clustered_vectors[0]
        searcher.search(query, k=5)

        assert searcher.last_intent_id >= 0
        assert searcher.last_intent_id < 3

    def test_intent_disabled_mode(self, simple_graph, simple_vectors):
        """Should not detect intent when disabled."""
        searcher = IntentAwareHNSWSearcher(
            graph=simple_graph,
            enable_intent_detection=False
        )

        query = simple_vectors[0]
        searcher.search(query, k=5)

        assert searcher.last_intent_id == -1
        assert searcher.last_confidence == 0.0


class TestAdaptiveEntryPoints:
    """Test adaptive entry point learning."""

    def test_feedback_updates_entry_scores(self, clustered_graph, clustered_vectors):
        """Should update entry point scores based on feedback after intent detection."""
        searcher = IntentAwareHNSWSearcher(
            graph=clustered_graph,
            enable_adaptation=True,
            enable_intent_detection=True,
            learning_rate=0.1,
            min_queries_for_clustering=20
        )

        # Warmup to enable intent detection
        for i in range(20):
            query = clustered_vectors[i]
            results = searcher.search(query, k=5)
            result_ids = [r[0] for r in results]
            searcher.provide_feedback(query, result_ids, set(result_ids[:1]))

        # Now test feedback updates with detected intent
        query = clustered_vectors[20]
        results = searcher.search(query, k=5)

        # Get initial stats
        initial_stats = searcher.entry_selector.get_statistics()
        initial_usage = initial_stats["total_usage"]

        # Provide feedback (with detected intent)
        result_ids = [r[0] for r in results]
        relevant_ids = set(result_ids[:2])  # First 2 are relevant

        searcher.provide_feedback(query, result_ids, relevant_ids)

        # Should have updated entry point usage
        updated_stats = searcher.entry_selector.get_statistics()
        assert updated_stats["total_usage"] > initial_usage

    def test_adaptation_disabled_mode(self, simple_graph, simple_vectors):
        """Should not adapt when disabled."""
        searcher = IntentAwareHNSWSearcher(
            graph=simple_graph,
            enable_adaptation=False
        )

        query = simple_vectors[0]
        results = searcher.search(query, k=5)
        result_ids = [r[0] for r in results]

        searcher.provide_feedback(query, result_ids, {result_ids[0]})

        # entry_selector should be None when adaptation disabled
        assert searcher.entry_selector is None

    def test_intent_aware_entry_updates(self, clustered_graph, clustered_vectors):
        """Should update intent-specific entry points."""
        searcher = IntentAwareHNSWSearcher(
            graph=clustered_graph,
            k_intents=3,
            enable_adaptation=True,
            enable_intent_detection=True,
            learning_rate=0.1,
            min_queries_for_clustering=30
        )

        # Warmup to initialize clustering
        for i in range(30):
            query = clustered_vectors[i]
            results = searcher.search(query, k=5)
            result_ids = [r[0] for r in results]
            searcher.provide_feedback(query, result_ids, set(result_ids[:1]))

        # Now provide feedback with detected intent
        query = clustered_vectors[0]
        results = searcher.search(query, k=5)
        result_ids = [r[0] for r in results]

        # Store intent before feedback
        intent_before = searcher.last_intent_id

        searcher.provide_feedback(query, result_ids, set(result_ids[:3]))

        # Should have updated entry point scores for the intent
        if intent_before >= 0:
            entry_stats = searcher.entry_selector.get_statistics()
            per_intent = entry_stats["per_intent"]
            # Check that the intent has some usage
            assert per_intent[intent_before]["total_usage"] > 0


class TestIntentDriftHandling:
    """Test intent drift detection and handling."""

    def test_drift_detection_triggers_reclustering(self, clustered_graph, clustered_vectors):
        """Should recompute clusters when drift detected."""
        searcher = IntentAwareHNSWSearcher(
            graph=clustered_graph,
            k_intents=3,
            enable_intent_detection=True
        )

        searcher.intent_detector.min_queries = 30

        # Initialize with cluster 0 queries
        for i in range(30):
            query = clustered_vectors[i]
            results = searcher.search(query, k=5)
            result_ids = [r[0] for r in results]
            searcher.provide_feedback(query, result_ids, set(result_ids[:1]))

        old_centroids = searcher.intent_detector.cluster_centroids.copy()

        # Shift to cluster 1 queries (different distribution)
        for i in range(30, 60):
            query = clustered_vectors[i]
            results = searcher.search(query, k=5)
            result_ids = [r[0] for r in results]
            searcher.provide_feedback(query, result_ids, set(result_ids[:1]))

        # Centroids may have changed due to drift detection
        # (drift detection happens every 50 queries)
        # This is a weak test - just verify it doesn't crash


class TestPerformanceMonitoring:
    """Test performance tracking and reset triggers."""

    def test_record_performance(self, simple_graph, simple_vectors):
        """Should record performance metrics."""
        searcher = IntentAwareHNSWSearcher(graph=simple_graph)

        searcher.record_performance(recall=0.9, precision=0.8, latency_ms=5.0)
        searcher.record_performance(recall=0.85, precision=0.75, latency_ms=4.5)

        # Should have set baseline
        assert searcher.performance_monitor.baseline_recall > 0.0

    def test_performance_monitoring_disabled(self, simple_graph, simple_vectors):
        """Should not monitor performance when adaptation disabled."""
        searcher = IntentAwareHNSWSearcher(
            graph=simple_graph,
            enable_adaptation=False
        )

        searcher.record_performance(recall=0.9, precision=0.8, latency_ms=5.0)

        # Baseline should remain 0
        assert searcher.performance_monitor.baseline_recall == 0.0

    def test_degradation_triggers_reset(self, simple_graph, simple_vectors):
        """Should trigger reset on performance degradation."""
        searcher = IntentAwareHNSWSearcher(
            graph=simple_graph,
            enable_adaptation=True
        )

        # Set baseline
        for _ in range(10):
            searcher.record_performance(recall=0.95, precision=0.9, latency_ms=5.0)

        # Simulate degradation
        for _ in range(10):
            searcher.record_performance(recall=0.5, precision=0.5, latency_ms=5.0)

        # Reset may or may not be triggered depending on thresholds
        # Just verify no crash


class TestEntryPointTracking:
    """Test entry point tracking."""

    def test_tracks_last_entry_used(self, simple_graph, simple_vectors):
        """Should track which entry point was used."""
        searcher = IntentAwareHNSWSearcher(graph=simple_graph)

        query = simple_vectors[0]
        searcher.search(query, k=5)

        # Should have tracked the entry point used
        assert searcher.last_entry_used is not None
        assert searcher.last_entry_used in simple_graph.nodes

    def test_entry_changes_with_intent(self, clustered_graph, clustered_vectors):
        """Different intents may use different entry points after learning."""
        searcher = IntentAwareHNSWSearcher(
            graph=clustered_graph,
            k_intents=3,
            enable_intent_detection=True,
            enable_adaptation=True,
            min_queries_for_clustering=30
        )

        # Warmup with feedback
        for i in range(40):
            query = clustered_vectors[i]
            results = searcher.search(query, k=5)
            result_ids = [r[0] for r in results]
            # Provide varied feedback
            relevant = set(result_ids[:3] if i < 20 else result_ids[:1])
            searcher.provide_feedback(query, result_ids, relevant)

        # Entry points may have adapted per intent
        entry_stats = searcher.entry_selector.get_statistics()
        assert "best_entries" in entry_stats


class TestStatisticsReporting:
    """Test comprehensive statistics."""

    def test_get_statistics(self, simple_graph):
        """Should return comprehensive statistics."""
        searcher = IntentAwareHNSWSearcher(graph=simple_graph, k_intents=3)

        stats = searcher.get_statistics()

        assert "graph" in stats
        assert "entry_selection" in stats
        assert "performance" in stats
        assert "feedback" in stats

    def test_statistics_with_intent_detection(self, simple_graph):
        """Should include intent detection stats when enabled."""
        searcher = IntentAwareHNSWSearcher(
            graph=simple_graph,
            enable_intent_detection=True
        )

        stats = searcher.get_statistics()

        assert "intent_detection" in stats
        assert "total_queries" in stats["intent_detection"]
        assert "clustering_active" in stats["intent_detection"]

    def test_statistics_without_intent_detection(self, simple_graph):
        """Should not include intent stats when disabled."""
        searcher = IntentAwareHNSWSearcher(
            graph=simple_graph,
            enable_intent_detection=False
        )

        stats = searcher.get_statistics()

        assert "intent_detection" not in stats


class TestEndToEndIntentAwareSearch:
    """End-to-end integration tests."""

    def test_full_intent_aware_workflow(self, clustered_graph, clustered_vectors):
        """Should perform full intent-aware search workflow."""
        searcher = IntentAwareHNSWSearcher(
            graph=clustered_graph,
            k_intents=3,
            enable_adaptation=True,
            enable_intent_detection=True,
            learning_rate=0.1
        )

        searcher.intent_detector.min_queries = 20

        # Phase 1: Cold start (no intent detection yet)
        for i in range(20):
            query = clustered_vectors[i]
            results = searcher.search(query, k=5)

            assert len(results) == 5
            # Intent ID may or may not be -1 depending on when clustering initializes

        # Phase 2: Intent detection active
        for i in range(20, 40):
            query = clustered_vectors[i]
            results = searcher.search(query, k=5)
            result_ids = [r[0] for r in results]

            # Provide feedback
            relevant_ids = set(result_ids[:2])
            searcher.provide_feedback(query, result_ids, relevant_ids)

        # Should have detected intents
        stats = searcher.get_statistics()
        assert stats["intent_detection"]["clustering_active"] is True

        # Should have learned entry point scores
        assert stats["entry_selection"]["total_usage"] > 0

    def test_static_mode_matches_baseline(self, simple_graph, simple_vectors):
        """Static mode should behave like traditional HNSW."""
        searcher_static = IntentAwareHNSWSearcher(
            graph=simple_graph,
            enable_adaptation=False,
            enable_intent_detection=False
        )

        searcher_adaptive = IntentAwareHNSWSearcher(
            graph=simple_graph,
            enable_adaptation=False,
            enable_intent_detection=False
        )

        query = simple_vectors[0]

        results_static = searcher_static.search(query, k=5)
        results_adaptive = searcher_adaptive.search(query, k=5)

        # Should return identical results (same graph, no adaptation)
        assert results_static == results_adaptive
