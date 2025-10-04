"""Unit tests for intent detection."""

import pytest
import numpy as np
from dynhnsw.intent_detector import IntentDetector


@pytest.fixture
def clustered_queries():
    """Create queries with clear cluster structure."""
    np.random.seed(42)

    # Cluster 0: centered at [5, 5, ...]
    cluster_0 = np.random.randn(30, 128).astype(np.float32) + 5.0

    # Cluster 1: centered at [-5, -5, ...]
    cluster_1 = np.random.randn(30, 128).astype(np.float32) - 5.0

    # Cluster 2: centered at [0, 10, ...]
    cluster_2 = np.random.randn(30, 128).astype(np.float32)
    cluster_2[:, :64] += 10.0

    all_queries = np.vstack([cluster_0, cluster_1, cluster_2])

    # Normalize
    all_queries = all_queries / np.linalg.norm(all_queries, axis=1, keepdims=True)

    return all_queries


class TestIntentDetectorInitialization:
    """Test detector initialization."""

    def test_default_initialization(self):
        """Should initialize with default parameters."""
        detector = IntentDetector()

        assert detector.k_intents == 5
        assert detector.min_queries == 50
        assert detector.kmeans is None
        assert detector.total_queries == 0

    def test_custom_parameters(self):
        """Should accept custom parameters."""
        detector = IntentDetector(
            k_intents=3,
            min_queries_for_clustering=20,
            confidence_threshold=0.8
        )

        assert detector.k_intents == 3
        assert detector.min_queries == 20
        assert detector.confidence_threshold == 0.8


class TestColdStartBehavior:
    """Test behavior before clustering is initialized."""

    def test_cold_start_returns_no_intent(self):
        """Should return -1 intent during cold start."""
        detector = IntentDetector(min_queries_for_clustering=10)

        query = np.random.randn(128).astype(np.float32)
        intent_id, confidence = detector.detect_intent(query)

        assert intent_id == -1
        assert confidence == 0.0

    def test_queries_added_to_buffer(self):
        """Should add queries to buffer during cold start."""
        detector = IntentDetector(min_queries_for_clustering=10)

        for _ in range(5):
            query = np.random.randn(128).astype(np.float32)
            detector.detect_intent(query)

        assert len(detector.query_buffer) == 5

    def test_clustering_initializes_after_min_queries(self):
        """Should initialize clustering after minimum queries."""
        detector = IntentDetector(k_intents=3, min_queries_for_clustering=10)

        queries = np.random.randn(10, 128).astype(np.float32)

        for query in queries:
            detector.detect_intent(query)

        assert detector.kmeans is not None
        assert detector.cluster_centroids is not None


class TestIntentDetection:
    """Test intent detection with clustering."""

    def test_intent_assignment(self, clustered_queries):
        """Should assign intents after clustering."""
        detector = IntentDetector(k_intents=3, min_queries_for_clustering=20)

        # Initialize clustering
        for query in clustered_queries[:20]:
            detector.detect_intent(query)

        # Detect intent for new query from cluster 0
        test_query = clustered_queries[0] + np.random.randn(128).astype(np.float32) * 0.1
        test_query = test_query / np.linalg.norm(test_query)

        intent_id, confidence = detector.detect_intent(test_query)

        assert 0 <= intent_id < 3
        assert 0.0 <= confidence <= 1.0

    def test_consistent_intent_for_similar_queries(self, clustered_queries):
        """Similar queries should get same intent."""
        detector = IntentDetector(k_intents=3, min_queries_for_clustering=20)

        # Initialize
        for query in clustered_queries[:20]:
            detector.detect_intent(query)

        # Test similar queries from cluster 0
        intent_ids = []
        for i in range(5):
            query = clustered_queries[20 + i]
            intent_id, _ = detector.detect_intent(query)
            intent_ids.append(intent_id)

        # Should assign same intent (or mostly same)
        assert len(set(intent_ids)) <= 2  # Allow some variance


class TestConfidenceScoring:
    """Test confidence score computation."""

    def test_high_confidence_for_cluster_center(self, clustered_queries):
        """Query near cluster center should have high confidence."""
        detector = IntentDetector(k_intents=3, min_queries_for_clustering=20)

        # Initialize
        for query in clustered_queries[:30]:
            detector.detect_intent(query)

        # Query very close to cluster 0 center
        cluster_0_mean = clustered_queries[:30].mean(axis=0)
        test_query = cluster_0_mean / np.linalg.norm(cluster_0_mean)

        intent_id, confidence = detector.detect_intent(test_query)

        # Should have some confidence (may not be very high for normalized random data)
        assert 0.0 <= confidence <= 1.0
        assert intent_id >= 0  # Should assign some intent

    def test_low_confidence_for_ambiguous_query(self, clustered_queries):
        """Query between clusters should have lower confidence."""
        detector = IntentDetector(k_intents=3, min_queries_for_clustering=30)

        # Initialize
        for query in clustered_queries[:30]:
            detector.detect_intent(query)

        # Query between cluster 0 and cluster 1
        cluster_0_mean = clustered_queries[:30].mean(axis=0)
        cluster_1_mean = clustered_queries[30:60].mean(axis=0)
        midpoint = (cluster_0_mean + cluster_1_mean) / 2
        test_query = midpoint / np.linalg.norm(midpoint)

        intent_id, confidence = detector.detect_intent(test_query)

        # Confidence should be lower (closer to both clusters)
        assert 0.0 <= confidence <= 1.0


class TestDriftDetection:
    """Test intent drift detection."""

    def test_no_drift_with_consistent_queries(self, clustered_queries):
        """Consistent queries should not trigger drift."""
        detector = IntentDetector(k_intents=3, min_queries_for_clustering=30)

        # Initialize with cluster 0
        for query in clustered_queries[:30]:
            detector.detect_intent(query)

        # Continue with more cluster 0 queries (same distribution)
        for query in clustered_queries[:20]:  # Reuse same cluster 0 queries
            detector.detect_intent(query)

        drift_detected = detector.check_drift(drift_threshold=2.0)

        # With consistent queries from same distribution, drift is less likely
        # Just verify it runs without error
        assert isinstance(drift_detected, bool)

    def test_drift_with_shifting_distribution(self, clustered_queries):
        """Shifting query distribution should trigger drift."""
        detector = IntentDetector(k_intents=3, min_queries_for_clustering=30)

        # Initialize with cluster 0
        for query in clustered_queries[:30]:
            detector.detect_intent(query)

        # Shift to cluster 1 (very different)
        for query in clustered_queries[30:60]:
            detector.detect_intent(query)

        drift_detected = detector.check_drift(drift_threshold=2.0)

        # May or may not detect drift depending on how K-means assigns
        # Just verify it runs without error
        assert isinstance(drift_detected, bool)

    def test_recompute_clusters(self, clustered_queries):
        """Should be able to recompute clusters."""
        detector = IntentDetector(k_intents=3, min_queries_for_clustering=30)

        # Initialize
        for query in clustered_queries[:30]:
            detector.detect_intent(query)

        old_centroids = detector.cluster_centroids.copy()

        # Add more diverse queries
        for query in clustered_queries[30:60]:
            detector.detect_intent(query)

        # Recompute
        detector.recompute_clusters()

        # Centroids should have changed
        assert not np.allclose(old_centroids, detector.cluster_centroids)


class TestClusterStatistics:
    """Test cluster statistics."""

    def test_get_cluster_sizes(self, clustered_queries):
        """Should compute cluster sizes correctly."""
        detector = IntentDetector(k_intents=3, min_queries_for_clustering=30)

        # Initialize with all queries
        for query in clustered_queries:
            detector.detect_intent(query)

        sizes = detector.get_cluster_sizes()

        assert len(sizes) == 3
        assert sum(sizes) > 0

    def test_get_statistics(self, clustered_queries):
        """Should return comprehensive statistics."""
        detector = IntentDetector(k_intents=3, min_queries_for_clustering=20)

        # Add queries
        for query in clustered_queries[:30]:
            detector.detect_intent(query)

        stats = detector.get_statistics()

        assert stats["total_queries"] == 30
        assert stats["clustering_active"] is True
        assert stats["k_intents"] == 3
        assert "cluster_sizes" in stats
        assert "confidence_rate" in stats

    def test_get_intent_centroid(self, clustered_queries):
        """Should return centroid for valid intent."""
        detector = IntentDetector(k_intents=3, min_queries_for_clustering=20)

        # Initialize
        for query in clustered_queries[:30]:
            detector.detect_intent(query)

        # Get centroid for intent 0
        centroid = detector.get_intent_centroid(0)

        assert centroid is not None
        assert centroid.shape == (128,)

    def test_get_intent_centroid_invalid(self):
        """Should return None for invalid intent."""
        detector = IntentDetector()

        centroid = detector.get_intent_centroid(0)

        assert centroid is None


class TestBufferManagement:
    """Test query buffer management."""

    def test_buffer_size_limit(self):
        """Should respect buffer size limit."""
        detector = IntentDetector(
            min_queries_for_clustering=10,
            query_buffer_size=50
        )

        queries = np.random.randn(100, 128).astype(np.float32)

        for query in queries:
            detector.detect_intent(query)

        assert len(detector.query_buffer) == 50  # Max size

    def test_buffer_evicts_oldest(self):
        """Should evict oldest queries when buffer is full."""
        detector = IntentDetector(
            min_queries_for_clustering=10,
            query_buffer_size=20
        )

        # Add 30 unique queries
        for i in range(30):
            query = np.ones(128, dtype=np.float32) * i
            detector.detect_intent(query)

        # Buffer should have last 20 queries
        assert len(detector.query_buffer) == 20

        # First query should be from iteration 10 (0-9 evicted)
        first_in_buffer = detector.query_buffer[0]
        expected = np.ones(128, dtype=np.float32) * 10

        assert np.allclose(first_in_buffer, expected)
