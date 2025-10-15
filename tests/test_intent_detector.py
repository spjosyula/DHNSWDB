"""
Unit tests for difficulty-based intent detector.

Tests clustering of query difficulties into intent tiers.
"""

import pytest
import numpy as np
from dynhnsw.intent_detector import IntentDetector


class TestIntentDetectorColdStart:
    """Tests for cold start behavior."""

    def test_cold_start_returns_minus_one(self):
        """Test that detector returns -1 during cold start."""
        detector = IntentDetector(n_intents=5, min_queries_for_clustering=10)

        # First 9 queries should return -1
        for i in range(9):
            detector.add_query_difficulty(0.5 + i * 0.1)
            intent = detector.detect_intent(0.5)
            assert intent == -1, f"Query {i}: Expected -1 during cold start"

    def test_clustering_activates_at_threshold(self):
        """Test that clustering activates after min_queries."""
        detector = IntentDetector(n_intents=3, min_queries_for_clustering=10)

        # Add 10 queries with varying difficulty
        for i in range(10):
            detector.add_query_difficulty(i * 0.1)  # 0.0 to 0.9

        # Clustering should now be active
        assert detector.is_active()

        # Should return valid intent
        intent = detector.detect_intent(0.5)
        assert 0 <= intent <= 2


class TestIntentDetectorClustering:
    """Tests for difficulty clustering."""

    def test_easy_and_hard_queries_separate(self):
        """Test that easy and hard queries get different intents."""
        detector = IntentDetector(n_intents=3, min_queries_for_clustering=10)

        # Add 5 easy queries (low difficulty)
        for _ in range(5):
            detector.add_query_difficulty(0.1 + np.random.rand() * 0.1)

        # Add 5 hard queries (high difficulty)
        for _ in range(5):
            detector.add_query_difficulty(0.8 + np.random.rand() * 0.2)

        # Clustering should be active now
        assert detector.is_active()

        # Easy query should get low intent
        easy_intent = detector.detect_intent(0.15)
        assert easy_intent == 0, "Easy query should get intent 0"

        # Hard query should get high intent
        hard_intent = detector.detect_intent(0.95)
        assert hard_intent == 2, "Hard query should get intent 2"

    def test_five_intent_tiers(self):
        """Test clustering with 5 intent tiers."""
        detector = IntentDetector(n_intents=5, min_queries_for_clustering=20)

        # Generate data across full difficulty range
        for i in range(20):
            difficulty = i / 20.0  # 0.0 to 0.95
            detector.add_query_difficulty(difficulty)

        assert detector.is_active()

        # Test boundary queries
        very_easy = detector.detect_intent(0.05)
        medium = detector.detect_intent(0.5)
        very_hard = detector.detect_intent(0.95)

        assert very_easy < medium < very_hard
        assert 0 <= very_easy <= 4
        assert 0 <= medium <= 4
        assert 0 <= very_hard <= 4

    def test_centroid_ordering(self):
        """Test that centroids are ordered easy to hard."""
        detector = IntentDetector(n_intents=3, min_queries_for_clustering=10)

        # Add queries
        for i in range(10):
            detector.add_query_difficulty(i * 0.1)

        centroids = detector.get_cluster_centroids()

        # Centroids should be in ascending order
        assert centroids is not None
        assert len(centroids) == 3
        assert centroids[0] < centroids[1] < centroids[2]


class TestIntentDetectorStatistics:
    """Tests for statistics and introspection."""

    def test_get_statistics_before_clustering(self):
        """Test statistics during cold start."""
        detector = IntentDetector(n_intents=5)

        for i in range(5):
            detector.add_query_difficulty(0.5)

        stats = detector.get_statistics()

        assert stats["total_queries"] == 5
        assert stats["clustering_active"] is False
        assert stats["n_intents"] == 5
        assert "centroids" not in stats  # Not available during cold start

    def test_get_statistics_after_clustering(self):
        """Test statistics after clustering active."""
        detector = IntentDetector(n_intents=3, min_queries_for_clustering=10)

        for i in range(10):
            detector.add_query_difficulty(i * 0.1)

        stats = detector.get_statistics()

        assert stats["total_queries"] == 10
        assert stats["clustering_active"] is True
        assert stats["n_intents"] == 3
        assert "centroids" in stats
        assert len(stats["centroids"]) == 3
        assert sum(stats["cluster_sizes"]) == 10  # All queries assigned

    def test_get_cluster_sizes(self):
        """Test cluster size computation."""
        detector = IntentDetector(n_intents=2, min_queries_for_clustering=10)

        # Add 6 easy, 4 hard queries
        for _ in range(6):
            detector.add_query_difficulty(0.1)
        for _ in range(4):
            detector.add_query_difficulty(0.9)

        sizes = detector.get_cluster_sizes()

        assert len(sizes) == 2
        assert sum(sizes) == 10
        # Exact distribution depends on K-means, but should be roughly [6, 4]
        assert 4 <= sizes[0] <= 7  # Easy cluster
        assert 3 <= sizes[1] <= 6  # Hard cluster

    def test_get_intent_difficulty_range(self):
        """Test difficulty range per intent."""
        detector = IntentDetector(n_intents=2, min_queries_for_clustering=10)

        # Add easy queries (0.1-0.3)
        for i in range(5):
            detector.add_query_difficulty(0.1 + i * 0.04)

        # Add hard queries (0.7-0.9)
        for i in range(5):
            detector.add_query_difficulty(0.7 + i * 0.04)

        # Get ranges
        easy_range = detector.get_intent_difficulty_range(0)
        hard_range = detector.get_intent_difficulty_range(1)

        assert easy_range is not None
        assert hard_range is not None

        # Easy range should be lower than hard range
        assert easy_range[1] < hard_range[0]  # Max easy < Min hard


class TestIntentDetectorEdgeCases:
    """Tests for edge cases and error handling."""

    def test_all_same_difficulty(self):
        """Test with all queries having same difficulty."""
        detector = IntentDetector(n_intents=3, min_queries_for_clustering=10)

        # All queries have difficulty 0.5
        for _ in range(10):
            detector.add_query_difficulty(0.5)

        # Should still work (all in one cluster)
        assert detector.is_active()
        intent = detector.detect_intent(0.5)
        assert 0 <= intent <= 2

    def test_very_few_intents(self):
        """Test with only 2 intents."""
        detector = IntentDetector(n_intents=2, min_queries_for_clustering=10)

        for i in range(10):
            detector.add_query_difficulty(i * 0.1)

        assert detector.is_active()
        centroids = detector.get_cluster_centroids()
        assert len(centroids) == 2
        assert centroids[0] < centroids[1]

    def test_more_intents_than_queries(self):
        """Test when n_intents > n_queries (edge case for K-means)."""
        # K-means requires n_clusters <= n_samples
        # This should still work as we require min 10 queries
        detector = IntentDetector(n_intents=3, min_queries_for_clustering=10)

        for i in range(10):
            detector.add_query_difficulty(i * 0.1)

        assert detector.is_active()

    def test_buffer_overflow(self):
        """Test that buffer respects max size."""
        detector = IntentDetector(
            n_intents=3,
            min_queries_for_clustering=10,
            difficulty_buffer_size=50
        )

        # Add 100 queries (exceeds buffer size)
        for i in range(100):
            detector.add_query_difficulty(i * 0.01)

        # Buffer should have only last 50
        assert len(detector.difficulty_buffer) == 50
        assert detector.total_queries == 100


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
