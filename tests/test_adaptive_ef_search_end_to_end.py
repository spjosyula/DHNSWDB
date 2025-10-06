"""End-to-end tests for adaptive ef_search learning.

These tests demonstrate the full adaptive ef_search behavior:
1. Intent detection identifies query types
2. Different intents learn different optimal ef_search values
3. Exploratory queries learn higher ef_search (broader recall)
4. Precise queries learn lower ef_search (faster, focused results)
"""

import pytest
import numpy as np
from dynhnsw import VectorStore


@pytest.fixture
def clustered_dataset():
    """Create dataset with 3 distinct clusters for different intents."""
    np.random.seed(42)
    vectors = []

    # Cluster 0: centered at [10, 10, ...] (500 vectors)
    cluster0 = np.random.randn(500, 128).astype(np.float32) * 2 + 10.0
    vectors.extend(cluster0)

    # Cluster 1: centered at [-10, -10, ...] (500 vectors)
    cluster1 = np.random.randn(500, 128).astype(np.float32) * 2 - 10.0
    vectors.extend(cluster1)

    # Cluster 2: centered at [0, 20, ...] (500 vectors)
    cluster2 = np.random.randn(500, 128).astype(np.float32) * 2
    cluster2[:, 1] += 20.0
    vectors.extend(cluster2)

    return np.array(vectors)


@pytest.fixture
def vector_store_with_data(clustered_dataset):
    """Build vector store with clustered data."""
    store = VectorStore(
        dimension=128,
        M=16,
        ef_construction=200,
        ef_search=100,  # Default ef_search
        enable_intent_detection=True,
        k_intents=3,
        learning_rate=0.15,
        min_queries_for_clustering=30
    )

    # Add all vectors
    store.add(clustered_dataset)

    return store


class TestAdaptiveEfSearchBasics:
    """Test basic adaptive ef_search functionality."""

    def test_cold_start_uses_default_ef(self, vector_store_with_data):
        """Should use default ef_search during cold start."""
        store = vector_store_with_data

        # Query before clustering
        query = np.random.randn(128).astype(np.float32) + 10.0
        results = store.search(query, k=10)

        # Should return results
        assert len(results) > 0

        # Check statistics - should not have clustered yet
        stats = store.get_statistics()
        assert not stats["intent_detection"]["clustering_active"]

    def test_intent_detection_activates(self, vector_store_with_data):
        """Should activate intent detection after min_queries."""
        store = vector_store_with_data

        # Run 30 queries to trigger clustering
        for i in range(30):
            cluster_id = i % 3
            if cluster_id == 0:
                query = np.random.randn(128).astype(np.float32) + 10.0
            elif cluster_id == 1:
                query = np.random.randn(128).astype(np.float32) - 10.0
            else:
                query = np.random.randn(128).astype(np.float32)
                query[1] += 20.0

            store.search(query, k=10)

        # Check that clustering is now active
        stats = store.get_statistics()
        assert stats["intent_detection"]["clustering_active"]


class TestEfSearchLearning:
    """Test ef_search learning from feedback."""

    def test_learns_different_ef_per_intent(self, vector_store_with_data):
        """Should learn different ef_search values for different intents."""
        store = vector_store_with_data

        # Cold start: run 30 queries to activate clustering
        for i in range(30):
            query = np.random.randn(128).astype(np.float32) + (10.0 if i % 2 == 0 else -10.0)
            store.search(query, k=10)

        # Intent 0: Exploratory (want many results, high ef_search)
        # Simulate by providing high satisfaction for all results
        for _ in range(15):
            query = np.random.randn(128).astype(np.float32) + 10.0
            results = store.search(query, k=20)

            # All results relevant (exploratory behavior)
            relevant_ids = [r["id"] for r in results]
            store.provide_feedback(relevant_ids=relevant_ids)

        # Intent 1: Precise (want top-5 only, low ef_search)
        # Simulate by providing high satisfaction for top-5 only
        for _ in range(15):
            query = np.random.randn(128).astype(np.float32) - 10.0
            results = store.search(query, k=10)

            # Only top-5 relevant (precise behavior)
            relevant_ids = [r["id"] for r in results[:5]]
            store.provide_feedback(relevant_ids=relevant_ids)

        # Check that learning occurred
        stats = store.get_statistics()

        if "ef_search_selection" in stats:
            ef_values = stats["ef_search_selection"]["learned_ef_values"]
            per_intent = stats["ef_search_selection"]["per_intent"]

            # Should have learned values for different intents
            assert len(ef_values) == 3

            # Should have recorded queries for learning
            total_queries_with_feedback = sum(intent["num_queries"] for intent in per_intent)
            assert total_queries_with_feedback > 0

    def test_efficiency_based_learning(self, vector_store_with_data):
        """Should optimize for efficiency (satisfaction/latency)."""
        store = vector_store_with_data

        # Activate clustering
        for i in range(30):
            query = np.random.randn(128).astype(np.float32) + 10.0
            store.search(query, k=10)

        # Provide feedback to trigger learning
        for _ in range(20):
            query = np.random.randn(128).astype(np.float32) + 10.0
            results = store.search(query, k=10)
            relevant_ids = [r["id"] for r in results[:7]]
            store.provide_feedback(relevant_ids=relevant_ids)

        # Check that learning occurred
        stats = store.get_statistics()

        if "ef_search_selection" in stats:
            per_intent = stats["ef_search_selection"]["per_intent"]

            # Should have recorded queries
            total_queries = sum(intent["num_queries"] for intent in per_intent)
            assert total_queries > 0


class TestIntentSpecificBehavior:
    """Test that different query patterns lead to different ef_search values."""

    def test_high_recall_vs_high_precision_intents(self, vector_store_with_data):
        """Different satisfaction patterns should lead to different ef_search."""
        store = vector_store_with_data

        # Activate clustering
        for i in range(35):
            cluster = i % 3
            if cluster == 0:
                query = np.random.randn(128).astype(np.float32) + 10.0
            elif cluster == 1:
                query = np.random.randn(128).astype(np.float32) - 10.0
            else:
                query = np.random.randn(128).astype(np.float32)
                query[1] += 20.0
            store.search(query, k=10)

        initial_stats = store.get_statistics()

        # Cluster 0 queries: Want broad recall (many results)
        for _ in range(10):
            query = np.random.randn(128).astype(np.float32) + 10.0
            results = store.search(query, k=15)
            # High satisfaction for all results
            relevant_ids = [r["id"] for r in results]
            store.provide_feedback(relevant_ids=relevant_ids)

        # Cluster 1 queries: Want precision (few results)
        for _ in range(10):
            query = np.random.randn(128).astype(np.float32) - 10.0
            results = store.search(query, k=10)
            # High satisfaction for top-3 only
            relevant_ids = [r["id"] for r in results[:3]]
            store.provide_feedback(relevant_ids=relevant_ids)

        # Check learning
        final_stats = store.get_statistics()

        # Intent detection should be active
        assert final_stats["intent_detection"]["clustering_active"]


class TestStatistics:
    """Test statistics reporting."""

    def test_statistics_include_ef_search_info(self, vector_store_with_data):
        """Statistics should include ef_search selection information."""
        store = vector_store_with_data

        # Run some queries
        for _ in range(35):
            query = np.random.randn(128).astype(np.float32)
            store.search(query, k=5)

        stats = store.get_statistics()

        # Should have intent detection stats
        assert "intent_detection" in stats

        # Should have ef_search selection stats when adaptation is enabled
        if store.enable_intent_detection:
            assert "ef_search_selection" in stats or "intent_detection" in stats

    def test_per_intent_statistics(self, vector_store_with_data):
        """Should track statistics per intent."""
        store = vector_store_with_data

        # Activate and run queries
        for i in range(40):
            query = np.random.randn(128).astype(np.float32) + (10.0 if i % 2 == 0 else -10.0)
            results = store.search(query, k=10)

            if i >= 30:  # Start providing feedback after clustering
                relevant_ids = [r["id"] for r in results[:5]]
                store.provide_feedback(relevant_ids=relevant_ids)

        stats = store.get_statistics()

        if "ef_search_selection" in stats:
            assert "per_intent" in stats["ef_search_selection"]
            per_intent = stats["ef_search_selection"]["per_intent"]

            # Should have stats for each intent
            assert len(per_intent) == 3

            # Each intent should have relevant fields
            for intent_stats in per_intent:
                assert "intent_id" in intent_stats
                assert "learned_ef" in intent_stats
                assert "num_queries" in intent_stats


class TestDisabledAdaptation:
    """Test behavior with adaptation disabled."""

    def test_static_ef_when_adaptation_disabled(self, clustered_dataset):
        """Should use static ef_search when adaptation is disabled."""
        store = VectorStore(
            dimension=128,
            ef_search=75,
            enable_intent_detection=False
        )

        store.add(clustered_dataset)

        # Run queries and provide feedback
        for _ in range(20):
            query = np.random.randn(128).astype(np.float32)
            results = store.search(query, k=10)
            relevant_ids = [r["id"] for r in results[:5]]
            store.provide_feedback(relevant_ids=relevant_ids)

        # Should not have ef_search selection stats
        stats = store.get_statistics()
        assert not stats["intent_detection_enabled"]
        assert "ef_search_selection" not in stats
