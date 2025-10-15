"""
Unit tests for metrics module.

Tests recall@k, difficulty computation, and ground truth functions.
"""

import pytest
import numpy as np
from dynhnsw.metrics import (
    compute_recall_at_k,
    compute_query_difficulty,
    compute_ground_truth_brute_force,
    compute_precision_at_k,
    compute_mean_reciprocal_rank
)


class TestRecallAtK:
    """Tests for recall@k computation."""

    def test_perfect_recall(self):
        """Test recall@10 with perfect retrieval."""
        retrieved = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        ground_truth = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        recall = compute_recall_at_k(retrieved, ground_truth, k=10)
        assert recall == 1.0, "Perfect retrieval should give recall=1.0"

    def test_partial_recall(self):
        """Test recall@10 with 70% correct retrieval."""
        retrieved = [1, 2, 3, 4, 5, 6, 7, 99, 98, 97]
        ground_truth = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        recall = compute_recall_at_k(retrieved, ground_truth, k=10)
        assert recall == 0.7, "7/10 correct should give recall=0.7"

    def test_zero_recall(self):
        """Test recall@10 with no correct retrievals."""
        retrieved = [99, 98, 97, 96, 95, 94, 93, 92, 91, 90]
        ground_truth = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        recall = compute_recall_at_k(retrieved, ground_truth, k=10)
        assert recall == 0.0, "No correct retrievals should give recall=0.0"

    def test_recall_at_different_k(self):
        """Test recall with different k values."""
        retrieved = [1, 2, 3, 99, 98]
        ground_truth = [1, 2, 3, 4, 5]

        # k=3: all correct
        recall_3 = compute_recall_at_k(retrieved, ground_truth, k=3)
        assert recall_3 == 1.0

        # k=5: 3/5 correct
        recall_5 = compute_recall_at_k(retrieved, ground_truth, k=5)
        assert recall_5 == 0.6

    def test_order_independence(self):
        """Test that recall doesn't depend on order."""
        retrieved_1 = [1, 2, 3, 4, 5]
        retrieved_2 = [5, 4, 3, 2, 1]  # Reversed
        ground_truth = [1, 2, 3, 4, 5]

        recall_1 = compute_recall_at_k(retrieved_1, ground_truth, k=5)
        recall_2 = compute_recall_at_k(retrieved_2, ground_truth, k=5)

        assert recall_1 == recall_2 == 1.0

    def test_empty_lists(self):
        """Test edge case with empty lists."""
        recall = compute_recall_at_k([], [], k=0)
        assert recall == 0.0


class TestGroundTruthBruteForce:
    """Tests for brute force ground truth computation."""

    def test_simple_case(self):
        """Test brute force k-NN on simple 2D data."""
        # Query at origin
        query = np.array([0.0, 0.0])

        # 4 points: (1,0), (0,1), (2,0), (0,2)
        database = np.array([
            [1.0, 0.0],
            [0.0, 1.0],
            [2.0, 0.0],
            [0.0, 2.0]
        ])

        ids, dists = compute_ground_truth_brute_force(query, database, k=2)

        # Nearest should be points 0 and 1 (distance 1.0 each)
        assert set(ids[:2]) == {0, 1}
        assert np.allclose(dists[:2], [1.0, 1.0], atol=0.01)

    def test_returns_sorted_by_distance(self):
        """Test that results are sorted by distance."""
        query = np.array([0.0, 0.0])
        database = np.array([
            [3.0, 0.0],  # Distance 3.0
            [1.0, 0.0],  # Distance 1.0
            [2.0, 0.0],  # Distance 2.0
        ])

        ids, dists = compute_ground_truth_brute_force(query, database, k=3)

        # Should be sorted: [1, 2, 0]
        assert ids == [1, 2, 0]
        assert np.allclose(dists, [1.0, 2.0, 3.0])

    def test_k_larger_than_database(self):
        """Test when k > number of vectors."""
        query = np.array([0.0, 0.0])
        database = np.array([[1.0, 0.0], [2.0, 0.0]])

        ids, dists = compute_ground_truth_brute_force(query, database, k=10)

        # Should return all vectors (2 in this case)
        assert len(ids) == 2
        assert len(dists) == 2

    def test_high_dimensional(self):
        """Test on high-dimensional vectors."""
        dim = 128
        query = np.random.randn(dim)
        database = np.random.randn(100, dim)

        ids, dists = compute_ground_truth_brute_force(query, database, k=10)

        # Basic sanity checks
        assert len(ids) == 10
        assert len(dists) == 10
        assert all(0 <= i < 100 for i in ids)
        assert all(d >= 0 for d in dists)
        assert dists == sorted(dists)  # Should be sorted


class TestQueryDifficulty:
    """Tests for query difficulty computation."""

    def test_difficulty_increases_with_distance(self):
        """Test that outlier queries have higher difficulty."""
        # Mock index that returns known distances
        class MockIndex:
            def search(self, query, k, ef_search):
                # Return fake results with increasing distances
                return [(i, float(i)) for i in range(k)]

        index = MockIndex()
        query = np.array([1.0, 2.0, 3.0])

        difficulty = compute_query_difficulty(index, query, k=10)

        # Difficulty should be 10th neighbor distance = 9.0
        assert difficulty == 9.0

    def test_inf_difficulty_when_not_enough_neighbors(self):
        """Test that difficulty is inf when fewer than k neighbors found."""
        class MockIndex:
            def search(self, query, k, ef_search):
                # Return only 5 neighbors instead of 10
                return [(i, float(i)) for i in range(5)]

        index = MockIndex()
        query = np.array([1.0])

        difficulty = compute_query_difficulty(index, query, k=10)

        # Should return inf since we got < 10 neighbors
        assert difficulty == float('inf')


class TestPrecisionAtK:
    """Tests for precision@k computation."""

    def test_precision_equals_recall_for_knn(self):
        """For k-NN, precision@k should equal recall@k."""
        retrieved = [1, 2, 3, 99, 98]
        ground_truth = [1, 2, 3, 4, 5]

        precision = compute_precision_at_k(retrieved, ground_truth, k=5)
        recall = compute_recall_at_k(retrieved, ground_truth, k=5)

        assert precision == recall == 0.6


class TestMeanReciprocalRank:
    """Tests for MRR computation."""

    def test_first_result_correct(self):
        """Test MRR when first result is correct."""
        retrieved = [1, 2, 3, 4, 5]
        ground_truth = [1, 99, 98, 97, 96]

        mrr = compute_mean_reciprocal_rank(retrieved, ground_truth)
        assert mrr == 1.0  # 1/1

    def test_third_result_correct(self):
        """Test MRR when first correct result is at position 3."""
        retrieved = [99, 98, 1, 2, 3]
        ground_truth = [1, 2, 3, 4, 5]

        mrr = compute_mean_reciprocal_rank(retrieved, ground_truth)
        assert np.isclose(mrr, 1.0/3.0)  # 1/3

    def test_no_correct_results(self):
        """Test MRR when no correct results found."""
        retrieved = [99, 98, 97, 96, 95]
        ground_truth = [1, 2, 3, 4, 5]

        mrr = compute_mean_reciprocal_rank(retrieved, ground_truth)
        assert mrr == 0.0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
