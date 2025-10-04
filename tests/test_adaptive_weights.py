"""Integration tests for adaptive edge weight learning."""

import time
import math
import numpy as np
import pytest
from dynhnsw.adaptive_weights import EdgeWeightLearner
from dynhnsw.feedback import QueryFeedback


class TestEdgeWeightLearner:
    """Test EdgeWeightLearner initialization and basic functionality."""

    def test_initialization_with_defaults(self):
        """Should initialize with default parameters."""
        learner = EdgeWeightLearner()

        assert learner.learning_rate == 0.05
        assert learner.get_total_edges() == 0

    def test_initialization_with_custom_params(self):
        """Should accept custom learning rate and decay parameters."""
        learner = EdgeWeightLearner(learning_rate=0.1, decay_half_life_seconds=3600)

        assert learner.learning_rate == 0.1
        # Decay rate should be ln(2) / half_life
        expected_decay_rate = math.log(2) / 3600
        assert learner.decay_rate == pytest.approx(expected_decay_rate)

    def test_invalid_learning_rate_raises_error(self):
        """Invalid learning rate should raise ValueError."""
        with pytest.raises(ValueError, match="learning_rate must be in"):
            EdgeWeightLearner(learning_rate=0.0)

        with pytest.raises(ValueError, match="learning_rate must be in"):
            EdgeWeightLearner(learning_rate=1.0)

        with pytest.raises(ValueError, match="learning_rate must be in"):
            EdgeWeightLearner(learning_rate=-0.1)

    def test_invalid_decay_half_life_raises_error(self):
        """Invalid decay half life should raise ValueError."""
        with pytest.raises(ValueError, match="decay_half_life_seconds must be positive"):
            EdgeWeightLearner(decay_half_life_seconds=0.0)

        with pytest.raises(ValueError, match="decay_half_life_seconds must be positive"):
            EdgeWeightLearner(decay_half_life_seconds=-100)

    def test_get_edge_weight_creates_new_edge(self):
        """Getting a non-existent edge should create it with neutral weight."""
        learner = EdgeWeightLearner()

        edge = learner.get_edge_weight(1, 2)

        assert edge.weight == 1.0
        assert learner.get_total_edges() == 1

    def test_get_edge_weight_returns_existing_edge(self):
        """Getting an existing edge should return the same object."""
        learner = EdgeWeightLearner()

        edge1 = learner.get_edge_weight(1, 2)
        edge1.weight = 2.5  # Modify it

        edge2 = learner.get_edge_weight(1, 2)

        assert edge2.weight == 2.5  # Should be same object
        assert edge1 is edge2


class TestEffectiveDistance:
    """Test effective distance computation with learned weights."""

    def test_effective_distance_neutral_weight(self):
        """Neutral weight should return base distance unchanged."""
        learner = EdgeWeightLearner()

        effective = learner.get_effective_distance(1, 2, base_distance=10.0)

        assert effective == pytest.approx(10.0)

    def test_effective_distance_with_learned_weight(self):
        """Should apply learned weight to base distance."""
        learner = EdgeWeightLearner()

        # Manually set a weight
        edge = learner.get_edge_weight(1, 2)
        edge.weight = 2.0

        effective = learner.get_effective_distance(1, 2, base_distance=5.0)

        assert effective == pytest.approx(10.0)  # 2.0 * 5.0


class TestFeedbackProcessing:
    """Test updating edge weights from user feedback."""

    def test_positive_feedback_decreases_weights(self):
        """Good feedback should make edges faster (lower weight)."""
        learner = EdgeWeightLearner(learning_rate=0.5)  # High learning rate for fast test

        # Create positive feedback (all results relevant)
        query_vec = np.array([1.0, 2.0], dtype=np.float32)
        feedback = QueryFeedback(
            query_vector=query_vec,
            result_ids=[1, 2, 3],
            relevant_ids={1, 2, 3},  # 100% satisfaction
            timestamp=time.time(),
        )

        # Simulate traversing these edges
        traversed_edges = [(0, 1), (1, 2), (2, 3)]

        learner.update_from_feedback(feedback, traversed_edges)

        # All edges should have weight < 1.0 (preferred)
        for edge_id in traversed_edges:
            edge = learner.get_edge_weight(edge_id[0], edge_id[1])
            assert edge.weight < 1.0

    def test_negative_feedback_increases_weights(self):
        """Bad feedback should make edges slower (higher weight)."""
        learner = EdgeWeightLearner(learning_rate=0.5)

        # Create negative feedback (no results relevant)
        query_vec = np.array([1.0, 2.0], dtype=np.float32)
        feedback = QueryFeedback(
            query_vector=query_vec,
            result_ids=[1, 2, 3],
            relevant_ids=set(),  # 0% satisfaction
            timestamp=time.time(),
        )

        traversed_edges = [(0, 1), (1, 2)]

        learner.update_from_feedback(feedback, traversed_edges)

        # All edges should have weight > 1.0 (discouraged)
        for edge_id in traversed_edges:
            edge = learner.get_edge_weight(edge_id[0], edge_id[1])
            assert edge.weight > 1.0

    def test_neutral_feedback_maintains_weights(self):
        """50% satisfaction should keep weights near neutral."""
        learner = EdgeWeightLearner(learning_rate=0.1)

        # Create neutral feedback (half results relevant)
        query_vec = np.array([1.0, 2.0], dtype=np.float32)
        feedback = QueryFeedback(
            query_vector=query_vec,
            result_ids=[1, 2, 3, 4],
            relevant_ids={1, 2},  # 50% satisfaction
            timestamp=time.time(),
        )

        traversed_edges = [(0, 1), (1, 2)]

        learner.update_from_feedback(feedback, traversed_edges)

        # Weights should be close to 1.0
        for edge_id in traversed_edges:
            edge = learner.get_edge_weight(edge_id[0], edge_id[1])
            assert edge.weight == pytest.approx(1.0, abs=0.1)

    def test_empty_edge_list_no_updates(self):
        """Empty traversed_edges should not cause errors."""
        learner = EdgeWeightLearner()

        query_vec = np.array([1.0], dtype=np.float32)
        feedback = QueryFeedback(
            query_vector=query_vec,
            result_ids=[1],
            relevant_ids={1},
            timestamp=time.time(),
        )

        # Should not raise an error
        learner.update_from_feedback(feedback, traversed_edges=[])

        assert learner.get_total_edges() == 0

    def test_usage_count_incremented(self):
        """Edges should track usage count after updates."""
        learner = EdgeWeightLearner()

        query_vec = np.array([1.0], dtype=np.float32)
        feedback = QueryFeedback(
            query_vector=query_vec,
            result_ids=[1],
            relevant_ids={1},
            timestamp=time.time(),
        )

        traversed_edges = [(0, 1), (1, 2)]

        learner.update_from_feedback(feedback, traversed_edges)

        # Check usage counts
        edge1 = learner.get_edge_weight(0, 1)
        edge2 = learner.get_edge_weight(1, 2)

        assert edge1.usage_count == 1
        assert edge2.usage_count == 1

        # Update again
        learner.update_from_feedback(feedback, traversed_edges)

        assert edge1.usage_count == 2
        assert edge2.usage_count == 2


class TestSatisfactionToReward:
    """Test the satisfaction -> reward conversion."""

    def test_perfect_satisfaction_gives_low_reward(self):
        """100% satisfaction should give reward < 1.0 (decrease weights)."""
        learner = EdgeWeightLearner()

        reward = learner._satisfaction_to_reward(1.0)

        assert reward == pytest.approx(0.8)  # 1.2 - 0.4 * 1.0

    def test_zero_satisfaction_gives_high_reward(self):
        """0% satisfaction should give reward > 1.0 (increase weights)."""
        learner = EdgeWeightLearner()

        reward = learner._satisfaction_to_reward(0.0)

        assert reward == pytest.approx(1.2)  # 1.2 - 0.4 * 0.0

    def test_neutral_satisfaction_gives_neutral_reward(self):
        """50% satisfaction should give reward = 1.0 (no change)."""
        learner = EdgeWeightLearner()

        reward = learner._satisfaction_to_reward(0.5)

        assert reward == pytest.approx(1.0)  # 1.2 - 0.4 * 0.5


class TestTemporalDecayIntegration:
    """Test temporal decay in learning context."""

    def test_apply_decay_to_all_edges(self):
        """apply_decay_to_all should decay all edge weights."""
        learner = EdgeWeightLearner(decay_half_life_seconds=1.0)

        # Create some edges with non-neutral weights
        edge1 = learner.get_edge_weight(0, 1)
        edge1.weight = 5.0
        edge1.last_update_time = time.time() - 2.0  # 2 seconds ago

        edge2 = learner.get_edge_weight(1, 2)
        edge2.weight = 0.5
        edge2.last_update_time = time.time() - 2.0

        learner.apply_decay_to_all()

        # Both should have moved toward 1.0
        assert 1.0 < edge1.weight < 5.0
        assert 0.5 < edge2.weight < 1.0

    def test_decay_applied_before_update(self):
        """Feedback update should apply decay first."""
        learner = EdgeWeightLearner(learning_rate=0.1, decay_half_life_seconds=1.0)

        # Set an edge weight in the past
        edge = learner.get_edge_weight(0, 1)
        edge.weight = 5.0
        edge.last_update_time = time.time() - 5.0  # 5 seconds ago

        # Apply feedback
        query_vec = np.array([1.0], dtype=np.float32)
        feedback = QueryFeedback(
            query_vector=query_vec,
            result_ids=[1],
            relevant_ids={1},
            timestamp=time.time(),
        )

        learner.update_from_feedback(feedback, [(0, 1)])

        # Weight should be affected by decay before the update
        # With 5 seconds elapsed and half_life=1, decay is significant
        # Weight should be much closer to 1.0 than original 5.0
        assert edge.weight < 3.0  # Decayed significantly


class TestWeightStatistics:
    """Test weight statistics computation."""

    def test_statistics_empty_learner(self):
        """Empty learner should return neutral statistics."""
        learner = EdgeWeightLearner()

        stats = learner.get_weight_statistics()

        assert stats["min"] == 1.0
        assert stats["max"] == 1.0
        assert stats["mean"] == 1.0
        assert stats["count"] == 0

    def test_statistics_with_weights(self):
        """Should compute correct statistics."""
        learner = EdgeWeightLearner()

        # Create edges with known weights
        learner.get_edge_weight(0, 1).weight = 0.5
        learner.get_edge_weight(1, 2).weight = 1.0
        learner.get_edge_weight(2, 3).weight = 2.0

        stats = learner.get_weight_statistics()

        assert stats["min"] == pytest.approx(0.5)
        assert stats["max"] == pytest.approx(2.0)
        assert stats["mean"] == pytest.approx(1.166666, abs=0.01)
        assert stats["count"] == 3


class TestEndToEndLearning:
    """End-to-end learning scenarios."""

    def test_consistent_positive_feedback_converges(self):
        """Consistent good feedback should converge weights to low values."""
        learner = EdgeWeightLearner(learning_rate=0.1)

        query_vec = np.array([1.0], dtype=np.float32)
        traversed_edges = [(0, 1), (1, 2)]

        # Apply consistent positive feedback
        for _ in range(50):
            feedback = QueryFeedback(
                query_vector=query_vec,
                result_ids=[1, 2, 3],
                relevant_ids={1, 2, 3},  # 100% satisfaction
                timestamp=time.time(),
            )
            learner.update_from_feedback(feedback, traversed_edges)

        # Weights should converge to 0.8 (perfect satisfaction reward)
        edge1 = learner.get_edge_weight(0, 1)
        edge2 = learner.get_edge_weight(1, 2)

        assert edge1.weight == pytest.approx(0.8, abs=0.05)
        assert edge2.weight == pytest.approx(0.8, abs=0.05)

    def test_alternating_feedback_stays_stable(self):
        """Alternating feedback should keep weights near neutral."""
        learner = EdgeWeightLearner(learning_rate=0.1)

        query_vec = np.array([1.0], dtype=np.float32)
        traversed_edges = [(0, 1)]

        # Alternate between good and bad feedback
        for i in range(100):
            if i % 2 == 0:
                relevant = {1, 2, 3}  # Good
            else:
                relevant = set()  # Bad

            feedback = QueryFeedback(
                query_vector=query_vec,
                result_ids=[1, 2, 3],
                relevant_ids=relevant,
                timestamp=time.time(),
            )
            learner.update_from_feedback(feedback, traversed_edges)

        # Weight should be near neutral (1.0)
        edge = learner.get_edge_weight(0, 1)
        assert edge.weight == pytest.approx(1.0, abs=0.2)
