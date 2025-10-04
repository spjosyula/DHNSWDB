"""Unit tests for IntentAwareWeightLearner."""

import pytest
import numpy as np
import time
from dynhnsw.intent_aware_weights import IntentAwareWeightLearner
from dynhnsw.feedback import FeedbackCollector


@pytest.fixture
def learner():
    """Create weight learner with default settings."""
    return IntentAwareWeightLearner(
        k_intents=3,
        learning_rate=0.1,
        decay_half_life_seconds=7 * 24 * 3600
    )


@pytest.fixture
def feedback_collector():
    """Create feedback collector for test data."""
    return FeedbackCollector(buffer_size=100)


class TestIntentAwareWeightLearnerInitialization:
    """Test learner initialization."""

    def test_default_initialization(self):
        """Should initialize with default parameters."""
        learner = IntentAwareWeightLearner()

        assert learner.k_intents == 5
        assert learner.learning_rate == 0.05
        assert learner.decay_rate > 0
        assert len(learner.edge_weights) == 0

    def test_custom_parameters(self):
        """Should accept custom parameters."""
        learner = IntentAwareWeightLearner(
            k_intents=10,
            learning_rate=0.2,
            decay_half_life_seconds=86400.0  # 1 day
        )

        assert learner.k_intents == 10
        assert learner.learning_rate == 0.2

    def test_invalid_learning_rate(self):
        """Should reject invalid learning rate."""
        with pytest.raises(ValueError, match="learning_rate must be in"):
            IntentAwareWeightLearner(learning_rate=0.0)

        with pytest.raises(ValueError, match="learning_rate must be in"):
            IntentAwareWeightLearner(learning_rate=1.5)

    def test_invalid_decay_half_life(self):
        """Should reject invalid decay half life."""
        with pytest.raises(ValueError, match="decay_half_life_seconds must be positive"):
            IntentAwareWeightLearner(decay_half_life_seconds=-100.0)


class TestEdgeWeightAccess:
    """Test edge weight creation and retrieval."""

    def test_get_edge_weight_creates_new(self, learner):
        """Should create edge weight if it doesn't exist."""
        edge_weight = learner.get_edge_weight(node_u=10, node_v=20)

        assert edge_weight is not None
        assert edge_weight.k_intents == 3
        assert len(learner.edge_weights) == 1

    def test_get_edge_weight_returns_existing(self, learner):
        """Should return existing edge weight."""
        # First access
        edge_weight_1 = learner.get_edge_weight(node_u=5, node_v=15)
        edge_weight_1.global_weight = 1.5

        # Second access
        edge_weight_2 = learner.get_edge_weight(node_u=5, node_v=15)

        assert edge_weight_1 is edge_weight_2
        assert edge_weight_2.global_weight == 1.5

    def test_edge_order_doesnt_matter(self, learner):
        """Should treat (u, v) same as (v, u)."""
        edge_weight_1 = learner.get_edge_weight(node_u=3, node_v=7)
        edge_weight_2 = learner.get_edge_weight(node_u=7, node_v=3)

        # Should be different weights (order preserved in key)
        assert edge_weight_1 is not edge_weight_2


class TestEffectiveDistanceComputation:
    """Test intent-aware distance computation."""

    def test_effective_distance_cold_start(self, learner):
        """Should use global weight during cold start."""
        edge_weight = learner.get_edge_weight(node_u=1, node_v=2)
        edge_weight.global_weight = 0.8

        distance = learner.get_effective_distance(
            node_u=1,
            node_v=2,
            base_distance=0.5,
            intent_id=-1,
            confidence=0.0
        )

        # Expected: 0.5 * 0.8 = 0.4
        assert np.isclose(distance, 0.4)

    def test_effective_distance_with_intent(self, learner):
        """Should use intent-specific weight for high confidence."""
        edge_weight = learner.get_edge_weight(node_u=10, node_v=20)
        edge_weight.intent_weights[1] = 2.0

        distance = learner.get_effective_distance(
            node_u=10,
            node_v=20,
            base_distance=0.3,
            intent_id=1,
            confidence=1.0
        )

        # Expected: 0.3 * 2.0 = 0.6
        assert np.isclose(distance, 0.6)

    def test_effective_distance_with_blending(self, learner):
        """Should blend weights for moderate confidence."""
        edge_weight = learner.get_edge_weight(node_u=5, node_v=8)
        edge_weight.global_weight = 1.0
        edge_weight.intent_weights[0] = 2.0

        distance = learner.get_effective_distance(
            node_u=5,
            node_v=8,
            base_distance=0.4,
            intent_id=0,
            confidence=0.5
        )

        # Effective weight: 0.5 * 2.0 + 0.5 * 1.0 = 1.5
        # Distance: 0.4 * 1.5 = 0.6
        assert np.isclose(distance, 0.6)


class TestFeedbackUpdate:
    """Test weight updates from feedback."""

    def test_update_from_feedback_cold_start(self, learner, feedback_collector):
        """Should update global weight during cold start."""
        query = np.random.randn(128).astype(np.float32)
        result_ids = [1, 2, 3]
        relevant_ids = {1, 2}  # High satisfaction

        feedback = feedback_collector.add_feedback(
            query_vector=query,
            result_ids=result_ids,
            relevant_ids=relevant_ids,
            timestamp=time.time()
        )

        traversed_edges = [(0, 1), (1, 2), (2, 3)]

        learner.update_from_feedback(
            feedback=feedback,
            traversed_edges=traversed_edges,
            intent_id=-1,
            confidence=0.0
        )

        # Should have updated global weights for all edges
        for edge_id in traversed_edges:
            edge_weight = learner.get_edge_weight(edge_id[0], edge_id[1])
            # High satisfaction -> reward < 1.0 -> weight should decrease
            assert edge_weight.global_weight < 1.0

    def test_update_from_feedback_with_intent(self, learner, feedback_collector):
        """Should update intent-specific weight for high confidence."""
        query = np.random.randn(128).astype(np.float32)
        result_ids = [1, 2, 3, 4, 5]
        relevant_ids = {1}  # Low satisfaction

        feedback = feedback_collector.add_feedback(
            query_vector=query,
            result_ids=result_ids,
            relevant_ids=relevant_ids,
            timestamp=time.time()
        )

        traversed_edges = [(5, 10), (10, 15)]

        learner.update_from_feedback(
            feedback=feedback,
            traversed_edges=traversed_edges,
            intent_id=1,
            confidence=0.9
        )

        # Should have updated intent-specific weights
        for edge_id in traversed_edges:
            edge_weight = learner.get_edge_weight(edge_id[0], edge_id[1])
            # Low satisfaction -> reward > 1.0 -> weight should increase
            assert edge_weight.intent_weights[1] > 1.0

    def test_satisfaction_to_reward_conversion(self, learner):
        """Should convert satisfaction to correct reward signal."""
        # Perfect satisfaction
        reward = learner._satisfaction_to_reward(1.0)
        assert np.isclose(reward, 0.8)  # 1.2 - 0.4 * 1.0

        # Neutral satisfaction
        reward = learner._satisfaction_to_reward(0.5)
        assert np.isclose(reward, 1.0)  # 1.2 - 0.4 * 0.5

        # No satisfaction
        reward = learner._satisfaction_to_reward(0.0)
        assert np.isclose(reward, 1.2)  # 1.2 - 0.4 * 0.0

    def test_update_increments_usage(self, learner, feedback_collector):
        """Should track usage per intent."""
        query = np.random.randn(128).astype(np.float32)
        feedback = feedback_collector.add_feedback(
            query_vector=query,
            result_ids=[1, 2],
            relevant_ids={1},
            timestamp=time.time()
        )

        traversed_edges = [(1, 2), (2, 3)]

        learner.update_from_feedback(
            feedback=feedback,
            traversed_edges=traversed_edges,
            intent_id=2,
            confidence=0.8
        )

        # Check usage incremented
        for edge_id in traversed_edges:
            edge_weight = learner.get_edge_weight(edge_id[0], edge_id[1])
            assert edge_weight.usage_counts[2] == 1

    def test_empty_edges_doesnt_crash(self, learner, feedback_collector):
        """Should handle empty traversed edges gracefully."""
        query = np.random.randn(128).astype(np.float32)
        feedback = feedback_collector.add_feedback(
            query_vector=query,
            result_ids=[1],
            relevant_ids={1},
            timestamp=time.time()
        )

        learner.update_from_feedback(
            feedback=feedback,
            traversed_edges=[],
            intent_id=0,
            confidence=0.9
        )

        # Should not crash, no weights updated
        assert len(learner.edge_weights) == 0


class TestTemporalDecayApplication:
    """Test temporal decay application."""

    def test_apply_decay_to_all(self, learner):
        """Should apply decay to all edge weights."""
        # Create some edges with non-neutral weights
        for i in range(5):
            edge_weight = learner.get_edge_weight(i, i+1)
            edge_weight.global_weight = 1.5
            edge_weight.intent_weights[:] = 2.0
            edge_weight.last_update_time = time.time() - 1000.0

        learner.apply_decay_to_all()

        # All weights should have decayed toward 1.0
        for edge_weight in learner.edge_weights.values():
            assert edge_weight.global_weight < 1.5
            assert edge_weight.global_weight > 1.0
            assert np.all(edge_weight.intent_weights < 2.0)
            assert np.all(edge_weight.intent_weights > 1.0)


class TestWeightStatistics:
    """Test weight statistics computation."""

    def test_statistics_for_empty_learner(self):
        """Should return valid statistics even with no edges."""
        learner = IntentAwareWeightLearner(k_intents=3)

        stats = learner.get_weight_statistics()

        assert stats["total_edges"] == 0
        assert stats["global_weights"]["mean"] == 1.0
        assert "intent_weights" in stats

    def test_statistics_with_edges(self, learner):
        """Should compute correct statistics."""
        # Create edges with different weights
        edge1 = learner.get_edge_weight(1, 2)
        edge1.global_weight = 0.8

        edge2 = learner.get_edge_weight(3, 4)
        edge2.global_weight = 1.2

        edge3 = learner.get_edge_weight(5, 6)
        edge3.global_weight = 1.0

        stats = learner.get_weight_statistics()

        assert stats["total_edges"] == 3
        assert np.isclose(stats["global_weights"]["mean"], 1.0)
        assert stats["global_weights"]["min"] == 0.8
        assert stats["global_weights"]["max"] == 1.2

    def test_intent_weight_statistics(self, learner):
        """Should compute per-intent statistics."""
        edge1 = learner.get_edge_weight(1, 2)
        edge1.intent_weights[0] = 0.5
        edge1.intent_weights[1] = 1.0
        edge1.intent_weights[2] = 1.5

        edge2 = learner.get_edge_weight(3, 4)
        edge2.intent_weights[0] = 0.7
        edge2.intent_weights[1] = 1.2
        edge2.intent_weights[2] = 1.3

        stats = learner.get_weight_statistics()

        # Check intent_0 stats (use approximate comparison for float32)
        intent_0 = stats["intent_weights"]["intent_0"]
        assert np.isclose(intent_0["min"], 0.5, rtol=1e-5)
        assert np.isclose(intent_0["max"], 0.7, rtol=1e-5)
        assert np.isclose(intent_0["mean"], 0.6, rtol=1e-5)


class TestIntentUsageStatistics:
    """Test intent usage statistics."""

    def test_usage_statistics_empty(self):
        """Should return zero usage for empty learner."""
        learner = IntentAwareWeightLearner(k_intents=3)

        usage = learner.get_intent_usage_statistics()

        assert len(usage) == 3
        for intent_id in range(3):
            assert usage[intent_id]["total_traversals"] == 0
            assert usage[intent_id]["edges_used"] == 0

    def test_usage_statistics_with_data(self, learner):
        """Should compute correct usage statistics."""
        # Simulate usage
        edge1 = learner.get_edge_weight(1, 2)
        edge1.usage_counts[0] = 10
        edge1.usage_counts[1] = 5

        edge2 = learner.get_edge_weight(3, 4)
        edge2.usage_counts[0] = 8
        edge2.usage_counts[2] = 3

        usage = learner.get_intent_usage_statistics()

        # Intent 0: 10 + 8 = 18 traversals, 2 edges
        assert usage[0]["total_traversals"] == 18
        assert usage[0]["edges_used"] == 2

        # Intent 1: 5 traversals, 1 edge
        assert usage[1]["total_traversals"] == 5
        assert usage[1]["edges_used"] == 1

        # Intent 2: 3 traversals, 1 edge
        assert usage[2]["total_traversals"] == 3
        assert usage[2]["edges_used"] == 1


class TestTotalEdgeCount:
    """Test total edge tracking."""

    def test_get_total_edges(self, learner):
        """Should return correct total edge count."""
        assert learner.get_total_edges() == 0

        learner.get_edge_weight(1, 2)
        assert learner.get_total_edges() == 1

        learner.get_edge_weight(3, 4)
        learner.get_edge_weight(5, 6)
        assert learner.get_total_edges() == 3

        # Accessing same edge shouldn't increase count
        learner.get_edge_weight(1, 2)
        assert learner.get_total_edges() == 3
