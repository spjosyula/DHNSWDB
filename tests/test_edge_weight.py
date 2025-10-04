"""Unit tests for EdgeWeight data structure."""

import time
import math
import pytest
from dynhnsw.models.edge_weight import EdgeWeight


class TestEdgeWeightBounds:
    """Test that edge weights stay within valid bounds."""

    def test_initial_weight_is_neutral(self):
        """New edge weight should start at 1.0 (neutral)."""
        edge = EdgeWeight()
        assert edge.weight == 1.0

    def test_weight_below_minimum_gets_clamped(self):
        """Weights below MIN_WEIGHT should be clamped."""
        edge = EdgeWeight(weight=0.01)  # Below 0.1 minimum
        assert edge.weight == EdgeWeight.MIN_WEIGHT

    def test_weight_above_maximum_gets_clamped(self):
        """Weights above MAX_WEIGHT should be clamped."""
        edge = EdgeWeight(weight=100.0)  # Above 10.0 maximum
        assert edge.weight == EdgeWeight.MAX_WEIGHT

    def test_update_enforces_minimum_bound(self):
        """Update should not allow weight to go below minimum."""
        edge = EdgeWeight()
        # Try to push weight very low with multiple updates
        for _ in range(100):
            edge.update(reward_signal=0.01, learning_rate=0.5)

        assert edge.weight >= EdgeWeight.MIN_WEIGHT

    def test_update_enforces_maximum_bound(self):
        """Update should not allow weight to go above maximum."""
        edge = EdgeWeight()
        # Try to push weight very high with multiple updates
        for _ in range(100):
            edge.update(reward_signal=50.0, learning_rate=0.5)

        assert edge.weight <= EdgeWeight.MAX_WEIGHT


class TestEdgeWeightUpdate:
    """Test edge weight update mechanics."""

    def test_update_with_zero_learning_rate_no_change(self):
        """Zero learning rate should freeze the weight."""
        edge = EdgeWeight(weight=2.0)
        original_weight = edge.weight

        edge.update(reward_signal=0.5, learning_rate=0.0)

        assert edge.weight == original_weight

    def test_update_with_full_learning_rate_immediate_change(self):
        """Learning rate of 1.0 should immediately set to reward signal."""
        edge = EdgeWeight(weight=2.0)

        edge.update(reward_signal=1.5, learning_rate=1.0)

        assert edge.weight == pytest.approx(1.5)

    def test_update_exponential_smoothing_behavior(self):
        """Update should blend old weight with new signal."""
        edge = EdgeWeight(weight=2.0)
        learning_rate = 0.1
        reward = 1.0

        edge.update(reward_signal=reward, learning_rate=learning_rate)

        expected = 2.0 * (1 - learning_rate) + reward * learning_rate
        assert edge.weight == pytest.approx(expected)

    def test_repeated_updates_converge_to_signal(self):
        """Repeated updates should converge toward reward signal."""
        edge = EdgeWeight(weight=5.0)
        target_signal = 1.5
        learning_rate = 0.1

        # Apply many updates
        for _ in range(100):
            edge.update(reward_signal=target_signal, learning_rate=learning_rate)

        # Should be very close to target
        assert edge.weight == pytest.approx(target_signal, abs=0.01)


class TestTemporalDecay:
    """Test temporal decay functionality."""

    def test_decay_moves_weight_toward_neutral(self):
        """Decay should move weights back toward 1.0."""
        edge = EdgeWeight(weight=5.0)
        edge.last_update_time = time.time() - 100  # 100 seconds ago

        decay_rate = 0.01  # Moderate decay
        edge.apply_temporal_decay(decay_rate)

        # Weight should be between original (5.0) and neutral (1.0)
        assert 1.0 < edge.weight < 5.0

    def test_decay_below_neutral_moves_up(self):
        """Decay should move low weights up toward 1.0."""
        edge = EdgeWeight(weight=0.5)
        edge.last_update_time = time.time() - 100

        decay_rate = 0.01
        edge.apply_temporal_decay(decay_rate)

        # Weight should be between original (0.5) and neutral (1.0)
        assert 0.5 < edge.weight < 1.0

    def test_no_time_elapsed_no_decay(self):
        """If no time passed, decay should not change weight."""
        edge = EdgeWeight(weight=2.0)
        edge.last_update_time = time.time()  # Right now

        decay_rate = 0.01
        edge.apply_temporal_decay(decay_rate)

        # Weight should be essentially unchanged (might have tiny floating point difference)
        assert edge.weight == pytest.approx(2.0, abs=0.01)

    def test_decay_formula_correctness(self):
        """Verify exponential decay formula is correct."""
        edge = EdgeWeight(weight=4.0)
        time_elapsed = 100.0
        edge.last_update_time = time.time() - time_elapsed

        decay_rate = 0.01
        edge.apply_temporal_decay(decay_rate)

        # Manual calculation: w_new = w_old * exp(-λt) + 1.0 * (1 - exp(-λt))
        decay_factor = math.exp(-decay_rate * time_elapsed)
        expected = 4.0 * decay_factor + 1.0 * (1 - decay_factor)

        assert edge.weight == pytest.approx(expected)

    def test_decay_updates_last_update_time(self):
        """Decay should update the last_update_time."""
        edge = EdgeWeight()
        old_time = edge.last_update_time

        time.sleep(0.01)  # Small delay
        edge.apply_temporal_decay(0.01)

        assert edge.last_update_time > old_time


class TestEdgeWeightUsage:
    """Test usage tracking and distance computation."""

    def test_initial_usage_count_is_zero(self):
        """New edge should have zero usage count."""
        edge = EdgeWeight()
        assert edge.usage_count == 0

    def test_increment_usage_increases_count(self):
        """increment_usage should increase count by 1."""
        edge = EdgeWeight()

        edge.increment_usage()
        assert edge.usage_count == 1

        edge.increment_usage()
        assert edge.usage_count == 2

    def test_effective_distance_computation(self):
        """Effective distance should be weight * base_distance."""
        edge = EdgeWeight(weight=2.5)
        base_distance = 10.0

        effective = edge.get_effective_distance(base_distance)

        assert effective == pytest.approx(25.0)

    def test_effective_distance_with_neutral_weight(self):
        """Neutral weight (1.0) should return unchanged distance."""
        edge = EdgeWeight(weight=1.0)
        base_distance = 7.5

        effective = edge.get_effective_distance(base_distance)

        assert effective == pytest.approx(7.5)

    def test_effective_distance_with_low_weight(self):
        """Low weight makes distance shorter (preferred path)."""
        edge = EdgeWeight(weight=0.5)
        base_distance = 10.0

        effective = edge.get_effective_distance(base_distance)

        assert effective == pytest.approx(5.0)
        assert effective < base_distance  # Shorter = preferred
