"""Unit tests for IntentAwareEdgeWeight model."""

import pytest
import numpy as np
import time
from dynhnsw.models.intent_aware_edge_weight import IntentAwareEdgeWeight


class TestIntentAwareEdgeWeightInitialization:
    """Test edge weight initialization."""

    def test_default_initialization(self):
        """Should initialize with default parameters."""
        edge_weight = IntentAwareEdgeWeight(k_intents=5)

        assert edge_weight.k_intents == 5
        assert edge_weight.global_weight == 1.0
        assert len(edge_weight.intent_weights) == 5
        assert np.allclose(edge_weight.intent_weights, 1.0)
        assert len(edge_weight.usage_counts) == 5
        assert np.all(edge_weight.usage_counts == 0)
        assert edge_weight.last_update_time > 0

    def test_custom_k_intents(self):
        """Should support custom number of intents."""
        edge_weight = IntentAwareEdgeWeight(k_intents=10)

        assert edge_weight.k_intents == 10
        assert len(edge_weight.intent_weights) == 10
        assert len(edge_weight.usage_counts) == 10


class TestEffectiveWeightComputation:
    """Test effective weight computation with blending."""

    def test_cold_start_uses_global_weight(self):
        """Should use global weight during cold start."""
        edge_weight = IntentAwareEdgeWeight(k_intents=5)
        edge_weight.global_weight = 0.8

        # Cold start: intent_id = -1
        weight = edge_weight.get_effective_weight(intent_id=-1, confidence=0.0)

        assert weight == 0.8

    def test_low_confidence_uses_global_weight(self):
        """Should use global weight for very low confidence."""
        edge_weight = IntentAwareEdgeWeight(k_intents=5)
        edge_weight.global_weight = 0.7
        edge_weight.intent_weights[0] = 1.5

        # Very low confidence
        weight = edge_weight.get_effective_weight(intent_id=0, confidence=0.05)

        assert weight == 0.7

    def test_high_confidence_uses_intent_weight(self):
        """Should use intent-specific weight for high confidence."""
        edge_weight = IntentAwareEdgeWeight(k_intents=5)
        edge_weight.global_weight = 1.0
        edge_weight.intent_weights[2] = 1.5

        # High confidence (100%)
        weight = edge_weight.get_effective_weight(intent_id=2, confidence=1.0)

        assert np.isclose(weight, 1.5)

    def test_moderate_confidence_blends_weights(self):
        """Should blend global and intent weights for moderate confidence."""
        edge_weight = IntentAwareEdgeWeight(k_intents=5)
        edge_weight.global_weight = 1.0
        edge_weight.intent_weights[1] = 2.0

        # 50% confidence
        weight = edge_weight.get_effective_weight(intent_id=1, confidence=0.5)

        # Expected: 0.5 * 2.0 + 0.5 * 1.0 = 1.5
        assert np.isclose(weight, 1.5)

    def test_blending_formula_correctness(self):
        """Should apply correct blending formula."""
        edge_weight = IntentAwareEdgeWeight(k_intents=5)
        edge_weight.global_weight = 0.8
        edge_weight.intent_weights[3] = 1.6

        # 70% confidence
        weight = edge_weight.get_effective_weight(intent_id=3, confidence=0.7)

        # Expected: 0.7 * 1.6 + 0.3 * 0.8 = 1.12 + 0.24 = 1.36
        assert np.isclose(weight, 1.36)


class TestIntentWeightUpdates:
    """Test intent-specific weight updates."""

    def test_update_global_weight_during_cold_start(self):
        """Should update global weight when intent_id is -1."""
        edge_weight = IntentAwareEdgeWeight(k_intents=5)
        edge_weight.global_weight = 1.0

        # Cold start update
        edge_weight.update_intent_weight(
            intent_id=-1,
            reward_signal=1.2,
            learning_rate=0.1,
            confidence=0.0
        )

        # Expected: 1.0 * 0.9 + 1.2 * 0.1 = 0.9 + 0.12 = 1.02
        assert np.isclose(edge_weight.global_weight, 1.02)

    def test_update_global_weight_for_low_confidence(self):
        """Should update global weight for low confidence."""
        edge_weight = IntentAwareEdgeWeight(k_intents=5)
        edge_weight.global_weight = 1.0

        # Low confidence (< 0.5)
        edge_weight.update_intent_weight(
            intent_id=2,
            reward_signal=0.8,
            learning_rate=0.1,
            confidence=0.4
        )

        # Should update global, not intent-specific
        assert np.isclose(edge_weight.global_weight, 0.98)
        assert np.isclose(edge_weight.intent_weights[2], 1.0)  # Unchanged

    def test_update_intent_weight_for_high_confidence(self):
        """Should update intent-specific weight for high confidence."""
        edge_weight = IntentAwareEdgeWeight(k_intents=5)
        edge_weight.intent_weights[1] = 1.0

        # High confidence
        edge_weight.update_intent_weight(
            intent_id=1,
            reward_signal=1.4,
            learning_rate=0.1,
            confidence=0.8
        )

        # Expected: 1.0 * 0.9 + 1.4 * 0.1 = 0.9 + 0.14 = 1.04
        assert np.isclose(edge_weight.intent_weights[1], 1.04)

    def test_weight_bounds_enforcement(self):
        """Should enforce min/max weight bounds."""
        edge_weight = IntentAwareEdgeWeight(k_intents=5)

        # Try to set very high reward
        edge_weight.update_intent_weight(
            intent_id=0,
            reward_signal=20.0,  # Way above MAX_WEIGHT
            learning_rate=0.5,
            confidence=0.9
        )

        # Should be clamped to MAX_WEIGHT (10.0)
        assert edge_weight.intent_weights[0] <= 10.0

        # Try to set very low reward
        edge_weight.update_intent_weight(
            intent_id=1,
            reward_signal=0.01,  # Way below MIN_WEIGHT
            learning_rate=0.5,
            confidence=0.9
        )

        # Should be clamped to MIN_WEIGHT (0.1)
        assert edge_weight.intent_weights[1] >= 0.1


class TestTemporalDecay:
    """Test temporal decay toward neutral weights."""

    def test_decay_moves_toward_one(self):
        """Should decay weights toward 1.0."""
        edge_weight = IntentAwareEdgeWeight(k_intents=5)
        edge_weight.global_weight = 2.0
        edge_weight.intent_weights[:] = 1.5
        edge_weight.last_update_time = time.time() - 1000.0  # 1000 seconds ago

        # Decay rate: log(2) / 604800 ≈ 1.1e-6 per second
        decay_rate = 0.693 / 604800.0

        edge_weight.apply_temporal_decay(decay_rate)

        # After decay, weights should be closer to 1.0
        assert edge_weight.global_weight < 2.0
        assert edge_weight.global_weight > 1.0
        assert np.all(edge_weight.intent_weights < 1.5)
        assert np.all(edge_weight.intent_weights > 1.0)

    def test_decay_updates_timestamp(self):
        """Should update last_update_time after decay."""
        edge_weight = IntentAwareEdgeWeight(k_intents=5)
        old_time = edge_weight.last_update_time

        time.sleep(0.01)  # Small delay

        edge_weight.apply_temporal_decay(decay_rate=0.001)

        assert edge_weight.last_update_time > old_time

    def test_decay_respects_bounds(self):
        """Should keep weights within bounds after decay."""
        edge_weight = IntentAwareEdgeWeight(k_intents=5)
        edge_weight.global_weight = 10.0  # MAX
        edge_weight.intent_weights[:] = 0.1  # MIN

        edge_weight.apply_temporal_decay(decay_rate=0.01)

        assert 0.1 <= edge_weight.global_weight <= 10.0
        assert np.all(edge_weight.intent_weights >= 0.1)
        assert np.all(edge_weight.intent_weights <= 10.0)


class TestUsageTracking:
    """Test intent usage tracking."""

    def test_increment_usage_for_valid_intent(self):
        """Should increment usage count for valid intent."""
        edge_weight = IntentAwareEdgeWeight(k_intents=5)

        edge_weight.increment_usage(intent_id=2)
        edge_weight.increment_usage(intent_id=2)
        edge_weight.increment_usage(intent_id=3)

        assert edge_weight.usage_counts[2] == 2
        assert edge_weight.usage_counts[3] == 1
        assert edge_weight.usage_counts[0] == 0

    def test_ignore_invalid_intent_id(self):
        """Should ignore usage increment for invalid intent."""
        edge_weight = IntentAwareEdgeWeight(k_intents=5)

        # Negative intent
        edge_weight.increment_usage(intent_id=-1)

        # Out of range
        edge_weight.increment_usage(intent_id=10)

        # Should not crash, all counts remain 0
        assert np.all(edge_weight.usage_counts == 0)


class TestEffectiveDistance:
    """Test effective distance computation."""

    def test_effective_distance_scales_by_weight(self):
        """Should scale distance by effective weight."""
        edge_weight = IntentAwareEdgeWeight(k_intents=5)
        edge_weight.intent_weights[1] = 2.0

        base_distance = 0.5
        effective_dist = edge_weight.get_effective_distance(
            base_distance=base_distance,
            intent_id=1,
            confidence=1.0
        )

        # Expected: 0.5 * 2.0 = 1.0
        assert np.isclose(effective_dist, 1.0)

    def test_effective_distance_with_blending(self):
        """Should blend weights when computing distance."""
        edge_weight = IntentAwareEdgeWeight(k_intents=5)
        edge_weight.global_weight = 1.0
        edge_weight.intent_weights[0] = 2.0

        base_distance = 0.4

        # 50% confidence
        effective_dist = edge_weight.get_effective_distance(
            base_distance=base_distance,
            intent_id=0,
            confidence=0.5
        )

        # Effective weight: 0.5 * 2.0 + 0.5 * 1.0 = 1.5
        # Effective distance: 0.4 * 1.5 = 0.6
        assert np.isclose(effective_dist, 0.6)


class TestWeightStatistics:
    """Test weight statistics reporting."""

    def test_get_statistics(self):
        """Should return comprehensive statistics."""
        edge_weight = IntentAwareEdgeWeight(k_intents=3)
        edge_weight.global_weight = 1.2
        edge_weight.intent_weights[:] = [0.8, 1.0, 1.4]
        edge_weight.usage_counts[:] = [10, 5, 15]

        stats = edge_weight.get_weight_statistics()

        assert np.isclose(stats["global_weight"], 1.2)
        # Check intent weights approximately (float32 precision)
        assert np.allclose(stats["intent_weights"], [0.8, 1.0, 1.4], rtol=1e-5)
        assert np.isclose(stats["mean_intent_weight"], 1.0666, atol=0.01)
        assert stats["std_intent_weight"] > 0
        assert stats["usage_counts"] == [10, 5, 15]
        assert stats["total_usage"] == 30
