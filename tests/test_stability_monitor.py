"""Unit tests for stability monitoring and oscillation detection."""

import pytest
from dynhnsw.stability_monitor import WeightHistory, StabilityMonitor


class TestWeightHistory:
    """Test weight history tracking."""

    def test_initial_history_empty(self):
        """New history should be empty."""
        history = WeightHistory(window_size=5)
        assert len(history.history) == 0

    def test_add_weight_stores_value(self):
        """Adding weight should store it in history."""
        history = WeightHistory(window_size=5)
        history.add_weight(1.5)

        assert len(history.history) == 1
        assert history.history[0] == 1.5

    def test_window_size_limit_enforced(self):
        """Should not exceed window size."""
        history = WeightHistory(window_size=3)

        for i in range(10):
            history.add_weight(float(i))

        assert len(history.history) == 3

    def test_window_evicts_oldest(self):
        """Exceeding window should remove oldest values."""
        history = WeightHistory(window_size=3)

        for i in range(5):
            history.add_weight(float(i))

        # Should have [2, 3, 4] (oldest evicted)
        assert list(history.history) == [2.0, 3.0, 4.0]


class TestVarianceComputation:
    """Test variance computation for oscillation detection."""

    def test_variance_insufficient_data(self):
        """Insufficient data should return 0.0."""
        history = WeightHistory()
        history.add_weight(1.0)

        assert history.compute_variance() == 0.0

    def test_variance_constant_weights(self):
        """Constant weights should have zero variance."""
        history = WeightHistory()
        for _ in range(5):
            history.add_weight(1.0)

        assert history.compute_variance() == pytest.approx(0.0)

    def test_variance_varying_weights(self):
        """Varying weights should have non-zero variance."""
        history = WeightHistory()
        weights = [0.5, 1.0, 1.5, 2.0, 2.5]

        for w in weights:
            history.add_weight(w)

        variance = history.compute_variance()
        assert variance > 0.0

    def test_variance_oscillating_pattern(self):
        """Oscillating pattern should have high variance."""
        history = WeightHistory()
        # Oscillate between 0.5 and 2.0
        oscillating = [0.5, 2.0, 0.5, 2.0, 0.5, 2.0]

        for w in oscillating:
            history.add_weight(w)

        variance = history.compute_variance()
        assert variance > 0.5  # High variance expected


class TestOscillationDetection:
    """Test oscillation detection logic."""

    def test_stable_weights_not_oscillating(self):
        """Stable weights should not trigger oscillation."""
        history = WeightHistory()
        for _ in range(10):
            history.add_weight(1.0)

        assert not history.is_oscillating()

    def test_small_variations_not_oscillating(self):
        """Small variations should not trigger oscillation."""
        history = WeightHistory()
        weights = [1.0, 1.05, 0.95, 1.02, 0.98]

        for w in weights:
            history.add_weight(w)

        assert not history.is_oscillating()

    def test_large_oscillations_detected(self):
        """Large oscillations should be detected."""
        history = WeightHistory()
        # Large swings
        oscillating = [0.5, 2.5, 0.5, 2.5, 0.5, 2.5]

        for w in oscillating:
            history.add_weight(w)

        assert history.is_oscillating()

    def test_custom_threshold(self):
        """Should respect custom oscillation threshold."""
        history = WeightHistory()
        weights = [0.8, 1.2, 0.8, 1.2]

        for w in weights:
            history.add_weight(w)

        # Default threshold (0.25): should not oscillate
        assert not history.is_oscillating(threshold=0.25)

        # Lower threshold (0.01): should oscillate
        assert history.is_oscillating(threshold=0.01)


class TestTrendDetection:
    """Test weight trend detection."""

    def test_increasing_trend(self):
        """Increasing weights should be detected."""
        history = WeightHistory()
        weights = [0.5, 0.7, 0.9, 1.1, 1.3, 1.5]

        for w in weights:
            history.add_weight(w)

        assert history.get_trend() == "increasing"

    def test_decreasing_trend(self):
        """Decreasing weights should be detected."""
        history = WeightHistory()
        weights = [2.0, 1.8, 1.6, 1.4, 1.2, 1.0]

        for w in weights:
            history.add_weight(w)

        assert history.get_trend() == "decreasing"

    def test_stable_trend(self):
        """Stable weights should be detected."""
        history = WeightHistory()
        weights = [1.0, 1.02, 0.98, 1.01, 0.99, 1.0]

        for w in weights:
            history.add_weight(w)

        assert history.get_trend() == "stable"

    def test_unknown_trend_insufficient_data(self):
        """Insufficient data should return unknown."""
        history = WeightHistory()
        history.add_weight(1.0)

        assert history.get_trend() == "unknown"


class TestStabilityMonitor:
    """Test overall stability monitoring."""

    def test_initial_state(self):
        """New monitor should have clean state."""
        monitor = StabilityMonitor()

        assert monitor.total_updates == 0
        assert monitor.oscillation_count == 0

    def test_record_weight_update(self):
        """Recording updates should increment counter."""
        monitor = StabilityMonitor()

        monitor.record_weight_update(1, 2, 1.5)

        assert monitor.total_updates == 1
        assert (1, 2) in monitor.edge_histories

    def test_multiple_updates_same_edge(self):
        """Multiple updates to same edge should accumulate history."""
        monitor = StabilityMonitor()

        for i in range(5):
            monitor.record_weight_update(1, 2, float(i))

        assert monitor.total_updates == 5
        assert len(monitor.edge_histories[(1, 2)].history) == 5

    def test_oscillation_detection_increments_count(self):
        """Detecting oscillation should increment count."""
        monitor = StabilityMonitor(oscillation_threshold=0.1)

        # Create oscillating pattern
        for i in range(10):
            weight = 0.5 if i % 2 == 0 else 2.5
            monitor.record_weight_update(1, 2, weight)

        # Should have detected oscillation
        assert monitor.oscillation_count > 0


class TestOscillationTracking:
    """Test oscillation tracking across edges."""

    def test_get_oscillating_edges_empty(self):
        """No updates should have no oscillating edges."""
        monitor = StabilityMonitor()

        oscillating = monitor.get_oscillating_edges()

        assert len(oscillating) == 0

    def test_get_oscillating_edges_stable(self):
        """Stable edges should not be listed."""
        monitor = StabilityMonitor()

        for _ in range(10):
            monitor.record_weight_update(1, 2, 1.0)

        oscillating = monitor.get_oscillating_edges()

        assert len(oscillating) == 0

    def test_get_oscillating_edges_detected(self):
        """Oscillating edges should be listed."""
        monitor = StabilityMonitor(oscillation_threshold=0.1)

        # Edge (1,2): oscillating
        for i in range(10):
            weight = 0.5 if i % 2 == 0 else 2.5
            monitor.record_weight_update(1, 2, weight)

        # Edge (3,4): stable
        for _ in range(10):
            monitor.record_weight_update(3, 4, 1.0)

        oscillating = monitor.get_oscillating_edges()

        assert (1, 2) in oscillating
        assert (3, 4) not in oscillating

    def test_oscillation_rate_computation(self):
        """Should compute correct oscillation rate."""
        monitor = StabilityMonitor(oscillation_threshold=0.1)

        # 5 stable updates
        for _ in range(5):
            monitor.record_weight_update(1, 2, 1.0)

        # 5 oscillating updates
        for i in range(5):
            weight = 0.5 if i % 2 == 0 else 2.5
            monitor.record_weight_update(3, 4, weight)

        rate = monitor.get_oscillation_rate()

        # Rate should be 5/10 = 0.5
        assert rate == pytest.approx(0.5, abs=0.1)


class TestStabilityScore:
    """Test overall stability score computation."""

    def test_stability_score_empty_monitor(self):
        """Empty monitor should have perfect stability."""
        monitor = StabilityMonitor()

        score = monitor.compute_stability_score()

        assert score == 1.0

    def test_stability_score_all_stable(self):
        """All stable edges should have high score."""
        monitor = StabilityMonitor()

        for i in range(3):
            for _ in range(10):
                monitor.record_weight_update(i, i + 1, 1.0)

        score = monitor.compute_stability_score()

        assert score > 0.9  # Very stable

    def test_stability_score_all_oscillating(self):
        """All oscillating edges should have low score."""
        monitor = StabilityMonitor(oscillation_threshold=0.1)

        for i in range(3):
            for j in range(10):
                weight = 0.5 if j % 2 == 0 else 2.5
                monitor.record_weight_update(i, i + 1, weight)

        score = monitor.compute_stability_score()

        assert score < 0.5  # Very unstable

    def test_stability_score_mixed(self):
        """Mixed stability should give moderate score."""
        monitor = StabilityMonitor(oscillation_threshold=0.1)

        # Stable edge
        for _ in range(10):
            monitor.record_weight_update(1, 2, 1.0)

        # Oscillating edge
        for i in range(10):
            weight = 0.5 if i % 2 == 0 else 2.5
            monitor.record_weight_update(3, 4, weight)

        score = monitor.compute_stability_score()

        assert 0.3 < score < 0.8  # Moderate


class TestUnstableEdges:
    """Test detection of high-variance edges."""

    def test_get_unstable_edges_none(self):
        """Low variance should have no unstable edges."""
        monitor = StabilityMonitor()

        for _ in range(10):
            monitor.record_weight_update(1, 2, 1.0)

        unstable = monitor.get_unstable_edges(min_variance=0.5)

        assert len(unstable) == 0

    def test_get_unstable_edges_detected(self):
        """High variance should be detected."""
        monitor = StabilityMonitor()

        # High variance edge
        weights = [0.5, 2.0, 0.5, 2.0, 0.5, 2.0]
        for w in weights:
            monitor.record_weight_update(1, 2, w)

        unstable = monitor.get_unstable_edges(min_variance=0.3)

        assert (1, 2) in unstable


class TestEdgeTrend:
    """Test trend tracking for specific edges."""

    def test_get_edge_trend_increasing(self):
        """Should detect increasing trend for edge."""
        monitor = StabilityMonitor()

        weights = [0.5, 0.7, 0.9, 1.1, 1.3]
        for w in weights:
            monitor.record_weight_update(1, 2, w)

        trend = monitor.get_edge_trend(1, 2)

        assert trend == "increasing"

    def test_get_edge_trend_nonexistent_edge(self):
        """Non-existent edge should return unknown."""
        monitor = StabilityMonitor()

        trend = monitor.get_edge_trend(99, 100)

        assert trend == "unknown"


class TestStatistics:
    """Test statistics reporting."""

    def test_get_statistics(self):
        """Should return comprehensive statistics."""
        monitor = StabilityMonitor()

        for _ in range(10):
            monitor.record_weight_update(1, 2, 1.0)

        stats = monitor.get_statistics()

        assert stats["total_updates"] == 10
        assert stats["total_edges_tracked"] == 1
        assert "stability_score" in stats

    def test_reset_statistics(self):
        """Reset should clear all tracking data."""
        monitor = StabilityMonitor()

        for _ in range(10):
            monitor.record_weight_update(1, 2, 1.0)

        monitor.reset_statistics()

        assert monitor.total_updates == 0
        assert len(monitor.edge_histories) == 0
