"""Unit tests for performance monitoring and reset mechanism."""

import pytest
from dynhnsw.performance_monitor import (
    PerformanceMetrics,
    PerformanceMonitor,
    AdaptiveResetManager,
)


class TestPerformanceMetrics:
    """Test performance metrics container."""

    def test_initialization(self):
        """Should initialize with provided values."""
        metrics = PerformanceMetrics(
            recall_at_k=0.85, precision_at_k=0.90, latency_ms=15.5
        )

        assert metrics.recall_at_k == 0.85
        assert metrics.precision_at_k == 0.90
        assert metrics.latency_ms == 15.5

    def test_default_initialization(self):
        """Should have default values of 0.0."""
        metrics = PerformanceMetrics()

        assert metrics.recall_at_k == 0.0
        assert metrics.precision_at_k == 0.0
        assert metrics.latency_ms == 0.0


class TestPerformanceMonitor:
    """Test performance monitoring functionality."""

    def test_initial_state(self):
        """New monitor should have clean state."""
        monitor = PerformanceMonitor(baseline_recall=0.9)

        assert monitor.baseline_recall == 0.9
        assert not monitor.degradation_detected
        assert monitor.degradation_count == 0

    def test_set_baseline(self):
        """Should be able to set/update baseline."""
        monitor = PerformanceMonitor()

        monitor.set_baseline(0.85)

        assert monitor.baseline_recall == 0.85

    def test_record_query_performance(self):
        """Should record performance metrics."""
        monitor = PerformanceMonitor()

        metrics = PerformanceMetrics(recall_at_k=0.80, precision_at_k=0.85)
        monitor.record_query_performance(metrics)

        assert len(monitor.recall_history) == 1
        assert monitor.recall_history[0] == 0.80

    def test_window_size_limit(self):
        """Should respect window size limit."""
        monitor = PerformanceMonitor(window_size=5)

        for i in range(10):
            metrics = PerformanceMetrics(recall_at_k=float(i))
            monitor.record_query_performance(metrics)

        assert len(monitor.recall_history) == 5


class TestCurrentPerformance:
    """Test current performance computation."""

    def test_get_current_recall_empty(self):
        """Empty history should return 0.0."""
        monitor = PerformanceMonitor()

        assert monitor.get_current_recall() == 0.0

    def test_get_current_recall_average(self):
        """Should compute average of recent recalls."""
        monitor = PerformanceMonitor()

        recalls = [0.80, 0.85, 0.90]
        for r in recalls:
            metrics = PerformanceMetrics(recall_at_k=r)
            monitor.record_query_performance(metrics)

        current = monitor.get_current_recall()

        assert current == pytest.approx(0.85)  # Average

    def test_get_current_precision(self):
        """Should compute average precision."""
        monitor = PerformanceMonitor()

        precisions = [0.70, 0.80, 0.90]
        for p in precisions:
            metrics = PerformanceMetrics(precision_at_k=p)
            monitor.record_query_performance(metrics)

        current = monitor.get_current_precision()

        assert current == pytest.approx(0.80)

    def test_get_average_latency(self):
        """Should compute average latency."""
        monitor = PerformanceMonitor()

        latencies = [10.0, 20.0, 30.0]
        for lat in latencies:
            metrics = PerformanceMetrics(latency_ms=lat)
            monitor.record_query_performance(metrics)

        avg_latency = monitor.get_average_latency()

        assert avg_latency == pytest.approx(20.0)


class TestDegradationDetection:
    """Test performance degradation detection."""

    def test_no_degradation_above_threshold(self):
        """Performance above threshold should not trigger degradation."""
        monitor = PerformanceMonitor(
            baseline_recall=0.90, degradation_threshold=0.95
        )

        # Current recall: 0.87 (above 0.90 * 0.95 = 0.855)
        for _ in range(5):
            metrics = PerformanceMetrics(recall_at_k=0.87)
            monitor.record_query_performance(metrics)

        is_degraded = monitor.check_for_degradation()

        assert not is_degraded

    def test_degradation_below_threshold(self):
        """Performance below threshold should trigger degradation."""
        monitor = PerformanceMonitor(
            baseline_recall=0.90, degradation_threshold=0.95
        )

        # Current recall: 0.80 (below 0.90 * 0.95 = 0.855)
        for _ in range(5):
            metrics = PerformanceMetrics(recall_at_k=0.80)
            monitor.record_query_performance(metrics)

        is_degraded = monitor.check_for_degradation()

        assert is_degraded
        assert monitor.degradation_detected

    def test_degradation_count_increments(self):
        """Each degradation detection should increment counter."""
        monitor = PerformanceMonitor(
            baseline_recall=0.90, degradation_threshold=0.95
        )

        metrics = PerformanceMetrics(recall_at_k=0.70)
        monitor.record_query_performance(metrics)

        monitor.check_for_degradation()
        count1 = monitor.degradation_count

        monitor.check_for_degradation()
        count2 = monitor.degradation_count

        assert count2 > count1

    def test_no_baseline_no_degradation(self):
        """No baseline set should not detect degradation."""
        monitor = PerformanceMonitor(baseline_recall=0.0)

        metrics = PerformanceMetrics(recall_at_k=0.50)
        monitor.record_query_performance(metrics)

        is_degraded = monitor.check_for_degradation()

        assert not is_degraded


class TestDegradationSeverity:
    """Test degradation severity computation."""

    def test_no_degradation_zero_severity(self):
        """Performance at baseline should have zero severity."""
        monitor = PerformanceMonitor(baseline_recall=0.90)

        for _ in range(5):
            metrics = PerformanceMetrics(recall_at_k=0.90)
            monitor.record_query_performance(metrics)

        severity = monitor.get_degradation_severity()

        assert severity == pytest.approx(0.0)

    def test_degradation_severity_percentage(self):
        """Severity should be percentage drop from baseline."""
        monitor = PerformanceMonitor(baseline_recall=0.90)

        # Current: 0.80, baseline: 0.90, drop: 0.10, severity: 0.10/0.90 ≈ 0.11
        for _ in range(5):
            metrics = PerformanceMetrics(recall_at_k=0.80)
            monitor.record_query_performance(metrics)

        severity = monitor.get_degradation_severity()

        assert severity == pytest.approx(0.111, abs=0.01)

    def test_improved_performance_zero_severity(self):
        """Performance above baseline should have zero severity."""
        monitor = PerformanceMonitor(baseline_recall=0.90)

        for _ in range(5):
            metrics = PerformanceMetrics(recall_at_k=0.95)
            monitor.record_query_performance(metrics)

        severity = monitor.get_degradation_severity()

        assert severity == pytest.approx(0.0)


class TestResetTrigger:
    """Test reset trigger logic."""

    def test_no_reset_without_degradation(self):
        """Should not trigger reset without degradation."""
        monitor = PerformanceMonitor(baseline_recall=0.90)

        for _ in range(5):
            metrics = PerformanceMetrics(recall_at_k=0.90)
            monitor.record_query_performance(metrics)

        should_reset = monitor.should_trigger_reset(consecutive_checks=3)

        assert not should_reset

    def test_no_reset_insufficient_consecutive(self):
        """Should not trigger reset without enough consecutive degradation."""
        monitor = PerformanceMonitor(
            baseline_recall=0.90, degradation_threshold=0.95
        )

        # Only 2 degraded queries, need 3 consecutive
        metrics_bad = PerformanceMetrics(recall_at_k=0.70)
        metrics_good = PerformanceMetrics(recall_at_k=0.90)

        monitor.record_query_performance(metrics_bad)
        monitor.record_query_performance(metrics_bad)
        monitor.record_query_performance(metrics_good)  # Breaks streak

        monitor.check_for_degradation()
        should_reset = monitor.should_trigger_reset(consecutive_checks=3)

        assert not should_reset

    def test_trigger_reset_consecutive_degradation(self):
        """Should trigger reset after consecutive degradation."""
        monitor = PerformanceMonitor(
            baseline_recall=0.90, degradation_threshold=0.95
        )

        # 3 consecutive degraded queries
        for _ in range(3):
            metrics = PerformanceMetrics(recall_at_k=0.70)
            monitor.record_query_performance(metrics)

        monitor.check_for_degradation()
        should_reset = monitor.should_trigger_reset(consecutive_checks=3)

        assert should_reset


class TestPerformanceSummary:
    """Test performance summary reporting."""

    def test_get_performance_summary(self):
        """Should return comprehensive summary."""
        monitor = PerformanceMonitor(baseline_recall=0.90)

        for _ in range(5):
            metrics = PerformanceMetrics(
                recall_at_k=0.85, precision_at_k=0.80, latency_ms=15.0
            )
            monitor.record_query_performance(metrics)

        summary = monitor.get_performance_summary()

        assert summary["baseline_recall"] == 0.90
        assert summary["current_recall"] == pytest.approx(0.85)
        assert summary["current_precision"] == pytest.approx(0.80)
        assert summary["average_latency_ms"] == pytest.approx(15.0)
        assert "degradation_severity" in summary

    def test_reset_degradation_state(self):
        """Reset should clear degradation flag."""
        monitor = PerformanceMonitor(baseline_recall=0.90)

        monitor.degradation_detected = True

        monitor.reset_degradation_state()

        assert not monitor.degradation_detected


class TestAdaptiveResetManager:
    """Test adaptive reset manager."""

    def test_initial_state_not_resetting(self):
        """New manager should not be resetting."""
        manager = AdaptiveResetManager()

        assert not manager.is_resetting()
        assert manager.current_step == 0

    def test_start_reset(self):
        """Starting reset should activate it."""
        manager = AdaptiveResetManager()

        manager.start_reset()

        assert manager.is_resetting()
        assert manager.current_step == 0

    def test_apply_reset_step_when_inactive(self):
        """Reset step when inactive should not change weight."""
        manager = AdaptiveResetManager()

        weight = manager.apply_reset_step(2.0)

        assert weight == 2.0  # Unchanged

    def test_apply_reset_step_moves_toward_neutral(self):
        """Reset step should move weight toward 1.0."""
        manager = AdaptiveResetManager(reset_rate=0.5)

        manager.start_reset()
        weight = manager.apply_reset_step(2.0)

        # Should move 50% toward 1.0: 2.0 + (1.0 - 2.0) * 0.5 = 1.5
        assert weight == pytest.approx(1.5)

    def test_apply_reset_step_below_neutral(self):
        """Reset should work for weights below 1.0."""
        manager = AdaptiveResetManager(reset_rate=0.5)

        manager.start_reset()
        weight = manager.apply_reset_step(0.6)

        # Should move 50% toward 1.0: 0.6 + (1.0 - 0.6) * 0.5 = 0.8
        assert weight == pytest.approx(0.8)

    def test_advance_step_increments(self):
        """Advancing step should increment counter."""
        manager = AdaptiveResetManager(reset_steps=10)

        manager.start_reset()
        manager.advance_step()

        assert manager.current_step == 1

    def test_complete_reset_after_steps(self):
        """Reset should complete after specified steps."""
        manager = AdaptiveResetManager(reset_steps=5)

        manager.start_reset()

        for _ in range(5):
            manager.advance_step()

        assert not manager.is_resetting()

    def test_get_reset_progress(self):
        """Should compute correct reset progress."""
        manager = AdaptiveResetManager(reset_steps=10)

        manager.start_reset()

        assert manager.get_reset_progress() == pytest.approx(0.0)

        for _ in range(5):
            manager.advance_step()

        assert manager.get_reset_progress() == pytest.approx(0.5)

    def test_get_reset_progress_when_inactive(self):
        """Progress when inactive should be 0.0."""
        manager = AdaptiveResetManager()

        progress = manager.get_reset_progress()

        assert progress == 0.0

    def test_complete_reset_manually(self):
        """Manual completion should deactivate reset."""
        manager = AdaptiveResetManager()

        manager.start_reset()
        manager.complete_reset()

        assert not manager.is_resetting()
        assert manager.current_step == 0


class TestGradualResetBehavior:
    """Test gradual reset behavior over multiple steps."""

    def test_gradual_convergence_to_neutral(self):
        """Weight should gradually converge to 1.0."""
        manager = AdaptiveResetManager(reset_rate=0.2, reset_steps=100)

        manager.start_reset()
        weight = 5.0

        for _ in range(20):
            weight = manager.apply_reset_step(weight)
            manager.advance_step()

        # Should be closer to 1.0 than original 5.0
        assert 1.0 < weight < 3.0

    def test_multiple_reset_cycles(self):
        """Should handle multiple reset cycles correctly."""
        manager = AdaptiveResetManager(reset_steps=3)

        # First reset
        manager.start_reset()
        for _ in range(3):
            manager.advance_step()

        assert not manager.is_resetting()

        # Second reset
        manager.start_reset()

        assert manager.is_resetting()
        assert manager.current_step == 0
