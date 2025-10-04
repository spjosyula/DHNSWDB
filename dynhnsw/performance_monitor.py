"""Performance monitoring and degradation detection for adaptive HNSW.

This module tracks search quality metrics and triggers a reset mechanism
if adaptation causes performance to degrade below acceptable thresholds.
"""

from typing import Dict, List, Optional
from collections import deque
import numpy as np


class PerformanceMetrics:
    """Container for search performance metrics."""

    def __init__(
        self,
        recall_at_k: float = 0.0,
        precision_at_k: float = 0.0,
        latency_ms: float = 0.0,
    ) -> None:
        """Initialize performance metrics.

        Args:
            recall_at_k: Fraction of true neighbors found in top-k results
            precision_at_k: Fraction of returned results that are relevant
            latency_ms: Search latency in milliseconds
        """
        self.recall_at_k = recall_at_k
        self.precision_at_k = precision_at_k
        self.latency_ms = latency_ms


class PerformanceMonitor:
    """Monitors search performance and detects degradation.

    Tracks performance metrics over time and triggers reset mechanism
    if quality drops below baseline thresholds.
    """

    def __init__(
        self,
        baseline_recall: float = 0.0,
        degradation_threshold: float = 0.95,
        window_size: int = 50,
    ) -> None:
        """Initialize performance monitor.

        Args:
            baseline_recall: Initial recall@k performance (static HNSW)
            degradation_threshold: Minimum acceptable performance (fraction of baseline)
            window_size: Number of recent queries to track
        """
        self.baseline_recall = baseline_recall
        self.degradation_threshold = degradation_threshold
        self.window_size = window_size

        # Track recent performance metrics
        self.recall_history: deque = deque(maxlen=window_size)
        self.precision_history: deque = deque(maxlen=window_size)
        self.latency_history: deque = deque(maxlen=window_size)

        # Degradation tracking
        self.degradation_detected = False
        self.degradation_count = 0

    def set_baseline(self, recall_at_k: float) -> None:
        """Set or update baseline recall performance.

        Args:
            recall_at_k: Baseline recall@k from static HNSW
        """
        self.baseline_recall = recall_at_k

    def record_query_performance(self, metrics: PerformanceMetrics) -> None:
        """Record performance metrics for a query.

        Args:
            metrics: Performance metrics from the query
        """
        self.recall_history.append(metrics.recall_at_k)
        self.precision_history.append(metrics.precision_at_k)
        self.latency_history.append(metrics.latency_ms)

    def get_current_recall(self) -> float:
        """Get average recall over recent window.

        Returns:
            Average recall@k from recent queries
        """
        if not self.recall_history:
            return 0.0

        return float(np.mean(list(self.recall_history)))

    def get_current_precision(self) -> float:
        """Get average precision over recent window.

        Returns:
            Average precision@k from recent queries
        """
        if not self.precision_history:
            return 0.0

        return float(np.mean(list(self.precision_history)))

    def get_average_latency(self) -> float:
        """Get average latency over recent window.

        Returns:
            Average query latency in milliseconds
        """
        if not self.latency_history:
            return 0.0

        return float(np.mean(list(self.latency_history)))

    def check_for_degradation(self) -> bool:
        """Check if performance has degraded below threshold.

        Returns:
            True if current recall is below acceptable threshold
        """
        if self.baseline_recall == 0.0:
            return False  # No baseline set yet

        current_recall = self.get_current_recall()
        threshold = self.baseline_recall * self.degradation_threshold

        is_degraded = current_recall < threshold

        if is_degraded:
            self.degradation_detected = True
            self.degradation_count += 1

        return is_degraded

    def get_degradation_severity(self) -> float:
        """Compute severity of performance degradation.

        Returns:
            Degradation as percentage drop from baseline (0.0 = no degradation)
        """
        if self.baseline_recall == 0.0:
            return 0.0

        current_recall = self.get_current_recall()
        drop = self.baseline_recall - current_recall

        return max(0.0, drop / self.baseline_recall)

    def should_trigger_reset(self, consecutive_checks: int = 3) -> bool:
        """Determine if reset mechanism should be triggered.

        Reset is triggered if degradation persists across multiple checks.

        Args:
            consecutive_checks: Number of consecutive degradation detections needed

        Returns:
            True if reset should be triggered
        """
        if not self.degradation_detected:
            return False

        # Check if recent queries consistently show degradation
        if len(self.recall_history) < consecutive_checks:
            return False

        recent_recalls = list(self.recall_history)[-consecutive_checks:]
        threshold = self.baseline_recall * self.degradation_threshold

        all_degraded = all(r < threshold for r in recent_recalls)

        return all_degraded

    def get_performance_summary(self) -> Dict[str, float]:
        """Get summary of current performance.

        Returns:
            Dictionary with performance metrics
        """
        return {
            "baseline_recall": self.baseline_recall,
            "current_recall": self.get_current_recall(),
            "current_precision": self.get_current_precision(),
            "average_latency_ms": self.get_average_latency(),
            "degradation_severity": self.get_degradation_severity(),
            "degradation_detected": float(self.degradation_detected),
            "degradation_count": float(self.degradation_count),
        }

    def reset_degradation_state(self) -> None:
        """Reset degradation detection state after recovery."""
        self.degradation_detected = False
        # Keep degradation_count for historical tracking


class AdaptiveResetManager:
    """Manages gradual reset of edge weights when performance degrades.

    Implements a gradual reversion strategy that slowly moves weights
    back toward neutral (1.0) rather than an abrupt reset.
    """

    def __init__(self, reset_steps: int = 100, reset_rate: float = 0.1) -> None:
        """Initialize reset manager.

        Args:
            reset_steps: Number of queries over which to apply gradual reset
            reset_rate: Fraction of distance to neutral to cover per step
        """
        self.reset_steps = reset_steps
        self.reset_rate = reset_rate

        self.reset_active = False
        self.current_step = 0

    def start_reset(self) -> None:
        """Begin gradual reset process."""
        self.reset_active = True
        self.current_step = 0

    def is_resetting(self) -> bool:
        """Check if reset is currently active.

        Returns:
            True if reset is in progress
        """
        return self.reset_active

    def apply_reset_step(self, current_weight: float) -> float:
        """Apply one step of gradual reset to a weight.

        Args:
            current_weight: Current edge weight

        Returns:
            Weight moved toward neutral (1.0)
        """
        if not self.reset_active:
            return current_weight

        # Move weight toward 1.0 by reset_rate
        distance_to_neutral = 1.0 - current_weight
        adjusted_weight = current_weight + distance_to_neutral * self.reset_rate

        return adjusted_weight

    def advance_step(self) -> None:
        """Advance one step in the reset process."""
        if not self.reset_active:
            return

        self.current_step += 1

        if self.current_step >= self.reset_steps:
            self.complete_reset()

    def complete_reset(self) -> None:
        """Complete the reset process."""
        self.reset_active = False
        self.current_step = 0

    def get_reset_progress(self) -> float:
        """Get progress of current reset.

        Returns:
            Progress as fraction in [0.0, 1.0]
        """
        if not self.reset_active:
            return 0.0

        return min(1.0, self.current_step / self.reset_steps)
