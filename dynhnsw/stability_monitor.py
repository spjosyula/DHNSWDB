"""Stability monitoring for adaptive edge weights.

This module detects problematic patterns in weight updates such as oscillations,
extreme changes, or unstable behavior that could degrade search quality.
"""

from typing import Dict, List, Tuple
from collections import deque
import numpy as np


# Type alias for edge identification
EdgeId = Tuple[int, int]


class WeightHistory:
    """Tracks weight history for an edge to detect oscillations."""

    def __init__(self, window_size: int = 10) -> None:
        """Initialize weight history tracker.

        Args:
            window_size: Number of recent weight values to track
        """
        self.window_size = window_size
        self.history: deque = deque(maxlen=window_size)

    def add_weight(self, weight: float) -> None:
        """Record a new weight value.

        Args:
            weight: Current weight value
        """
        self.history.append(weight)

    def compute_variance(self) -> float:
        """Compute variance of recent weights.

        High variance indicates oscillation or instability.

        Returns:
            Variance of weights in history window
        """
        if len(self.history) < 2:
            return 0.0

        weights = np.array(list(self.history))
        return float(np.var(weights))

    def is_oscillating(self, threshold: float = 0.25) -> bool:
        """Check if weights are oscillating.

        Oscillation is detected when variance exceeds threshold.

        Args:
            threshold: Variance threshold for oscillation detection

        Returns:
            True if weights are oscillating
        """
        return self.compute_variance() > threshold

    def get_trend(self) -> str:
        """Determine weight trend direction.

        Returns:
            "increasing", "decreasing", "stable", or "unknown"
        """
        if len(self.history) < 3:
            return "unknown"

        # Compare first half to second half
        mid = len(self.history) // 2
        first_half_avg = np.mean(list(self.history)[:mid])
        second_half_avg = np.mean(list(self.history)[mid:])

        diff = second_half_avg - first_half_avg

        if diff > 0.1:
            return "increasing"
        elif diff < -0.1:
            return "decreasing"
        else:
            return "stable"


class StabilityMonitor:
    """Monitors edge weight stability across the entire graph.

    Detects oscillations, extreme changes, and other unstable patterns
    that could harm search performance.
    """

    def __init__(
        self, oscillation_threshold: float = 0.25, history_window: int = 10
    ) -> None:
        """Initialize stability monitor.

        Args:
            oscillation_threshold: Variance threshold for oscillation detection
            history_window: Number of weight updates to track per edge
        """
        self.oscillation_threshold = oscillation_threshold
        self.history_window = history_window

        # Track weight history per edge
        self.edge_histories: Dict[EdgeId, WeightHistory] = {}

        # Statistics
        self.total_updates = 0
        self.oscillation_count = 0

    def record_weight_update(
        self, node_u: int, node_v: int, new_weight: float
    ) -> None:
        """Record a weight update for monitoring.

        Args:
            node_u: First node of edge
            node_v: Second node of edge
            new_weight: Updated weight value
        """
        edge_id = (node_u, node_v)

        if edge_id not in self.edge_histories:
            self.edge_histories[edge_id] = WeightHistory(self.history_window)

        self.edge_histories[edge_id].add_weight(new_weight)
        self.total_updates += 1

        # Check for oscillation
        if self.edge_histories[edge_id].is_oscillating(self.oscillation_threshold):
            self.oscillation_count += 1

    def get_oscillating_edges(self) -> List[EdgeId]:
        """Get list of edges currently showing oscillation.

        Returns:
            List of edge IDs with oscillating weights
        """
        oscillating = []

        for edge_id, history in self.edge_histories.items():
            if history.is_oscillating(self.oscillation_threshold):
                oscillating.append(edge_id)

        return oscillating

    def get_oscillation_rate(self) -> float:
        """Compute overall oscillation rate.

        Returns:
            Fraction of updates that triggered oscillation detection
        """
        if self.total_updates == 0:
            return 0.0

        return self.oscillation_count / self.total_updates

    def compute_stability_score(self) -> float:
        """Compute overall stability score for the graph.

        Higher score = more stable. Score in [0, 1].

        Returns:
            Stability score based on oscillation rate and variance
        """
        if not self.edge_histories:
            return 1.0  # Perfect stability (no edges to be unstable)

        # Compute average variance across all edges
        variances = [h.compute_variance() for h in self.edge_histories.values()]
        avg_variance = np.mean(variances) if variances else 0.0

        # Normalize variance to [0, 1] score (lower variance = higher score)
        # Max expected variance is ~1.0 for weights in [0.1, 10.0]
        variance_score = 1.0 - min(avg_variance, 1.0)

        # Factor in oscillation rate
        oscillation_score = 1.0 - self.get_oscillation_rate()

        # Combined score (equal weighting)
        return (variance_score + oscillation_score) / 2.0

    def get_unstable_edges(self, min_variance: float = 0.5) -> List[EdgeId]:
        """Get edges with high weight variance.

        Args:
            min_variance: Minimum variance to be considered unstable

        Returns:
            List of edge IDs with high variance
        """
        unstable = []

        for edge_id, history in self.edge_histories.items():
            if history.compute_variance() >= min_variance:
                unstable.append(edge_id)

        return unstable

    def get_edge_trend(self, node_u: int, node_v: int) -> str:
        """Get weight trend for a specific edge.

        Args:
            node_u: First node of edge
            node_v: Second node of edge

        Returns:
            Trend direction: "increasing", "decreasing", "stable", or "unknown"
        """
        edge_id = (node_u, node_v)

        if edge_id not in self.edge_histories:
            return "unknown"

        return self.edge_histories[edge_id].get_trend()

    def get_statistics(self) -> Dict[str, float]:
        """Get stability statistics.

        Returns:
            Dictionary with stability metrics
        """
        oscillating_count = len(self.get_oscillating_edges())
        total_edges = len(self.edge_histories)

        return {
            "total_updates": self.total_updates,
            "total_edges_tracked": total_edges,
            "oscillating_edges": oscillating_count,
            "oscillation_rate": self.get_oscillation_rate(),
            "stability_score": self.compute_stability_score(),
        }

    def reset_statistics(self) -> None:
        """Reset all tracking statistics and history."""
        self.edge_histories.clear()
        self.total_updates = 0
        self.oscillation_count = 0
