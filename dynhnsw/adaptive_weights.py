"""Edge weight learning using feedback signals.

This module implements the core learning mechanism that updates edge weights
based on user feedback. It converts satisfaction scores into weight adjustments
that improve future search performance.
"""

import math
from typing import Dict, Tuple, List
from dynhnsw.models.edge_weight import EdgeWeight
from dynhnsw.feedback import QueryFeedback


# Type alias for edge identification
EdgeId = Tuple[int, int]  # (node_u, node_v)


class EdgeWeightLearner:
    """Manages learning and updating of edge weights based on feedback.

    This class maintains a mapping of edges to their weights and provides
    methods to update weights based on user feedback signals.
    """

    def __init__(
        self,
        learning_rate: float = 0.05,
        decay_half_life_seconds: float = 604800.0,  # 7 days in seconds
    ) -> None:
        """Initialize the edge weight learner.

        Args:
            learning_rate: Alpha parameter for exponential smoothing (0.0-1.0)
                Higher values = faster adaptation, but more noise sensitivity
            decay_half_life_seconds: Time for old patterns to decay by 50%
                Default is 7 days (604800 seconds)
        """
        if not 0.0 < learning_rate < 1.0:
            raise ValueError(f"learning_rate must be in (0, 1), got {learning_rate}")

        if decay_half_life_seconds <= 0:
            raise ValueError(
                f"decay_half_life_seconds must be positive, got {decay_half_life_seconds}"
            )

        self.learning_rate = learning_rate
        self.decay_rate = math.log(2) / decay_half_life_seconds

        # Map from edge to its weight
        self.edge_weights: Dict[EdgeId, EdgeWeight] = {}

    def get_edge_weight(self, node_u: int, node_v: int) -> EdgeWeight:
        """Get or create weight for an edge.

        Args:
            node_u: Source node ID
            node_v: Destination node ID

        Returns:
            EdgeWeight object for this edge
        """
        edge_id = (node_u, node_v)

        if edge_id not in self.edge_weights:
            self.edge_weights[edge_id] = EdgeWeight()

        return self.edge_weights[edge_id]

    def get_effective_distance(
        self, node_u: int, node_v: int, base_distance: float
    ) -> float:
        """Compute weight-modified distance for an edge.

        Args:
            node_u: Source node ID
            node_v: Destination node ID
            base_distance: Original Euclidean distance

        Returns:
            Effective distance = weight * base_distance
        """
        edge_weight = self.get_edge_weight(node_u, node_v)
        return edge_weight.get_effective_distance(base_distance)

    def update_from_feedback(
        self, feedback: QueryFeedback, traversed_edges: List[EdgeId]
    ) -> None:
        """Update edge weights based on query feedback.

        Edges used in successful queries get lower weights (preferred).
        Edges used in unsuccessful queries get higher weights (discouraged).

        Args:
            feedback: User feedback for the query
            traversed_edges: List of edges used during search for this query
        """
        if not traversed_edges:
            return

        # Convert satisfaction to reward signal
        # High satisfaction → reward < 1.0 (decrease weights, prefer these edges)
        # Low satisfaction → reward > 1.0 (increase weights, avoid these edges)
        satisfaction = feedback.get_satisfaction_score()
        reward_signal = self._satisfaction_to_reward(satisfaction)

        # Update all edges that were traversed
        for edge_id in traversed_edges:
            edge_weight = self.get_edge_weight(edge_id[0], edge_id[1])

            # Apply temporal decay before update
            edge_weight.apply_temporal_decay(self.decay_rate)

            # Update weight based on feedback
            edge_weight.update(reward_signal, self.learning_rate)

            # Track usage
            edge_weight.increment_usage()

    def _satisfaction_to_reward(self, satisfaction: float) -> float:
        """Convert satisfaction score to weight update target.

        Args:
            satisfaction: Score in [0.0, 1.0] from feedback

        Returns:
            Reward signal:
            - satisfaction = 1.0 → reward = 0.8 (reduce weight by 20%)
            - satisfaction = 0.5 → reward = 1.0 (no change)
            - satisfaction = 0.0 → reward = 1.2 (increase weight by 20%)
        """
        # Linear mapping: satisfaction [0,1] → reward [1.2, 0.8]
        return 1.2 - 0.4 * satisfaction

    def apply_decay_to_all(self) -> None:
        """Apply temporal decay to all edge weights.

        This should be called periodically to ensure old patterns decay
        even if edges aren't being actively used.
        """
        for edge_weight in self.edge_weights.values():
            edge_weight.apply_temporal_decay(self.decay_rate)

    def get_weight_statistics(self) -> Dict[str, float]:
        """Compute statistics about current edge weights.

        Returns:
            Dictionary with min, max, mean, and count of weights
        """
        if not self.edge_weights:
            return {"min": 1.0, "max": 1.0, "mean": 1.0, "count": 0}

        weights = [ew.weight for ew in self.edge_weights.values()]

        return {
            "min": min(weights),
            "max": max(weights),
            "mean": sum(weights) / len(weights),
            "count": len(weights),
        }

    def get_total_edges(self) -> int:
        """Get total number of edges being tracked.

        Returns:
            Number of edges with learned weights
        """
        return len(self.edge_weights)
