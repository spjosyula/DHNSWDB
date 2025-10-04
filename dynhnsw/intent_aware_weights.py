"""Intent-aware edge weight learning.

This module extends the EdgeWeightLearner to support per-intent weight tracking.
Each edge maintains separate weights for different query intents, enabling
context-aware adaptation based on detected intent patterns.
"""

import math
from typing import Dict, Tuple, List
from dynhnsw.models.intent_aware_edge_weight import IntentAwareEdgeWeight
from dynhnsw.feedback import QueryFeedback


# Type alias for edge identification
EdgeId = Tuple[int, int]


class IntentAwareWeightLearner:
    """Manages intent-aware edge weight learning from feedback.

    This learner maintains separate edge weights for each detected intent,
    allowing the same graph structure to adapt differently based on query context.
    """

    def __init__(
        self,
        k_intents: int = 5,
        learning_rate: float = 0.05,
        decay_half_life_seconds: float = 604800.0,  # 7 days
    ) -> None:
        """Initialize intent-aware weight learner.

        Args:
            k_intents: Number of intent clusters
            learning_rate: Alpha parameter for exponential smoothing
            decay_half_life_seconds: Time for old patterns to decay by 50%
        """
        if not 0.0 < learning_rate < 1.0:
            raise ValueError(f"learning_rate must be in (0, 1), got {learning_rate}")

        if decay_half_life_seconds <= 0:
            raise ValueError(
                f"decay_half_life_seconds must be positive, got {decay_half_life_seconds}"
            )

        self.k_intents = k_intents
        self.learning_rate = learning_rate
        self.decay_rate = math.log(2) / decay_half_life_seconds

        # Map from edge to intent-aware weight
        self.edge_weights: Dict[EdgeId, IntentAwareEdgeWeight] = {}

    def get_edge_weight(self, node_u: int, node_v: int) -> IntentAwareEdgeWeight:
        """Get or create intent-aware weight for an edge.

        Args:
            node_u: Source node ID
            node_v: Destination node ID

        Returns:
            IntentAwareEdgeWeight object for this edge
        """
        edge_id = (node_u, node_v)

        if edge_id not in self.edge_weights:
            self.edge_weights[edge_id] = IntentAwareEdgeWeight(k_intents=self.k_intents)

        return self.edge_weights[edge_id]

    def get_effective_distance(
        self,
        node_u: int,
        node_v: int,
        base_distance: float,
        intent_id: int,
        confidence: float
    ) -> float:
        """Compute intent-aware weighted distance for an edge.

        Args:
            node_u: Source node ID
            node_v: Destination node ID
            base_distance: Original Euclidean distance
            intent_id: Detected intent cluster ID (-1 for cold start)
            confidence: Intent detection confidence [0, 1]

        Returns:
            Effective distance = intent_weight * base_distance
        """
        edge_weight = self.get_edge_weight(node_u, node_v)
        return edge_weight.get_effective_distance(base_distance, intent_id, confidence)

    def update_from_feedback(
        self,
        feedback: QueryFeedback,
        traversed_edges: List[EdgeId],
        intent_id: int,
        confidence: float
    ) -> None:
        """Update intent-specific edge weights based on query feedback.

        Args:
            feedback: User feedback for the query
            traversed_edges: List of edges used during search
            intent_id: Detected intent for this query (-1 for cold start)
            confidence: Intent detection confidence
        """
        if not traversed_edges:
            return

        # Convert satisfaction to reward signal
        satisfaction = feedback.get_satisfaction_score()
        reward_signal = self._satisfaction_to_reward(satisfaction)

        # Update all edges that were traversed
        for edge_id in traversed_edges:
            edge_weight = self.get_edge_weight(edge_id[0], edge_id[1])

            # Apply temporal decay before update
            edge_weight.apply_temporal_decay(self.decay_rate)

            # Update intent-specific weight
            edge_weight.update_intent_weight(
                intent_id, reward_signal, self.learning_rate, confidence
            )

            # Track usage
            edge_weight.increment_usage(intent_id)

    def _satisfaction_to_reward(self, satisfaction: float) -> float:
        """Convert satisfaction score to weight update target.

        Args:
            satisfaction: Score in [0.0, 1.0] from feedback

        Returns:
            Reward signal:
            - satisfaction = 1.0 → reward = 0.8 (reduce weight 20%)
            - satisfaction = 0.5 → reward = 1.0 (no change)
            - satisfaction = 0.0 → reward = 1.2 (increase weight 20%)
        """
        return 1.2 - 0.4 * satisfaction

    def apply_decay_to_all(self) -> None:
        """Apply temporal decay to all edge weights.

        This should be called periodically to ensure old patterns decay
        even if edges aren't being actively used.
        """
        for edge_weight in self.edge_weights.values():
            edge_weight.apply_temporal_decay(self.decay_rate)

    def get_weight_statistics(self) -> Dict[str, any]:
        """Compute statistics about current edge weights.

        Returns:
            Dictionary with global and per-intent statistics
        """
        if not self.edge_weights:
            return {
                "total_edges": 0,
                "global_weights": {"min": 1.0, "max": 1.0, "mean": 1.0},
                "intent_weights": {},
            }

        # Global weight statistics
        global_weights = [ew.global_weight for ew in self.edge_weights.values()]

        # Per-intent weight statistics
        intent_stats = {}
        for intent_id in range(self.k_intents):
            intent_weights = [
                ew.intent_weights[intent_id]
                for ew in self.edge_weights.values()
            ]
            intent_stats[f"intent_{intent_id}"] = {
                "min": float(min(intent_weights)),
                "max": float(max(intent_weights)),
                "mean": float(sum(intent_weights) / len(intent_weights)),
            }

        return {
            "total_edges": len(self.edge_weights),
            "global_weights": {
                "min": min(global_weights),
                "max": max(global_weights),
                "mean": sum(global_weights) / len(global_weights),
            },
            "intent_weights": intent_stats,
        }

    def get_intent_usage_statistics(self) -> Dict[int, Dict[str, int]]:
        """Get usage statistics per intent.

        Returns:
            Dictionary mapping intent_id to usage stats
        """
        intent_usage = {}

        for intent_id in range(self.k_intents):
            total_usage = sum(
                ew.usage_counts[intent_id]
                for ew in self.edge_weights.values()
            )

            edges_used = sum(
                1 for ew in self.edge_weights.values()
                if ew.usage_counts[intent_id] > 0
            )

            intent_usage[intent_id] = {
                "total_traversals": int(total_usage),
                "edges_used": int(edges_used),
            }

        return intent_usage

    def get_total_edges(self) -> int:
        """Get total number of edges being tracked.

        Returns:
            Number of edges with learned weights
        """
        return len(self.edge_weights)
