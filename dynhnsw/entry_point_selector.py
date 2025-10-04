"""Entry point selection for intent-aware HNSW search.

This module implements intent-specific entry point learning. Instead of always
starting search from the same entry point, we learn optimal entry points for
each query intent based on feedback.

Mathematical foundation:
- Track satisfaction scores: S[intent_id, entry_point]
- Update via exponential moving average: S_new = (1-α)S_old + α·satisfaction
- Select: best_entry[intent] = argmax_{entry} S[intent, entry]
"""

from typing import Dict, List, Any, Optional
import numpy as np
import numpy.typing as npt

from dynhnsw.hnsw.graph import HNSWGraph

Vector = npt.NDArray[np.float32]


class EntryPointSelector:
    """Learns optimal entry points for each query intent.

    Strategy:
    1. Identify candidate entry points (high-layer nodes)
    2. Track which entries lead to high satisfaction per intent
    3. Select best entry based on historical performance
    4. Fall back to default for low confidence or cold start
    """

    def __init__(
        self,
        k_intents: int,
        graph: HNSWGraph,
        min_layer_for_entry: int = 1,
        learning_rate: float = 0.1,
        exploration_rate: float = 0.1,
    ) -> None:
        """Initialize entry point selector.

        Args:
            k_intents: Number of intent clusters
            graph: HNSW graph to search
            min_layer_for_entry: Minimum layer for entry point candidates
            learning_rate: Learning rate for score updates
            exploration_rate: Probability of random exploration (epsilon-greedy)
        """
        self.k_intents = k_intents
        self.graph = graph
        self.min_layer_for_entry = min_layer_for_entry
        self.learning_rate = learning_rate
        self.exploration_rate = exploration_rate

        # Get candidate entry points (high-layer nodes)
        self.candidate_entries = self._get_high_layer_nodes()

        # Score matrix: [k_intents, num_candidates]
        # Initialized to 0.5 (neutral)
        num_candidates = len(self.candidate_entries)
        self.entry_scores = np.full(
            (k_intents, num_candidates),
            0.5,
            dtype=np.float32
        )

        # Current best entry per intent
        self.best_entries: Dict[int, int] = {
            i: graph.entry_point for i in range(k_intents)
        }

        # Usage tracking
        self.entry_usage_counts = np.zeros(
            (k_intents, num_candidates),
            dtype=np.int32
        )

    def _get_high_layer_nodes(self) -> List[int]:
        """Get nodes at high layers as entry point candidates.

        High-layer nodes are better entry points because they have
        long-range connections and can navigate quickly.

        Returns:
            List of node IDs at layer >= min_layer_for_entry
        """
        candidates = []

        for node_id, node in self.graph.nodes.items():
            if node.level >= self.min_layer_for_entry:
                candidates.append(node_id)

        # Fallback: if no high-layer nodes, use all nodes
        if not candidates:
            candidates = list(self.graph.nodes.keys())

        # Always include graph's default entry point
        if self.graph.entry_point not in candidates:
            candidates.append(self.graph.entry_point)

        return candidates

    def select_entry(
        self,
        intent_id: int,
        confidence: float,
        confidence_threshold: float = 0.5
    ) -> int:
        """Select entry point for given intent using epsilon-greedy strategy.

        Args:
            intent_id: Query intent cluster ID
            confidence: Confidence in intent assignment
            confidence_threshold: Minimum confidence to use intent-specific entry

        Returns:
            Node ID to use as entry point
        """
        # Cold start or low confidence: use default entry
        if intent_id < 0 or confidence < confidence_threshold:
            return self.graph.entry_point

        # Out of range: use default
        if intent_id >= self.k_intents:
            return self.graph.entry_point

        # Epsilon-greedy: explore with probability exploration_rate
        if np.random.random() < self.exploration_rate:
            # Explore: select random candidate entry
            return int(np.random.choice(self.candidate_entries))
        else:
            # Exploit: return best entry for this intent
            return self.best_entries[intent_id]

    def update_from_feedback(
        self,
        intent_id: int,
        entry_used: int,
        satisfaction: float,
    ) -> None:
        """Update entry point scores based on feedback.

        Args:
            intent_id: Intent that was detected
            entry_used: Entry point that was used
            satisfaction: Satisfaction score from feedback (0-1)
        """
        # Validate inputs
        if intent_id < 0 or intent_id >= self.k_intents:
            return

        if entry_used not in self.candidate_entries:
            return

        # Get index of entry point
        entry_idx = self.candidate_entries.index(entry_used)

        # Update score using exponential moving average
        old_score = self.entry_scores[intent_id, entry_idx]
        new_score = (
            (1 - self.learning_rate) * old_score +
            self.learning_rate * satisfaction
        )
        self.entry_scores[intent_id, entry_idx] = new_score

        # Update usage count
        self.entry_usage_counts[intent_id, entry_idx] += 1

        # Update best entry for this intent (argmax of scores)
        best_idx = int(np.argmax(self.entry_scores[intent_id]))
        self.best_entries[intent_id] = self.candidate_entries[best_idx]

    def get_entry_score(self, intent_id: int, entry_id: int) -> float:
        """Get current score for a specific (intent, entry) pair.

        Args:
            intent_id: Intent cluster ID
            entry_id: Entry point node ID

        Returns:
            Current score (0-1), or 0.5 if not found
        """
        if intent_id < 0 or intent_id >= self.k_intents:
            return 0.5

        if entry_id not in self.candidate_entries:
            return 0.5

        entry_idx = self.candidate_entries.index(entry_id)
        return float(self.entry_scores[intent_id, entry_idx])

    def get_statistics(self) -> Dict[str, Any]:
        """Get entry point selection statistics.

        Returns:
            Dictionary with usage and performance metrics
        """
        stats = {
            "num_candidates": len(self.candidate_entries),
            "candidate_entries": self.candidate_entries,
            "best_entries": self.best_entries,
            "entry_scores": self.entry_scores.tolist(),
            "total_usage": int(self.entry_usage_counts.sum()),
        }

        # Per-intent statistics
        intent_stats = []
        for intent_id in range(self.k_intents):
            best_entry = self.best_entries[intent_id]
            best_idx = self.candidate_entries.index(best_entry)
            best_score = float(self.entry_scores[intent_id, best_idx])
            usage_count = int(self.entry_usage_counts[intent_id].sum())

            intent_stats.append({
                "intent_id": intent_id,
                "best_entry": best_entry,
                "best_score": best_score,
                "total_usage": usage_count,
            })

        stats["per_intent"] = intent_stats

        return stats

    def reset_scores(self) -> None:
        """Reset all entry point scores to neutral (0.5).

        Useful for performance degradation recovery.
        """
        self.entry_scores.fill(0.5)

        # Reset best entries to default
        for intent_id in range(self.k_intents):
            self.best_entries[intent_id] = self.graph.entry_point
