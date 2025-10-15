"""Adaptive ef_search selection per intent using Q-learning.

This module implements contextual multi-armed bandit learning for ef_search selection.

Theoretical Foundation:
- **Problem**: Contextual Multi-Armed Bandit
- **Arms (Actions)**: Different ef_search values [50, 75, 100, 150, 200, 250]
- **Context**: Query intent (from K-means clustering on query difficulty)
- **Reward**: Recall@k (fraction of ground truth neighbors retrieved)
- **Goal**: Learn Q(intent, ef_search) = expected recall

Algorithm: Action-Value Q-Learning with Phased Cold Start
- Phase 1 (queries 1-20): Uniform random exploration
- Phase 2 (queries 21-50): Epsilon-greedy with ε=0.3
- Phase 3 (queries 51+): Full UCB1 or epsilon-greedy
- Track Q(intent, ef) = average recall for each (intent, ef_search) pair
- Update: Q(intent, ef) ← running average of observed recall values
"""

from typing import Dict, List, Any
from collections import defaultdict
import numpy as np


class EfSearchSelector:
    """Q-learning based ef_search selector for intent-aware adaptation.

    Uses action-value Q-learning to learn which ef_search values maximize
    recall@k for each query intent. Implements phased cold start:
    - Phase 1 (queries 1-20): Uniform random exploration
    - Phase 2 (queries 21-50): Epsilon-greedy with ε=0.3
    - Phase 3 (queries 51+): Full UCB1 or epsilon-greedy
    """

    # Phased cold start thresholds
    PHASE_1_THRESHOLD = 20  # Uniform random
    PHASE_2_THRESHOLD = 50  # ε-greedy with high epsilon

    def __init__(
        self,
        k_intents: int,
        default_ef: int = 100,
        learning_rate: float = 0.1,  # Not used in Q-learning, kept for compatibility
        min_ef: int = 20,
        max_ef: int = 300,
        exploration_rate: float = 0.15,
        ef_candidates: List[int] = None,
        epsilon_decay_mode: str = "multiplicative",  # "multiplicative", "glie", or "none"
        min_epsilon: float = 0.01,
        use_ucb1: bool = False,
        ucb1_c: float = 1.414,
        use_warm_start: bool = False,
    ) -> None:
        """Initialize Q-learning based ef_search selector.

        Args:
            k_intents: Number of intent clusters
            default_ef: ef_search value for cold start / low confidence
            learning_rate: Kept for compatibility (not used in Q-learning)
            min_ef: Minimum allowed ef_search value
            max_ef: Maximum allowed ef_search value
            exploration_rate: Epsilon for epsilon-greedy (probability of exploration)
            ef_candidates: List of ef_search values to try (actions/arms)
            epsilon_decay_mode: "multiplicative" (ε *= 0.95), "glie" (ε = 1/t), or "none" (fixed)
            min_epsilon: Minimum epsilon value (for multiplicative decay)
            use_ucb1: Use UCB1 exploration strategy instead of epsilon-greedy
            ucb1_c: UCB1 exploration constant (typically sqrt(2) = 1.414)
            use_warm_start: Initialize Q-table with HNSW-theory priors instead of cold start
        """
        self.k_intents = k_intents
        self.default_ef = default_ef
        self.learning_rate = learning_rate  # Unused, kept for compatibility
        self.min_ef = min_ef
        self.max_ef = max_ef
        self.exploration_rate = exploration_rate
        self.initial_exploration_rate = exploration_rate  # Store initial value
        self.epsilon_decay_mode = epsilon_decay_mode
        self.min_epsilon = min_epsilon
        self.use_ucb1 = use_ucb1
        self.ucb1_c = ucb1_c
        self.use_warm_start = use_warm_start

        # Arms/Actions: Candidate ef_search values to explore
        self.ef_candidates = ef_candidates if ef_candidates else [50, 75, 100, 150, 200, 250]

        # Q-table: Q(intent, ef_search) = List of observed efficiencies
        # Structure: {intent_id: {ef_value: [efficiency_1, efficiency_2, ...]}}
        if use_warm_start:
            # Warm start with HNSW-theory priors
            self.q_table: Dict[int, Dict[int, List[float]]] = {
                intent_id: {ef: [self._get_prior_q_value(ef)] for ef in self.ef_candidates}
                for intent_id in range(k_intents)
            }
        else:
            # Cold start with empty Q-table
            self.q_table: Dict[int, Dict[int, List[float]]] = {
                intent_id: {ef: [] for ef in self.ef_candidates}
                for intent_id in range(k_intents)
            }

        # Action counts: Number of times each (intent, ef) pair was tried
        self.action_counts: Dict[int, Dict[int, int]] = {
            intent_id: {ef: 0 for ef in self.ef_candidates}
            for intent_id in range(k_intents)
        }

        # Track total feedback count for GLIE decay
        self.total_updates = 0

    def _get_prior_q_value(self, ef: int) -> float:
        """Get HNSW-theory based prior Q-value for warm start.

        Priors based on HNSW characteristics:
        - Low ef (20-75): Faster but lower recall → moderate efficiency
        - Medium ef (100-150): Balanced speed/recall → higher efficiency
        - High ef (200-350): High recall but slower → moderate efficiency

        Args:
            ef: ef_search value

        Returns:
            Prior Q-value (efficiency estimate)
        """
        if ef < 100:
            # Low ef: fast, lower recall
            return 150.0
        elif ef <= 150:
            # Medium ef: balanced (sweet spot for many workloads)
            return 200.0
        else:
            # High ef: high recall, slower
            return 180.0

    def select_ef(self, intent_id: int, confidence: float, confidence_threshold: float = 0.5) -> int:
        """Select ef_search using phased cold start strategy.

        Phase 1 (queries 1-20): Uniform random exploration
        Phase 2 (queries 21-50): Epsilon-greedy with ε=0.3
        Phase 3 (queries 51+): Full UCB1 or epsilon-greedy

        Args:
            intent_id: Query intent cluster ID
            confidence: Confidence in intent assignment
            confidence_threshold: Minimum confidence to use learned value

        Returns:
            ef_search value to use for this query
        """
        # Cold start or low confidence: use default
        if intent_id < 0 or confidence < confidence_threshold:
            return self.default_ef

        # Out of range: use default
        if intent_id >= self.k_intents:
            return self.default_ef

        # Phase 1: Uniform random exploration (queries 1-20)
        if self.total_updates < self.PHASE_1_THRESHOLD:
            return self._select_ef_uniform_random()

        # Phase 2: ε-greedy with high epsilon (queries 21-50)
        elif self.total_updates < self.PHASE_2_THRESHOLD:
            return self._select_ef_epsilon_greedy(intent_id, epsilon_override=0.3)

        # Phase 3: Full UCB1 or standard ε-greedy (queries 51+)
        else:
            if self.use_ucb1:
                return self._select_ef_ucb1(intent_id)
            else:
                return self._select_ef_epsilon_greedy(intent_id)

    def _select_ef_uniform_random(self) -> int:
        """Select ef_search uniformly at random (Phase 1).

        Returns:
            ef_search value selected randomly
        """
        import random
        return random.choice(self.ef_candidates)

    def _select_ef_epsilon_greedy(self, intent_id: int, epsilon_override: float = None) -> int:
        """Select ef_search using epsilon-greedy strategy.

        Args:
            intent_id: Query intent cluster ID
            epsilon_override: Optional epsilon value to override self.exploration_rate

        Returns:
            ef_search value to use for this query
        """
        import random

        # Use override epsilon if provided (for Phase 2), otherwise use self.exploration_rate
        epsilon = epsilon_override if epsilon_override is not None else self.exploration_rate

        # Epsilon-greedy strategy
        if random.random() < epsilon:
            # EXPLORATION: Try random ef_search value
            return random.choice(self.ef_candidates)

        # EXPLOITATION: Select ef with highest Q-value
        q_values = {}
        for ef in self.ef_candidates:
            if len(self.q_table[intent_id][ef]) > 0:
                # Q(intent, ef) = average recall observed
                q_values[ef] = np.mean(self.q_table[intent_id][ef])
            else:
                # Optimistic initialization: unexplored actions get high value
                # This encourages trying each action at least once
                q_values[ef] = float('inf')

        # Return ef_search with maximum Q-value
        best_ef = max(q_values.items(), key=lambda x: x[1])[0]
        return best_ef

    def _select_ef_ucb1(self, intent_id: int) -> int:
        """Select ef_search using UCB1 strategy.

        UCB1 formula: Q(a) + c * sqrt(ln(N) / n(a))
        where:
        - Q(a): average recall for action a
        - c: exploration constant (typically sqrt(2))
        - N: total actions taken for this intent
        - n(a): number of times action a was taken

        Args:
            intent_id: Query intent cluster ID

        Returns:
            ef_search value to use for this query
        """
        import math

        # Total actions taken for this intent
        total_actions = sum(self.action_counts[intent_id].values())

        # Cold start: try each action once first
        for ef in self.ef_candidates:
            if self.action_counts[intent_id][ef] == 0:
                return ef

        # Compute UCB1 values for all actions
        ucb_values = {}
        for ef in self.ef_candidates:
            n_a = self.action_counts[intent_id][ef]
            if n_a > 0 and len(self.q_table[intent_id][ef]) > 0:
                # Q(a): average recall observed
                q_mean = np.mean(self.q_table[intent_id][ef])

                # Exploration bonus: c * sqrt(ln(N) / n(a))
                exploration_bonus = self.ucb1_c * math.sqrt(math.log(total_actions) / n_a)

                # UCB1 value
                ucb_values[ef] = q_mean + exploration_bonus
            else:
                # Should not happen after cold start, but handle gracefully
                ucb_values[ef] = float('inf')

        # Select action with highest UCB1 value
        best_ef = max(ucb_values.items(), key=lambda x: x[1])[0]
        return best_ef

    def update_from_feedback(
        self,
        intent_id: int,
        ef_used: int,
        recall: float,
    ) -> None:
        """Update Q-table based on observed reward (recall@k).

        This is the Q-learning update: append the observed recall to
        Q(intent, ef_used). The Q-value is the running average of all
        observed recall values for this (intent, ef) pair.

        Args:
            intent_id: Intent that was detected
            ef_used: ef_search value that was used
            recall: Recall@k value (fraction of ground truth neighbors retrieved, 0-1)
        """
        # Validate inputs
        if intent_id < 0 or intent_id >= self.k_intents:
            return

        if not (0.0 <= recall <= 1.0):
            return

        # Reward is recall directly (no latency involved)
        reward = recall

        # Update Q-table: Add this recall observation for Q(intent, ef_used)
        if ef_used in self.q_table[intent_id]:
            self.q_table[intent_id][ef_used].append(reward)
            self.action_counts[intent_id][ef_used] += 1
        else:
            # ef_used not in candidates (e.g., user override)
            # We can still track it for logging, but won't use in selection
            pass

        # Increment total updates (used for phased cold start)
        self.total_updates += 1

    def get_current_phase(self) -> int:
        """Get current phase of cold start.

        Returns:
            1: Uniform random (queries 1-20)
            2: Epsilon-greedy with high ε (queries 21-50)
            3: Full UCB1 or standard ε-greedy (queries 51+)
        """
        if self.total_updates < self.PHASE_1_THRESHOLD:
            return 1
        elif self.total_updates < self.PHASE_2_THRESHOLD:
            return 2
        else:
            return 3

    def decay_exploration(self, min_rate: float = None, decay_factor: float = 0.95) -> None:
        """Gradually reduce exploration rate over time (Phase 3 only).

        Supports three modes:
        1. None: epsilon stays fixed (RECOMMENDED based on A/B testing)
        2. Multiplicative decay: eps(t+1) = max(min_eps, eps(t) * decay_factor)
        3. GLIE decay: eps(t) = eps_0 / (1 + t/100), guarantees convergence

        Args:
            min_rate: Minimum exploration rate (multiplicative mode only)
            decay_factor: Multiplicative decay factor (multiplicative mode only)

        Note: A/B testing with 350+ queries showed GLIE decay provides no significant
        improvement over fixed epsilon=0.15 (-0.4% recall change). The theoretical
        benefit exists but is negligible in practice for small action spaces with fast
        convergence. Recommended: use epsilon_decay_mode="none" with eps=0.15.
        """
        if min_rate is None:
            min_rate = self.min_epsilon

        if self.epsilon_decay_mode == "none":
            # No decay: epsilon stays fixed (RECOMMENDED)
            pass
        elif self.epsilon_decay_mode == "glie":
            # GLIE decay: eps(t) = eps_0 / (1 + t/100)
            # The /100 slows down decay to allow sufficient early exploration
            self.exploration_rate = self.initial_exploration_rate / (1 + self.total_updates / 100.0)
        else:
            # Multiplicative decay: eps(t+1) = max(min_eps, eps(t) * decay_factor)
            self.exploration_rate = max(min_rate, self.exploration_rate * decay_factor)

    def get_statistics(self) -> Dict[str, Any]:
        """Get ef_search selection statistics with Q-table values.

        Returns:
            Dictionary with Q-values, best ef per intent, action counts, and current phase
        """
        current_phase = self.get_current_phase()
        phase_names = {
            1: "uniform_random",
            2: "epsilon_greedy_high",
            3: "full_exploration"
        }

        stats = {
            "default_ef": self.default_ef,
            "k_intents": self.k_intents,
            "current_phase": current_phase,
            "phase_name": phase_names[current_phase],
            "exploration_strategy": "ucb1" if self.use_ucb1 else "epsilon_greedy",
            "exploration_rate": self.exploration_rate,
            "initial_exploration_rate": self.initial_exploration_rate,
            "epsilon_decay_mode": self.epsilon_decay_mode,
            "ucb1_enabled": self.use_ucb1,
            "ucb1_c": self.ucb1_c,
            "total_updates": self.total_updates,
            "ef_candidates": self.ef_candidates,
        }

        # Per-intent statistics with Q-values
        intent_stats = []
        for intent_id in range(self.k_intents):
            # Find best ef_search based on Q-values
            q_values = {}
            for ef in self.ef_candidates:
                if len(self.q_table[intent_id][ef]) > 0:
                    q_values[ef] = float(np.mean(self.q_table[intent_id][ef]))
                else:
                    q_values[ef] = None

            # Get best ef (highest Q-value among explored actions)
            explored_q = {ef: q for ef, q in q_values.items() if q is not None}
            best_ef = max(explored_q.items(), key=lambda x: x[1])[0] if explored_q else self.default_ef

            # Total queries for this intent
            total_queries = sum(self.action_counts[intent_id].values())

            intent_stats.append({
                "intent_id": intent_id,
                "learned_ef": best_ef,
                "num_queries": total_queries,
                "q_values": q_values,
                "action_counts": self.action_counts[intent_id].copy(),
            })

        stats["per_intent"] = intent_stats

        return stats

    def reset_ef(self, intent_id: int) -> None:
        """Reset Q-table for a specific intent.

        Args:
            intent_id: Intent to reset
        """
        if intent_id >= 0 and intent_id < self.k_intents:
            # Clear all Q-values for this intent
            for ef in self.ef_candidates:
                self.q_table[intent_id][ef] = []
                self.action_counts[intent_id][ef] = 0

    def reset_all(self) -> None:
        """Reset all Q-table values for all intents."""
        for intent_id in range(self.k_intents):
            self.reset_ef(intent_id)
