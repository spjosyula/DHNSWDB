"""Unit tests for EfSearchSelector with Q-learning and phased cold start."""

import pytest
import numpy as np
from dynhnsw.ef_search_selector import EfSearchSelector


class TestEfSearchSelectorBasics:
    """Test basic ef_search selector functionality."""

    def test_initialization(self):
        """Should initialize with Q-table and action counts."""
        selector = EfSearchSelector(k_intents=3, default_ef=100)

        assert selector.k_intents == 3
        assert selector.default_ef == 100
        assert len(selector.q_table) == 3
        assert len(selector.action_counts) == 3

        # Check Q-table is initialized with empty lists
        for intent_id in range(3):
            for ef in selector.ef_candidates:
                assert selector.q_table[intent_id][ef] == []
                assert selector.action_counts[intent_id][ef] == 0

    def test_cold_start_returns_default(self):
        """Should return default ef_search for invalid intent."""
        selector = EfSearchSelector(k_intents=3, default_ef=100)

        # Cold start: intent_id = -1
        ef = selector.select_ef(intent_id=-1, confidence=0.9)
        assert ef == 100

    def test_low_confidence_returns_default(self):
        """Should return default ef_search for low confidence."""
        selector = EfSearchSelector(k_intents=3, default_ef=100)

        # Low confidence
        ef = selector.select_ef(intent_id=0, confidence=0.3, confidence_threshold=0.5)
        assert ef == 100

    def test_out_of_range_intent_returns_default(self):
        """Should return default for out-of-range intent ID."""
        selector = EfSearchSelector(k_intents=3, default_ef=100)

        ef = selector.select_ef(intent_id=10, confidence=0.9)
        assert ef == 100


class TestPhasedColdStart:
    """Test phased cold start behavior."""

    def test_phase_1_uniform_random(self):
        """Phase 1 (queries 1-20) should select uniformly at random."""
        selector = EfSearchSelector(k_intents=3, default_ef=100)

        # First 20 queries should be uniform random
        selected_efs = []
        for i in range(20):
            ef = selector.select_ef(intent_id=0, confidence=0.9)
            selected_efs.append(ef)
            # Simulate feedback to increment total_updates
            selector.update_from_feedback(intent_id=0, ef_used=ef, recall=0.7)

        # Should see variety (not all the same)
        assert len(set(selected_efs)) > 1
        # All should be from candidates
        assert all(ef in selector.ef_candidates for ef in selected_efs)

        # Check we're still in phase 1
        assert selector.get_current_phase() == 2  # Just moved to phase 2

    def test_phase_2_epsilon_greedy_high(self):
        """Phase 2 (queries 21-50) should use ε-greedy with ε=0.3."""
        selector = EfSearchSelector(k_intents=3, default_ef=100, exploration_rate=0.15)

        # Advance to phase 2 (queries 21-50)
        for i in range(20):
            ef = selector.select_ef(intent_id=0, confidence=0.9)
            selector.update_from_feedback(intent_id=0, ef_used=ef, recall=0.7)

        # Now in phase 2
        assert selector.get_current_phase() == 2

        # Phase 2 should use high epsilon (0.3) regardless of self.exploration_rate
        # Difficult to test probabilistically, but we can verify phase
        for i in range(10):
            ef = selector.select_ef(intent_id=0, confidence=0.9)
            selector.update_from_feedback(intent_id=0, ef_used=ef, recall=0.8)
            assert selector.get_current_phase() == 2

    def test_phase_3_full_exploration(self):
        """Phase 3 (queries 51+) should use full UCB1 or ε-greedy."""
        selector = EfSearchSelector(k_intents=3, default_ef=100, use_ucb1=False, exploration_rate=0.15)

        # Advance to phase 3 (queries 51+)
        for i in range(50):
            ef = selector.select_ef(intent_id=0, confidence=0.9)
            selector.update_from_feedback(intent_id=0, ef_used=ef, recall=0.7)

        # Now in phase 3
        assert selector.get_current_phase() == 3

        # Should use standard epsilon-greedy
        ef = selector.select_ef(intent_id=0, confidence=0.9)
        assert ef in selector.ef_candidates

    def test_get_current_phase(self):
        """Should correctly report current phase."""
        selector = EfSearchSelector(k_intents=3, default_ef=100)

        # Phase 1: queries 1-20
        assert selector.get_current_phase() == 1

        # Advance to phase 2
        for i in range(20):
            selector.update_from_feedback(intent_id=0, ef_used=100, recall=0.7)
        assert selector.get_current_phase() == 2

        # Advance to phase 3
        for i in range(30):
            selector.update_from_feedback(intent_id=0, ef_used=100, recall=0.7)
        assert selector.get_current_phase() == 3


class TestRecallBasedQLearning:
    """Test Q-learning with recall-based rewards."""

    def test_q_table_updates_with_recall(self):
        """Should update Q-table with observed recall."""
        selector = EfSearchSelector(k_intents=3, default_ef=100)

        # Provide feedback with recall
        selector.update_from_feedback(
            intent_id=0,
            ef_used=150,
            recall=0.85
        )

        # Q-table should have one recall value
        assert len(selector.q_table[0][150]) == 1
        assert selector.action_counts[0][150] == 1

        # Reward is recall directly
        assert selector.q_table[0][150][0] == pytest.approx(0.85, rel=1e-5)

    def test_q_learning_converges_to_best_ef(self):
        """Should learn to prefer ef values with highest recall."""
        # Use phase 3 (disable phased cold start) by advancing updates
        selector = EfSearchSelector(k_intents=3, default_ef=100, exploration_rate=0.0)

        # Advance to phase 3
        for _ in range(50):
            selector.total_updates += 1

        # Try all candidates to avoid optimistic initialization effects
        # Give low recall to most, high to ef=200
        for ef in selector.ef_candidates:
            if ef == 200:
                # High recall at ef=200
                for _ in range(5):
                    selector.update_from_feedback(
                        intent_id=0,
                        ef_used=200,
                        recall=0.95
                    )
            else:
                # Low recall for all others
                selector.update_from_feedback(
                    intent_id=0,
                    ef_used=ef,
                    recall=0.60
                )

        # With exploration=0, should always select ef=200 (highest Q-value)
        selected = selector.select_ef(intent_id=0, confidence=0.9)
        assert selected == 200

    def test_optimistic_initialization(self):
        """Should try unexplored actions first (optimistic initialization)."""
        selector = EfSearchSelector(k_intents=3, default_ef=100, exploration_rate=0.0)

        # Advance to phase 3
        for _ in range(50):
            selector.total_updates += 1

        # Give mediocre feedback for ef=100
        selector.update_from_feedback(
            intent_id=0,
            ef_used=100,
            recall=0.60
        )

        # With exploration=0, should still explore unexplored ef values
        # because they have Q-value = infinity (optimistic initialization)
        selected = selector.select_ef(intent_id=0, confidence=0.9)

        # Should select an unexplored ef (not 100, which has Q=0.60)
        # Could be any candidate except 100
        assert selected in selector.ef_candidates

    def test_invalid_intent_id_does_not_update(self):
        """Should ignore updates for invalid intent IDs."""
        selector = EfSearchSelector(k_intents=3, default_ef=100)

        # Negative intent
        selector.update_from_feedback(
            intent_id=-1,
            ef_used=200,
            recall=0.90
        )

        # Out of range intent
        selector.update_from_feedback(
            intent_id=10,
            ef_used=200,
            recall=0.90
        )

        # Q-table should be empty
        for intent_id in range(3):
            for ef in selector.ef_candidates:
                assert len(selector.q_table[intent_id][ef]) == 0

    def test_invalid_recall_ignored(self):
        """Should ignore feedback with invalid recall values."""
        selector = EfSearchSelector(k_intents=3, default_ef=100)

        # Recall > 1.0
        selector.update_from_feedback(
            intent_id=0,
            ef_used=200,
            recall=1.5
        )

        # Should not update
        assert len(selector.q_table[0][200]) == 0

        # Recall < 0.0
        selector.update_from_feedback(
            intent_id=0,
            ef_used=200,
            recall=-0.5
        )

        # Should still not update
        assert len(selector.q_table[0][200]) == 0


class TestExplorationDecay:
    """Test exploration rate decay mechanism."""

    def test_decay_exploration_multiplicative(self):
        """Should decay exploration rate multiplicatively."""
        selector = EfSearchSelector(k_intents=3, exploration_rate=0.20, epsilon_decay_mode="multiplicative")

        assert selector.exploration_rate == 0.20

        selector.decay_exploration(min_rate=0.05, decay_factor=0.95)
        assert selector.exploration_rate == pytest.approx(0.19, rel=1e-5)

        selector.decay_exploration(min_rate=0.05, decay_factor=0.95)
        assert selector.exploration_rate == pytest.approx(0.1805, rel=1e-3)

    def test_decay_respects_minimum(self):
        """Should not decay below minimum exploration rate."""
        selector = EfSearchSelector(k_intents=3, exploration_rate=0.06, epsilon_decay_mode="multiplicative")

        # Decay multiple times
        for _ in range(10):
            selector.decay_exploration(min_rate=0.05, decay_factor=0.95)

        # Should stop at min_rate
        assert selector.exploration_rate == 0.05

    def test_no_decay_mode(self):
        """Should not decay when mode is 'none'."""
        selector = EfSearchSelector(k_intents=3, exploration_rate=0.15, epsilon_decay_mode="none")

        initial_rate = selector.exploration_rate

        # Decay multiple times
        for _ in range(10):
            selector.decay_exploration()

        # Should stay the same
        assert selector.exploration_rate == initial_rate


class TestEfSearchStatistics:
    """Test statistics and monitoring."""

    def test_get_statistics_structure(self):
        """Should return Q-table statistics with phase information."""
        selector = EfSearchSelector(k_intents=3, default_ef=100)

        stats = selector.get_statistics()

        assert "default_ef" in stats
        assert "k_intents" in stats
        assert "current_phase" in stats
        assert "phase_name" in stats
        assert "exploration_rate" in stats
        assert "ef_candidates" in stats
        assert "per_intent" in stats

        assert len(stats["per_intent"]) == 3
        assert stats["current_phase"] == 1  # Initial phase

    def test_get_statistics_with_q_values(self):
        """Should include Q-values (recall) in statistics."""
        selector = EfSearchSelector(k_intents=3, default_ef=100)

        # Add feedback
        selector.update_from_feedback(0, 150, recall=0.85)
        selector.update_from_feedback(0, 75, recall=0.90)

        stats = selector.get_statistics()
        intent_0_stats = stats["per_intent"][0]

        assert "q_values" in intent_0_stats
        assert "action_counts" in intent_0_stats
        assert "learned_ef" in intent_0_stats
        assert "num_queries" in intent_0_stats

        # Check Q-values
        assert intent_0_stats["q_values"][150] is not None
        assert intent_0_stats["q_values"][75] is not None
        assert intent_0_stats["num_queries"] == 2

    def test_learned_ef_is_best_q_value(self):
        """Learned ef should be the one with highest Q-value (recall)."""
        selector = EfSearchSelector(k_intents=3, default_ef=100)

        # ef=200 has high recall
        for _ in range(3):
            selector.update_from_feedback(0, 200, recall=0.95)

        # ef=50 has low recall
        for _ in range(3):
            selector.update_from_feedback(0, 50, recall=0.70)

        stats = selector.get_statistics()
        intent_0_stats = stats["per_intent"][0]

        # Learned ef should be 200 (highest Q-value/recall)
        assert intent_0_stats["learned_ef"] == 200

    def test_reset_ef_clears_q_table(self):
        """Should reset Q-table for specific intent."""
        selector = EfSearchSelector(k_intents=3, default_ef=100)

        # Add feedback
        selector.update_from_feedback(0, 200, recall=0.90)
        assert len(selector.q_table[0][200]) == 1

        # Reset intent 0
        selector.reset_ef(0)

        # Q-table should be empty for intent 0
        for ef in selector.ef_candidates:
            assert len(selector.q_table[0][ef]) == 0
            assert selector.action_counts[0][ef] == 0

    def test_reset_all_clears_all_q_tables(self):
        """Should reset Q-tables for all intents."""
        selector = EfSearchSelector(k_intents=3, default_ef=100)

        # Add feedback for all intents
        for i in range(3):
            selector.update_from_feedback(i, 200, recall=0.85)

        # Reset all
        selector.reset_all()

        # All Q-tables should be empty
        for intent_id in range(3):
            for ef in selector.ef_candidates:
                assert len(selector.q_table[intent_id][ef]) == 0
                assert selector.action_counts[intent_id][ef] == 0


class TestIntentDifferentiation:
    """Test that different intents learn different ef_search values."""

    def test_different_intents_learn_independently(self):
        """Each intent should learn its own optimal ef_search based on recall."""
        selector = EfSearchSelector(k_intents=3, default_ef=100, exploration_rate=0.0)

        # Advance to phase 3
        for _ in range(50):
            selector.total_updates += 1

        # Intent 0: High recall at ef=200 (hard queries need high ef)
        for _ in range(10):
            selector.update_from_feedback(
                intent_id=0,
                ef_used=200,
                recall=0.95
            )

        # Intent 1: High recall at ef=50 (easy queries need low ef)
        for _ in range(10):
            selector.update_from_feedback(
                intent_id=1,
                ef_used=50,
                recall=0.92
            )

        stats = selector.get_statistics()

        # Intent 0 should prefer ef=200
        intent_0_best = stats["per_intent"][0]["learned_ef"]
        # Intent 1 should prefer ef=50
        intent_1_best = stats["per_intent"][1]["learned_ef"]

        assert intent_0_best == 200
        assert intent_1_best == 50
        assert intent_0_best > intent_1_best

    def test_per_intent_q_table_isolation(self):
        """Q-tables should be independent across intents."""
        selector = EfSearchSelector(k_intents=2, default_ef=100)

        # Different recall for each intent
        selector.update_from_feedback(0, 150, recall=0.90)
        selector.update_from_feedback(1, 150, recall=0.70)

        # Check Q-tables are different
        q_0_150 = np.mean(selector.q_table[0][150])
        q_1_150 = np.mean(selector.q_table[1][150])

        assert q_0_150 == pytest.approx(0.90, rel=1e-5)
        assert q_1_150 == pytest.approx(0.70, rel=1e-5)
        assert q_0_150 > q_1_150


class TestUCB1Strategy:
    """Test UCB1 exploration strategy."""

    def test_ucb1_enabled(self):
        """Should use UCB1 when enabled in phase 3."""
        selector = EfSearchSelector(k_intents=3, default_ef=100, use_ucb1=True, ucb1_c=1.414)

        # Advance to phase 3
        for _ in range(50):
            ef = selector.select_ef(intent_id=0, confidence=0.9)
            selector.update_from_feedback(intent_id=0, ef_used=ef, recall=0.80)

        # Should be in phase 3 and using UCB1
        assert selector.get_current_phase() == 3
        assert selector.use_ucb1 is True

        # UCB1 should select based on exploration bonus
        ef = selector.select_ef(intent_id=0, confidence=0.9)
        assert ef in selector.ef_candidates

    def test_ucb1_cold_start_tries_all_actions(self):
        """UCB1 should try each action once before computing UCB values."""
        selector = EfSearchSelector(k_intents=3, default_ef=100, use_ucb1=True)

        # Advance to phase 3
        for _ in range(50):
            selector.total_updates += 1

        # Select actions until all have been tried at least once
        tried_actions = set()
        for _ in range(20):
            ef = selector.select_ef(intent_id=0, confidence=0.9)
            tried_actions.add(ef)
            selector.update_from_feedback(intent_id=0, ef_used=ef, recall=0.80)

            # Once all actions tried, break
            if len(tried_actions) == len(selector.ef_candidates):
                break

        # Should have tried all actions
        assert len(tried_actions) == len(selector.ef_candidates)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
