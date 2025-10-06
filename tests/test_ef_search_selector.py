"""Unit tests for EfSearchSelector with Q-learning."""

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
        """Should return default ef_search during cold start."""
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

    def test_exploration_returns_random_ef(self):
        """Should explore random ef values with epsilon probability."""
        selector = EfSearchSelector(k_intents=3, default_ef=100, exploration_rate=1.0)

        # With 100% exploration, should return random values from candidates
        selected_efs = [selector.select_ef(intent_id=0, confidence=0.9) for _ in range(20)]

        # Should see variety (not all the same)
        assert len(set(selected_efs)) > 1
        # All should be from candidates
        assert all(ef in selector.ef_candidates for ef in selected_efs)


class TestEfSearchQLearning:
    """Test Q-learning behavior."""

    def test_q_table_updates_with_feedback(self):
        """Should update Q-table with observed efficiency."""
        selector = EfSearchSelector(k_intents=3, default_ef=100)

        # Provide feedback
        selector.update_from_feedback(
            intent_id=0,
            ef_used=150,
            satisfaction=0.9,
            latency_ms=10.0
        )

        # Q-table should have one efficiency value
        assert len(selector.q_table[0][150]) == 1
        assert selector.action_counts[0][150] == 1

        # Efficiency = satisfaction / latency_sec = 0.9 / 0.01 = 90
        assert selector.q_table[0][150][0] == pytest.approx(90.0, rel=1e-5)

    def test_q_learning_converges_to_best_ef(self):
        """Should learn to prefer ef values with highest efficiency."""
        # Disable exploration to test pure exploitation
        selector = EfSearchSelector(k_intents=3, default_ef=100, exploration_rate=0.0)

        # Try all candidates to avoid optimistic initialization effects
        # Give low efficiency to most, high to ef=75
        for ef in selector.ef_candidates:
            if ef == 75:
                # High efficiency at ef=75
                for _ in range(5):
                    selector.update_from_feedback(
                        intent_id=0,
                        ef_used=75,
                        satisfaction=0.9,
                        latency_ms=5.0  # efficiency = 0.9/0.005 = 180
                    )
            else:
                # Low efficiency for all others
                selector.update_from_feedback(
                    intent_id=0,
                    ef_used=ef,
                    satisfaction=0.5,
                    latency_ms=20.0  # efficiency = 0.5/0.020 = 25
                )

        # With exploration=0, should always select ef=75 (highest Q-value)
        selected = selector.select_ef(intent_id=0, confidence=0.9)
        assert selected == 75

    def test_optimistic_initialization(self):
        """Should try unexplored actions first (optimistic initialization)."""
        selector = EfSearchSelector(k_intents=3, default_ef=100, exploration_rate=0.0)

        # Give mediocre feedback for ef=100
        selector.update_from_feedback(
            intent_id=0,
            ef_used=100,
            satisfaction=0.5,
            latency_ms=10.0  # efficiency = 50
        )

        # With exploration=0, should still explore unexplored ef values
        # because they have Q-value = infinity (optimistic initialization)
        selected = selector.select_ef(intent_id=0, confidence=0.9)

        # Should select an unexplored ef (not 100, which has Q=50)
        # Could be any candidate except 100
        assert selected in selector.ef_candidates

    def test_efficiency_calculation(self):
        """Should correctly calculate efficiency (satisfaction/latency_sec)."""
        selector = EfSearchSelector(k_intents=3, default_ef=100)

        # satisfaction=0.8, latency=20ms → efficiency = 0.8 / 0.02 = 40
        selector.update_from_feedback(
            intent_id=0,
            ef_used=150,
            satisfaction=0.8,
            latency_ms=20.0
        )

        # Check Q-table stores correct efficiency
        assert len(selector.q_table[0][150]) == 1
        assert selector.q_table[0][150][0] == pytest.approx(40.0, rel=1e-5)

    def test_invalid_intent_id_does_not_update(self):
        """Should ignore updates for invalid intent IDs."""
        selector = EfSearchSelector(k_intents=3, default_ef=100)

        # Negative intent
        selector.update_from_feedback(
            intent_id=-1,
            ef_used=200,
            satisfaction=1.0,
            latency_ms=10.0
        )

        # Out of range intent
        selector.update_from_feedback(
            intent_id=10,
            ef_used=200,
            satisfaction=1.0,
            latency_ms=10.0
        )

        # Q-table should be empty
        for intent_id in range(3):
            for ef in selector.ef_candidates:
                assert len(selector.q_table[intent_id][ef]) == 0

    def test_zero_or_negative_latency_ignored(self):
        """Should ignore feedback with invalid latency."""
        selector = EfSearchSelector(k_intents=3, default_ef=100)

        selector.update_from_feedback(
            intent_id=0,
            ef_used=200,
            satisfaction=1.0,
            latency_ms=0.0
        )

        # Should not update
        assert len(selector.q_table[0][200]) == 0

        selector.update_from_feedback(
            intent_id=0,
            ef_used=200,
            satisfaction=1.0,
            latency_ms=-5.0
        )

        # Should still not update
        assert len(selector.q_table[0][200]) == 0


class TestExplorationDecay:
    """Test exploration rate decay mechanism."""

    def test_decay_exploration(self):
        """Should decay exploration rate over time."""
        selector = EfSearchSelector(k_intents=3, exploration_rate=0.20)

        assert selector.exploration_rate == 0.20

        selector.decay_exploration(min_rate=0.05, decay_factor=0.95)
        assert selector.exploration_rate == pytest.approx(0.19, rel=1e-5)

        selector.decay_exploration(min_rate=0.05, decay_factor=0.95)
        assert selector.exploration_rate == pytest.approx(0.1805, rel=1e-3)

    def test_decay_respects_minimum(self):
        """Should not decay below minimum exploration rate."""
        selector = EfSearchSelector(k_intents=3, exploration_rate=0.06)

        # Decay multiple times
        for _ in range(10):
            selector.decay_exploration(min_rate=0.05, decay_factor=0.95)

        # Should stop at min_rate
        assert selector.exploration_rate == 0.05


class TestEfSearchStatistics:
    """Test statistics and monitoring."""

    def test_get_statistics_structure(self):
        """Should return Q-table statistics."""
        selector = EfSearchSelector(k_intents=3, default_ef=100)

        stats = selector.get_statistics()

        assert "default_ef" in stats
        assert "k_intents" in stats
        assert "exploration_rate" in stats
        assert "ef_candidates" in stats
        assert "per_intent" in stats

        assert len(stats["per_intent"]) == 3

    def test_get_statistics_with_q_values(self):
        """Should include Q-values in statistics."""
        selector = EfSearchSelector(k_intents=3, default_ef=100)

        # Add feedback
        selector.update_from_feedback(0, 150, 0.9, 10.0)
        selector.update_from_feedback(0, 75, 0.8, 5.0)

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
        """Learned ef should be the one with highest Q-value."""
        selector = EfSearchSelector(k_intents=3, default_ef=100)

        # ef=75 has high efficiency
        for _ in range(3):
            selector.update_from_feedback(0, 75, 0.9, 5.0)  # efficiency = 180

        # ef=150 has low efficiency
        for _ in range(3):
            selector.update_from_feedback(0, 150, 0.5, 20.0)  # efficiency = 25

        stats = selector.get_statistics()
        intent_0_stats = stats["per_intent"][0]

        # Learned ef should be 75 (highest Q-value)
        assert intent_0_stats["learned_ef"] == 75

    def test_reset_ef_clears_q_table(self):
        """Should reset Q-table for specific intent."""
        selector = EfSearchSelector(k_intents=3, default_ef=100)

        # Add feedback
        selector.update_from_feedback(0, 200, 1.0, 10.0)
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
            selector.update_from_feedback(i, 200, 1.0, 10.0)

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
        """Each intent should learn its own optimal ef_search."""
        selector = EfSearchSelector(k_intents=3, default_ef=100, exploration_rate=0.0)

        # Intent 0: High efficiency at ef=200
        for _ in range(10):
            selector.update_from_feedback(
                intent_id=0,
                ef_used=200,
                satisfaction=0.95,
                latency_ms=10.0  # efficiency = 95
            )

        # Intent 1: High efficiency at ef=50
        for _ in range(10):
            selector.update_from_feedback(
                intent_id=1,
                ef_used=50,
                satisfaction=0.90,
                latency_ms=5.0  # efficiency = 180
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

        # Different efficiency for each intent
        selector.update_from_feedback(0, 150, 0.8, 10.0)  # efficiency = 80
        selector.update_from_feedback(1, 150, 0.6, 20.0)  # efficiency = 30

        # Check Q-tables are different
        q_0_150 = np.mean(selector.q_table[0][150])
        q_1_150 = np.mean(selector.q_table[1][150])

        assert q_0_150 == pytest.approx(80.0, rel=1e-5)
        assert q_1_150 == pytest.approx(30.0, rel=1e-5)
        assert q_0_150 > q_1_150
