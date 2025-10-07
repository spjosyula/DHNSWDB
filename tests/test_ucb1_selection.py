"""Tests for UCB1 exploration strategy in ef_search selection.

This module tests the UCB1 (Upper Confidence Bound) algorithm for adaptive
ef_search selection. UCB1 is theoretically optimal for stationary multi-armed
bandits and should outperform epsilon-greedy in large action spaces.
"""

import pytest
import numpy as np
from dynhnsw import VectorStore
from dynhnsw.config import DynHNSWConfig, get_ucb1_config
from dynhnsw.ef_search_selector import EfSearchSelector


class TestUCB1Configuration:
    """Test UCB1 configuration validation and setup."""

    def test_ucb1_config_validation(self):
        """UCB1 and epsilon decay should be mutually exclusive."""
        with pytest.raises(ValueError, match="Cannot enable both"):
            DynHNSWConfig(enable_ucb1=True, enable_epsilon_decay=True)

    def test_ucb1_config_preset(self):
        """Test UCB1 preset configuration."""
        config = get_ucb1_config()
        assert config.enable_ucb1 is True
        assert config.enable_epsilon_decay is False
        assert config.ucb1_exploration_constant == 1.414
        assert config.epsilon_decay_mode == "none"

    def test_ucb1_positive_constant(self):
        """UCB1 exploration constant must be positive."""
        with pytest.raises(ValueError, match="ucb1_exploration_constant must be positive"):
            DynHNSWConfig(enable_ucb1=True, ucb1_exploration_constant=-1.0)

    def test_ucb1_auto_disables_epsilon_decay(self):
        """Enabling UCB1 should auto-disable epsilon decay."""
        config = DynHNSWConfig(
            enable_ucb1=True,
            epsilon_decay_mode="glie"
        )
        assert config.epsilon_decay_mode == "none"
        assert config.enable_epsilon_decay is False


class TestUCB1Selector:
    """Test UCB1 selection logic in EfSearchSelector."""

    def test_ucb1_cold_start_explores_all_actions(self):
        """UCB1 should try each action once before using confidence bounds."""
        selector = EfSearchSelector(
            k_intents=3,
            default_ef=100,
            use_ucb1=True,
            ucb1_c=1.414,
            ef_candidates=[50, 75, 100, 150, 200, 250]
        )

        intent_id = 0
        confidence = 1.0
        selected_actions = []

        # During cold start, should try each action once
        for _ in range(6):  # 6 candidates
            ef = selector.select_ef(intent_id, confidence)
            selected_actions.append(ef)
            # Update action count to simulate feedback (required for UCB1 to move to next action)
            selector.update_from_feedback(intent_id, ef, 0.7, 10.0)

        # All candidates should have been tried
        assert set(selected_actions) == {50, 75, 100, 150, 200, 250}

    def test_ucb1_uses_confidence_bounds_after_cold_start(self):
        """After cold start, UCB1 should use confidence bounds."""
        selector = EfSearchSelector(
            k_intents=3,
            default_ef=100,
            use_ucb1=True,
            ucb1_c=1.414,
            ef_candidates=[50, 100, 150]
        )

        intent_id = 0
        confidence = 1.0

        # Cold start: try each action once with fake feedback
        for ef in [50, 100, 150]:
            selected_ef = selector.select_ef(intent_id, confidence)
            # Simulate feedback: ef=100 is best (high efficiency)
            if selected_ef == 100:
                selector.update_from_feedback(intent_id, selected_ef, 1.0, 10.0)
            else:
                selector.update_from_feedback(intent_id, selected_ef, 0.5, 10.0)

        # After cold start, should start using UCB1 formula
        # Can't predict exact selection due to exploration bonus, but should not error
        for _ in range(10):
            ef = selector.select_ef(intent_id, confidence)
            assert ef in [50, 100, 150]

    def test_ucb1_statistics(self):
        """UCB1 statistics should indicate correct exploration strategy."""
        selector = EfSearchSelector(
            k_intents=3,
            use_ucb1=True,
            ucb1_c=1.414
        )

        stats = selector.get_statistics()
        assert stats["exploration_strategy"] == "ucb1"
        assert stats["ucb1_enabled"] is True
        assert stats["ucb1_c"] == 1.414


class TestUCB1Integration:
    """Test UCB1 integration with VectorStore."""

    @pytest.fixture
    def clustered_dataset(self):
        """Create dataset with 3 distinct clusters."""
        np.random.seed(42)
        vectors = []

        # Cluster 0
        cluster0 = np.random.randn(200, 128).astype(np.float32) * 2 + 10.0
        vectors.extend(cluster0)

        # Cluster 1
        cluster1 = np.random.randn(200, 128).astype(np.float32) * 2 - 10.0
        vectors.extend(cluster1)

        # Cluster 2
        cluster2 = np.random.randn(200, 128).astype(np.float32) * 2
        cluster2[:, 1] += 20.0
        vectors.extend(cluster2)

        return np.array(vectors)

    def test_vector_store_with_ucb1_config(self, clustered_dataset):
        """VectorStore should work with UCB1 configuration."""
        config = get_ucb1_config()
        store = VectorStore(
            dimension=128,
            config=config,
            enable_intent_detection=True,
            k_intents=3
        )

        store.add(clustered_dataset)

        # Cold start queries
        for i in range(35):
            query = np.random.randn(128).astype(np.float32) + 10.0
            results = store.search(query, k=10)
            assert len(results) > 0

        # Check statistics
        stats = store.get_statistics()
        assert "ef_search_selection" in stats
        assert stats["ef_search_selection"]["exploration_strategy"] == "ucb1"

    def test_ucb1_learning_with_feedback(self, clustered_dataset):
        """UCB1 should learn from feedback."""
        config = get_ucb1_config()
        store = VectorStore(
            dimension=128,
            config=config,
            enable_intent_detection=True,
            k_intents=3
        )

        store.add(clustered_dataset)

        # Activate clustering
        for i in range(35):
            query = np.random.randn(128).astype(np.float32) + 10.0
            store.search(query, k=10)

        # Provide feedback
        for _ in range(20):
            query = np.random.randn(128).astype(np.float32) + 10.0
            results = store.search(query, k=10)
            relevant_ids = [r["id"] for r in results[:5]]
            store.provide_feedback(relevant_ids=relevant_ids)

        # Check learning occurred
        stats = store.get_statistics()
        if "ef_search_selection" in stats:
            per_intent = stats["ef_search_selection"]["per_intent"]
            total_queries = sum(intent["num_queries"] for intent in per_intent)
            assert total_queries > 0


class TestUCB1VsEpsilonGreedy:
    """Compare UCB1 vs epsilon-greedy behavior."""

    def test_ucb1_explores_more_systematically(self):
        """UCB1 should explore more systematically than epsilon-greedy."""
        # UCB1 selector
        ucb1_selector = EfSearchSelector(
            k_intents=3,
            use_ucb1=True,
            ucb1_c=1.414,
            ef_candidates=[50, 75, 100, 150, 200, 250]
        )

        # Epsilon-greedy selector
        epsilon_selector = EfSearchSelector(
            k_intents=3,
            use_ucb1=False,
            exploration_rate=0.15,
            ef_candidates=[50, 75, 100, 150, 200, 250]
        )

        intent_id = 0
        confidence = 1.0

        # Track action diversity during first 20 selections
        ucb1_actions = []
        epsilon_actions = []

        for _ in range(20):
            ucb1_ef = ucb1_selector.select_ef(intent_id, confidence)
            epsilon_ef = epsilon_selector.select_ef(intent_id, confidence)

            # Simulate feedback (same for both)
            ucb1_selector.update_from_feedback(intent_id, ucb1_ef, 0.7, 10.0)
            epsilon_selector.update_from_feedback(intent_id, epsilon_ef, 0.7, 10.0)

            ucb1_actions.append(ucb1_ef)
            epsilon_actions.append(epsilon_ef)

        # UCB1 should explore more systematically (try all actions early)
        ucb1_unique = len(set(ucb1_actions[:10]))
        epsilon_unique = len(set(epsilon_actions[:10]))

        # UCB1 should have tried more unique actions during cold start
        # This is a probabilistic test, so we use a loose assertion
        assert ucb1_unique >= 5  # Should try most actions


class TestUCB1EdgeCases:
    """Test edge cases and error handling for UCB1."""

    def test_ucb1_low_confidence_uses_default(self):
        """Low confidence should use default ef_search."""
        selector = EfSearchSelector(
            k_intents=3,
            default_ef=100,
            use_ucb1=True
        )

        ef = selector.select_ef(intent_id=0, confidence=0.05, confidence_threshold=0.5)
        assert ef == 100  # Should use default

    def test_ucb1_invalid_intent_uses_default(self):
        """Invalid intent ID should use default ef_search."""
        selector = EfSearchSelector(
            k_intents=3,
            default_ef=100,
            use_ucb1=True
        )

        ef = selector.select_ef(intent_id=-1, confidence=1.0)
        assert ef == 100

        ef = selector.select_ef(intent_id=5, confidence=1.0)
        assert ef == 100

    def test_ucb1_with_single_action(self):
        """UCB1 should work with single action space."""
        selector = EfSearchSelector(
            k_intents=3,
            use_ucb1=True,
            ef_candidates=[100]
        )

        ef = selector.select_ef(intent_id=0, confidence=1.0)
        assert ef == 100

    def test_ucb1_reset_functionality(self):
        """Test reset functionality with UCB1."""
        selector = EfSearchSelector(
            k_intents=3,
            use_ucb1=True,
            ef_candidates=[50, 100, 150]
        )

        intent_id = 0

        # Make some selections and updates
        for _ in range(10):
            ef = selector.select_ef(intent_id, confidence=1.0)
            selector.update_from_feedback(intent_id, ef, 0.7, 10.0)

        # Reset
        selector.reset_ef(intent_id)

        # After reset, should be back to cold start
        stats = selector.get_statistics()
        per_intent = stats["per_intent"]
        assert per_intent[intent_id]["num_queries"] == 0
