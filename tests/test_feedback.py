"""Unit tests for feedback collection and processing."""

import time
import numpy as np
import pytest
from dynhnsw.feedback import FeedbackCollector, QueryFeedback


class TestQueryFeedback:
    """Test QueryFeedback data structure."""

    def test_satisfaction_score_all_relevant(self):
        """All results relevant should give satisfaction = 1.0."""
        query_vec = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        result_ids = [1, 2, 3]
        relevant_ids = {1, 2, 3}

        feedback = QueryFeedback(
            query_vector=query_vec,
            result_ids=result_ids,
            relevant_ids=relevant_ids,
            timestamp=time.time(),
        )

        assert feedback.get_satisfaction_score() == pytest.approx(1.0)

    def test_satisfaction_score_none_relevant(self):
        """No results relevant should give satisfaction = 0.0."""
        query_vec = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        result_ids = [1, 2, 3]
        relevant_ids = set()  # Nothing marked relevant

        feedback = QueryFeedback(
            query_vector=query_vec,
            result_ids=result_ids,
            relevant_ids=relevant_ids,
            timestamp=time.time(),
        )

        assert feedback.get_satisfaction_score() == pytest.approx(0.0)

    def test_satisfaction_score_partial_relevance(self):
        """Partial relevance should give fractional satisfaction."""
        query_vec = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        result_ids = [1, 2, 3, 4, 5]
        relevant_ids = {1, 3}  # 2 out of 5 relevant

        feedback = QueryFeedback(
            query_vector=query_vec,
            result_ids=result_ids,
            relevant_ids=relevant_ids,
            timestamp=time.time(),
        )

        assert feedback.get_satisfaction_score() == pytest.approx(0.4)

    def test_satisfaction_score_empty_results(self):
        """Empty results should give satisfaction = 0.0."""
        query_vec = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        result_ids = []
        relevant_ids = set()

        feedback = QueryFeedback(
            query_vector=query_vec,
            result_ids=result_ids,
            relevant_ids=relevant_ids,
            timestamp=time.time(),
        )

        assert feedback.get_satisfaction_score() == pytest.approx(0.0)

    def test_is_result_relevant_true(self):
        """Should correctly identify relevant results."""
        query_vec = np.array([1.0], dtype=np.float32)
        feedback = QueryFeedback(
            query_vector=query_vec,
            result_ids=[1, 2, 3],
            relevant_ids={1, 3},
            timestamp=time.time(),
        )

        assert feedback.is_result_relevant(1) is True
        assert feedback.is_result_relevant(3) is True

    def test_is_result_relevant_false(self):
        """Should correctly identify non-relevant results."""
        query_vec = np.array([1.0], dtype=np.float32)
        feedback = QueryFeedback(
            query_vector=query_vec,
            result_ids=[1, 2, 3],
            relevant_ids={1},
            timestamp=time.time(),
        )

        assert feedback.is_result_relevant(2) is False
        assert feedback.is_result_relevant(3) is False


class TestFeedbackCollector:
    """Test FeedbackCollector functionality."""

    def test_initial_state_empty(self):
        """New collector should have no feedback."""
        collector = FeedbackCollector()
        assert collector.get_feedback_count() == 0

    def test_add_feedback_stores_entry(self):
        """Adding feedback should store it in history."""
        collector = FeedbackCollector()
        query_vec = np.array([1.0, 2.0], dtype=np.float32)

        collector.add_feedback(
            query_vector=query_vec,
            result_ids=[1, 2, 3],
            relevant_ids={1, 2},
            timestamp=time.time(),
        )

        assert collector.get_feedback_count() == 1

    def test_add_multiple_feedback_entries(self):
        """Should accumulate multiple feedback entries."""
        collector = FeedbackCollector()
        query_vec = np.array([1.0], dtype=np.float32)

        for i in range(5):
            collector.add_feedback(
                query_vector=query_vec,
                result_ids=[i],
                relevant_ids={i},
                timestamp=time.time(),
            )

        assert collector.get_feedback_count() == 5

    def test_buffer_size_limit_enforced(self):
        """Should not exceed buffer_size."""
        buffer_size = 10
        collector = FeedbackCollector(buffer_size=buffer_size)
        query_vec = np.array([1.0], dtype=np.float32)

        # Add more than buffer_size entries
        for i in range(20):
            collector.add_feedback(
                query_vector=query_vec,
                result_ids=[i],
                relevant_ids={i},
                timestamp=time.time(),
            )

        assert collector.get_feedback_count() == buffer_size

    def test_buffer_evicts_oldest_entries(self):
        """Exceeding buffer should remove oldest entries."""
        buffer_size = 3
        collector = FeedbackCollector(buffer_size=buffer_size)
        query_vec = np.array([1.0], dtype=np.float32)

        # Add entries with distinct result_ids to track them
        for i in range(5):
            collector.add_feedback(
                query_vector=query_vec,
                result_ids=[i],
                relevant_ids={i},
                timestamp=time.time() + i,  # Different timestamps
            )

        # Should only have last 3 entries (result_ids: 2, 3, 4)
        recent = collector.get_recent_feedback(count=10)
        result_ids_in_buffer = [fb.result_ids[0] for fb in recent]

        assert len(result_ids_in_buffer) == 3
        assert result_ids_in_buffer == [4, 3, 2]  # Newest first

    def test_get_recent_feedback_returns_newest_first(self):
        """get_recent_feedback should return newest entries first."""
        collector = FeedbackCollector()
        query_vec = np.array([1.0], dtype=np.float32)

        # Add entries with distinct timestamps
        timestamps = []
        for i in range(5):
            ts = time.time() + i
            timestamps.append(ts)
            collector.add_feedback(
                query_vector=query_vec,
                result_ids=[i],
                relevant_ids={i},
                timestamp=ts,
            )

        recent = collector.get_recent_feedback(count=3)

        # Should get newest 3 in reverse chronological order
        assert len(recent) == 3
        assert recent[0].timestamp == timestamps[4]
        assert recent[1].timestamp == timestamps[3]
        assert recent[2].timestamp == timestamps[2]

    def test_get_recent_feedback_count_exceeds_history(self):
        """Requesting more than available should return all entries."""
        collector = FeedbackCollector()
        query_vec = np.array([1.0], dtype=np.float32)

        for i in range(3):
            collector.add_feedback(
                query_vector=query_vec,
                result_ids=[i],
                relevant_ids={i},
                timestamp=time.time(),
            )

        recent = collector.get_recent_feedback(count=100)

        assert len(recent) == 3

    def test_compute_average_satisfaction_all_perfect(self):
        """All perfect results should give average = 1.0."""
        collector = FeedbackCollector()
        query_vec = np.array([1.0], dtype=np.float32)

        for i in range(10):
            collector.add_feedback(
                query_vector=query_vec,
                result_ids=[i],
                relevant_ids={i},  # All relevant
                timestamp=time.time(),
            )

        avg_satisfaction = collector.compute_average_satisfaction()
        assert avg_satisfaction == pytest.approx(1.0)

    def test_compute_average_satisfaction_all_bad(self):
        """All bad results should give average = 0.0."""
        collector = FeedbackCollector()
        query_vec = np.array([1.0], dtype=np.float32)

        for i in range(10):
            collector.add_feedback(
                query_vector=query_vec,
                result_ids=[i],
                relevant_ids=set(),  # Nothing relevant
                timestamp=time.time(),
            )

        avg_satisfaction = collector.compute_average_satisfaction()
        assert avg_satisfaction == pytest.approx(0.0)

    def test_compute_average_satisfaction_mixed(self):
        """Mixed results should give correct average."""
        collector = FeedbackCollector()
        query_vec = np.array([1.0], dtype=np.float32)

        # Add 5 entries with 100% satisfaction
        for i in range(5):
            collector.add_feedback(
                query_vector=query_vec,
                result_ids=[i],
                relevant_ids={i},
                timestamp=time.time(),
            )

        # Add 5 entries with 0% satisfaction
        for i in range(5):
            collector.add_feedback(
                query_vector=query_vec,
                result_ids=[i],
                relevant_ids=set(),
                timestamp=time.time(),
            )

        avg_satisfaction = collector.compute_average_satisfaction()
        assert avg_satisfaction == pytest.approx(0.5)

    def test_compute_average_satisfaction_empty_collector(self):
        """Empty collector should return 0.0."""
        collector = FeedbackCollector()

        avg_satisfaction = collector.compute_average_satisfaction()
        assert avg_satisfaction == pytest.approx(0.0)

    def test_compute_average_satisfaction_respects_window(self):
        """Should only consider entries within window_size."""
        collector = FeedbackCollector()
        query_vec = np.array([1.0], dtype=np.float32)

        # Add 10 bad entries
        for i in range(10):
            collector.add_feedback(
                query_vector=query_vec,
                result_ids=[i],
                relevant_ids=set(),
                timestamp=time.time(),
            )

        # Add 5 perfect entries
        for i in range(5):
            collector.add_feedback(
                query_vector=query_vec,
                result_ids=[i],
                relevant_ids={i},
                timestamp=time.time(),
            )

        # Window of 5 should only see the perfect entries
        avg_satisfaction = collector.compute_average_satisfaction(window_size=5)
        assert avg_satisfaction == pytest.approx(1.0)
