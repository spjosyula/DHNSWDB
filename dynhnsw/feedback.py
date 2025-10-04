"""Feedback collection and processing for adaptive HNSW.

This module handles user feedback signals that drive edge weight learning.
Supports binary relevance (relevant/not relevant) per result.
"""

from dataclasses import dataclass
from typing import List, Set
import numpy as np
import numpy.typing as npt


# Type aliases for clarity
DocumentId = int
Vector = npt.NDArray[np.float32]


@dataclass
class QueryFeedback:
    """Feedback for a single query and its results.

    Attributes:
        query_vector: The original query vector
        result_ids: Document IDs returned by search
        relevant_ids: Set of IDs marked as relevant by user
        timestamp: When feedback was provided
    """

    query_vector: Vector
    result_ids: List[DocumentId]
    relevant_ids: Set[DocumentId]
    timestamp: float

    def get_satisfaction_score(self) -> float:
        """Compute query satisfaction as fraction of relevant results.

        Returns:
            Score in [0.0, 1.0] where 1.0 means all results were relevant
        """
        if not self.result_ids:
            return 0.0

        relevant_count = len(self.relevant_ids.intersection(set(self.result_ids)))
        return relevant_count / len(self.result_ids)

    def is_result_relevant(self, doc_id: DocumentId) -> bool:
        """Check if a specific result was marked relevant.

        Args:
            doc_id: Document ID to check

        Returns:
            True if this document was marked relevant
        """
        return doc_id in self.relevant_ids


class FeedbackCollector:
    """Collects and aggregates user feedback over time.

    Maintains a buffer of recent feedback for analysis and pattern detection.
    """

    def __init__(self, buffer_size: int = 1000) -> None:
        """Initialize feedback collector.

        Args:
            buffer_size: Maximum number of feedback entries to keep in memory
        """
        self.buffer_size = buffer_size
        self.feedback_history: List[QueryFeedback] = []

    def add_feedback(
        self,
        query_vector: Vector,
        result_ids: List[DocumentId],
        relevant_ids: Set[DocumentId],
        timestamp: float,
    ) -> QueryFeedback:
        """Record feedback for a query.

        Args:
            query_vector: The query that was issued
            result_ids: Results returned by search
            relevant_ids: Which results the user found relevant
            timestamp: When this feedback was provided

        Returns:
            The created QueryFeedback object
        """
        feedback = QueryFeedback(
            query_vector=query_vector,
            result_ids=result_ids,
            relevant_ids=relevant_ids,
            timestamp=timestamp,
        )

        self.feedback_history.append(feedback)

        # Maintain buffer size by removing oldest entries
        if len(self.feedback_history) > self.buffer_size:
            self.feedback_history.pop(0)

        return feedback

    def get_recent_feedback(self, count: int = 100) -> List[QueryFeedback]:
        """Retrieve most recent feedback entries.

        Args:
            count: Number of recent entries to retrieve

        Returns:
            List of recent QueryFeedback objects (newest first)
        """
        return self.feedback_history[-count:][::-1]

    def compute_average_satisfaction(self, window_size: int = 100) -> float:
        """Compute average satisfaction over recent queries.

        Args:
            window_size: Number of recent queries to consider

        Returns:
            Average satisfaction score in [0.0, 1.0]
        """
        recent = self.get_recent_feedback(window_size)
        if not recent:
            return 0.0

        total_satisfaction = sum(fb.get_satisfaction_score() for fb in recent)
        return total_satisfaction / len(recent)

    def get_feedback_count(self) -> int:
        """Get total number of feedback entries collected.

        Returns:
            Number of feedback entries in history
        """
        return len(self.feedback_history)
