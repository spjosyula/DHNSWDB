"""
Query intent detection based on difficulty clustering.

This module clusters queries by difficulty (k-th NN distance) into intent tiers.
Easier queries need lower ef_search, harder queries need higher ef_search.
"""

from typing import Optional, Dict, List
from collections import deque
import numpy as np
from sklearn.cluster import KMeans


class IntentDetector:
    """
    Detects query intent by clustering difficulty scores into tiers.

    Intents represent difficulty levels:
        0: Very Easy (low k-NN distance)
        1: Easy
        2: Medium
        3: Hard
        4: Very Hard (high k-NN distance)

    Uses K-means clustering on 1D difficulty values to automatically
    discover difficulty boundaries from data.
    """

    def __init__(
        self,
        n_intents: int = 5,
        min_queries_for_clustering: int = 10,
        difficulty_buffer_size: int = 1000,
        random_state: int = 42
    ) -> None:
        """
        Initialize difficulty-based intent detector.

        Args:
            n_intents: Number of difficulty tiers (default: 5)
            min_queries_for_clustering: Minimum queries before clustering starts
            difficulty_buffer_size: Max difficulty values to keep in memory
            random_state: Random seed for reproducibility
        """
        self.n_intents = n_intents
        self.min_queries = min_queries_for_clustering
        self.random_state = random_state

        # Buffer of difficulty values for clustering
        self.difficulty_buffer: deque = deque(maxlen=difficulty_buffer_size)

        # K-means model (None until enough data)
        self.cluster_model: Optional[KMeans] = None

        # Sorted cluster order (0=easiest, n-1=hardest)
        self.cluster_order: Optional[np.ndarray] = None

        # Statistics
        self.total_queries = 0

    def add_query_difficulty(self, difficulty: float) -> None:
        """
        Record a query difficulty for later clustering.

        Automatically fits clustering model once minimum queries reached.

        Args:
            difficulty: k-th nearest neighbor distance
        """
        self.difficulty_buffer.append(difficulty)
        self.total_queries += 1

        # Trigger clustering if we just reached minimum threshold
        if len(self.difficulty_buffer) == self.min_queries:
            self._fit_clusters()

    def _fit_clusters(self) -> None:
        """
        Fit K-means clustering on collected difficulty values.

        Clusters are sorted by centroid value so that:
            - Intent 0 = lowest difficulty (easiest queries)
            - Intent n-1 = highest difficulty (hardest queries)
        """
        # Convert buffer to 2D array for sklearn
        difficulties = np.array(list(self.difficulty_buffer)).reshape(-1, 1)

        # Fit K-means
        self.cluster_model = KMeans(
            n_clusters=self.n_intents,
            random_state=self.random_state,
            n_init=10
        )
        self.cluster_model.fit(difficulties)

        # Sort clusters by centroid (0=easiest, n-1=hardest)
        centroids = self.cluster_model.cluster_centers_.flatten()
        self.cluster_order = np.argsort(centroids)

    def detect_intent(self, difficulty: float) -> int:
        """
        Assign intent based on difficulty score.

        Args:
            difficulty: k-th nearest neighbor distance

        Returns:
            Intent ID (0 to n_intents-1), or -1 if cold start

        Example:
            >>> detector = IntentDetector(n_intents=3)
            >>> # ... add 10 queries ...
            >>> detector.detect_intent(0.15)  # Easy query
            0
            >>> detector.detect_intent(0.95)  # Hard query
            2
        """
        # Cold start: not enough data yet
        if self.cluster_model is None:
            return -1

        # Predict cluster for this difficulty
        difficulty_2d = np.array([[difficulty]])
        cluster_id = self.cluster_model.predict(difficulty_2d)[0]

        # Map to sorted order (0=easiest)
        intent_id = np.where(self.cluster_order == cluster_id)[0][0]

        return int(intent_id)

    def get_cluster_centroids(self) -> Optional[np.ndarray]:
        """
        Get difficulty centroids for each intent (sorted by difficulty).

        Returns:
            Array of centroids [intent_0_centroid, ..., intent_n_centroid]
            where centroids are in ascending order (easy to hard).
            Returns None if clustering not initialized.
        """
        if self.cluster_model is None:
            return None

        # Get all centroids and sort by difficulty
        all_centroids = self.cluster_model.cluster_centers_.flatten()
        sorted_centroids = all_centroids[self.cluster_order]

        return sorted_centroids

    def get_intent_difficulty_range(self, intent_id: int) -> Optional[tuple]:
        """
        Get min/max difficulty range for an intent.

        Args:
            intent_id: Intent to query (0 to n_intents-1)

        Returns:
            (min_difficulty, max_difficulty) tuple, or None if not initialized
        """
        if self.cluster_model is None or intent_id < 0 or intent_id >= self.n_intents:
            return None

        # Get all difficulties assigned to this intent
        difficulties = np.array(list(self.difficulty_buffer)).reshape(-1, 1)
        labels = self.cluster_model.predict(difficulties)

        # Map labels to sorted order
        sorted_labels = np.array([
            np.where(self.cluster_order == label)[0][0]
            for label in labels
        ])

        # Filter difficulties for this intent
        intent_difficulties = difficulties[sorted_labels == intent_id].flatten()

        if len(intent_difficulties) == 0:
            return None

        return (float(intent_difficulties.min()), float(intent_difficulties.max()))

    def get_cluster_sizes(self) -> np.ndarray:
        """
        Get number of queries in each intent cluster.

        Returns:
            Array of cluster sizes (length = n_intents)
        """
        if self.cluster_model is None:
            return np.zeros(self.n_intents, dtype=int)

        # Predict intents for all buffered difficulties
        difficulties = np.array(list(self.difficulty_buffer)).reshape(-1, 1)
        labels = self.cluster_model.predict(difficulties)

        # Map to sorted order
        sorted_labels = np.array([
            np.where(self.cluster_order == label)[0][0]
            for label in labels
        ])

        # Count per intent
        sizes = np.bincount(sorted_labels, minlength=self.n_intents)
        return sizes

    def get_statistics(self) -> Dict:
        """
        Get intent detection statistics.

        Returns:
            Dictionary with:
                - total_queries: Total queries seen
                - clustering_active: Whether clustering is initialized
                - n_intents: Number of intent tiers
                - cluster_sizes: Queries per intent
                - centroids: Difficulty centroid per intent (if active)
        """
        cluster_sizes = self.get_cluster_sizes()
        centroids = self.get_cluster_centroids()

        stats = {
            "total_queries": self.total_queries,
            "clustering_active": self.cluster_model is not None,
            "n_intents": self.n_intents,
            "cluster_sizes": cluster_sizes.tolist(),
            "buffer_size": len(self.difficulty_buffer),
        }

        if centroids is not None:
            stats["centroids"] = centroids.tolist()

        return stats

    def is_active(self) -> bool:
        """
        Check if clustering is active (past cold start).

        Returns:
            True if enough queries collected and clustering fitted
        """
        return self.cluster_model is not None
