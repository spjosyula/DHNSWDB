"""Query intent detection using K-means clustering.

This module implements intent detection to enable context-aware adaptation.
Queries are clustered in embedding space to identify distinct search patterns.
"""

from typing import Tuple, Optional, List, Dict
from collections import deque
import numpy as np
import numpy.typing as npt
from sklearn.cluster import KMeans, MiniBatchKMeans

Vector = npt.NDArray[np.float32]


class IntentDetector:
    """Detects query intent using K-means clustering on query embeddings.

    The detector maintains a sliding window of recent queries and performs
    clustering to identify distinct intent patterns. Each query is assigned
    to the nearest cluster with a confidence score.
    """

    def __init__(
        self,
        k_intents: int = 5,
        min_queries_for_clustering: int = 50,
        query_buffer_size: int = 1000,
        confidence_threshold: float = 0.7,
        random_state: int = 42,
        seed_centroids: Optional[np.ndarray] = None,
        use_incremental: bool = False,
    ) -> None:
        """Initialize intent detector.

        Args:
            k_intents: Number of intent clusters
            min_queries_for_clustering: Minimum queries before clustering starts
            query_buffer_size: Max queries to keep in memory
            confidence_threshold: Minimum confidence for intent assignment
            random_state: Random seed for reproducibility
            seed_centroids: Optional predefined centroids for bootstrap (k_intents × dim)
            use_incremental: Use MiniBatchKMeans for incremental updates
        """
        self.k_intents = k_intents
        self.min_queries = min_queries_for_clustering
        self.confidence_threshold = confidence_threshold
        self.random_state = random_state
        self.seed_centroids = seed_centroids
        self.use_incremental = use_incremental

        # Query buffer for clustering
        self.query_buffer: deque = deque(maxlen=query_buffer_size)

        # K-means model (None until enough queries)
        self.kmeans: Optional[KMeans] = None

        # Cluster centroids for drift detection
        self.cluster_centroids: Optional[np.ndarray] = None

        # Retroactive assignments (populated when clustering initializes)
        self.retroactive_assignments: Dict[int, int] = {}  # query_index -> intent_id
        self.buffer_start_index = 0  # Track position in buffer

        # Statistics
        self.total_queries = 0
        self.confident_detections = 0

    def detect_intent(self, query_vector: Vector) -> Tuple[int, float]:
        """Detect intent for a query vector.

        Args:
            query_vector: Query embedding vector

        Returns:
            (intent_id, confidence_score) tuple
            - intent_id: Cluster ID (0 to k_intents-1), or -1 if cold start
            - confidence: Confidence score in [0, 1]
        """
        # Add to buffer
        self.query_buffer.append(query_vector.copy())
        self.total_queries += 1

        # Cold start: not enough data yet
        if self.kmeans is None:
            if len(self.query_buffer) >= self.min_queries:
                self._initialize_clustering()
            else:
                return -1, 0.0  # No intent assigned during cold start

        # Predict cluster
        query_2d = query_vector.reshape(1, -1)
        intent_id = int(self.kmeans.predict(query_2d)[0])

        # Compute confidence based on distance to centroids
        confidence = self._compute_confidence(query_vector)

        if confidence >= self.confidence_threshold:
            self.confident_detections += 1

        return intent_id, confidence

    def _initialize_clustering(self) -> None:
        """Initialize K-means clustering on query buffer.

        Uses seed centroids if provided, otherwise k-means++ initialization.
        Performs retroactive intent assignment for all buffered queries.
        """
        queries = np.array(list(self.query_buffer))

        # Determine initialization method
        if self.seed_centroids is not None:
            # Use provided seed centroids
            init_centroids = self.seed_centroids
        else:
            # Use k-means++ for data-driven initialization
            init_centroids = 'k-means++'

        # Choose clustering algorithm
        if self.use_incremental:
            # MiniBatchKMeans for incremental updates
            self.kmeans = MiniBatchKMeans(
                n_clusters=self.k_intents,
                init=init_centroids,
                random_state=self.random_state,
                batch_size=100,
                max_iter=100
            )
        else:
            # Standard KMeans for better quality
            self.kmeans = KMeans(
                n_clusters=self.k_intents,
                init=init_centroids,
                n_init=10 if isinstance(init_centroids, str) else 1,
                random_state=self.random_state
            )

        self.kmeans.fit(queries)
        self.cluster_centroids = self.kmeans.cluster_centers_.copy()

        # Retroactive assignment: assign intents to all buffered queries
        self._retroactive_assign_intents(queries)

    def _compute_confidence(self, query_vector: Vector) -> float:
        """Compute confidence score for intent assignment.

        Confidence is based on relative distance to nearest vs second-nearest centroid.
        High confidence = query is much closer to one centroid than others.

        Args:
            query_vector: Query embedding

        Returns:
            Confidence score in [0, 1]
        """
        if self.kmeans is None:
            return 0.0

        # Compute distances to all centroids
        distances = np.linalg.norm(
            self.kmeans.cluster_centers_ - query_vector,
            axis=1
        )

        # Sort distances
        sorted_distances = np.sort(distances)

        # Avoid division by zero
        if sorted_distances[1] < 1e-8:
            return 1.0

        # Relative confidence: how much closer to nearest vs second-nearest
        # If nearest is much closer than second, confidence is high
        confidence = 1.0 - (sorted_distances[0] / sorted_distances[1])

        return np.clip(confidence, 0.0, 1.0)

    def _retroactive_assign_intents(self, queries: np.ndarray) -> None:
        """Assign intents to all buffered queries after clustering initializes.

        This ensures we don't waste the first batch of queries during cold start.
        The assignments can be used for backfilling historical learning.

        Args:
            queries: Array of buffered query vectors
        """
        if self.kmeans is None:
            return

        # Predict intents for all buffered queries
        intent_labels = self.kmeans.predict(queries)

        # Store assignments (keyed by absolute query index)
        base_index = self.total_queries - len(queries)
        for i, intent_id in enumerate(intent_labels):
            query_index = base_index + i
            self.retroactive_assignments[query_index] = int(intent_id)

    def get_retroactive_assignments(self) -> Dict[int, int]:
        """Get retroactive intent assignments for buffered queries.

        Returns:
            Dictionary mapping query_index -> intent_id for buffered queries
        """
        return self.retroactive_assignments.copy()

    def check_drift(self, drift_threshold: float = 2.0) -> bool:
        """Check if intent distribution has drifted significantly.

        Uses statistical test: centroid shift normalized by query variance.

        Args:
            drift_threshold: Number of standard deviations for drift detection

        Returns:
            True if drift detected (re-clustering recommended)
        """
        if self.kmeans is None or len(self.query_buffer) < self.min_queries:
            return False

        # Compute current centroids from recent queries
        queries = np.array(list(self.query_buffer))
        current_labels = self.kmeans.predict(queries)

        drift_detected = False

        for intent_id in range(self.k_intents):
            # Get queries assigned to this intent
            intent_queries = queries[current_labels == intent_id]

            if len(intent_queries) < 5:  # Too few samples
                continue

            # Compute new centroid from recent queries
            new_centroid = intent_queries.mean(axis=0)
            old_centroid = self.cluster_centroids[intent_id]

            # Centroid shift
            centroid_shift = np.linalg.norm(new_centroid - old_centroid)

            # Query variance around new centroid (for normalization)
            distances_to_centroid = np.linalg.norm(
                intent_queries - new_centroid,
                axis=1
            )
            query_std = np.std(distances_to_centroid)

            # Avoid division by zero
            if query_std < 1e-8:
                continue

            # Normalized drift metric (in standard deviations)
            normalized_drift = centroid_shift / query_std

            if normalized_drift > drift_threshold:
                drift_detected = True
                break

        return drift_detected

    def recompute_clusters(self) -> None:
        """Recompute intent clusters on current query buffer.

        Called when drift is detected to update cluster definitions.
        """
        if len(self.query_buffer) >= self.min_queries:
            self._initialize_clustering()

    def get_cluster_sizes(self) -> np.ndarray:
        """Get number of queries in each cluster.

        Returns:
            Array of cluster sizes
        """
        if self.kmeans is None:
            return np.zeros(self.k_intents, dtype=int)

        queries = np.array(list(self.query_buffer))
        labels = self.kmeans.predict(queries)

        sizes = np.bincount(labels, minlength=self.k_intents)
        return sizes

    def get_statistics(self) -> dict:
        """Get intent detection statistics.

        Returns:
            Dictionary with detection metrics
        """
        cluster_sizes = self.get_cluster_sizes()

        return {
            "total_queries": self.total_queries,
            "confident_detections": self.confident_detections,
            "confidence_rate": (
                self.confident_detections / self.total_queries
                if self.total_queries > 0
                else 0.0
            ),
            "clustering_active": self.kmeans is not None,
            "k_intents": self.k_intents,
            "cluster_sizes": cluster_sizes.tolist(),
            "buffer_size": len(self.query_buffer),
        }

    def get_intent_centroid(self, intent_id: int) -> Optional[Vector]:
        """Get centroid vector for a specific intent.

        Args:
            intent_id: Intent cluster ID

        Returns:
            Centroid vector, or None if clustering not initialized
        """
        if self.kmeans is None or intent_id < 0 or intent_id >= self.k_intents:
            return None

        return self.kmeans.cluster_centers_[intent_id].astype(np.float32)
