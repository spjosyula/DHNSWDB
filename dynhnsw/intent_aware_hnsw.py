"""Intent-aware adaptive HNSW searcher.

This module integrates intent detection with adaptive ef_search selection.
Different query intents learn optimal ef_search values for improved search efficiency.

Mathematical Foundation:
- Detect query intent using K-means clustering on query embeddings
- Learn optimal ef_search values per intent using Q-learning
- Reward function: efficiency = satisfaction / latency (maximize quality per time)
- Search uses pure HNSW algorithm with intent-specific ef_search parameter
- No edge weight modulation - adaptation is purely through search parameter tuning
"""

import time
from typing import List, Tuple, Set, Optional, Dict, Any
import numpy as np
import numpy.typing as npt

from dynhnsw.hnsw.graph import HNSWGraph
from dynhnsw.hnsw.distance import cosine_distance
from dynhnsw.intent_detector import IntentDetector
from dynhnsw.ef_search_selector import EfSearchSelector
from dynhnsw.feedback import FeedbackCollector, QueryFeedback
from dynhnsw.performance_monitor import PerformanceMonitor, PerformanceMetrics
from dynhnsw.config import DynHNSWConfig, get_default_config

Vector = npt.NDArray[np.float32]
DocumentId = int
EdgeId = Tuple[int, int]


class IntentAwareHNSWSearcher:
    """Intent-aware HNSW searcher with adaptive ef_search per intent.

    This searcher detects query intent and learns optimal ef_search values for
    each intent based on feedback. Different query types (exploratory vs precise)
    benefit from different search breadths.

    Core adaptation:
    - Learns ef_search values that maximize efficiency (satisfaction/latency) per intent
    - Exploratory queries → higher ef_search (broader recall)
    - Precise queries → lower ef_search (faster, focused results)
    """

    def __init__(
        self,
        graph: HNSWGraph,
        ef_search: int = 50,
        k_intents: int = 5,
        learning_rate: float = 0.1,
        enable_adaptation: bool = True,
        enable_intent_detection: bool = True,
        confidence_threshold: float = None,
        min_queries_for_clustering: int = 30,
        config: Optional[DynHNSWConfig] = None,
    ) -> None:
        """Initialize intent-aware HNSW searcher.

        Args:
            graph: HNSW graph structure
            ef_search: Default search candidate list size
            k_intents: Number of intent clusters
            learning_rate: ef_search learning rate (legacy, not used in Q-learning)
            enable_adaptation: If False, behaves like static HNSW
            enable_intent_detection: If False, uses default ef_search
            confidence_threshold: Minimum confidence for intent-specific ef_search
            min_queries_for_clustering: Queries needed before clustering starts
            config: DynHNSWConfig object for advanced configuration
        """
        # Use provided config or create default
        if config is None:
            config = get_default_config()
        self.config = config

        # Use config values if parameters not explicitly provided
        if confidence_threshold is None:
            confidence_threshold = self.config.confidence_threshold

        self.graph = graph
        self.ef_search = ef_search
        self.enable_adaptation = enable_adaptation
        self.enable_intent_detection = enable_intent_detection and enable_adaptation
        self.k_intents = k_intents
        self.confidence_threshold = confidence_threshold

        # Intent detection
        self.intent_detector = IntentDetector(
            k_intents=k_intents,
            min_queries_for_clustering=min_queries_for_clustering,
            confidence_threshold=confidence_threshold
        ) if enable_intent_detection else None

        # Adaptive ef_search selection using Q-learning
        # Use config to determine epsilon decay mode
        epsilon_decay_mode = "none"
        if self.config.enable_epsilon_decay:
            epsilon_decay_mode = self.config.epsilon_decay_mode

        self.ef_selector = EfSearchSelector(
            k_intents=k_intents,
            default_ef=ef_search,
            learning_rate=learning_rate,
            exploration_rate=self.config.exploration_rate,
            epsilon_decay_mode=epsilon_decay_mode,
            min_epsilon=self.config.min_epsilon,
        ) if enable_adaptation else None

        # Feedback and monitoring
        self.feedback_collector = FeedbackCollector(buffer_size=1000)
        self.performance_monitor = PerformanceMonitor(
            baseline_recall=0.0,
            degradation_threshold=0.95,
            window_size=50
        )

        # Track current intent and ef_search used
        self.last_intent_id: int = -1
        self.last_confidence: float = 0.0
        self.last_ef_used: int = ef_search
        self.query_count: int = 0

    def search(
        self, query: Vector, k: int, ef_search: Optional[int] = None
    ) -> List[Tuple[DocumentId, float]]:
        """Search for k nearest neighbors with intent-aware ef_search adaptation.

        Flow:
        1. Detect query intent (if enabled)
        2. Select optimal ef_search for that intent (if adaptation enabled)
        3. Perform HNSW search with adaptive ef_search
        4. Track state for feedback learning

        Args:
            query: Query vector
            k: Number of results to return
            ef_search: Override default ef_search for this query

        Returns:
            List of (node_id, distance) tuples sorted by distance
        """
        if self.graph.size() == 0:
            return []

        # Step 1: Detect intent
        if self.enable_intent_detection and self.intent_detector:
            self.last_intent_id, self.last_confidence = \
                self.intent_detector.detect_intent(query)
        else:
            self.last_intent_id = -1
            self.last_confidence = 0.0

        # Step 2: Select ef_search based on intent
        if self.enable_adaptation and self.ef_selector and ef_search is None:
            # Use learned ef_search for this intent
            ef = self.ef_selector.select_ef(
                intent_id=self.last_intent_id,
                confidence=self.last_confidence,
                confidence_threshold=self.confidence_threshold
            )
            self.last_ef_used = ef
        else:
            # Use provided ef_search or default
            ef = ef_search if ef_search is not None else self.ef_search
            self.last_ef_used = ef

        # Ensure ef is at least k
        ef = max(ef, k)

        # Step 3: Standard HNSW search with adaptive ef_search
        entry_point = self.graph.entry_point
        entry_node = self.graph.get_node(entry_point)

        # Search from top layer down to layer 1
        current_nearest = [entry_point]
        for layer in range(entry_node.level, 0, -1):
            current_nearest = self._search_layer(
                query=query, entry_points=current_nearest, num_closest=1, layer=layer
            )

        # At layer 0, expand search with adaptive ef_search
        candidates = self._search_layer(
            query=query, entry_points=current_nearest, num_closest=ef, layer=0
        )

        # Calculate final distances and return top k
        results = []
        for node_id in candidates:
            node = self.graph.get_node(node_id)
            dist = self._get_distance(query, node.vector)
            results.append((node_id, dist))

        results.sort(key=lambda x: x[1])

        # Track query count for drift detection
        self.query_count += 1

        # Periodic drift check
        if (self.query_count % 50 == 0 and
            self.enable_intent_detection and
            self.intent_detector):
            if self.intent_detector.check_drift(drift_threshold=2.0):
                self._handle_intent_drift()

        return results[:k]

    def _search_layer(
        self, query: Vector, entry_points: List[int], num_closest: int, layer: int
    ) -> List[int]:
        """Greedy search at a single layer (pure HNSW algorithm).

        Args:
            query: Query vector
            entry_points: Starting nodes
            num_closest: Number of results to return
            layer: Layer to search

        Returns:
            List of node IDs closest to query
        """
        visited: Set[int] = set(entry_points)
        candidates: List[Tuple[float, int]] = []

        # Initialize with entry points
        for node_id in entry_points:
            node = self.graph.get_node(node_id)
            dist = self._get_distance(query, node.vector)
            candidates.append((dist, node_id))

        best_results: List[Tuple[float, int]] = list(candidates)

        while candidates:
            candidates.sort(key=lambda x: x[0])
            current_dist, current_id = candidates.pop(0)

            # Early termination
            if len(best_results) >= num_closest:
                best_results.sort(key=lambda x: x[0])
                worst_dist = best_results[-1][0]
                if current_dist > worst_dist:
                    break

            # Explore neighbors
            current_node = self.graph.get_node(current_id)
            for neighbor_id in current_node.get_neighbors(layer):
                if neighbor_id in visited:
                    continue

                visited.add(neighbor_id)
                neighbor_node = self.graph.get_node(neighbor_id)

                # Use standard distance (no modulation)
                dist = self._get_distance(query, neighbor_node.vector)

                # Update results
                if len(best_results) < num_closest:
                    best_results.append((dist, neighbor_id))
                    candidates.append((dist, neighbor_id))
                else:
                    best_results.sort(key=lambda x: x[0])
                    if dist < best_results[-1][0]:
                        best_results[-1] = (dist, neighbor_id)
                        candidates.append((dist, neighbor_id))

        best_results.sort(key=lambda x: x[0])
        return [node_id for _, node_id in best_results[:num_closest]]

    def _get_distance(self, vec1: Vector, vec2: Vector) -> float:
        """Compute distance between vectors (pure HNSW, no modulation)."""
        return cosine_distance(vec1, vec2)

    def provide_feedback(
        self,
        query: Vector,
        result_ids: List[DocumentId],
        relevant_ids: Set[DocumentId],
        latency_ms: float,
    ) -> None:
        """Provide feedback for the last query to update ef_search learning.

        Args:
            query: The query vector
            result_ids: Result IDs returned by search
            relevant_ids: Which results were relevant (user feedback)
            latency_ms: Query latency in milliseconds
        """
        if not self.enable_adaptation:
            return

        # Create feedback object
        feedback = self.feedback_collector.add_feedback(
            query_vector=query,
            result_ids=result_ids,
            relevant_ids=relevant_ids,
            timestamp=time.time()
        )

        # Update ef_search learning based on satisfaction and latency
        if self.ef_selector:
            satisfaction = feedback.get_satisfaction_score()
            self.ef_selector.update_from_feedback(
                intent_id=self.last_intent_id,
                ef_used=self.last_ef_used,
                satisfaction=satisfaction,
                latency_ms=latency_ms
            )
            # Decay exploration rate over time (more exploration early, less later)
            self.ef_selector.decay_exploration()

    def _handle_intent_drift(self) -> None:
        """Handle detected intent drift by re-clustering.

        Re-computes intent clusters when query patterns shift significantly.
        """
        if self.intent_detector:
            self.intent_detector.recompute_clusters()

    def record_performance(self, recall: float, precision: float, latency_ms: float) -> None:
        """Record performance metrics for monitoring.

        Args:
            recall: Recall@k for the query
            precision: Precision@k for the query
            latency_ms: Query latency in milliseconds
        """
        if not self.enable_adaptation:
            return

        metrics = PerformanceMetrics(
            recall_at_k=recall, precision_at_k=precision, latency_ms=latency_ms
        )

        self.performance_monitor.record_query_performance(metrics)

        # Set baseline if not set
        if self.performance_monitor.baseline_recall == 0.0:
            avg_recall = self.performance_monitor.get_current_recall()
            if avg_recall > 0.0:
                self.performance_monitor.set_baseline(avg_recall)

        # Check for degradation and trigger reset if needed
        if self.performance_monitor.check_for_degradation():
            if self.performance_monitor.should_trigger_reset(consecutive_checks=3):
                self._trigger_reset()

    def _trigger_reset(self) -> None:
        """Trigger reset of learned ef_search values due to performance degradation."""
        if self.ef_selector:
            self.ef_selector.reset_all()
        self.performance_monitor.reset_degradation_state()

    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive statistics including intent detection.

        Returns:
            Dictionary with all system statistics
        """
        base_stats = {
            "graph": {
                "nodes": self.graph.size(),
                "max_level": self.graph.get_max_level(),
                "entry_point": self.graph.entry_point,
            },
            "query_count": self.query_count,
            "performance": self.performance_monitor.get_performance_summary(),
            "feedback": {
                "total_queries": self.feedback_collector.get_feedback_count(),
                "avg_satisfaction": self.feedback_collector.compute_average_satisfaction(),
            },
        }

        # Add intent detection statistics
        if self.enable_intent_detection and self.intent_detector:
            base_stats["intent_detection"] = self.intent_detector.get_statistics()

        # Add ef_search selection statistics
        if self.enable_adaptation and self.ef_selector:
            base_stats["ef_search_selection"] = self.ef_selector.get_statistics()

        return base_stats
