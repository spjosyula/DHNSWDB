"""Intent-aware adaptive HNSW searcher.

This module integrates intent detection with adaptive entry point selection.
Different query intents learn optimal entry points for improved search quality.

Mathematical Foundation:
- Detect query intent using K-means clustering
- Learn optimal entry points per intent from feedback
- Search uses pure HNSW algorithm, but starts from intent-specific entry
- Avoids semantic mismatch of edge weight modulation
"""

import time
from typing import List, Tuple, Set, Optional, Dict, Any
import numpy as np
import numpy.typing as npt

from dynhnsw.hnsw.graph import HNSWGraph
from dynhnsw.hnsw.distance import cosine_distance
from dynhnsw.intent_detector import IntentDetector
from dynhnsw.entry_point_selector import EntryPointSelector
from dynhnsw.feedback import FeedbackCollector, QueryFeedback
from dynhnsw.performance_monitor import PerformanceMonitor, PerformanceMetrics
from dynhnsw.graph_validator import GraphValidator

Vector = npt.NDArray[np.float32]
DocumentId = int
EdgeId = Tuple[int, int]


class IntentAwareHNSWSearcher:
    """Intent-aware HNSW searcher with per-intent entry point selection.

    This searcher detects query intent and selects optimal entry points for
    each intent based on feedback. Uses pure HNSW search algorithm but adapts
    where search starts based on query patterns.

    Key difference from edge-weight approach:
    - Semantically correct: learns entry points (graph navigation)
    - Not edge weights (which don't apply to query-to-node distances)
    """

    def __init__(
        self,
        graph: HNSWGraph,
        ef_search: int = 50,
        k_intents: int = 5,
        learning_rate: float = 0.1,
        enable_adaptation: bool = True,
        enable_intent_detection: bool = True,
        confidence_threshold: float = 0.7,
        min_queries_for_clustering: int = 50,
    ) -> None:
        """Initialize intent-aware HNSW searcher.

        Args:
            graph: HNSW graph structure
            ef_search: Default search candidate list size
            k_intents: Number of intent clusters
            learning_rate: Entry point score learning rate
            enable_adaptation: If False, behaves like static HNSW
            enable_intent_detection: If False, uses default entry point
            confidence_threshold: Minimum confidence for intent-specific entry
            min_queries_for_clustering: Queries needed before clustering starts
        """
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

        # Entry point selection (replaces edge weight learning)
        self.entry_selector = EntryPointSelector(
            k_intents=k_intents,
            graph=graph,
            learning_rate=learning_rate
        ) if enable_adaptation else None

        # Feedback and monitoring
        self.feedback_collector = FeedbackCollector(buffer_size=1000)
        self.performance_monitor = PerformanceMonitor(
            baseline_recall=0.0,
            degradation_threshold=0.95,
            window_size=50
        )

        # Graph validation (keep for potential future use)
        self.graph_validator = GraphValidator()
        self._initialize_validator()

        # Track current intent and entry used
        self.last_intent_id: int = -1
        self.last_confidence: float = 0.0
        self.last_entry_used: int = graph.entry_point
        self.query_count: int = 0

    def _initialize_validator(self) -> None:
        """Initialize graph validator with current graph structure."""
        for node_id, node in self.graph.nodes.items():
            for neighbor_id in node.get_neighbors(layer=0):
                if node_id < neighbor_id:
                    self.graph_validator.add_edge(node_id, neighbor_id)

    def search(
        self, query: Vector, k: int, ef_search: Optional[int] = None
    ) -> List[Tuple[DocumentId, float]]:
        """Search for k nearest neighbors with intent-aware entry point selection.

        Flow:
        1. Detect query intent (if enabled)
        2. Select optimal entry point for that intent (if adaptation enabled)
        3. Perform standard HNSW search from selected entry point
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

        # Step 2: Select entry point based on intent
        if self.enable_adaptation and self.entry_selector:
            self.last_entry_used = self.entry_selector.select_entry(
                intent_id=self.last_intent_id,
                confidence=self.last_confidence,
                confidence_threshold=self.confidence_threshold
            )
        else:
            self.last_entry_used = self.graph.entry_point

        # Step 3: Standard HNSW search from selected entry
        ef = ef_search if ef_search is not None else self.ef_search
        ef = max(ef, k)

        entry_node = self.graph.get_node(self.last_entry_used)

        # Search from top layer down to layer 1
        current_nearest = [self.last_entry_used]
        for layer in range(entry_node.level, 0, -1):
            current_nearest = self._search_layer(
                query=query, entry_points=current_nearest, num_closest=1, layer=layer
            )

        # At layer 0, expand search with ef_search
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
        self, query: Vector, result_ids: List[DocumentId], relevant_ids: Set[DocumentId]
    ) -> None:
        """Provide feedback for the last query to update entry point scores.

        Args:
            query: The query vector
            result_ids: Result IDs returned by search
            relevant_ids: Which results were relevant (user feedback)
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

        # Update entry point scores based on satisfaction
        if self.entry_selector:
            satisfaction = feedback.get_satisfaction_score()
            self.entry_selector.update_from_feedback(
                intent_id=self.last_intent_id,
                entry_used=self.last_entry_used,
                satisfaction=satisfaction
            )

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
        """Trigger reset of entry point scores due to performance degradation."""
        if self.entry_selector:
            self.entry_selector.reset_scores()
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

        # Add entry point selection statistics
        if self.enable_adaptation and self.entry_selector:
            base_stats["entry_selection"] = self.entry_selector.get_statistics()

        return base_stats
