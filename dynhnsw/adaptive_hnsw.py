"""Adaptive HNSW search with learnable edge weights.

This module integrates edge weight learning, stability monitoring, and performance
tracking with the core HNSW search algorithm. The system adapts based on user
feedback to improve search quality over time.
"""

import time
from typing import List, Tuple, Set, Optional
import numpy as np
import numpy.typing as npt

from dynhnsw.hnsw.graph import HNSWGraph, HNSWNode
from dynhnsw.hnsw.distance import cosine_distance
from dynhnsw.adaptive_weights import EdgeWeightLearner
from dynhnsw.feedback import FeedbackCollector, QueryFeedback
from dynhnsw.stability_monitor import StabilityMonitor
from dynhnsw.performance_monitor import PerformanceMonitor, PerformanceMetrics, AdaptiveResetManager
from dynhnsw.graph_validator import GraphValidator

Vector = npt.NDArray[np.float32]
DocumentId = int
EdgeId = Tuple[int, int]


class AdaptiveHNSWSearcher:
    """HNSW searcher with adaptive edge weights based on user feedback.

    This class extends the standard HNSW search to incorporate:
    - Edge weight learning from feedback
    - Stability monitoring for oscillation detection
    - Performance tracking and degradation recovery
    - Graph connectivity validation
    """

    def __init__(
        self,
        graph: HNSWGraph,
        ef_search: int = 50,
        learning_rate: float = 0.05,
        enable_adaptation: bool = True,
    ) -> None:
        """Initialize adaptive HNSW searcher.

        Args:
            graph: HNSW graph structure
            ef_search: Default search candidate list size
            learning_rate: Edge weight learning rate
            enable_adaptation: If False, behaves like static HNSW
        """
        self.graph = graph
        self.ef_search = ef_search
        self.enable_adaptation = enable_adaptation

        # Stage 1: Edge weight learning
        self.weight_learner = EdgeWeightLearner(learning_rate=learning_rate)
        self.feedback_collector = FeedbackCollector(buffer_size=1000)

        # Stage 2: Stability and performance monitoring
        self.stability_monitor = StabilityMonitor(
            oscillation_threshold=0.25, history_window=10
        )
        self.performance_monitor = PerformanceMonitor(
            baseline_recall=0.0,  # Will be set after first queries
            degradation_threshold=0.95,  # 5% degradation tolerance
            window_size=50,
        )
        self.reset_manager = AdaptiveResetManager(reset_steps=100, reset_rate=0.1)

        # Graph validation for safety
        self.graph_validator = GraphValidator()
        self._initialize_validator()

        # Track traversed edges for feedback
        self.last_traversed_edges: List[EdgeId] = []

    def _initialize_validator(self) -> None:
        """Initialize graph validator with current graph structure."""
        # Build validator graph from HNSW graph (layer 0 only for connectivity)
        for node_id, node in self.graph.nodes.items():
            for neighbor_id in node.get_neighbors(layer=0):
                # Only add each edge once (avoid duplicates)
                if node_id < neighbor_id:
                    self.graph_validator.add_edge(node_id, neighbor_id)

    def search(
        self, query: Vector, k: int, ef_search: Optional[int] = None
    ) -> List[Tuple[DocumentId, float]]:
        """Search for k nearest neighbors with adaptive edge weights.

        Args:
            query: Query vector
            k: Number of results to return
            ef_search: Override default ef_search for this query

        Returns:
            List of (node_id, distance) tuples sorted by distance
        """
        if self.graph.size() == 0:
            return []

        # Clear edge tracking for this query
        self.last_traversed_edges = []

        # Use provided ef_search or default
        ef = ef_search if ef_search is not None else self.ef_search
        ef = max(ef, k)

        # Start from entry point
        entry_point = self.graph.entry_point
        entry_node = self.graph.get_node(entry_point)

        # Search from top layer down to layer 1
        current_nearest = [entry_point]
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
        return results[:k]

    def _search_layer(
        self, query: Vector, entry_points: List[int], num_closest: int, layer: int
    ) -> List[int]:
        """Greedy search at a single layer with adaptive edge weights.

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

            # Early termination if current is worse than worst result
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

                # Track edge traversal (only at layer 0)
                if layer == 0:
                    edge = (min(current_id, neighbor_id), max(current_id, neighbor_id))
                    self.last_traversed_edges.append(edge)

                neighbor_node = self.graph.get_node(neighbor_id)

                # Use adaptive distance if enabled
                dist = self._get_adaptive_distance(
                    current_id, neighbor_id, query, neighbor_node.vector
                )

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
        """Compute base distance between vectors.

        Args:
            vec1: First vector
            vec2: Second vector

        Returns:
            Cosine distance
        """
        return cosine_distance(vec1, vec2)

    def _get_adaptive_distance(
        self, node_u: int, node_v: int, vec1: Vector, vec2: Vector
    ) -> float:
        """Compute distance with adaptive edge weights.

        Args:
            node_u: Source node ID
            node_v: Destination node ID
            vec1: First vector
            vec2: Second vector

        Returns:
            Weight-modified distance
        """
        base_distance = self._get_distance(vec1, vec2)

        if not self.enable_adaptation:
            return base_distance

        # Apply reset if active
        if self.reset_manager.is_resetting():
            edge_weight = self.weight_learner.get_edge_weight(node_u, node_v)
            edge_weight.weight = self.reset_manager.apply_reset_step(edge_weight.weight)

        # Get adaptive distance
        return self.weight_learner.get_effective_distance(node_u, node_v, base_distance)

    def provide_feedback(
        self, query: Vector, result_ids: List[DocumentId], relevant_ids: Set[DocumentId]
    ) -> None:
        """Provide feedback for the last query to update edge weights.

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
            timestamp=time.time(),
        )

        # Update edge weights based on feedback
        self.weight_learner.update_from_feedback(feedback, self.last_traversed_edges)

        # Monitor stability
        for edge_id in self.last_traversed_edges:
            edge_weight = self.weight_learner.get_edge_weight(edge_id[0], edge_id[1])
            self.stability_monitor.record_weight_update(
                edge_id[0], edge_id[1], edge_weight.weight
            )

        # Advance reset if active
        if self.reset_manager.is_resetting():
            self.reset_manager.advance_step()

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
        """Trigger gradual reset of edge weights due to performance degradation."""
        if not self.reset_manager.is_resetting():
            self.reset_manager.start_reset()
            self.performance_monitor.reset_degradation_state()

    def get_statistics(self) -> dict:
        """Get comprehensive statistics about the adaptive system.

        Returns:
            Dictionary with all system statistics
        """
        return {
            "graph": {
                "nodes": self.graph.size(),
                "max_level": self.graph.get_max_level(),
            },
            "weights": self.weight_learner.get_weight_statistics(),
            "stability": self.stability_monitor.get_statistics(),
            "performance": self.performance_monitor.get_performance_summary(),
            "reset": {
                "active": self.reset_manager.is_resetting(),
                "progress": self.reset_manager.get_reset_progress(),
            },
            "feedback": {
                "total_queries": self.feedback_collector.get_feedback_count(),
                "avg_satisfaction": self.feedback_collector.compute_average_satisfaction(),
            },
        }
