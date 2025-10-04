"""
HNSW search algorithm.

This module handles querying the HNSW graph to find approximate nearest neighbors.
The search algorithm:
1. Starts at the entry point (top layer)
2. Greedily navigates down through layers to get closer to the query
3. At layer 0, expands the search using ef_search parameter
4. Returns the k nearest neighbors

The ef_search parameter controls the accuracy-speed tradeoff:
- Higher ef_search = better recall, slower search
- Lower ef_search = faster search, lower recall
"""

from typing import List, Set, Tuple
import numpy as np
import numpy.typing as npt

from dynhnsw.hnsw.graph import HNSWGraph
from dynhnsw.hnsw.distance import cosine_distance

Vector = npt.NDArray[np.float32]


class HNSWSearcher:
    """
    Handles search queries on the HNSW graph.

    This class provides the search functionality to find k nearest neighbors
    for a given query vector.
    """

    def __init__(self, graph: HNSWGraph, ef_search: int = 50) -> None:
        """
        Initialize searcher with a graph.

        Args:
            graph: The HNSWGraph to search in
            ef_search: Size of candidate list during search (higher = better recall)
        """
        self.graph = graph
        self.ef_search = ef_search

    def search(self, query: Vector, k: int, ef_search: int | None = None) -> List[Tuple[int, float]]:
        """
        Search for k nearest neighbors to the query vector.

        Args:
            query: Query vector to search for
            k: Number of nearest neighbors to return
            ef_search: Override default ef_search for this query

        Returns:
            List of (node_id, distance) tuples, sorted by distance (closest first)
        """
        # Handle empty graph
        if self.graph.size() == 0:
            return []

        # Use provided ef_search or default
        ef = ef_search if ef_search is not None else self.ef_search

        # Ensure ef is at least k
        ef = max(ef, k)

        # Start from entry point
        entry_point = self.graph.entry_point
        entry_node = self.graph.get_node(entry_point)

        # Search from top layer down to layer 1, keeping only 1 closest
        current_nearest = [entry_point]
        for layer in range(entry_node.level, 0, -1):
            current_nearest = self._search_layer(
                query=query, entry_points=current_nearest, num_closest=1, layer=layer
            )

        # At layer 0, expand search with ef_search
        candidates = self._search_layer(
            query=query, entry_points=current_nearest, num_closest=ef, layer=0
        )

        # Calculate distances and return top k
        results = []
        for node_id in candidates:
            node = self.graph.get_node(node_id)
            dist = cosine_distance(query, node.vector)
            results.append((node_id, dist))

        # Sort by distance and return top k
        results.sort(key=lambda x: x[1])
        return results[:k]

    def _search_layer(
        self,
        query: Vector,
        entry_points: List[int],
        num_closest: int,
        layer: int,
    ) -> List[int]:
        """
        Greedy search for nearest neighbors at a single layer.

        This is the core search subroutine used at each layer of the HNSW graph.

        Args:
            query: Query vector to search for
            entry_points: Starting node IDs for the search
            num_closest: Number of closest nodes to return
            layer: Which layer to search on

        Returns:
            List of node IDs (up to num_closest), sorted by distance to query
        """
        # Track visited nodes to avoid cycles
        visited: Set[int] = set(entry_points)

        # Candidates to explore (start with entry points)
        # Store as (distance, node_id) for easy sorting
        candidates: List[Tuple[float, int]] = []
        for node_id in entry_points:
            node = self.graph.get_node(node_id)
            dist = cosine_distance(query, node.vector)
            candidates.append((dist, node_id))

        # Best results found so far
        best_results: List[Tuple[float, int]] = list(candidates)

        while candidates:
            # Get closest candidate
            candidates.sort(key=lambda x: x[0])
            current_dist, current_id = candidates.pop(0)

            # If this candidate is farther than our worst result, stop
            if len(best_results) >= num_closest:
                best_results.sort(key=lambda x: x[0])
                worst_dist = best_results[-1][0]
                if current_dist > worst_dist:
                    break

            # Explore neighbors of current node
            current_node = self.graph.get_node(current_id)
            for neighbor_id in current_node.get_neighbors(layer):
                if neighbor_id in visited:
                    continue

                visited.add(neighbor_id)
                neighbor_node = self.graph.get_node(neighbor_id)
                dist = cosine_distance(query, neighbor_node.vector)

                # Add to results if better than worst result or if we need more results
                if len(best_results) < num_closest:
                    best_results.append((dist, neighbor_id))
                    candidates.append((dist, neighbor_id))
                else:
                    best_results.sort(key=lambda x: x[0])
                    if dist < best_results[-1][0]:
                        best_results[-1] = (dist, neighbor_id)
                        candidates.append((dist, neighbor_id))

        # Return only node IDs, sorted by distance
        best_results.sort(key=lambda x: x[0])
        return [node_id for _, node_id in best_results[:num_closest]]
