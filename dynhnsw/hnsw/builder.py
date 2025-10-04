"""
HNSW graph construction and insertion logic.

This module handles adding new nodes to the HNSW graph. The insertion algorithm:
1. Assigns a random layer to the new node (using geometric distribution)
2. Finds nearest neighbors by greedy search from the entry point
3. Connects the new node to its neighbors at each layer
4. Prunes connections to maintain the M constraint (max edges per node)
5. Updates existing nodes' connections bidirectionally

The key insight: start search at the top (sparse) layer and progressively
zoom in through denser layers until reaching the target layer.
"""

from typing import List, Set, Tuple
import numpy as np
import numpy.typing as npt

from dynhnsw.hnsw.graph import HNSWGraph
from dynhnsw.hnsw.distance import cosine_distance
from dynhnsw.hnsw.utils import assign_layer, select_neighbors_simple

Vector = npt.NDArray[np.float32]


class HNSWBuilder:
    """
    Handles insertion of nodes into the HNSW graph.

    This class encapsulates the logic for adding new vectors to the index,
    including neighbor search, connection creation, and pruning.
    """

    def __init__(self, graph: HNSWGraph) -> None:
        """
        Initialize builder with a graph to operate on.

        Args:
            graph: The HNSWGraph to insert nodes into
        """
        self.graph = graph

    def insert(self, vector: Vector, node_id: int, level: int) -> None:
        """
        Insert a new node into the graph at a specific level.

        This is the main insertion algorithm from the HNSW paper.

        Args:
            vector: Vector data for the new node
            node_id: ID to assign to the new node
            level: Maximum layer for this node
        """
        # Add the node to the graph structure (not connected yet)
        actual_id = self.graph.add_node(vector, level)
        assert actual_id == node_id, f"Expected ID {node_id}, got {actual_id}"

        # Special case: first node in the graph
        if self.graph.size() == 1:
            return  # No neighbors to connect to

        # Find entry points for search
        # Start from the current entry point at the top layer
        entry_point = self.graph.entry_point
        if entry_point == node_id:
            # This node became the new entry point (higher layer than all others)
            # Find the previous entry point (second highest)
            entry_point = self._find_previous_entry_point(node_id)
            if entry_point is None:
                return  # Only one node exists

        # Search from top layer down to level+1, keeping only 1 closest candidate
        current_nearest = [entry_point]
        entry_node = self.graph.get_node(entry_point)

        for layer in range(entry_node.level, level, -1):
            # Greedy search at this layer to find closest node
            current_nearest = self._search_layer(
                query=vector, entry_points=current_nearest, num_closest=1, layer=layer
            )

        # Insert at layers from level down to 0
        for layer in range(level, -1, -1):
            # Search for ef_construction nearest neighbors at this layer
            candidates = self._search_layer(
                query=vector,
                entry_points=current_nearest,
                num_closest=self.graph.M if layer > 0 else self.graph.M_L,
                layer=layer,
            )

            # Select M (or M_L for layer 0) neighbors from candidates
            M = self.graph.M if layer > 0 else self.graph.M_L
            neighbors = self._select_neighbors(candidates, vector, M)

            # Filter neighbors to only those that exist at this layer
            # (a node at level L exists in layers 0 through L)
            valid_neighbors = []
            for neighbor_id in neighbors:
                neighbor_node = self.graph.get_node(neighbor_id)
                if neighbor_node.level >= layer:
                    valid_neighbors.append(neighbor_id)

            # Connect the new node to selected neighbors
            for neighbor_id in valid_neighbors:
                self.graph.add_edge(node_id, neighbor_id, layer)

            # Prune neighbors of the connected nodes if they exceed M
            for neighbor_id in valid_neighbors:
                self._prune_neighbors(neighbor_id, layer)

    def _search_layer(
        self,
        query: Vector,
        entry_points: List[int],
        num_closest: int,
        layer: int,
    ) -> List[int]:
        """
        Greedy search for nearest neighbors at a single layer.

        This is a core subroutine used during both insertion and search.
        It performs a greedy best-first search on the graph at a specific layer.

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

    def _select_neighbors(
        self, candidates: List[int], query: Vector, M: int
    ) -> List[int]:
        """
        Select M neighbors from candidates for connection.

        Uses simple strategy: pick M nearest neighbors.
        Can be enhanced with heuristic selection later.

        Args:
            candidates: List of candidate node IDs
            query: Query vector (for distance calculation)
            M: Number of neighbors to select

        Returns:
            List of selected neighbor IDs (up to M)
        """
        # Calculate distances to all candidates
        distances = []
        for node_id in candidates:
            node = self.graph.get_node(node_id)
            dist = cosine_distance(query, node.vector)
            distances.append(dist)

        # Use simple neighbor selection
        return select_neighbors_simple(candidates, distances, M)

    def _prune_neighbors(self, node_id: int, layer: int) -> None:
        """
        Prune connections of a node if it exceeds M constraint.

        When a new edge is added to a node, it might exceed the maximum
        allowed connections (M). We keep only the M nearest neighbors.

        Args:
            node_id: Node to prune
            layer: Which layer to prune at
        """
        node = self.graph.get_node(node_id)
        neighbors = node.get_neighbors(layer)

        M = self.graph.M if layer > 0 else self.graph.M_L

        # If within limit, no pruning needed
        if len(neighbors) <= M:
            return

        # Calculate distances to all neighbors
        distances = []
        for neighbor_id in neighbors:
            neighbor_node = self.graph.get_node(neighbor_id)
            dist = cosine_distance(node.vector, neighbor_node.vector)
            distances.append(dist)

        # Select M nearest neighbors
        selected = select_neighbors_simple(neighbors, distances, M)

        # Remove connections to pruned neighbors
        pruned = set(neighbors) - set(selected)
        for pruned_id in pruned:
            # Remove from this node's neighbor list
            node.neighbors[layer].remove(pruned_id)

            # Remove reverse connection from pruned neighbor
            pruned_node = self.graph.get_node(pruned_id)
            if node_id in pruned_node.neighbors[layer]:
                pruned_node.neighbors[layer].remove(node_id)

    def _find_previous_entry_point(self, exclude_id: int) -> int | None:
        """
        Find the entry point excluding a specific node.

        Used when a new node becomes the entry point to find the previous one.

        Args:
            exclude_id: Node ID to exclude from search

        Returns:
            ID of the highest-level node (excluding exclude_id), or None if none exists
        """
        max_level = -1
        entry_id = None

        for node_id, node in self.graph.nodes.items():
            if node_id == exclude_id:
                continue
            if node.level > max_level:
                max_level = node.level
                entry_id = node_id

        return entry_id
