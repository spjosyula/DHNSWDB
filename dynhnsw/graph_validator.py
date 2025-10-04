"""Graph validation and connectivity checks for adaptive HNSW.

This module ensures that graph modifications (edge removal, weight changes)
do not break the fundamental properties of the HNSW graph, particularly
connectivity and reachability.
"""

from typing import Set, Dict, List, Tuple
from collections import deque


# Type alias for edge identification
EdgeId = Tuple[int, int]


class GraphValidator:
    """Validates graph structure and connectivity properties.

    This class performs checks to ensure the HNSW graph remains valid
    even as edge weights are adapted or edges are added/removed.
    """

    def __init__(self) -> None:
        """Initialize the graph validator."""
        # Graph structure: adjacency list representation
        self.graph: Dict[int, Set[int]] = {}

    def add_edge(self, node_u: int, node_v: int) -> None:
        """Add an undirected edge to the graph.

        Args:
            node_u: First node ID
            node_v: Second node ID
        """
        if node_u not in self.graph:
            self.graph[node_u] = set()
        if node_v not in self.graph:
            self.graph[node_v] = set()

        self.graph[node_u].add(node_v)
        self.graph[node_v].add(node_u)

    def remove_edge(self, node_u: int, node_v: int) -> None:
        """Remove an undirected edge from the graph.

        Args:
            node_u: First node ID
            node_v: Second node ID
        """
        if node_u in self.graph:
            self.graph[node_u].discard(node_v)
        if node_v in self.graph:
            self.graph[node_v].discard(node_u)

    def is_connected(self, node_u: int, node_v: int) -> bool:
        """Check if two nodes are connected via any path.

        Uses BFS to determine reachability.

        Args:
            node_u: Start node
            node_v: Target node

        Returns:
            True if there exists a path from node_u to node_v
        """
        if node_u not in self.graph or node_v not in self.graph:
            return False

        if node_u == node_v:
            return True

        visited: Set[int] = set()
        queue: deque = deque([node_u])
        visited.add(node_u)

        while queue:
            current = queue.popleft()

            if current == node_v:
                return True

            for neighbor in self.graph.get(current, set()):
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)

        return False

    def would_disconnect_graph(self, node_u: int, node_v: int) -> bool:
        """Check if removing an edge would disconnect the graph.

        This is a critical safety check before edge removal.

        Args:
            node_u: First node of edge to test
            node_v: Second node of edge to test

        Returns:
            True if removing this edge would break connectivity
        """
        # Temporarily remove the edge
        self.remove_edge(node_u, node_v)

        # Check if nodes are still connected via alternative path
        still_connected = self.is_connected(node_u, node_v)

        # Restore the edge
        self.add_edge(node_u, node_v)

        return not still_connected

    def get_node_degree(self, node_id: int) -> int:
        """Get the degree (number of neighbors) of a node.

        Args:
            node_id: Node to query

        Returns:
            Number of edges connected to this node
        """
        return len(self.graph.get(node_id, set()))

    def get_all_nodes(self) -> Set[int]:
        """Get all node IDs in the graph.

        Returns:
            Set of all node IDs
        """
        return set(self.graph.keys())

    def get_neighbors(self, node_id: int) -> Set[int]:
        """Get all neighbors of a node.

        Args:
            node_id: Node to query

        Returns:
            Set of neighbor node IDs
        """
        return self.graph.get(node_id, set()).copy()

    def compute_average_path_length(self, sample_size: int = 100) -> float:
        """Compute average shortest path length between random node pairs.

        This is a graph health metric - shorter average path length indicates
        better small-world properties.

        Args:
            sample_size: Number of random node pairs to sample

        Returns:
            Average path length, or -1.0 if graph is empty
        """
        nodes = list(self.graph.keys())
        if len(nodes) < 2:
            return -1.0

        import random

        total_length = 0
        count = 0

        for _ in range(sample_size):
            node_u = random.choice(nodes)
            node_v = random.choice(nodes)

            if node_u == node_v:
                continue

            path_length = self._bfs_path_length(node_u, node_v)
            if path_length > 0:  # Valid path found
                total_length += path_length
                count += 1

        return total_length / count if count > 0 else -1.0

    def _bfs_path_length(self, start: int, target: int) -> int:
        """Compute shortest path length between two nodes using BFS.

        Args:
            start: Start node
            target: Target node

        Returns:
            Path length, or -1 if no path exists
        """
        if start == target:
            return 0

        visited: Set[int] = set()
        queue: deque = deque([(start, 0)])  # (node, distance)
        visited.add(start)

        while queue:
            current, distance = queue.popleft()

            if current == target:
                return distance

            for neighbor in self.graph.get(current, set()):
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, distance + 1))

        return -1  # No path exists

    def get_graph_statistics(self) -> Dict[str, float]:
        """Compute overall graph statistics.

        Returns:
            Dictionary with graph metrics (node_count, avg_degree, etc.)
        """
        if not self.graph:
            return {
                "node_count": 0,
                "edge_count": 0,
                "avg_degree": 0.0,
                "min_degree": 0,
                "max_degree": 0,
            }

        degrees = [self.get_node_degree(node) for node in self.graph.keys()]
        total_edges = sum(degrees) // 2  # Each edge counted twice

        return {
            "node_count": len(self.graph),
            "edge_count": total_edges,
            "avg_degree": sum(degrees) / len(degrees),
            "min_degree": min(degrees),
            "max_degree": max(degrees),
        }
