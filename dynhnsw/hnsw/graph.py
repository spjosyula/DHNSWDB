"""
HNSW graph data structures.

This module defines the core data structures for storing the HNSW graph:
- HNSWNode: Represents a single node (vector) in the graph with its connections
- HNSWGraph: Container for the entire graph structure

The graph is hierarchical: nodes at layer 0 form a dense graph with all vectors,
while higher layers contain progressively fewer nodes for faster coarse-grained search.
Each node stores connections (neighbors) at each layer it participates in.
"""

from typing import Dict, List, Optional
import numpy as np
import numpy.typing as npt

Vector = npt.NDArray[np.float32]


class HNSWNode:
    """
    Represents a single node in the HNSW graph.

    Each node contains a vector and its connections to other nodes across multiple layers.
    The node appears in layers 0 through its assigned 'level' (higher levels are sparser).
    """

    def __init__(self, node_id: int, vector: Vector, level: int) -> None:
        """
        Create a new HNSW node.

        Args:
            node_id: Unique identifier for this node
            vector: The vector data (1D numpy array)
            level: Maximum layer this node appears in (0 = base layer only)
        """
        self.id = node_id
        self.vector = vector
        self.level = level

        # Neighbors organized by layer: {layer_num: [neighbor_id1, neighbor_id2, ...]}
        # Initially empty - connections are added during graph construction
        self.neighbors: Dict[int, List[int]] = {layer: [] for layer in range(level + 1)}

    def add_neighbor(self, neighbor_id: int, layer: int) -> None:
        """
        Add a connection to another node at a specific layer.

        Args:
            neighbor_id: ID of the neighbor node to connect to
            layer: Which layer to add the connection at
        """
        if layer > self.level:
            raise ValueError(
                f"Cannot add neighbor at layer {layer} (node max level is {self.level})"
            )

        if neighbor_id not in self.neighbors[layer]:
            self.neighbors[layer].append(neighbor_id)

    def get_neighbors(self, layer: int) -> List[int]:
        """
        Get all neighbors at a specific layer.

        Args:
            layer: Which layer to query

        Returns:
            List of neighbor node IDs at that layer
        """
        if layer > self.level:
            return []

        return self.neighbors[layer]

    def __repr__(self) -> str:
        """String representation for debugging."""
        return f"HNSWNode(id={self.id}, level={self.level}, dim={len(self.vector)})"


class HNSWGraph:
    """
    Container for the entire HNSW graph structure.

    Manages all nodes, tracks the entry point for searches, and maintains
    graph parameters like maximum connections per layer.
    """

    def __init__(
        self,
        dimension: int,
        M: int = 16,
        M_L: Optional[int] = None,
        level_multiplier: Optional[float] = None,
    ) -> None:
        """
        Initialize an empty HNSW graph.

        Args:
            dimension: Dimensionality of vectors to store
            M: Maximum number of neighbors per node at layers > 0 (typical: 16-64)
            M_L: Maximum neighbors at layer 0 (default: 2*M for denser base layer)
            level_multiplier: Controls layer distribution (default: 1/ln(M) per HNSW paper)
        """
        self.dimension = dimension
        self.M = M
        self.M_L = M_L if M_L is not None else 2 * M  # Layer 0 has more connections

        # Calculate level_multiplier from M per HNSW paper (Malkov & Yashunin 2016)
        # Formula: mL = 1/ln(M) ensures exponential decay P(layer >= l) = (1/M)^l
        if level_multiplier is None:
            self.level_multiplier = 1.0 / np.log(M)
        else:
            self.level_multiplier = level_multiplier

        # Storage for all nodes
        self.nodes: Dict[int, HNSWNode] = {}

        # Entry point: the node at the highest layer where searches begin
        # None when graph is empty
        self.entry_point: Optional[int] = None

        # Counter for generating unique node IDs
        self._next_id = 0

    def add_node(self, vector: Vector, level: int, node_id: Optional[int] = None) -> int:
        """
        Add a new node to the graph structure (without connecting it yet).

        Args:
            vector: Vector data for the node
            level: Maximum layer this node should appear in
            node_id: Optional explicit node ID to use (if None, auto-generates)

        Returns:
            The ID assigned to the new node
        """
        # Validate vector dimension
        if len(vector) != self.dimension:
            raise ValueError(
                f"Vector dimension {len(vector)} doesn't match graph dimension {self.dimension}"
            )

        # Use provided node_id or generate new one
        if node_id is None:
            node_id = self._next_id
            self._next_id += 1
        else:
            # Update counter if provided ID is higher than current
            if node_id >= self._next_id:
                self._next_id = node_id + 1

        node = HNSWNode(node_id, vector, level)
        self.nodes[node_id] = node

        # Update entry point if this is the first node or if it's at a higher layer
        if self.entry_point is None or level > self.nodes[self.entry_point].level:
            self.entry_point = node_id

        return node_id

    def get_node(self, node_id: int) -> Optional[HNSWNode]:
        """
        Retrieve a node by its ID.

        Args:
            node_id: ID of the node to retrieve

        Returns:
            The HNSWNode, or None if not found
        """
        return self.nodes.get(node_id)

    def add_edge(self, node1_id: int, node2_id: int, layer: int) -> None:
        """
        Create a bidirectional connection between two nodes at a specific layer.

        Args:
            node1_id: First node ID
            node2_id: Second node ID
            layer: Layer at which to create the connection
        """
        node1 = self.nodes.get(node1_id)
        node2 = self.nodes.get(node2_id)

        if node1 is None or node2 is None:
            raise ValueError(f"Node not found: {node1_id} or {node2_id}")

        # Add bidirectional edges
        node1.add_neighbor(node2_id, layer)
        node2.add_neighbor(node1_id, layer)

    def get_max_level(self) -> int:
        """
        Get the maximum layer level in the graph (level of entry point).

        Returns:
            Maximum layer number, or -1 if graph is empty
        """
        if self.entry_point is None:
            return -1

        return self.nodes[self.entry_point].level

    def size(self) -> int:
        """
        Get the total number of nodes in the graph.

        Returns:
            Number of nodes
        """
        return len(self.nodes)

    def __repr__(self) -> str:
        """String representation for debugging."""
        return (
            f"HNSWGraph(nodes={self.size()}, max_level={self.get_max_level()}, "
            f"M={self.M}, dim={self.dimension})"
        )
