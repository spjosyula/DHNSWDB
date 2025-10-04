"""
Tests for HNSW graph data structures.

These tests verify the graph container and node structures work correctly:
- Node creation and neighbor management
- Graph initialization and node addition
- Edge creation and bidirectional connections
- Entry point tracking
"""

import numpy as np
import pytest
from dynhnsw.hnsw.graph import HNSWNode, HNSWGraph


def test_create_node():
    """Create a basic HNSW node"""
    vector = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    node = HNSWNode(node_id=42, vector=vector, level=2)

    assert node.id == 42
    assert np.allclose(node.vector, vector)
    assert node.level == 2
    # Should have empty neighbor lists for layers 0, 1, 2
    assert node.neighbors == {0: [], 1: [], 2: []}


def test_add_neighbor_to_node():
    """Add neighbors to a node at different layers"""
    vector = np.array([1.0, 2.0], dtype=np.float32)
    node = HNSWNode(node_id=1, vector=vector, level=2)

    # Add neighbors at different layers
    node.add_neighbor(neighbor_id=10, layer=0)
    node.add_neighbor(neighbor_id=20, layer=0)
    node.add_neighbor(neighbor_id=30, layer=1)

    assert node.get_neighbors(layer=0) == [10, 20]
    assert node.get_neighbors(layer=1) == [30]
    assert node.get_neighbors(layer=2) == []


def test_node_prevents_duplicate_neighbors():
    """Adding the same neighbor twice should not create duplicates"""
    vector = np.array([1.0], dtype=np.float32)
    node = HNSWNode(node_id=1, vector=vector, level=1)

    node.add_neighbor(neighbor_id=10, layer=0)
    node.add_neighbor(neighbor_id=10, layer=0)  # Duplicate

    assert node.get_neighbors(layer=0) == [10], "Should not have duplicate neighbors"


def test_node_invalid_layer():
    """Adding neighbor at layer higher than node level should fail"""
    vector = np.array([1.0], dtype=np.float32)
    node = HNSWNode(node_id=1, vector=vector, level=1)

    with pytest.raises(ValueError):
        node.add_neighbor(neighbor_id=10, layer=5)  # Layer 5 > node level 1


def test_get_neighbors_above_level():
    """Getting neighbors at layer > node level should return empty list"""
    vector = np.array([1.0], dtype=np.float32)
    node = HNSWNode(node_id=1, vector=vector, level=1)

    neighbors = node.get_neighbors(layer=5)
    assert neighbors == []


def test_create_empty_graph():
    """Initialize an empty HNSW graph"""
    graph = HNSWGraph(dimension=128, M=16)

    assert graph.dimension == 128
    assert graph.M == 16
    assert graph.M_L == 32  # Default is 2*M
    assert graph.size() == 0
    assert graph.entry_point is None
    assert graph.get_max_level() == -1


def test_add_node_to_graph():
    """Add a node to the graph"""
    graph = HNSWGraph(dimension=3, M=16)
    vector = np.array([1.0, 2.0, 3.0], dtype=np.float32)

    node_id = graph.add_node(vector, level=2)

    assert node_id == 0  # First node gets ID 0
    assert graph.size() == 1
    assert graph.entry_point == 0
    assert graph.get_max_level() == 2

    # Retrieve the node
    node = graph.get_node(node_id)
    assert node is not None
    assert node.id == 0
    assert node.level == 2


def test_add_multiple_nodes():
    """Add multiple nodes with auto-incrementing IDs"""
    graph = HNSWGraph(dimension=2, M=16)

    v1 = np.array([1.0, 0.0], dtype=np.float32)
    v2 = np.array([0.0, 1.0], dtype=np.float32)
    v3 = np.array([1.0, 1.0], dtype=np.float32)

    id1 = graph.add_node(v1, level=0)
    id2 = graph.add_node(v2, level=1)
    id3 = graph.add_node(v3, level=0)

    assert id1 == 0
    assert id2 == 1
    assert id3 == 2
    assert graph.size() == 3


def test_entry_point_updates():
    """Entry point should be the node at the highest level"""
    graph = HNSWGraph(dimension=2, M=16)

    v1 = np.array([1.0, 0.0], dtype=np.float32)
    v2 = np.array([0.0, 1.0], dtype=np.float32)
    v3 = np.array([1.0, 1.0], dtype=np.float32)

    # Add nodes with increasing levels
    id1 = graph.add_node(v1, level=0)
    assert graph.entry_point == id1
    assert graph.get_max_level() == 0

    id2 = graph.add_node(v2, level=2)
    assert graph.entry_point == id2  # Higher level, becomes entry point
    assert graph.get_max_level() == 2

    id3 = graph.add_node(v3, level=1)
    assert graph.entry_point == id2  # Still id2 (level 2 is highest)
    assert graph.get_max_level() == 2


def test_add_edge_bidirectional():
    """Adding an edge should create bidirectional connection"""
    graph = HNSWGraph(dimension=2, M=16)

    v1 = np.array([1.0, 0.0], dtype=np.float32)
    v2 = np.array([0.0, 1.0], dtype=np.float32)

    id1 = graph.add_node(v1, level=1)
    id2 = graph.add_node(v2, level=1)

    # Add edge at layer 0
    graph.add_edge(id1, id2, layer=0)

    node1 = graph.get_node(id1)
    node2 = graph.get_node(id2)

    assert id2 in node1.get_neighbors(layer=0)
    assert id1 in node2.get_neighbors(layer=0)


def test_add_edge_multiple_layers():
    """Nodes can be connected at multiple layers"""
    graph = HNSWGraph(dimension=2, M=16)

    v1 = np.array([1.0, 0.0], dtype=np.float32)
    v2 = np.array([0.0, 1.0], dtype=np.float32)

    id1 = graph.add_node(v1, level=2)
    id2 = graph.add_node(v2, level=2)

    # Connect at layers 0 and 1
    graph.add_edge(id1, id2, layer=0)
    graph.add_edge(id1, id2, layer=1)

    node1 = graph.get_node(id1)

    assert id2 in node1.get_neighbors(layer=0)
    assert id2 in node1.get_neighbors(layer=1)
    assert node1.get_neighbors(layer=2) == []  # Not connected at layer 2


def test_add_edge_invalid_node():
    """Adding edge with non-existent node should raise error"""
    graph = HNSWGraph(dimension=2, M=16)

    v1 = np.array([1.0, 0.0], dtype=np.float32)
    id1 = graph.add_node(v1, level=1)

    with pytest.raises(ValueError):
        graph.add_edge(id1, 999, layer=0)  # Node 999 doesn't exist


def test_dimension_validation():
    """Adding vector with wrong dimension should fail"""
    graph = HNSWGraph(dimension=3, M=16)

    wrong_vector = np.array([1.0, 2.0], dtype=np.float32)  # 2D, but graph expects 3D

    with pytest.raises(ValueError):
        graph.add_node(wrong_vector, level=0)


def test_get_nonexistent_node():
    """Getting a node that doesn't exist should return None"""
    graph = HNSWGraph(dimension=2, M=16)

    node = graph.get_node(999)
    assert node is None


def test_custom_M_L():
    """Graph should accept custom M_L parameter"""
    graph = HNSWGraph(dimension=2, M=16, M_L=48)

    assert graph.M == 16
    assert graph.M_L == 48
