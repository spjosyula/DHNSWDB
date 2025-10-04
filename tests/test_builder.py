"""
Tests for HNSW insertion algorithm.

These tests verify that the builder correctly inserts nodes into the graph:
- First node insertion (special case)
- Multiple node insertion with connections
- Neighbor selection and pruning
- Graph structure integrity after insertions
"""

import numpy as np
import pytest
from dynhnsw.hnsw.graph import HNSWGraph
from dynhnsw.hnsw.builder import HNSWBuilder


def test_insert_first_node():
    """Insert the first node into an empty graph"""
    graph = HNSWGraph(dimension=3, M=4)
    builder = HNSWBuilder(graph)

    vector = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    builder.insert(vector, node_id=0, level=2)

    assert graph.size() == 1
    assert graph.entry_point == 0
    assert graph.get_max_level() == 2

    # First node has no neighbors
    node = graph.get_node(0)
    assert node.get_neighbors(0) == []
    assert node.get_neighbors(1) == []
    assert node.get_neighbors(2) == []


def test_insert_two_nodes():
    """Insert two nodes and verify they connect"""
    graph = HNSWGraph(dimension=2, M=4)
    builder = HNSWBuilder(graph)

    v1 = np.array([1.0, 0.0], dtype=np.float32)
    v2 = np.array([0.9, 0.1], dtype=np.float32)

    builder.insert(v1, node_id=0, level=1)
    builder.insert(v2, node_id=1, level=1)

    # Nodes should be connected at layer 0 and layer 1
    node0 = graph.get_node(0)
    node1 = graph.get_node(1)

    assert 1 in node0.get_neighbors(0), "Node 0 should connect to node 1 at layer 0"
    assert 0 in node1.get_neighbors(0), "Node 1 should connect to node 0 at layer 0"

    assert 1 in node0.get_neighbors(1), "Node 0 should connect to node 1 at layer 1"
    assert 0 in node1.get_neighbors(1), "Node 1 should connect to node 0 at layer 1"


def test_insert_multiple_nodes():
    """Insert several nodes and verify graph growth"""
    graph = HNSWGraph(dimension=2, M=4)
    builder = HNSWBuilder(graph)

    # Create 5 nodes with different levels
    vectors = [
        np.array([1.0, 0.0], dtype=np.float32),
        np.array([0.9, 0.1], dtype=np.float32),
        np.array([0.8, 0.2], dtype=np.float32),
        np.array([0.0, 1.0], dtype=np.float32),
        np.array([0.1, 0.9], dtype=np.float32),
    ]

    levels = [0, 1, 0, 0, 0]

    for i, (vec, lvl) in enumerate(zip(vectors, levels)):
        builder.insert(vec, node_id=i, level=lvl)

    assert graph.size() == 5

    # Each node should have some neighbors
    for i in range(5):
        node = graph.get_node(i)
        neighbors = node.get_neighbors(0)
        # First node might have no neighbors, others should have at least one
        if i > 0:
            assert len(neighbors) > 0, f"Node {i} should have neighbors"


def test_insert_respects_M_constraint():
    """Verify that pruning maintains M constraint"""
    graph = HNSWGraph(dimension=2, M=2, M_L=3)  # Small M for easier testing
    builder = HNSWBuilder(graph)

    # Insert several similar nodes all at level 0
    # They should all want to connect, but pruning should limit connections
    vectors = [
        np.array([1.0, 0.0], dtype=np.float32),
        np.array([0.99, 0.01], dtype=np.float32),
        np.array([0.98, 0.02], dtype=np.float32),
        np.array([0.97, 0.03], dtype=np.float32),
        np.array([0.96, 0.04], dtype=np.float32),
    ]

    for i, vec in enumerate(vectors):
        builder.insert(vec, node_id=i, level=0)

    # Check that no node exceeds M_L connections at layer 0
    for i in range(5):
        node = graph.get_node(i)
        neighbors_count = len(node.get_neighbors(0))
        assert neighbors_count <= graph.M_L, (
            f"Node {i} has {neighbors_count} neighbors, exceeds M_L={graph.M_L}"
        )


def test_search_layer():
    """Test the internal search_layer function"""
    graph = HNSWGraph(dimension=2, M=4)
    builder = HNSWBuilder(graph)

    # Build a small graph
    vectors = [
        np.array([1.0, 0.0], dtype=np.float32),
        np.array([0.0, 1.0], dtype=np.float32),
        np.array([1.0, 1.0], dtype=np.float32),
    ]

    for i, vec in enumerate(vectors):
        builder.insert(vec, node_id=i, level=0)

    # Search for vector close to [1.0, 0.0]
    query = np.array([0.95, 0.05], dtype=np.float32)
    results = builder._search_layer(
        query=query, entry_points=[0], num_closest=2, layer=0
    )

    # Should return node 0 (closest) and possibly node 2
    assert 0 in results, "Node 0 should be in results (closest to query)"
    assert len(results) <= 2, "Should return at most 2 results"


def test_different_levels():
    """Insert nodes at different levels and verify layer connections"""
    graph = HNSWGraph(dimension=2, M=4)
    builder = HNSWBuilder(graph)

    # Node at level 0
    v1 = np.array([1.0, 0.0], dtype=np.float32)
    builder.insert(v1, node_id=0, level=0)

    # Node at level 2 (becomes entry point)
    v2 = np.array([0.0, 1.0], dtype=np.float32)
    builder.insert(v2, node_id=1, level=2)

    # Node at level 1
    v3 = np.array([0.5, 0.5], dtype=np.float32)
    builder.insert(v3, node_id=2, level=1)

    # Entry point should be node 1 (highest level)
    assert graph.entry_point == 1

    node1 = graph.get_node(1)
    node2 = graph.get_node(2)

    # Node 1 and 2 should connect at layer 1
    assert 2 in node1.get_neighbors(1) or 1 in node2.get_neighbors(1), (
        "High-level nodes should connect at their shared layers"
    )


def test_bidirectional_connections():
    """Verify all connections are bidirectional"""
    graph = HNSWGraph(dimension=2, M=4)
    builder = HNSWBuilder(graph)

    vectors = [
        np.array([1.0, 0.0], dtype=np.float32),
        np.array([0.0, 1.0], dtype=np.float32),
        np.array([1.0, 1.0], dtype=np.float32),
    ]

    for i, vec in enumerate(vectors):
        builder.insert(vec, node_id=i, level=0)

    # Check bidirectionality for all nodes
    for i in range(3):
        node = graph.get_node(i)
        for neighbor_id in node.get_neighbors(0):
            neighbor = graph.get_node(neighbor_id)
            assert i in neighbor.get_neighbors(0), (
                f"Connection from {i} to {neighbor_id} is not bidirectional"
            )


def test_node_with_higher_level_becomes_entry():
    """When inserting a node with higher level than current entry, it should become new entry"""
    graph = HNSWGraph(dimension=2, M=4)
    builder = HNSWBuilder(graph)

    # First node at level 0
    v1 = np.array([1.0, 0.0], dtype=np.float32)
    builder.insert(v1, node_id=0, level=0)
    assert graph.entry_point == 0

    # Second node at level 3 should become entry point
    v2 = np.array([0.0, 1.0], dtype=np.float32)
    builder.insert(v2, node_id=1, level=3)
    assert graph.entry_point == 1
    assert graph.get_max_level() == 3
