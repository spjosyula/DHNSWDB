"""
Tests for HNSW search algorithm.

These tests verify that the searcher correctly finds nearest neighbors:
- Empty graph handling
- Single and multiple node searches
- Distance-based ranking
- k parameter behavior
- ef_search parameter effects
"""

import numpy as np
import pytest
from dynhnsw.hnsw.graph import HNSWGraph
from dynhnsw.hnsw.builder import HNSWBuilder
from dynhnsw.hnsw.searcher import HNSWSearcher


def test_search_empty_graph():
    """Searching an empty graph should return empty results"""
    graph = HNSWGraph(dimension=2, M=4)
    searcher = HNSWSearcher(graph, ef_search=10)

    query = np.array([1.0, 0.0], dtype=np.float32)
    results = searcher.search(query, k=5)

    assert results == [], "Empty graph should return no results"


def test_search_single_node():
    """Searching with one node should return that node"""
    graph = HNSWGraph(dimension=2, M=4)
    builder = HNSWBuilder(graph)
    searcher = HNSWSearcher(graph, ef_search=10)

    vector = np.array([1.0, 0.0], dtype=np.float32)
    builder.insert(vector, node_id=0, level=0)

    query = np.array([0.9, 0.1], dtype=np.float32)
    results = searcher.search(query, k=5)

    assert len(results) == 1, "Should return the single node"
    assert results[0][0] == 0, "Should return node 0"


def test_search_returns_closest_nodes():
    """Search should return nodes sorted by distance (closest first)"""
    graph = HNSWGraph(dimension=2, M=4)
    builder = HNSWBuilder(graph)
    searcher = HNSWSearcher(graph, ef_search=10)

    # Insert nodes at different distances from query [1.0, 0.0]
    vectors = [
        np.array([1.0, 0.0], dtype=np.float32),   # Very close
        np.array([0.0, 1.0], dtype=np.float32),   # Orthogonal (far)
        np.array([0.9, 0.1], dtype=np.float32),   # Close
    ]

    for i, vec in enumerate(vectors):
        builder.insert(vec, node_id=i, level=0)

    query = np.array([1.0, 0.0], dtype=np.float32)
    results = searcher.search(query, k=3)

    # Results should be sorted by distance
    assert len(results) == 3
    assert results[0][0] == 0, "Node 0 should be closest (identical to query)"
    assert results[1][0] == 2, "Node 2 should be second closest"
    assert results[2][0] == 1, "Node 1 should be farthest (orthogonal)"

    # Verify distances are in ascending order
    distances = [dist for _, dist in results]
    assert distances == sorted(distances), "Distances should be sorted ascending"


def test_search_respects_k_parameter():
    """Search should return at most k results"""
    graph = HNSWGraph(dimension=2, M=4)
    builder = HNSWBuilder(graph)
    searcher = HNSWSearcher(graph, ef_search=10)

    # Insert 5 nodes
    for i in range(5):
        vector = np.array([float(i), 0.0], dtype=np.float32)
        builder.insert(vector, node_id=i, level=0)

    query = np.array([0.0, 0.0], dtype=np.float32)

    # Search for k=3
    results = searcher.search(query, k=3)
    assert len(results) == 3, "Should return exactly k=3 results"

    # Search for k=10 (more than available)
    results = searcher.search(query, k=10)
    assert len(results) == 5, "Should return all 5 nodes when k > num_nodes"


def test_search_with_different_ef_search():
    """Higher ef_search should potentially find better results"""
    graph = HNSWGraph(dimension=3, M=4)
    builder = HNSWBuilder(graph)

    # Build a larger graph
    np.random.seed(42)
    for i in range(20):
        vector = np.random.rand(3).astype(np.float32)
        builder.insert(vector, node_id=i, level=0)

    query = np.random.rand(3).astype(np.float32)

    # Search with low ef_search
    searcher_low = HNSWSearcher(graph, ef_search=5)
    results_low = searcher_low.search(query, k=5)

    # Search with high ef_search
    searcher_high = HNSWSearcher(graph, ef_search=20)
    results_high = searcher_high.search(query, k=5)

    # Both should return 5 results
    assert len(results_low) == 5
    assert len(results_high) == 5

    # Results should be sorted by distance
    for results in [results_low, results_high]:
        distances = [dist for _, dist in results]
        assert distances == sorted(distances)


def test_search_multilayer_graph():
    """Search should work correctly on graphs with multiple layers"""
    graph = HNSWGraph(dimension=2, M=4)
    builder = HNSWBuilder(graph)
    searcher = HNSWSearcher(graph, ef_search=10)

    # Insert nodes at different layers
    vectors = [
        (np.array([1.0, 0.0], dtype=np.float32), 0),
        (np.array([0.0, 1.0], dtype=np.float32), 2),  # Higher layer
        (np.array([0.5, 0.5], dtype=np.float32), 1),
        (np.array([0.9, 0.1], dtype=np.float32), 0),
    ]

    for i, (vec, level) in enumerate(vectors):
        builder.insert(vec, node_id=i, level=level)

    query = np.array([1.0, 0.0], dtype=np.float32)
    results = searcher.search(query, k=2)

    assert len(results) == 2
    # Should find the closest nodes regardless of their layers
    assert results[0][0] == 0  # Exact match
    assert results[1][0] == 3  # Second closest


def test_search_finds_exact_match():
    """Searching for an existing vector should return distance ~0"""
    graph = HNSWGraph(dimension=2, M=4)
    builder = HNSWBuilder(graph)
    searcher = HNSWSearcher(graph, ef_search=10)

    vector = np.array([0.7, 0.3], dtype=np.float32)
    builder.insert(vector, node_id=0, level=0)

    # Add more nodes
    for i in range(1, 5):
        other = np.random.rand(2).astype(np.float32)
        builder.insert(other, node_id=i, level=0)

    # Search for exact vector
    results = searcher.search(vector, k=1)

    assert len(results) == 1
    assert results[0][0] == 0, "Should find the exact match"
    assert results[0][1] < 0.01, "Distance should be very close to 0"


def test_override_ef_search_per_query():
    """Should be able to override ef_search for individual queries"""
    graph = HNSWGraph(dimension=2, M=4)
    builder = HNSWBuilder(graph)
    searcher = HNSWSearcher(graph, ef_search=10)  # Default ef_search=10

    # Insert nodes
    for i in range(10):
        vector = np.random.rand(2).astype(np.float32)
        builder.insert(vector, node_id=i, level=0)

    query = np.random.rand(2).astype(np.float32)

    # Search with override
    results = searcher.search(query, k=5, ef_search=20)

    assert len(results) == 5, "Should return k results"
