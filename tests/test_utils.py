"""
Tests for HNSW utility functions.

These tests verify helper functions used in graph construction:
- Layer assignment (geometric distribution for hierarchical structure)
- Neighbor selection (choosing which edges to keep in the graph)
"""

import numpy as np
import pytest
from dynhnsw.hnsw.utils import assign_layer, select_neighbors_simple


def test_assign_layer_returns_non_negative():
    """Layer assignment should always return non-negative integers"""
    for _ in range(100):
        layer = assign_layer()
        assert isinstance(layer, int), "Layer should be an integer"
        assert layer >= 0, "Layer should be non-negative"


def test_assign_layer_distribution():
    """Most nodes should be at layer 0, fewer at higher layers (geometric distribution)"""
    np.random.seed(42)  # For reproducible test
    layers = [assign_layer() for _ in range(1000)]

    # Count frequency at each layer
    layer_0_count = layers.count(0)
    layer_1_count = layers.count(1)
    layer_2_count = layers.count(2)

    # Layer 0 should have the most nodes (roughly 50% with default multiplier)
    assert layer_0_count > layer_1_count, "Layer 0 should have more nodes than layer 1"
    assert layer_1_count > layer_2_count, "Layer 1 should have more nodes than layer 2"

    # At least 40% should be at layer 0 (allowing some variance)
    assert layer_0_count >= 400, "Most nodes should be at layer 0"


def test_assign_layer_with_custom_multiplier():
    """Higher multiplier should create more layers on average"""
    np.random.seed(42)

    # Low multiplier = fewer layers
    low_multiplier_layers = [assign_layer(level_multiplier=0.5) for _ in range(1000)]
    low_max_layer = max(low_multiplier_layers)

    # High multiplier = more layers
    high_multiplier_layers = [assign_layer(level_multiplier=2.0) for _ in range(1000)]
    high_max_layer = max(high_multiplier_layers)

    assert (
        high_max_layer >= low_max_layer
    ), "Higher multiplier should produce higher layers"


def test_select_neighbors_simple_basic():
    """Select M nearest neighbors from candidates"""
    candidates = [10, 20, 30, 40, 50]
    distances = [0.5, 0.2, 0.8, 0.3, 0.9]
    M = 3

    selected = select_neighbors_simple(candidates, distances, M)

    # Should select the 3 closest: nodes 20 (0.2), 40 (0.3), 10 (0.5)
    assert len(selected) == 3, f"Should select {M} neighbors"
    assert 20 in selected, "Node 20 (distance 0.2) should be selected"
    assert 40 in selected, "Node 40 (distance 0.3) should be selected"
    assert 10 in selected, "Node 10 (distance 0.5) should be selected"
    assert 30 not in selected, "Node 30 (distance 0.8) should not be selected"
    assert 50 not in selected, "Node 50 (distance 0.9) should not be selected"


def test_select_neighbors_simple_ordering():
    """Selected neighbors should be sorted by distance (closest first)"""
    candidates = [100, 200, 300]
    distances = [0.9, 0.1, 0.5]
    M = 3

    selected = select_neighbors_simple(candidates, distances, M)

    # Should be sorted: 200 (0.1), 300 (0.5), 100 (0.9)
    assert selected == [200, 300, 100], "Neighbors should be sorted by distance"


def test_select_neighbors_simple_fewer_than_M():
    """When candidates < M, return all candidates"""
    candidates = [10, 20]
    distances = [0.5, 0.3]
    M = 5

    selected = select_neighbors_simple(candidates, distances, M)

    assert len(selected) == 2, "Should return all candidates when fewer than M"
    assert 20 in selected and 10 in selected, "Should include all candidates"


def test_select_neighbors_simple_empty():
    """Empty candidate list should return empty result"""
    candidates = []
    distances = []
    M = 3

    selected = select_neighbors_simple(candidates, distances, M)

    assert selected == [], "Empty candidates should return empty list"


def test_select_neighbors_simple_single():
    """Single candidate should be selected"""
    candidates = [42]
    distances = [0.7]
    M = 3

    selected = select_neighbors_simple(candidates, distances, M)

    assert selected == [42], "Should select the single candidate"
