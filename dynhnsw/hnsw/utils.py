"""
Utility functions for HNSW graph construction and maintenance.

This module provides helper functions used during HNSW index building and search:
- Layer assignment: Determines which layers a new node should appear in
- Neighbor selection: Chooses which edges to keep when building the graph

The layer assignment uses a geometric distribution to create a hierarchical structure,
where most nodes are only in layer 0, and progressively fewer nodes appear in higher layers.
This hierarchy allows for efficient search by starting at sparse top layers and zooming in.
"""

import numpy as np
from typing import List, Tuple


def assign_layer(level_multiplier: float = 1.0 / np.log(2.0)) -> int:
    """
    Randomly assign a layer for a new node using geometric distribution.

    In HNSW, nodes are assigned to different layers probabilistically. Most nodes
    only appear in layer 0 (the base layer), while fewer nodes extend to higher layers.
    This creates a hierarchical structure for efficient multi-scale search.

    The probability of a node reaching layer L is: (1/exp(level_multiplier))^L

    Args:
        level_multiplier: Controls the layer distribution (default: 1/ln(2) ≈ 1.44)
                          Higher values = more layers = slower build, better search

    Returns:
        Layer number (0 = bottom layer, higher = sparser upper layers)

    Example:
        >>> # Most calls return 0, occasionally 1, rarely 2+
        >>> layers = [assign_layer() for _ in range(1000)]
        >>> layers.count(0)  # Should be ~500 (50%)
        >>> layers.count(1)  # Should be ~250 (25%)
    """
    # Draw from uniform distribution [0, 1)
    random_value = np.random.uniform(0, 1)

    # Apply inverse of geometric distribution to get layer
    # Formula: floor(-log(uniform_random) * level_multiplier)
    layer = int(-np.log(random_value) * level_multiplier)

    return layer


def select_neighbors_simple(
    candidates: List[int], distances: List[float], M: int
) -> List[int]:
    """
    Select M nearest neighbors from candidates based on distances.

    This is the "simple" neighbor selection strategy - just pick the M closest nodes.
    Later, we can implement more sophisticated heuristics that consider graph connectivity
    and diversity to improve search quality.

    Args:
        candidates: List of node IDs
        distances: List of distances (parallel to candidates, lower = closer)
        M: Maximum number of neighbors to select

    Returns:
        List of selected node IDs (up to M nodes, sorted by distance)

    Example:
        >>> candidates = [10, 20, 30, 40]
        >>> distances = [0.5, 0.2, 0.8, 0.3]
        >>> select_neighbors_simple(candidates, distances, M=2)
        [20, 40]  # The two closest nodes
    """
    # Handle edge case: no candidates
    if len(candidates) == 0:
        return []

    # Combine candidates and distances, then sort by distance
    paired = list(zip(candidates, distances))
    paired_sorted = sorted(paired, key=lambda x: x[1])  # Sort by distance (ascending)

    # Take the M closest neighbors
    selected = paired_sorted[:M]

    # Return just the node IDs (discard distances)
    return [node_id for node_id, _ in selected]


def select_neighbors_heuristic(
    candidates: List[int],
    distances: List[float],
    M: int,
    extend_candidates: bool = True,
    keep_pruned: bool = False,
) -> List[int]:
    """
    Select neighbors using heuristic that prioritizes diversity (FUTURE).

    This is a placeholder for the heuristic selection algorithm from the HNSW paper.
    It considers not just distance, but also graph connectivity to avoid creating
    clustered connections that hurt search quality.

    For now, this just calls select_neighbors_simple. We'll implement the full
    heuristic in a future iteration if needed.

    Args:
        candidates: List of node IDs
        distances: List of distances
        M: Maximum number of neighbors
        extend_candidates: Whether to extend candidates with their neighbors
        keep_pruned: Whether to keep pruned connections

    Returns:
        List of selected node IDs
    """
    # TODO: Implement full heuristic selection from HNSW paper
    # For now, use simple selection
    return select_neighbors_simple(candidates, distances, M)
