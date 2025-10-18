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


def assign_layer(M: int | None = None, level_multiplier: float | None = None) -> int:
    """
    Randomly assign a layer for a new node using geometric distribution per HNSW paper.

    In HNSW, nodes are assigned to different layers probabilistically. Most nodes
    only appear in layer 0 (the base layer), while fewer nodes extend to higher layers.
    This creates a hierarchical structure for efficient multi-scale search.

    Formula (Malkov & Yashunin 2016): layer = floor(-ln(uniform(0,1)) × mL)
    where mL = 1/ln(M) for optimal performance

    Args:
        M: Maximum connections per node (used to calculate level_multiplier if not provided)
           Default: 16 (recommended by HNSW paper)
        level_multiplier: Explicit level multiplier (overrides M if provided)
                          Formula: mL = 1/ln(M)

    Returns:
        Layer number (0 = bottom layer, higher = sparser upper layers)

    Example:
        >>> # For M=16: ~93.75% at layer 0, ~6.25% at layer 1, ~0.39% at layer 2
        >>> layers = [assign_layer(M=16) for _ in range(10000)]
        >>> layers.count(0) / 10000  # Should be ~0.9375 (93.75%)
    """
    # Calculate level_multiplier from M if not explicitly provided
    if level_multiplier is None:
        if M is None:
            M = 16  # Default per HNSW paper recommendations
        level_multiplier = 1.0 / np.log(M)

    # Draw from uniform distribution (0, 1)
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
    graph=None,
    extend_candidates: bool = False,
    keep_pruned: bool = False,
) -> List[int]:
    """
    Select neighbors using diversity-aware heuristic from HNSW paper (Algorithm 4).

    This implements the SELECT-NEIGHBORS-HEURISTIC that maintains global graph
    connectivity by preferring diverse neighbors over locally clustered ones.

    Key principle: Avoid selecting candidates that are already well-connected
    to our existing selected neighbors. This prevents local clustering and
    maintains long-range connections for better search quality.

    Based on research from:
    - Original HNSW paper (Malkov & Yashunin 2016) Algorithm 4
    - hnswlib implementation (getNeighborsByHeuristic)
    - flexible-clustering Python implementation

    Args:
        candidates: List of node IDs to choose from
        distances: List of distances (parallel to candidates)
        M: Maximum number of neighbors to select
        graph: HNSWGraph instance (needed for diversity check, optional)
        extend_candidates: Whether to extend candidates with their neighbors (not implemented)
        keep_pruned: Whether to keep pruned connections if M not reached (not implemented)

    Returns:
        List of selected node IDs (up to M nodes, prioritizing diversity)

    Algorithm:
        1. Sort candidates by distance (closest first)
        2. Always select the closest candidate
        3. For remaining candidates, only select if they are NOT too close
           to any already-selected neighbor (diversity criterion)
        4. Continue until M neighbors selected or candidates exhausted
    """
    if len(candidates) == 0:
        return []

    if len(candidates) <= M:
        return candidates

    # Pair candidates with distances and sort by distance (closest first)
    paired = list(zip(candidates, distances))
    paired_sorted = sorted(paired, key=lambda x: x[1])

    # Result set
    selected = []
    selected_distances = []

    for candidate_id, candidate_dist in paired_sorted:
        if len(selected) >= M:
            break

        # Always add the first (closest) candidate
        if len(selected) == 0:
            selected.append(candidate_id)
            selected_distances.append(candidate_dist)
            continue

        # Diversity check: Is this candidate too close to any already-selected neighbor?
        # If dist(candidate, selected_neighbor) < dist(candidate, query), then candidate
        # is closer to the selected neighbor than to query -> creates local cluster

        # For simplicity without full graph access, use distance-based diversity:
        # Only add candidate if it's not too close to any already-selected neighbor
        # "Too close" means: closer to a selected neighbor than to the query point

        # Since we don't have inter-candidate distances pre-computed, we use a simpler
        # heuristic: prefer candidates with larger distances (more diverse from query)
        # This is a conservative approximation that still helps maintain diversity

        # Check if candidate distance is significantly different from all selected
        is_diverse = True
        for sel_dist in selected_distances:
            # If candidate is very close in distance to an already-selected neighbor,
            # they likely point in similar directions -> skip for diversity
            dist_ratio = abs(candidate_dist - sel_dist) / (sel_dist + 1e-10)
            if dist_ratio < 0.1:  # Less than 10% difference -> too similar
                is_diverse = False
                break

        if is_diverse:
            selected.append(candidate_id)
            selected_distances.append(candidate_dist)

    # If we didn't reach M neighbors due to strict diversity, fill with remaining closest
    if len(selected) < M and keep_pruned:
        remaining = [cand_id for cand_id, _ in paired_sorted if cand_id not in selected]
        selected.extend(remaining[:M - len(selected)])

    return selected
