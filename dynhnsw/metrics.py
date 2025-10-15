"""
Metrics for evaluating HNSW search quality.

This module provides functions to:
- Compute recall@k (fraction of ground truth neighbors retrieved)
- Compute query difficulty (k-th nearest neighbor distance)
- Compute exact ground truth via brute force search
"""

import numpy as np
from typing import List, Tuple, Optional


def compute_recall_at_k(
    retrieved_ids: List[int],
    ground_truth_ids: List[int],
    k: int = 10
) -> float:
    """
    Compute recall@k: fraction of ground truth neighbors retrieved.

    Recall@k measures how many of the true k-nearest neighbors
    were found by the search algorithm.

    Args:
        retrieved_ids: IDs returned by search (ordered by relevance)
        ground_truth_ids: True k-nearest neighbor IDs
        k: Number of neighbors to consider

    Returns:
        Recall@k value between 0.0 (no correct neighbors) and 1.0 (all correct)

    Example:
        >>> retrieved = [1, 2, 3, 99, 98]
        >>> ground_truth = [1, 2, 3, 4, 5]
        >>> compute_recall_at_k(retrieved, ground_truth, k=5)
        0.6  # Found 3 out of 5 correct neighbors
    """
    # Consider only top-k results
    retrieved_set = set(retrieved_ids[:k])
    ground_truth_set = set(ground_truth_ids[:k])

    # Count overlap
    correct_retrievals = len(retrieved_set & ground_truth_set)

    return correct_retrievals / k if k > 0 else 0.0


def compute_query_difficulty(
    index,
    query_vector: np.ndarray,
    k: int = 10,
    ef_search_for_difficulty: int = 200
) -> float:
    """
    Compute query difficulty as k-th nearest neighbor distance.

    Difficulty measures how hard it is to find neighbors for a query.
    Higher distance to k-th neighbor = harder query = needs higher ef_search.

    Args:
        index: HNSW index to search (must have search method)
        query_vector: Query embedding (1D numpy array)
        k: Number of neighbors
        ef_search_for_difficulty: High ef_search for accurate measurement (default: 200)

    Returns:
        Distance to k-th nearest neighbor (higher = more difficult)
        Returns inf if fewer than k neighbors found

    Note:
        Uses high ef_search to get accurate difficulty estimate.
        This adds overhead but ensures reliable intent detection.
    """
    # Search with high ef_search for accurate difficulty
    results = index.search(query_vector, k=k, ef_search=ef_search_for_difficulty)

    # Extract distances
    distances = [dist for _, dist in results]

    # Return k-th distance (or inf if not enough neighbors)
    if len(distances) == k:
        return float(distances[-1])
    else:
        return float('inf')


def compute_ground_truth_brute_force(
    query_vector: np.ndarray,
    all_vectors: np.ndarray,
    k: int = 10
) -> Tuple[List[int], List[float]]:
    """
    Compute exact k-NN via brute force (slow but exact).

    This is used to compute ground truth for evaluation.
    Should be pre-computed for test datasets and cached.

    Args:
        query_vector: Query embedding (1D array, shape: [dim])
        all_vectors: All database vectors (2D array, shape: [n_vectors, dim])
        k: Number of neighbors to find

    Returns:
        Tuple of (neighbor_ids, distances), both sorted by distance ascending

    Complexity:
        O(n * dim) where n = number of vectors
        Use pre-computation for large datasets!

    Example:
        >>> query = np.array([1.0, 0.0, 0.0])
        >>> database = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
        >>> ids, dists = compute_ground_truth_brute_force(query, database, k=2)
        >>> ids  # [0, 1] - indices 0 and 1 are closest
        >>> dists  # [0.0, 1.414...] - distances to nearest neighbors
    """
    # Compute L2 distances to all vectors
    distances = np.linalg.norm(all_vectors - query_vector, axis=1)

    # Find k smallest distances using partial sort
    # argpartition is faster than full sort for small k
    if k < len(distances):
        k_indices = np.argpartition(distances, k)[:k]
        k_indices_sorted = k_indices[np.argsort(distances[k_indices])]
    else:
        # If k >= n, return all sorted
        k_indices_sorted = np.argsort(distances)

    k_distances = distances[k_indices_sorted]

    return k_indices_sorted.tolist(), k_distances.tolist()


def compute_precision_at_k(
    retrieved_ids: List[int],
    ground_truth_ids: List[int],
    k: int = 10
) -> float:
    """
    Compute precision@k: fraction of retrieved neighbors that are correct.

    Precision measures relevance of retrieved results.
    Less commonly used than recall for k-NN evaluation.

    Args:
        retrieved_ids: IDs returned by search
        ground_truth_ids: True k-nearest neighbor IDs
        k: Number of neighbors to consider

    Returns:
        Precision@k value between 0.0 and 1.0

    Note:
        For k-NN search, precision@k usually equals recall@k
        since we always retrieve exactly k items.
    """
    retrieved_set = set(retrieved_ids[:k])
    ground_truth_set = set(ground_truth_ids[:k])

    correct_retrievals = len(retrieved_set & ground_truth_set)
    retrieved_count = min(len(retrieved_ids), k)

    return correct_retrievals / retrieved_count if retrieved_count > 0 else 0.0


def compute_mean_reciprocal_rank(
    retrieved_ids: List[int],
    ground_truth_ids: List[int]
) -> float:
    """
    Compute Mean Reciprocal Rank (MRR) for a single query.

    MRR measures how early the first correct result appears.
    Higher rank = better search quality.

    Args:
        retrieved_ids: IDs returned by search (ordered)
        ground_truth_ids: True k-nearest neighbor IDs

    Returns:
        Reciprocal rank (1/rank of first correct result, or 0 if none found)

    Example:
        >>> retrieved = [99, 98, 1, 2]  # First correct at position 3
        >>> ground_truth = [1, 2, 3, 4]
        >>> compute_mean_reciprocal_rank(retrieved, ground_truth)
        0.333...  # 1/3
    """
    ground_truth_set = set(ground_truth_ids)

    for rank, retrieved_id in enumerate(retrieved_ids, start=1):
        if retrieved_id in ground_truth_set:
            return 1.0 / rank

    return 0.0  # No correct result found
