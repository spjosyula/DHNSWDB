"""
Distance and similarity metrics for vector comparisons.

This module provides functions to measure how similar or different two vectors are.
In vector databases, we use these metrics to find the most similar items to a query.

Cosine similarity measures the angle between vectors (ranges from -1 to 1, where 1 means
identical direction). It's commonly used for text embeddings since it ignores magnitude
and focuses on semantic similarity.
"""

import numpy as np
import numpy.typing as npt

Vector = npt.NDArray[np.float32]


def cosine_similarity(v1: Vector, v2: Vector) -> float:
    """
    Compute cosine similarity between two vectors.

    Cosine similarity measures the cosine of the angle between two vectors.
    It ranges from -1 (opposite directions) to 1 (same direction).
    For normalized vectors, this is equivalent to their dot product.

    Args:
        v1: First vector (1D numpy array)
        v2: Second vector (1D numpy array)

    Returns:
        Similarity score between -1 and 1 (higher means more similar)

    Example:
        >>> v1 = np.array([1.0, 0.0, 0.0])
        >>> v2 = np.array([1.0, 0.0, 0.0])
        >>> cosine_similarity(v1, v2)
        1.0
    """
    # Compute dot product
    dot_product = np.dot(v1, v2)

    # Compute magnitudes (L2 norms)
    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)

    # Handle edge case where one or both vectors are zero
    if norm_v1 == 0.0 or norm_v2 == 0.0:
        return 0.0

    # Cosine similarity = dot product / (magnitude1 * magnitude2)
    return float(dot_product / (norm_v1 * norm_v2))


def cosine_distance(v1: Vector, v2: Vector) -> float:
    """
    Compute cosine distance between two vectors.

    Cosine distance is defined as 1 - cosine_similarity, converting similarity
    to a distance metric. It ranges from 0 (identical) to 2 (opposite directions).

    Args:
        v1: First vector (1D numpy array)
        v2: Second vector (1D numpy array)

    Returns:
        Distance score between 0 and 2 (lower means more similar)

    Example:
        >>> v1 = np.array([1.0, 0.0, 0.0])
        >>> v2 = np.array([1.0, 0.0, 0.0])
        >>> cosine_distance(v1, v2)
        0.0
    """
    return 1.0 - cosine_similarity(v1, v2)


def normalize_vector(v: Vector) -> Vector:
    """
    Normalize a vector to unit length (L2 norm = 1).

    Normalization is useful for cosine similarity since it allows us to use
    simple dot products instead of the full cosine formula.

    Args:
        v: Input vector (1D numpy array)

    Returns:
        Normalized vector with L2 norm = 1

    Example:
        >>> v = np.array([3.0, 4.0])
        >>> normalized = normalize_vector(v)
        >>> np.linalg.norm(normalized)  # Should be 1.0
        1.0
    """
    norm = np.linalg.norm(v)

    # Avoid division by zero
    if norm == 0.0:
        return v

    return v / norm
