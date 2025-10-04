"""
Tests for distance and similarity metrics.

These tests verify that our distance functions correctly measure similarity between vectors.
We test with known vector pairs to ensure the math is correct.
"""

import numpy as np
import pytest
from dynhnsw.hnsw.distance import (
    cosine_similarity,
    cosine_distance,
    normalize_vector,
)


def test_cosine_similarity_identical_vectors():
    """Identical vectors should have similarity of 1.0"""
    v1 = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    v2 = np.array([1.0, 2.0, 3.0], dtype=np.float32)

    similarity = cosine_similarity(v1, v2)
    assert np.isclose(similarity, 1.0), "Identical vectors should have similarity 1.0"


def test_cosine_similarity_orthogonal_vectors():
    """Orthogonal (perpendicular) vectors should have similarity of 0.0"""
    v1 = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    v2 = np.array([0.0, 1.0, 0.0], dtype=np.float32)

    similarity = cosine_similarity(v1, v2)
    assert np.isclose(similarity, 0.0), "Orthogonal vectors should have similarity 0.0"


def test_cosine_similarity_opposite_vectors():
    """Opposite direction vectors should have similarity of -1.0"""
    v1 = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    v2 = np.array([-1.0, -2.0, -3.0], dtype=np.float32)

    similarity = cosine_similarity(v1, v2)
    assert np.isclose(
        similarity, -1.0
    ), "Opposite vectors should have similarity -1.0"


def test_cosine_similarity_zero_vector():
    """Zero vectors should return 0.0 (edge case handling)"""
    v1 = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    v2 = np.array([0.0, 0.0, 0.0], dtype=np.float32)

    similarity = cosine_similarity(v1, v2)
    assert similarity == 0.0, "Zero vector should return similarity 0.0"


def test_cosine_distance():
    """Cosine distance should be 1 - cosine_similarity"""
    v1 = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    v2 = np.array([1.0, 0.0, 0.0], dtype=np.float32)

    similarity = cosine_similarity(v1, v2)
    distance = cosine_distance(v1, v2)

    assert np.isclose(distance, 1.0 - similarity), "Distance should be 1 - similarity"
    assert np.isclose(distance, 0.0), "Identical vectors should have distance 0.0"


def test_normalize_vector():
    """Normalized vector should have L2 norm of 1.0"""
    v = np.array([3.0, 4.0], dtype=np.float32)  # Magnitude is 5.0
    normalized = normalize_vector(v)

    norm = np.linalg.norm(normalized)
    assert np.isclose(norm, 1.0), "Normalized vector should have norm 1.0"

    # Check direction is preserved (just scaled)
    expected = np.array([0.6, 0.8], dtype=np.float32)  # [3/5, 4/5]
    assert np.allclose(normalized, expected), "Direction should be preserved"


def test_normalize_zero_vector():
    """Normalizing zero vector should return zero vector (edge case)"""
    v = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    normalized = normalize_vector(v)

    assert np.allclose(normalized, v), "Zero vector should remain zero after normalization"
