"""
End-to-end tests for VectorStore.

These tests verify the complete functionality of the VectorStore API:
- Adding vectors (single and batch)
- Searching for nearest neighbors
- Dimension validation
- Normalization
- Integration of all HNSW components
"""

import numpy as np
import pytest
from dynhnsw import VectorStore


def test_create_vector_store():
    """Create a basic vector store"""
    store = VectorStore(dimension=128, M=16, ef_search=50)

    assert store.dimension == 128
    assert store.M == 16
    assert store.ef_search == 50
    assert store.size() == 0


def test_add_single_vector():
    """Add a single vector to the store"""
    store = VectorStore(dimension=3)

    vector = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    ids = store.add(vector)

    assert len(ids) == 1
    assert ids[0] == 0
    assert store.size() == 1


def test_add_multiple_vectors():
    """Add multiple vectors to the store"""
    store = VectorStore(dimension=2)

    vectors = [
        np.array([1.0, 0.0], dtype=np.float32),
        np.array([0.0, 1.0], dtype=np.float32),
        np.array([1.0, 1.0], dtype=np.float32),
    ]

    ids = store.add(vectors)

    assert len(ids) == 3
    assert ids == [0, 1, 2]
    assert store.size() == 3


def test_add_batch_vectors():
    """Add a batch of vectors"""
    store = VectorStore(dimension=4)

    # Create 10 random vectors
    vectors = [np.random.rand(4).astype(np.float32) for _ in range(10)]
    ids = store.add(vectors)

    assert len(ids) == 10
    assert store.size() == 10


def test_search_single_vector():
    """Search with a single vector in store"""
    store = VectorStore(dimension=2)

    vector = np.array([1.0, 0.0], dtype=np.float32)
    store.add(vector)

    query = np.array([0.9, 0.1], dtype=np.float32)
    results = store.search(query, k=1)

    assert len(results) == 1
    assert results[0]["id"] == 0
    assert "distance" in results[0]
    assert "vector" in results[0]


def test_search_returns_k_results():
    """Search should return k nearest neighbors"""
    store = VectorStore(dimension=2)

    # Add 5 vectors
    vectors = [
        np.array([1.0, 0.0], dtype=np.float32),
        np.array([0.0, 1.0], dtype=np.float32),
        np.array([1.0, 1.0], dtype=np.float32),
        np.array([0.5, 0.5], dtype=np.float32),
        np.array([0.7, 0.3], dtype=np.float32),
    ]
    store.add(vectors)

    query = np.array([1.0, 0.0], dtype=np.float32)
    results = store.search(query, k=3)

    assert len(results) == 3
    # Results should be sorted by distance
    assert results[0]["id"] == 0  # Exact match


def test_search_closest_vectors():
    """Search should return vectors sorted by similarity"""
    store = VectorStore(dimension=2)

    # Add vectors at different distances
    vectors = [
        np.array([1.0, 0.0], dtype=np.float32),   # id=0, very close
        np.array([0.0, 1.0], dtype=np.float32),   # id=1, far
        np.array([0.9, 0.1], dtype=np.float32),   # id=2, close
    ]
    store.add(vectors)

    query = np.array([1.0, 0.0], dtype=np.float32)
    results = store.search(query, k=3)

    # Verify order: 0 (closest), 2 (second), 1 (farthest)
    assert results[0]["id"] == 0
    assert results[1]["id"] == 2
    assert results[2]["id"] == 1

    # Distances should be ascending
    distances = [r["distance"] for r in results]
    assert distances == sorted(distances)


def test_dimension_validation_add():
    """Adding vectors with wrong dimension should fail"""
    store = VectorStore(dimension=3)

    wrong_vector = np.array([1.0, 2.0], dtype=np.float32)  # 2D instead of 3D

    with pytest.raises(ValueError):
        store.add(wrong_vector)


def test_dimension_validation_search():
    """Searching with wrong dimension should fail"""
    store = VectorStore(dimension=3)

    # Add valid vector
    store.add(np.array([1.0, 0.0, 0.0], dtype=np.float32))

    # Search with wrong dimension
    wrong_query = np.array([1.0, 0.0], dtype=np.float32)

    with pytest.raises(ValueError):
        store.search(wrong_query, k=1)


def test_normalization_enabled():
    """Vectors should be normalized when normalize=True"""
    store = VectorStore(dimension=2, normalize=True)

    # Add non-normalized vector
    vector = np.array([3.0, 4.0], dtype=np.float32)  # Magnitude = 5.0
    store.add(vector)

    # Search with non-normalized query
    query = np.array([3.0, 4.0], dtype=np.float32)
    results = store.search(query, k=1)

    # Should find the vector with very small distance (since both are normalized)
    assert results[0]["distance"] < 0.01


def test_normalization_disabled():
    """Vectors should not be normalized when normalize=False"""
    store = VectorStore(dimension=2, normalize=False)

    vector = np.array([3.0, 4.0], dtype=np.float32)
    store.add(vector)

    # The stored vector should still be [3.0, 4.0], not normalized
    query = np.array([3.0, 4.0], dtype=np.float32)
    results = store.search(query, k=1)

    # Distance should still be small (exact match)
    assert results[0]["distance"] < 0.01


def test_search_empty_store():
    """Searching an empty store should return empty results"""
    store = VectorStore(dimension=2)

    query = np.array([1.0, 0.0], dtype=np.float32)
    results = store.search(query, k=5)

    assert results == []


def test_override_ef_search():
    """Should be able to override ef_search per query"""
    store = VectorStore(dimension=2, ef_search=10)

    # Add vectors
    for i in range(10):
        vec = np.random.rand(2).astype(np.float32)
        store.add(vec)

    query = np.random.rand(2).astype(np.float32)

    # Search with overridden ef_search
    results = store.search(query, k=5, ef_search=20)

    assert len(results) == 5


def test_large_batch_insert_and_search():
    """Test with a larger dataset"""
    store = VectorStore(dimension=128, M=16, ef_search=50)

    # Add 100 random vectors
    np.random.seed(42)
    vectors = [np.random.rand(128).astype(np.float32) for _ in range(100)]
    ids = store.add(vectors)

    assert len(ids) == 100
    assert store.size() == 100

    # Search for nearest neighbors
    query = np.random.rand(128).astype(np.float32)
    results = store.search(query, k=10)

    assert len(results) == 10
    # Verify all results have required fields
    for result in results:
        assert "id" in result
        assert "distance" in result
        assert "vector" in result
        assert result["vector"].shape == (128,)


def test_incremental_additions():
    """Add vectors incrementally and search"""
    store = VectorStore(dimension=3)

    # Add first batch
    batch1 = [np.random.rand(3).astype(np.float32) for _ in range(5)]
    ids1 = store.add(batch1)
    assert store.size() == 5

    # Add second batch
    batch2 = [np.random.rand(3).astype(np.float32) for _ in range(5)]
    ids2 = store.add(batch2)
    assert store.size() == 10

    # IDs should be sequential
    assert ids1 == [0, 1, 2, 3, 4]
    assert ids2 == [5, 6, 7, 8, 9]

    # Search should find from both batches
    query = np.random.rand(3).astype(np.float32)
    results = store.search(query, k=10)
    assert len(results) == 10


def test_recall_with_brute_force():
    """Compare HNSW results with brute force search for recall"""
    from dynhnsw.hnsw.distance import cosine_distance

    store = VectorStore(dimension=10, M=8, ef_search=50)

    # Add 50 vectors
    np.random.seed(42)
    vectors = [np.random.rand(10).astype(np.float32) for _ in range(50)]
    store.add(vectors)

    # Search query
    query = np.random.rand(10).astype(np.float32)

    # HNSW search
    hnsw_results = store.search(query, k=10)
    hnsw_ids = set(r["id"] for r in hnsw_results)

    # Brute force search
    if store.normalize:
        from dynhnsw.hnsw.distance import normalize_vector
        query_norm = normalize_vector(query)
        vectors_norm = [normalize_vector(v) for v in vectors]
    else:
        query_norm = query
        vectors_norm = vectors

    distances = [(i, cosine_distance(query_norm, v)) for i, v in enumerate(vectors_norm)]
    distances.sort(key=lambda x: x[1])
    brute_force_ids = set(i for i, _ in distances[:10])

    # Calculate recall (intersection / k)
    recall = len(hnsw_ids & brute_force_ids) / 10

    # HNSW should have good recall (at least 80%)
    assert recall >= 0.8, f"Recall is too low: {recall}"
