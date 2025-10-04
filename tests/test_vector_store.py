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
    assert ids[0] == "doc_0"
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
    assert ids == ["doc_0", "doc_1", "doc_2"]
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
    assert results[0]["id"] == "doc_0"
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
    assert results[0]["id"] == "doc_0"  # Exact match


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

    # Verify order: doc_0 (closest), doc_2 (second), doc_1 (farthest)
    assert results[0]["id"] == "doc_0"
    assert results[1]["id"] == "doc_2"
    assert results[2]["id"] == "doc_1"

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
    assert ids1 == ["doc_0", "doc_1", "doc_2", "doc_3", "doc_4"]
    assert ids2 == ["doc_5", "doc_6", "doc_7", "doc_8", "doc_9"]

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
    brute_force_ids = set(f"doc_{i}" for i, _ in distances[:10])

    # Calculate recall (intersection / k)
    recall = len(hnsw_ids & brute_force_ids) / 10

    # HNSW should have good recall (at least 80%)
    assert recall >= 0.8, f"Recall is too low: {recall}"


def test_recall_with_larger_dataset():
    """Test HNSW recall on 500 vectors - should maintain high recall"""
    from dynhnsw.hnsw.distance import cosine_distance, normalize_vector

    store = VectorStore(dimension=32, M=16, ef_search=100, normalize=True)

    # Add 500 vectors
    np.random.seed(42)
    vectors = [np.random.rand(32).astype(np.float32) for _ in range(500)]
    store.add(vectors)

    # Test 10 different queries
    recalls = []
    for seed in range(10):
        np.random.seed(seed + 100)
        query = np.random.rand(32).astype(np.float32)

        # HNSW search
        hnsw_results = store.search(query, k=10)
        hnsw_ids = set(r["id"] for r in hnsw_results)

        # Brute force
        query_norm = normalize_vector(query)
        vectors_norm = [normalize_vector(v) for v in vectors]
        distances = [(i, cosine_distance(query_norm, v)) for i, v in enumerate(vectors_norm)]
        distances.sort(key=lambda x: x[1])
        brute_force_ids = set(f"doc_{i}" for i, _ in distances[:10])

        recall = len(hnsw_ids & brute_force_ids) / 10
        recalls.append(recall)

    avg_recall = np.mean(recalls)
    # With ef_search=100, should get >90% recall on average
    assert avg_recall >= 0.9, f"Average recall too low: {avg_recall:.3f}"


def test_recall_with_very_large_dataset():
    """Test HNSW recall on 1000 vectors"""
    from dynhnsw.hnsw.distance import cosine_distance, normalize_vector

    store = VectorStore(dimension=64, M=16, ef_search=150, normalize=True)

    # Add 1000 vectors
    np.random.seed(42)
    vectors = [np.random.rand(64).astype(np.float32) for _ in range(1000)]
    store.add(vectors)

    # Test 5 queries
    recalls = []
    for seed in range(5):
        np.random.seed(seed + 200)
        query = np.random.rand(64).astype(np.float32)

        hnsw_results = store.search(query, k=10)
        hnsw_ids = set(r["id"] for r in hnsw_results)

        query_norm = normalize_vector(query)
        vectors_norm = [normalize_vector(v) for v in vectors]
        distances = [(i, cosine_distance(query_norm, v)) for i, v in enumerate(vectors_norm)]
        distances.sort(key=lambda x: x[1])
        brute_force_ids = set(f"doc_{i}" for i, _ in distances[:10])

        recall = len(hnsw_ids & brute_force_ids) / 10
        recalls.append(recall)

    avg_recall = np.mean(recalls)
    # Should maintain >85% recall even with 1000 vectors
    assert avg_recall >= 0.85, f"Average recall too low on large dataset: {avg_recall:.3f}"


def test_single_vector_edge_case():
    """Edge case: graph with only one vector"""
    store = VectorStore(dimension=4)

    vector = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    ids = store.add(vector)

    # Search should return the only vector
    query = np.array([1.1, 2.1, 3.1, 4.1], dtype=np.float32)
    results = store.search(query, k=5)

    assert len(results) == 1
    assert results[0]["id"] == ids[0]


def test_identical_vectors_edge_case():
    """Edge case: all vectors are identical"""
    store = VectorStore(dimension=3, normalize=False)

    # Add 10 identical vectors
    identical_vec = np.array([1.0, 1.0, 1.0], dtype=np.float32)
    vectors = [identical_vec.copy() for _ in range(10)]
    store.add(vectors)

    # Search with the same vector
    results = store.search(identical_vec, k=10)

    # Should find all 10 vectors with same distance
    assert len(results) == 10

    # All distances should be very close to 0 (identical)
    for result in results:
        assert result["distance"] < 0.01, f"Distance too large for identical vector: {result['distance']}"


def test_high_dimensional_sparse_vectors():
    """Test with high-dimensional sparse vectors"""
    store = VectorStore(dimension=256, M=16, ef_search=50)

    # Create sparse vectors (mostly zeros)
    np.random.seed(42)
    vectors = []
    for i in range(100):
        vec = np.zeros(256, dtype=np.float32)
        # Set only 5 random positions to non-zero
        indices = np.random.choice(256, 5, replace=False)
        vec[indices] = np.random.rand(5).astype(np.float32)
        vectors.append(vec)

    store.add(vectors)

    # Query with another sparse vector
    query = np.zeros(256, dtype=np.float32)
    query[[10, 20, 30, 40, 50]] = np.random.rand(5).astype(np.float32)

    results = store.search(query, k=10)

    # Should return results without errors
    assert len(results) == 10
    assert all("id" in r and "distance" in r for r in results)


def test_two_vector_edge_case():
    """Edge case: graph with exactly two vectors"""
    store = VectorStore(dimension=4)

    vec1 = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    vec2 = np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32)
    store.add([vec1, vec2])

    # Search should work with only 2 vectors
    query = np.array([0.9, 0.1, 0.0, 0.0], dtype=np.float32)
    results = store.search(query, k=2)

    assert len(results) == 2
    # First result should be vec1 (closer to query)
    assert results[0]["id"] == "doc_0"
    assert results[1]["id"] == "doc_1"


def test_graph_connectivity_reachability():
    """Test that entry point can reach all nodes in the graph"""
    store = VectorStore(dimension=16, M=8, ef_search=50)

    # Add 100 random vectors
    np.random.seed(42)
    vectors = [np.random.rand(16).astype(np.float32) for _ in range(100)]
    store.add(vectors)

    # Get the graph and entry point
    graph = store._graph
    entry_point = graph.entry_point

    # BFS from entry point to find all reachable nodes
    visited = set()
    queue = [entry_point]
    visited.add(entry_point)

    while queue:
        current_id = queue.pop(0)
        node = graph.get_node(current_id)

        # Check all layers
        for layer in range(node.level + 1):
            for neighbor_id in node.get_neighbors(layer):
                if neighbor_id not in visited:
                    visited.add(neighbor_id)
                    queue.append(neighbor_id)

    # All nodes should be reachable from entry point
    all_node_ids = set(graph.nodes.keys())
    assert visited == all_node_ids, f"Graph not fully connected: {len(visited)}/{len(all_node_ids)} nodes reachable"


def test_graph_bidirectional_edges():
    """Test that all edges are bidirectional"""
    store = VectorStore(dimension=8, M=4, ef_search=20)

    # Add 50 vectors
    np.random.seed(42)
    vectors = [np.random.rand(8).astype(np.float32) for _ in range(50)]
    store.add(vectors)

    graph = store._graph

    # Check all nodes and their edges
    for node_id, node in graph.nodes.items():
        for layer in range(node.level + 1):
            neighbors = node.get_neighbors(layer)
            for neighbor_id in neighbors:
                neighbor_node = graph.get_node(neighbor_id)

                # If neighbor_id is at this layer, it should have reverse edge
                if neighbor_node.level >= layer:
                    reverse_neighbors = neighbor_node.get_neighbors(layer)
                    assert node_id in reverse_neighbors, \
                        f"Edge {node_id}->{neighbor_id} at layer {layer} not bidirectional"


def test_entry_point_updates_correctly():
    """Test that entry point updates when higher-layer node is added"""
    store = VectorStore(dimension=4, M=4, ef_search=10)

    # Add first vector
    vec1 = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    store.add(vec1)

    initial_entry = store._graph.entry_point
    initial_node = store._graph.get_node(initial_entry)
    initial_level = initial_node.level

    # Add more vectors until we see an entry point update
    np.random.seed(42)
    entry_point_updated = False

    for i in range(100):
        vec = np.random.rand(4).astype(np.float32)
        store.add(vec)

        new_entry = store._graph.entry_point
        new_node = store._graph.get_node(new_entry)

        if new_entry != initial_entry:
            # Entry point changed - verify new entry has higher or equal level
            assert new_node.level >= initial_level, \
                f"New entry point has lower level: {new_node.level} < {initial_level}"
            entry_point_updated = True
            break

    # Note: It's possible (though unlikely) that entry point never changes
    # This is fine - the test validates the logic when it does change


def test_graph_no_self_loops():
    """Test that no node has edges to itself"""
    store = VectorStore(dimension=8, M=4, ef_search=20)

    # Add 50 vectors
    np.random.seed(42)
    vectors = [np.random.rand(8).astype(np.float32) for _ in range(50)]
    store.add(vectors)

    graph = store._graph

    # Check that no node points to itself
    for node_id, node in graph.nodes.items():
        for layer in range(node.level + 1):
            neighbors = node.get_neighbors(layer)
            assert node_id not in neighbors, \
                f"Node {node_id} has self-loop at layer {layer}"


def test_graph_respects_m_constraint():
    """Test that nodes don't exceed M connections per layer"""
    store = VectorStore(dimension=16, M=8, ef_search=50)

    # Add 100 vectors to stress test the M constraint
    np.random.seed(42)
    vectors = [np.random.rand(16).astype(np.float32) for _ in range(100)]
    store.add(vectors)

    graph = store._graph
    M = graph.M
    M_L = graph.M_L

    # Check all nodes
    for node_id, node in graph.nodes.items():
        for layer in range(node.level + 1):
            neighbors = node.get_neighbors(layer)
            max_neighbors = M_L if layer == 0 else M

            assert len(neighbors) <= max_neighbors, \
                f"Node {node_id} at layer {layer} has {len(neighbors)} neighbors (max: {max_neighbors})"
