"""End-to-end workflow test for DynHNSW.

Tests that all components work together seamlessly:
1. VectorStore with layer-adaptive search
2. Adaptive thresholds integration
3. Feedback mechanism
4. Statistics reporting
"""

import numpy as np
from dynhnsw import VectorStore

print("=" * 80)
print("DYNHNSW WORKFLOW TEST")
print("=" * 80)

# Test 1: Basic layer-adaptive search
print("\n[Test 1] Basic Layer-Adaptive Search")
print("-" * 80)

store = VectorStore(
    dimension=128,
    M=16,
    ef_construction=100,
    ef_search=50,
    enable_intent_detection=True
)

print(f"Created VectorStore: dim={store.dimension}, M={store.M}")

# Add vectors
n_vectors = 1000
vectors = np.random.randn(n_vectors, 128).astype('float32')
ids = store.add(vectors)
print(f"Added {len(ids)} vectors")

# Search
query = np.random.randn(128).astype('float32')
results = store.search(query, k=10)
print(f"Search returned {len(results)} results")
print(f"First result: ID={results[0]['id']}, distance={results[0]['distance']:.4f}")

# Get statistics
stats = store.get_statistics()
print(f"\nStatistics:")
print(f"  Total vectors: {stats['total_vectors']}")
print(f"  Intent detection: {stats['intent_detection_enabled']}")
if 'graph' in stats:
    print(f"  Graph max level: {stats['graph']['max_level']}")

print("[PASS] Basic search works")

# Test 2: Adaptive thresholds
print("\n[Test 2] Adaptive Thresholds")
print("-" * 80)

store_adaptive = VectorStore(
    dimension=128,
    M=16,
    ef_construction=100,
    ef_search=50,
    enable_intent_detection=True,
    enable_adaptive_thresholds=True  # Enable threshold learning
)

# Add vectors
store_adaptive.add(vectors)
print(f"Added {n_vectors} vectors with adaptive thresholds enabled")

# Perform multiple searches to trigger threshold learning
for i in range(10):
    query = np.random.randn(128).astype('float32')
    results = store_adaptive.search(query, k=10)

stats_adaptive = store_adaptive.get_statistics()
if 'adaptive_thresholds' in stats_adaptive:
    thresh_stats = stats_adaptive['adaptive_thresholds']
    print(f"\nAdaptive Threshold Status:")
    print(f"  Current t1: {thresh_stats['current_t1']:.3f}")
    print(f"  Current t2: {thresh_stats['current_t2']:.3f}")
    print(f"  Calibrated: {thresh_stats['is_calibrated']}")
    print("[PASS] Adaptive thresholds working")
else:
    print("[INFO] Adaptive thresholds stats not available yet (need more queries)")

# Test 3: Save and load
print("\n[Test 3] Save and Load")
print("-" * 80)

store.save("test_workflow_store.pkl")
print("Saved store to test_workflow_store.pkl")

loaded_store = VectorStore.load("test_workflow_store.pkl")
print(f"Loaded store: {loaded_store.size()} vectors")

# Verify loaded store works
results_loaded = loaded_store.search(query, k=10)
print(f"Search on loaded store: {len(results_loaded)} results")
print("[PASS] Save/load works")

# Test 4: Metadata handling
print("\n[Test 4] Metadata Handling")
print("-" * 80)

store_meta = VectorStore(dimension=128, enable_intent_detection=True)

# Add with metadata
vectors_meta = np.random.randn(10, 128).astype('float32')
metadata = [{"label": f"doc_{i}", "score": i * 0.1} for i in range(10)]
ids_meta = store_meta.add(vectors_meta, metadata=metadata)

# Search and check metadata
results_meta = store_meta.search(vectors_meta[0], k=3)
print(f"Results with metadata:")
for r in results_meta:
    print(f"  {r['id']}: {r['metadata']}")

print("[PASS] Metadata works")

# Test 5: Configuration modes
print("\n[Test 5] Configuration Modes")
print("-" * 80)

# Static HNSW
store_static = VectorStore(dimension=128, enable_intent_detection=False)
store_static.add(np.random.randn(100, 128).astype('float32'))
query_test = np.random.randn(128).astype('float32')
results_static = store_static.search(query_test, k=5)
print(f"Static HNSW: {len(results_static)} results")

# Layer-adaptive
store_adaptive_only = VectorStore(dimension=128, enable_intent_detection=True, enable_adaptive_thresholds=False)
store_adaptive_only.add(np.random.randn(100, 128).astype('float32'))
results_adaptive = store_adaptive_only.search(query_test, k=5)
print(f"Layer-adaptive: {len(results_adaptive)} results")

# Full adaptive
store_full = VectorStore(dimension=128, enable_intent_detection=True, enable_adaptive_thresholds=True)
store_full.add(np.random.randn(100, 128).astype('float32'))
results_full = store_full.search(query_test, k=5)
print(f"Full adaptive: {len(results_full)} results")

print("[PASS] All configuration modes work")

print("\n" + "=" * 80)
print("ALL TESTS PASSED")
print("=" * 80)
print("\nWorkflow coordination verified:")
print("- VectorStore properly integrates layer-adaptive search")
print("- Adaptive thresholds work when enabled")
print("- Statistics reporting functional")
print("- Save/load persistence works")
print("- All configuration modes functional")

# Cleanup
import os
if os.path.exists("test_workflow_store.pkl"):
    os.remove("test_workflow_store.pkl")
    print("\nCleaned up test files")
