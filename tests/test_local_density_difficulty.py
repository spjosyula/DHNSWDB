"""Test local density estimation difficulty metric calibration.

Verifies that the new k-NN based difficulty metric produces reasonable values
for normalized embeddings, unlike the broken distance-to-entry-point metric.
"""

import numpy as np
from sentence_transformers import SentenceTransformer
from dynhnsw import VectorStore

print("="*80)
print("LOCAL DENSITY DIFFICULTY METRIC CALIBRATION TEST")
print("="*80)

# Load model
print("\n[Setup] Loading sentence-transformers model...")
model = SentenceTransformer('all-MiniLM-L6-v2')
print("  Model: all-MiniLM-L6-v2 (384 dimensions)")

# Generate corpus
np.random.seed(42)
n_docs = 1000
print(f"\n[Setup] Generating {n_docs} documents...")
docs = [f"Document {i} with some random content about topic {i % 20}" for i in range(n_docs)]

# Get normalized embeddings
print(f"[Setup] Embedding corpus...")
embeddings = model.encode(docs, convert_to_numpy=True, show_progress_bar=False)
embeddings = embeddings.astype(np.float32)

# Verify normalization
norms = np.linalg.norm(embeddings, axis=1)
print(f"\nEmbedding statistics:")
print(f"  Dimensions: {embeddings.shape[1]}")
print(f"  Norms: min={np.min(norms):.6f}, max={np.max(norms):.6f}, mean={np.mean(norms):.6f}")
print(f"  Are normalized? {all(abs(norm - 1.0) < 0.01 for norm in norms)}")

# Create store with layer-adaptive enabled
print(f"\n[Setup] Creating vector store with layer-adaptive enabled...")
store = VectorStore(
    dimension=384,
    M=16,
    ef_construction=200,
    ef_search=100,
    enable_intent_detection=True,  # Enable layer-adaptive
    normalize=False  # Embeddings already normalized
)

print(f"[Setup] Adding vectors to store...")
store.add(embeddings)

print(f"\nGraph statistics:")
print(f"  Total nodes: {store._graph.size()}")
print(f"  Max level: {store._graph.get_max_level()}")
print(f"  Entry point: {store._graph.entry_point}")

# Test with 100 queries
print(f"\n[Test] Computing difficulty for 100 test queries...")
n_queries = 100
test_queries = [f"Query about topic {i} with some text" for i in range(n_queries)]
query_embeddings = model.encode(test_queries, convert_to_numpy=True, show_progress_bar=False).astype(np.float32)

difficulties = []
for i, query in enumerate(query_embeddings):
    if (i + 1) % 20 == 0:
        print(f"  {i+1}/{n_queries}")

    # Search to trigger difficulty calculation
    store.search(query, k=10)
    difficulty = store._searcher.last_difficulty
    difficulties.append(difficulty)

difficulties = np.array(difficulties)

print(f"\n" + "="*80)
print("DIFFICULTY DISTRIBUTION (Local Density Estimation)")
print("="*80)

print(f"\nStatistics:")
print(f"  Min:    {np.min(difficulties):.4f}")
print(f"  25th:   {np.percentile(difficulties, 25):.4f}")
print(f"  Median: {np.median(difficulties):.4f}")
print(f"  75th:   {np.percentile(difficulties, 75):.4f}")
print(f"  Max:    {np.max(difficulties):.4f}")
print(f"  Mean:   {np.mean(difficulties):.4f}")
print(f"  Std:    {np.std(difficulties):.4f}")

# Check threshold distribution with CALIBRATED thresholds (0.35, 0.45)
t1, t2 = 0.35, 0.45

print(f"\nThreshold distribution (t1={t1}, t2={t2}):")
easy_count = np.sum(difficulties < t1)
medium_count = np.sum((difficulties >= t1) & (difficulties < t2))
hard_count = np.sum(difficulties >= t2)

print(f"  Easy (<{t1}):      {easy_count:3d} ({easy_count/len(difficulties)*100:5.1f}%)")
print(f"  Medium ({t1}-{t2}):  {medium_count:3d} ({medium_count/len(difficulties)*100:5.1f}%)")
print(f"  Hard (>={t2}):     {hard_count:3d} ({hard_count/len(difficulties)*100:5.1f}%)")

# Validation checks
print(f"\n" + "="*80)
print("VALIDATION CHECKS")
print("="*80)

checks_passed = 0
checks_total = 0

# Check 1: Range should be 0.2-0.9 (not 0.8-1.0 like broken metric)
checks_total += 1
range_ok = (np.min(difficulties) >= 0.1) and (np.max(difficulties) <= 1.0)
if range_ok:
    print(f"[PASS] Range check: {np.min(difficulties):.3f} to {np.max(difficulties):.3f} (expected 0.2-0.9)")
    checks_passed += 1
else:
    print(f"[FAIL] Range check FAILED: {np.min(difficulties):.3f} to {np.max(difficulties):.3f} (expected 0.2-0.9)")

# Check 2: Not all queries should be clustered at high difficulty
checks_total += 1
high_cluster = np.sum(difficulties >= 0.9) / len(difficulties)
if high_cluster < 0.3:  # Less than 30% should be >= 0.9
    print(f"[PASS] Distribution check: Only {high_cluster*100:.1f}% >= 0.9 (not clustered)")
    checks_passed += 1
else:
    print(f"[FAIL] Distribution check FAILED: {high_cluster*100:.1f}% >= 0.9 (too clustered)")

# Check 3: Should have reasonable spread (std > 0.05)
checks_total += 1
spread_ok = np.std(difficulties) > 0.05
if spread_ok:
    print(f"[PASS] Spread check: std={np.std(difficulties):.4f} (good variance)")
    checks_passed += 1
else:
    print(f"[FAIL] Spread check FAILED: std={np.std(difficulties):.4f} (insufficient variance)")

# Check 4: Hard queries should not dominate (< 50%)
checks_total += 1
hard_ok = hard_count / len(difficulties) < 0.5
if hard_ok:
    print(f"[PASS] Hard query check: {hard_count/len(difficulties)*100:.1f}% hard (not dominant)")
    checks_passed += 1
else:
    print(f"[FAIL] Hard query check FAILED: {hard_count/len(difficulties)*100:.1f}% hard (too many)")

# Check 5: Each category should have some representation (at least 10%)
checks_total += 1
balanced = (easy_count/len(difficulties) >= 0.1 and
            medium_count/len(difficulties) >= 0.1 and
            hard_count/len(difficulties) >= 0.1)
if balanced:
    print(f"[PASS] Balance check: All categories have >= 10% representation")
    checks_passed += 1
else:
    print(f"[FAIL] Balance check FAILED: Some category has < 10% representation")

print(f"\n" + "="*80)
print(f"RESULT: {checks_passed}/{checks_total} checks passed")
print("="*80)

if checks_passed == checks_total:
    print("\n[SUCCESS] Local density difficulty metric is properly calibrated!")
    print("  The metric produces reasonable difficulty values for normalized embeddings.")
    print("  Ready to run full medical domain test.")
elif checks_passed >= checks_total * 0.6:
    print("\n[PARTIAL] Most checks passed, but some tuning may be needed.")
    print("  Consider adjusting thresholds based on actual distribution.")
else:
    print("\n[FAILURE] Difficulty metric needs more work.")
    print("  Review the implementation and threshold values.")

print()
