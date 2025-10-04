# DynHNSW Library Usage Guide

## Table of Contents
1. [Quick Start](#quick-start)
2. [Core Concepts](#core-concepts)
3. [Building Your Index](#building-your-index)
4. [Search Modes](#search-modes)
5. [Adaptive Learning](#adaptive-learning)
6. [Performance Monitoring](#performance-monitoring)
7. [Best Practices](#best-practices)
8. [Production Checklist](#production-checklist)

---

## Quick Start

### Installation
```bash
pip install -e .
```

### Minimal Example
```python
import numpy as np
from dynhnsw.hnsw.graph import HNSWGraph
from dynhnsw.hnsw.builder import HNSWBuilder
from dynhnsw.adaptive_hnsw import AdaptiveHNSWSearcher

# 1. Create index
graph = HNSWGraph(dimension=128, M=16)
builder = HNSWBuilder(graph)

for i, vector in enumerate(vectors):
    level = int(np.random.geometric(p=0.5)) - 1
    builder.insert(vector, node_id=i, level=min(level, 5))

# 2. Create searcher
searcher = AdaptiveHNSWSearcher(graph, ef_search=50)

# 3. Search
results = searcher.search(query, k=10)

# 4. Provide feedback (optional, for adaptation)
searcher.provide_feedback(query, result_ids, relevant_ids)
```

---

## Core Concepts

### HNSW Graph Structure
- **Hierarchical Layers**: Multi-layer graph where higher layers are sparser
- **Layer 0**: Contains all vectors (dense base layer)
- **Higher Layers**: Progressively fewer nodes for faster coarse search
- **M Parameter**: Max connections per node (16-64 typical)

### Adaptive Learning
- **Edge Weights**: Modify effective distance = weight × base_distance
- **Feedback Signal**: User indicates which results are relevant
- **Weight Update**: Exponential smoothing with learning rate
- **Temporal Decay**: Old patterns gradually fade (7-day half-life default)

### In-Memory Design
- All data stored in RAM for low latency
- No persistence layer (add your own if needed)
- Build index once, search many times
- Typical latency: <5ms for 1M vectors

---

## Building Your Index

### Step 1: Create Graph
```python
from dynhnsw.hnsw.graph import HNSWGraph

graph = HNSWGraph(
    dimension=128,          # Vector dimension
    M=16,                   # Max edges at higher layers (16-64 recommended)
    M_L=None,              # Max edges at layer 0 (default: 2*M)
    level_multiplier=1.0/np.log(2.0)  # Layer distribution
)
```

**Parameter Guidelines:**
- `dimension`: Must match your vector embeddings
- `M`: Higher = better recall, slower build (16-32 for balanced)
- `M_L`: Layer 0 connectivity (2*M is good default)

### Step 2: Insert Vectors
```python
from dynhnsw.hnsw.builder import HNSWBuilder
import numpy as np

builder = HNSWBuilder(graph)

for i, vector in enumerate(vectors):
    # Assign layer using geometric distribution
    level = int(np.random.geometric(p=0.5)) - 1
    level = min(level, 5)  # Cap at reasonable max

    builder.insert(vector, node_id=i, level=level)
```

**Best Practices:**
- Insert in batches if you have millions of vectors
- Use consistent `node_id` mapping (e.g., database IDs)
- Layer assignment: geometric(p=0.5) works well empirically
- Cap max level at 5-10 to avoid very sparse top layers

### Step 3: Verify Index
```python
print(f"Indexed: {graph.size()} vectors")
print(f"Layers: {graph.get_max_level() + 1}")
print(f"Entry point: {graph.entry_point}")
```

---

## Search Modes

### 1. Static HNSW (Baseline)
```python
from dynhnsw.hnsw.searcher import HNSWSearcher

searcher = HNSWSearcher(graph, ef_search=50)
results = searcher.search(query, k=10)

# Results: [(node_id, distance), ...]
```

**When to use:**
- No need for adaptation
- Baseline comparison
- Maximum speed (no overhead)

**Parameters:**
- `ef_search`: Candidate list size (higher = better recall, slower)
- `k`: Number of results to return
- Typical: ef_search=50-100 for balanced recall/speed

### 2. Adaptive HNSW (Learning)
```python
from dynhnsw.adaptive_hnsw import AdaptiveHNSWSearcher

searcher = AdaptiveHNSWSearcher(
    graph,
    ef_search=50,
    learning_rate=0.05,     # How fast to adapt (0.01-0.1)
    enable_adaptation=True  # Can disable for A/B testing
)

results = searcher.search(query, k=10)
```

**When to use:**
- User feedback available
- Query patterns evolve over time
- Want to improve results incrementally
- Need performance monitoring

---

## Adaptive Learning

### Feedback Collection
```python
# After search
results = searcher.search(query, k=10)
result_ids = [r[0] for r in results]

# User indicates relevant results (your implementation)
relevant_ids = get_user_feedback(results)  # Returns set of relevant IDs

# Provide feedback to system
searcher.provide_feedback(query, result_ids, relevant_ids)
```

**Feedback Strategies:**
1. **Explicit**: User clicks "relevant/not relevant"
2. **Implicit**: Track clicks, dwell time, conversions
3. **Hybrid**: Combine explicit and implicit signals

### How Learning Works

**Mathematical Flow:**
1. User satisfaction = relevant_count / total_results
2. Reward signal = 1.2 - 0.4 × satisfaction
   - 100% satisfaction → reward = 0.8 (decrease weights 20%)
   - 50% satisfaction → reward = 1.0 (no change)
   - 0% satisfaction → reward = 1.2 (increase weights 20%)
3. Weight update: w_new = w_old × (1-α) + reward × α

**What Gets Learned:**
- Edges used in successful queries get lower weights (preferred paths)
- Edges used in unsuccessful queries get higher weights (avoided paths)
- Weights bounded [0.1, 10.0] to prevent extreme bias

### Temporal Decay
```python
# Weights automatically decay toward neutral (1.0) over time
# Default: 7-day half-life
learner = EdgeWeightLearner(
    learning_rate=0.05,
    decay_half_life_seconds=604800  # 7 days
)
```

**Why Decay?**
- User preferences change over time
- Prevents overfitting to old patterns
- Allows system to re-adapt to new patterns

---

## Performance Monitoring

### Recording Metrics
```python
# Compute your metrics
recall = compute_recall(retrieved, ground_truth, k)
precision = compute_precision(retrieved, ground_truth, k)
latency_ms = measure_latency()

# Record in system
searcher.record_performance(recall, precision, latency_ms)
```

### Automatic Baseline Setting
```python
# System automatically sets baseline from first ~10 queries
# Baseline used to detect performance degradation
```

### Degradation Detection & Reset
```python
# Automatic reset if performance drops >5% below baseline
# Reset is gradual (over 100 queries) to avoid disruption

# Check if reset is active
if searcher.reset_manager.is_resetting():
    print(f"Reset progress: {searcher.reset_manager.get_reset_progress():.0%}")
```

### System Statistics
```python
stats = searcher.get_statistics()

print(stats['graph'])        # nodes, max_level
print(stats['weights'])       # min, max, mean, count
print(stats['stability'])     # oscillation_rate, stability_score
print(stats['performance'])   # baseline_recall, current_recall
print(stats['reset'])         # active, progress
print(stats['feedback'])      # total_queries, avg_satisfaction
```

---

## Best Practices

### 1. Index Building
```python
# DO: Normalize vectors for cosine similarity
vectors = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)

# DO: Use appropriate M based on dataset size
# Small (<100K): M=8-16
# Medium (100K-1M): M=16-32
# Large (>1M): M=32-64

# DON'T: Insert vectors at random levels
# Use geometric distribution: level = int(np.random.geometric(p=0.5)) - 1
```

### 2. Search Quality
```python
# DO: Set ef_search >> k for better recall
ef_search = max(50, k * 5)  # At least 5x k

# DO: Monitor recall@k on ground truth
if recall < target_recall:
    # Increase ef_search
    searcher.ef_search = min(searcher.ef_search * 2, 500)

# DON'T: Use very low ef_search (<20)
# Results will be poor quality
```

### 3. Adaptive Learning
```python
# DO: Provide consistent feedback format
relevant_ids = set(user_marked_relevant)  # Always use set

# DO: Track satisfaction to monitor system health
satisfaction = len(relevant_ids) / len(result_ids)

# DON'T: Provide contradictory feedback for same query
# System will oscillate - use consistent criteria

# DON'T: Adapt too aggressively
learning_rate = 0.05  # Not 0.5 (too fast, unstable)
```

### 4. Stability Monitoring
```python
# DO: Check stability periodically
stats = searcher.get_statistics()
if stats['stability']['oscillating_edges'] > 0:
    print("Warning: Some edges are oscillating")
    # Consider reducing learning_rate

# DO: Monitor performance trends
if stats['performance']['degradation_severity'] > 0.1:
    print("Performance degraded >10%")
    # Reset will trigger automatically at 5%

# DON'T: Ignore stability warnings
# They indicate feedback quality issues
```

### 5. Memory Management
```python
# In-memory limits:
# 100K vectors × 128 dim × 4 bytes = ~50 MB
# 1M vectors × 128 dim × 4 bytes = ~500 MB
# Plus graph structure: ~100 bytes per node

# DO: Estimate memory before building
estimated_mb = (num_vectors * dimension * 4 + num_vectors * 100) / 1024 / 1024
print(f"Estimated memory: {estimated_mb:.1f} MB")

# DON'T: Exceed available RAM
# System will swap and become very slow
```

---

## Production Checklist

### Before Deployment

- [ ] **Index built successfully**
  ```python
  assert graph.size() == len(vectors)
  assert graph.entry_point is not None
  ```

- [ ] **Baseline performance measured**
  ```python
  # Run on validation set
  recalls = [compute_recall(...) for _ in range(100)]
  baseline_recall = np.mean(recalls)
  assert baseline_recall > target_recall  # e.g., 0.90
  ```

- [ ] **Search latency acceptable**
  ```python
  latencies = [measure_search_time(...) for _ in range(100)]
  p99_latency = np.percentile(latencies, 99)
  assert p99_latency < max_latency_ms  # e.g., 50ms
  ```

- [ ] **Adaptation tested** (if enabled)
  ```python
  # Simulate 1000 queries with feedback
  for query, ground_truth in test_set:
      results = searcher.search(query, k)
      searcher.provide_feedback(query, result_ids, ground_truth)

  stats = searcher.get_statistics()
  assert stats['stability']['stability_score'] > 0.8
  assert stats['weights']['mean'] > 0.5  # Not all penalized
  ```

- [ ] **Memory usage validated**
  ```python
  import psutil
  process = psutil.Process()
  memory_mb = process.memory_info().rss / 1024 / 1024
  assert memory_mb < max_memory_mb
  ```

### Monitoring in Production

```python
# Log key metrics every N queries
if query_count % 100 == 0:
    stats = searcher.get_statistics()

    logger.info(f"Queries: {stats['feedback']['total_queries']}")
    logger.info(f"Recall: {stats['performance']['current_recall']:.2%}")
    logger.info(f"Satisfaction: {stats['feedback']['avg_satisfaction']:.2%}")
    logger.info(f"Stability: {stats['stability']['stability_score']:.2%}")

    # Alert on degradation
    if stats['performance']['degradation_severity'] > 0.05:
        alert("Performance degraded >5%")

    # Alert on instability
    if stats['stability']['oscillation_rate'] > 0.1:
        alert("10% of updates causing oscillation")
```

### A/B Testing Adaptive vs Static

```python
# Group A: Static HNSW
static_searcher = HNSWSearcher(graph, ef_search=50)

# Group B: Adaptive HNSW
adaptive_searcher = AdaptiveHNSWSearcher(
    graph, ef_search=50, enable_adaptation=True
)

# Compare metrics after N queries
# - Recall@k
# - User satisfaction (CTR, dwell time, etc.)
# - Search latency
```

---

## Example: Production Setup

```python
import numpy as np
from dynhnsw.hnsw.graph import HNSWGraph
from dynhnsw.hnsw.builder import HNSWBuilder
from dynhnsw.adaptive_hnsw import AdaptiveHNSWSearcher

class VectorDatabase:
    """Production-ready vector database wrapper."""

    def __init__(self, vectors, dimension, enable_adaptation=True):
        # Build index
        self.graph = HNSWGraph(dimension=dimension, M=16)
        builder = HNSWBuilder(self.graph)

        for i, vector in enumerate(vectors):
            level = int(np.random.geometric(p=0.5)) - 1
            builder.insert(vector, node_id=i, level=min(level, 5))

        # Create searcher
        self.searcher = AdaptiveHNSWSearcher(
            self.graph,
            ef_search=50,
            learning_rate=0.05,
            enable_adaptation=enable_adaptation
        )

        # Baseline metrics
        self.query_count = 0

    def search(self, query, k=10):
        """Search for k nearest neighbors."""
        return self.searcher.search(query, k)

    def provide_feedback(self, query, result_ids, relevant_ids):
        """Learn from user feedback."""
        self.searcher.provide_feedback(query, result_ids, relevant_ids)
        self.query_count += 1

        # Periodic monitoring
        if self.query_count % 100 == 0:
            self._log_stats()

    def _log_stats(self):
        """Log system statistics."""
        stats = self.searcher.get_statistics()
        print(f"[Stats] Queries: {stats['feedback']['total_queries']}, "
              f"Recall: {stats['performance']['current_recall']:.2%}, "
              f"Stability: {stats['stability']['stability_score']:.2%}")

# Usage
vectors = np.random.randn(10000, 128).astype(np.float32)
vectors = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)

db = VectorDatabase(vectors, dimension=128)

# Search and learn loop
query = vectors[0]
results = db.search(query, k=10)
result_ids = [r[0] for r in results]
relevant_ids = {result_ids[0], result_ids[1]}  # User marks 2 as relevant

db.provide_feedback(query, result_ids, relevant_ids)
```

---

## Troubleshooting

### Low Recall
- Increase `ef_search` (try doubling)
- Increase `M` when building (rebuild index)
- Check vector normalization
- Verify query and documents are in same space

### High Latency
- Decrease `ef_search`
- Reduce index size (filter before indexing)
- Check if system is swapping (memory limit)

### Unstable Adaptation
- Reduce `learning_rate` (try 0.01)
- Check feedback consistency
- Increase `decay_half_life_seconds`
- Verify feedback signal quality

### Memory Issues
- Reduce number of vectors
- Reduce dimension (use PCA/compression)
- Decrease `M` (fewer edges)
- Consider approximate quantization

---

## Next Steps

- Run `examples/quickstart.py` for hands-on tutorial
- Run `examples/real_world_demo.py` for comprehensive demo
- See `INTEGRATION_SUMMARY.md` for technical details
- Check test files for advanced usage patterns
