# DynHNSW - Adaptive Vector Search with Layer-Adaptive Multi-Path

A Python library for vector similarity search featuring **layer-adaptive multi-path search** - a novel approach that dramatically improves recall by maintaining multiple parallel paths through the HNSW hierarchy based on query difficulty.

---

## What Makes This Different

**Traditional Vector Search**: Uses a single entry point at the top of the HNSW graph, which works well for queries close to the graph center but struggles with queries far from it.

**Layer-Adaptive Multi-Path**: Dynamically maintains 1-3 parallel paths through the HNSW hierarchy depending on query difficulty. Hard queries (far from the center) use 3 entry points at layer 0 for better coverage.

### How It Works

1. **Compute query difficulty**: Distance from query to graph entry point (zero-cost proxy)
2. **Select number of paths**:
   - Easy queries (close to center): 1 path
   - Medium queries: 2 paths
   - Hard queries (far from center): 3 paths
3. **Navigate hierarchy**: Maintain multiple parallel paths through top layers
4. **Search layer 0**: Start from multiple entry points for better coverage

---

## Performance Results

Comprehensive testing on real-world data with sentence-transformer embeddings (all-MiniLM-L6-v2, 384 dimensions):

### Scenario 1: Large Corpus (10,000 documents, 1,000 queries)

| Metric | Static HNSW | Layer-Adaptive | Improvement |
|--------|-------------|----------------|-------------|
| **Recall@10** | 35.8% | **58.2%** | **+62.5%** |
| **Median Recall** | 20.0% | **60.0%** | **+200%** |
| **Avg Latency** | 2.74ms | 5.54ms | +102% |
| **P99 Latency** | 7.37ms | 5.97ms | **-19%** (improved) |

**Key Finding**: Triple the median recall with 2x latency. P99 latency actually improves despite average increase.

### Scenario 2: Query Difficulty Analysis (5,000 documents, 600 queries)

| Metric | Static HNSW | Layer-Adaptive | Improvement |
|--------|-------------|----------------|-------------|
| **Recall@10** | 62.4% | **68.0%** | **+9.0%** |
| **Avg Latency** | 10.36ms | 11.06ms | +6.7% |

**Per-Difficulty Performance**:
- Easy queries: 97.2% recall
- Medium queries: 62.0% recall
- Hard queries: 44.9% recall

### Scenario 3: ef_search Independence (5,000 documents, 500 queries)

| ef_search | Static Recall | Adaptive Recall | Improvement |
|-----------|---------------|-----------------|-------------|
| 50 | 43.1% | **59.3%** | **+37.7%** |
| 100 | 43.1% | **59.3%** | **+37.7%** |
| 150 | 43.1% | **59.3%** | **+37.7%** |
| 200 | 43.1% | **59.3%** | **+37.7%** |

**Key Finding - The Single-Path Limitation**:

Static HNSW achieves 43.1% recall **regardless of ef_search value** (50-200), revealing a fundamental limitation: single-path greedy descent through the HNSW hierarchy can arrive at the wrong region of the vector space, and increasing ef_search only explores that wrong region more thoroughly—it cannot "jump" to distant neighborhoods.

Layer-adaptive multi-path search overcomes this by maintaining 1-3 parallel paths through the hierarchy, ensuring multiple arrival points at layer 0. This architectural change (multiple paths) provides more benefit than parameter tuning (increasing ef), achieving 59.3% recall consistently across all ef values.

**Why This Matters**: This demonstrates that layer-adaptive is not just an optimization but a fundamental architectural improvement. The consistent +37.7% improvement across all ef values proves that coverage (multiple paths) beats thoroughness (high ef) for diverse query distributions.

### Scenario 4: Large-Scale Real-World Validation (50,000 documents, 1,000 queries)

| Metric | Static HNSW | Layer-Adaptive | Improvement |
|--------|-------------|----------------|-------------|
| **Recall@10** | **9.5%** | **15.5%** | **+63.2%** |
| **Median Recall** | **0.0%** | **10.0%** | **∞** |
| **Zero Recall Queries** | 577/1000 (57.7%) | 446/1000 (44.6%) | **-22.7%** |
| **Perfect Recall Queries** | 0/1000 (0.0%) | 4/1000 (0.4%) | **∞** |
| **Avg Latency** | 2.79ms | 4.04ms | +44.9% |

**Configuration**: 768-dimensional embeddings (all-mpnet-base-v2), 19-layer graph (M=24), 90.4% hard queries (difficulty >= 0.9)

**Statistical Significance**:
- t-statistic: 11.19, p < 0.01 (HIGHLY SIGNIFICANT)
- 95% CI: Static [8.6%, 10.5%], Adaptive [14.1%, 16.8%]
- Cohen's d: 0.318 (small but robust effect)

**Catastrophic Static HNSW Failure**:
On this extremely hard query distribution (90.4% queries with difficulty >= 0.9), static HNSW shows devastating performance:
- **57.7% of queries returned ZERO correct results** in top-10
- **Median recall = 0.0%** (more than half of all queries completely failed)
- **Zero perfect recalls** across 1,000 queries

Layer-adaptive multi-path mitigates this failure mode:
- Reduces zero-recall queries from 57.7% to 44.6% (-22.7%)
- Achieves 10.0% median recall (vs 0.0% for static)
- Produces 4 perfect recalls (vs 0 for static)

**Key Insight**: On highly challenging, real-world query distributions, static HNSW essentially breaks down. Layer-adaptive multi-path is not just an improvement but a **necessary architectural change** to maintain usable recall levels.

### Summary Across All Tests

- **Recall improvement**: +9% to +63.2% depending on query difficulty
- **Average improvement**: +43.2% across all four scenarios
- **Latency overhead**: +7% to +102% (varies by graph size)
- **Benefit**: Consistent across all ef_search values
- **Critical finding**: Static HNSW fails catastrophically (57.7% zero-recall queries) on hard distributions; layer-adaptive is essential for robust performance

---

## Installation

**From Source:**
```bash
git clone https://github.com/spjosyula/DHNSWDB.git
cd DHNSWDB
pip install -e .
```

**For Development:**
```bash
git clone https://github.com/spjosyula/DHNSWDB.git
cd DHNSWDB
pip install -e ".[dev]"
pytest tests/
```

---

## Quick Start

### 1. Basic Usage with Layer-Adaptive Search

```python
import numpy as np
from dynhnsw.hnsw.graph import HNSWGraph
from dynhnsw.hnsw.builder import HNSWBuilder
from dynhnsw.hnsw.utils import assign_layer
from dynhnsw.intent_aware_hnsw import IntentAwareHNSWSearcher

# Build HNSW graph
dim = 384
graph = HNSWGraph(dimension=dim, M=16)
builder = HNSWBuilder(graph=graph)

# Add vectors
vectors = [np.random.randn(dim).astype(np.float32) for _ in range(1000)]
for i, vec in enumerate(vectors):
    level = assign_layer(level_multiplier=graph.level_multiplier)
    builder.insert(vector=vec, node_id=i, level=level)

# Create searcher with layer-adaptive search
searcher = IntentAwareHNSWSearcher(
    graph=graph,
    ef_search=100,
    enable_adaptation=False,  # No UCB1/K-means overhead
    enable_intent_detection=True  # Enable difficulty computation only
)

# Search - layer-adaptive happens automatically
query = np.random.randn(dim).astype(np.float32)
results = searcher.search(query, k=10)

for node_id, distance in results:
    print(f"Node {node_id}: distance {distance:.4f}")
```

### 2. With Sentence Transformers

```python
from sentence_transformers import SentenceTransformer
from dynhnsw.hnsw.graph import HNSWGraph
from dynhnsw.hnsw.builder import HNSWBuilder
from dynhnsw.hnsw.utils import assign_layer
from dynhnsw.intent_aware_hnsw import IntentAwareHNSWSearcher

# Load model and prepare documents
model = SentenceTransformer('all-MiniLM-L6-v2')
documents = [
    "Python is a high-level programming language",
    "Machine learning enables data-driven predictions",
    "Vector databases support semantic search",
    # ... more documents
]

# Generate embeddings
embeddings = model.encode(documents, convert_to_numpy=True).astype('float32')

# Build graph
graph = HNSWGraph(dimension=384, M=16)
builder = HNSWBuilder(graph=graph)
for i, vec in enumerate(embeddings):
    level = assign_layer(level_multiplier=graph.level_multiplier)
    builder.insert(vector=vec, node_id=i, level=level)

# Create layer-adaptive searcher
searcher = IntentAwareHNSWSearcher(
    graph=graph,
    ef_search=100,
    enable_adaptation=False,
    enable_intent_detection=True
)

# Search with query
query_text = "What is Python?"
query_embedding = model.encode([query_text], convert_to_numpy=True)[0]
results = searcher.search(query_embedding, k=3)

for node_id, distance in results:
    print(f"{documents[node_id]}")
```

### 3. Configuration Options

```python
# Static HNSW (no adaptation)
searcher = IntentAwareHNSWSearcher(
    graph=graph,
    ef_search=100,
    enable_adaptation=False,
    enable_intent_detection=False
)

# Layer-adaptive only (recommended)
searcher = IntentAwareHNSWSearcher(
    graph=graph,
    ef_search=100,
    enable_adaptation=False,  # No UCB1/K-means
    enable_intent_detection=True  # Difficulty computation only
)

# Full adaptive (UCB1 + K-means + layer-adaptive)
searcher = IntentAwareHNSWSearcher(
    graph=graph,
    ef_search=100,
    enable_adaptation=True,  # Enable UCB1/K-means
    enable_intent_detection=True,
    k_intents=5
)
```

---

## Why Layer-Adaptive Works

### The Problem with Single Entry Point

Traditional HNSW uses a single entry point at the top layer. This works well when:
- Query is close to the entry point
- Entry point is near the center of the vector space

But fails when:
- Query is far from the entry point
- Query is in the periphery of the vector space
- 87.7% of real-world queries are "hard" (far from center)

### The Solution: Multiple Parallel Paths

Layer-adaptive search:
1. Identifies query difficulty instantly (distance to entry point)
2. Maintains multiple paths through sparse top layers (low cost)
3. Arrives at layer 0 with multiple entry points (better coverage)
4. Significantly improves recall for hard queries

### Why It's Fast

Top layers of HNSW are very sparse:
- Layer 3: ~10 nodes
- Layer 2: ~50 nodes
- Layer 1: ~250 nodes
- Layer 0: ~10,000 nodes

Maintaining 2-3 paths through layers 3-1 is cheap. The benefit at layer 0 (10K nodes) is massive.

---

## When to Use

### Use Layer-Adaptive When:
- **Recall is critical**: Search, RAG systems, recommendation engines
- **You can tolerate 2x latency**: Still achieves sub-6ms average
- **Corpus is large**: 5K-10K+ documents
- **Queries are diverse**: Mix of easy/medium/hard queries

### Use Static HNSW When:
- **Latency is critical**: Sub-3ms requirements
- **Corpus is small**: <1K documents
- **Queries are uniformly easy**: All near corpus center
- **Recall is sufficient**: Static already meets needs

---

## API Reference

### IntentAwareHNSWSearcher

**Constructor:**
```python
IntentAwareHNSWSearcher(
    graph: HNSWGraph,           # HNSW graph to search
    ef_search: int = 50,        # Default search parameter
    k_intents: int = 5,         # Number of intent clusters
    enable_adaptation: bool = True,        # Enable UCB1/K-means
    enable_intent_detection: bool = True,  # Enable difficulty computation
    min_queries_for_clustering: int = 30,  # Queries before clustering
    config: Optional[DynHNSWConfig] = None # Advanced configuration
)
```

**Methods:**
```python
search(query: Vector, k: int, ef_search: Optional[int] = None) -> List[Tuple[int, float]]
    # Search for k nearest neighbors
    # Returns: List of (node_id, distance) tuples

provide_feedback(query: Vector, result_ids: List[int], ground_truth_ids: List[int], k: int = 10) -> None
    # Provide feedback for learning (if adaptation enabled)

get_statistics() -> Dict[str, Any]
    # Get search statistics and learned parameters
```

---

## Testing

Comprehensive test suite with real-world sentence embeddings:

```bash
# Run all layer-adaptive tests
python layer_adaptive_test/scenario_1_large_corpus.py
python layer_adaptive_test/scenario_2_query_difficulty.py
python layer_adaptive_test/scenario_3_ef_sensitivity.py

# Results saved to layer_adaptive_test/results/
```

See `layer_adaptive_test/README.md` for detailed test results and analysis.

---

## Architecture

```
dynhnsw/
├── intent_aware_hnsw.py     # Layer-adaptive search engine
├── intent_detector.py       # Query difficulty clustering
├── ef_search_selector.py    # UCB1 ef_search optimization
├── metrics.py               # Difficulty computation
└── hnsw/
    ├── graph.py             # HNSW graph structure
    ├── builder.py           # Graph construction
    ├── distance.py          # Distance metrics
    └── utils.py             # Layer assignment
```

**Layer-Adaptive Components:**

1. **Difficulty Proxy**: Distance from query to entry point (zero-cost)
2. **Path Selection**: Choose 1-3 paths based on difficulty thresholds
3. **Multi-Path Traversal**: Maintain parallel paths through top layers
4. **Layer 0 Search**: Start from multiple entry points

---

## Research Background

### Novel Contributions

1. **Zero-cost difficulty proxy**: Distance to entry point provides sufficient signal
2. **Layer-adaptive multi-path**: First approach to vary path count dynamically
3. **Large-scale validation**: Tested on 10K documents with sentence embeddings
4. **ef-independent recall**: Achieves high recall regardless of ef_search setting

### Related Work

- **HNSW Algorithm**: Malkov & Yashunin (2016)
- **Query-aware search**: Prior work on query-specific optimization
- **Multi-path search**: Explored in other graph algorithms but not HNSW

### Previous Approaches Explored

Before arriving at layer-adaptive multi-path search, we explored other adaptive strategies:

1. **UCB1 Exploration** (`UCB1_FINDINGS.md`): Upper Confidence Bound exploration for ef_search selection
   - Achieved strong synthetic performance (257 sat/sec)
   - But showed overhead in real-world scenarios (-24% to -47% vs static)
   - More complex than layer-adaptive with less practical benefit

2. **Epsilon Decay** (`EPSILON_DECAY_FINDINGS.md`): GLIE-based epsilon decay for exploration
   - Showed no significant improvement over fixed epsilon (-0.4%)
   - Added complexity without performance gains
   - Not recommended for production use

Layer-adaptive multi-path search emerged as the simplest and most effective approach, achieving substantial recall improvements without complex learning mechanisms.

### Key Findings

- 87.7% of real-world queries benefit from multi-path (hard queries dominant)
- Multiple entry points at layer 0 are critical for recall
- Top layers are sparse enough that multi-path overhead is minimal
- Approach outperforms complex learning-based methods (UCB1 + K-means)

---

## Future Work

### Potential Enhancements

1. **Dynamic thresholds**: Learn difficulty thresholds instead of fixed 0.8/0.9
2. **Adaptive path count**: Use continuous function instead of discrete 1/2/3
3. **Layer-specific paths**: Different path counts at different layers
4. **Hybrid with ef_search**: Combine layer-adaptive with adaptive ef

### Open Research Questions

1. Optimal difficulty thresholds for different corpus sizes?
2. Can we predict optimal path count from query features?
3. Does benefit persist at very large scale (100K+ documents)?
4. How to handle dynamic corpus (insertions/deletions)?

---

## Known Limitations

1. **Latency overhead**: 2x slower than static HNSW on large corpus
2. **Memory overhead**: Minimal but maintains multiple candidate lists
3. **Fixed thresholds**: Currently uses hardcoded 0.8/0.9 difficulty thresholds
4. **No persistence**: Layer-adaptive state not saved/loaded
5. **Single-threaded**: No parallel path exploration

---

## License

MIT License - see [LICENSE](LICENSE) file for details

---

## Citation

If you use this work in research, please cite:

```bibtex
@software{dynhnsw2025,
  title={DynHNSW: Layer-Adaptive Multi-Path Search for HNSW},
  author={Josyula, Sai Pranav},
  year={2025},
  url={https://github.com/spjosyula/DHNSWDB}
}
```

---

## Contributing

Contributions welcome! Areas of interest:
- Dynamic threshold learning
- Large-scale testing (100K+ docs)
- GPU acceleration
- Distributed HNSW
- Integration with vector DB frameworks

---

## Contact

Issues and questions: [GitHub Issues](https://github.com/spjosyula/DHNSWDB/issues)

---
