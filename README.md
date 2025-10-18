# DynHNSW - Experimental Query-Adaptive Vector Search

A research project exploring dynamic search strategies for vector similarity search using HNSW (Hierarchical Navigable Small World graphs).

## Status: Active Research

This is an experimental library currently under development. While the core HNSW implementation works reliably, the adaptive search features are still being refined and do not yet consistently outperform standard HNSW. The intent here is to experiment on multiple strategies and learn. So the README will be updated constantly.

## What This Project Does

DynHNSW attempts to improve vector search by adapting search behavior based on query difficulty. The goal is to automatically detect when a query is "hard" (requiring more thorough search) versus "easy" (where a simple search suffices), and allocate computational resources accordingly.

Think of it like a smart search assistant:
- For easy queries near dense clusters of similar items, use a fast single-path search
- For hard queries in sparse regions, use multiple search paths for better coverage
- Automatically classify query difficulty without manual tuning

## Why This Matters: Medical Information Retrieval

Consider a pharmacist searching a medical database containing thousands of drug information documents. When they query "best medicine for headache," the system needs to:

1. **High Recall**: Find all relevant medications, not miss important options (critical for patient safety)
2. **Reasonable Latency**: Return results quickly enough for practical use (seconds, not minutes)
3. **Trade-off**: Accept some latency overhead if it significantly improves recall

Missing a relevant drug could have serious consequences. Standard vector search (HNSW) is fast but sometimes misses results, especially for queries far from the database center. DynHNSW experiments with adaptive strategies to reduce these missed results.

## Current Approach: Local Density Estimation

The latest implementation uses local density estimation to classify query difficulty:

**How it works:**
1. For each query, find the 50 nearest neighbors in the vector database
2. Compute the mean distance to these neighbors
3. Low mean distance = dense region = easy query (use 1 search path)
4. High mean distance = sparse region = hard query (use 3 search paths)

**Path-based search:**
- Standard HNSW: Single entry point, may miss peripheral results
- Layer-adaptive: Maintains 1-3 parallel paths through the graph hierarchy
- More paths = better coverage but higher latency

## Installation

```bash
git clone https://github.com/spjosyula/DHNSWDB.git
cd DHNSWDB
pip install -e .
```

**Requirements:**
- Python 3.10+
- numpy
- sentence-transformers (for text embeddings)

## Basic Usage

```python
import numpy as np
from dynhnsw import VectorStore

# Create a vector store
store = VectorStore(
    dimension=384,                    # Vector size
    M=16,                             # HNSW connections per node
    ef_construction=200,              # Build quality (higher = better)
    ef_search=100,                    # Search quality (higher = better)
    enable_intent_detection=True      # Enable adaptive search
)

# Add vectors (can be any 384-dimensional vectors)
vectors = np.random.randn(1000, 384).astype('float32')
store.add(vectors)

# Search for similar vectors
query = np.random.randn(384).astype('float32')
results = store.search(query, k=10)

for result in results:
    print(f"ID: {result['id']}, Distance: {result['distance']:.4f}")
```

## Example: Medical Document Search

```python
from sentence_transformers import SentenceTransformer
from dynhnsw import VectorStore

# Load a model that converts text to vectors
model = SentenceTransformer('all-MiniLM-L6-v2')

# Example medical documents
documents = [
    "Ibuprofen is an NSAID used for pain relief and fever reduction",
    "Acetaminophen is recommended for mild to moderate pain",
    "Aspirin is used for pain relief and as a blood thinner",
    # ... thousands more documents
]

# Convert documents to vectors (embeddings)
embeddings = model.encode(documents, convert_to_numpy=True)

# Store vectors with adaptive search enabled
store = VectorStore(dimension=384, enable_intent_detection=True)
store.add(embeddings)

# Pharmacist searches for headache treatment
query = "best medicine for headache"
query_vector = model.encode([query], convert_to_numpy=True)[0]
results = store.search(query_vector, k=10)

# Show results
for result in results:
    doc_idx = int(result['id'].split('_')[1])
    print(f"{documents[doc_idx]}")
    print(f"  Relevance score: {1 - result['distance']:.3f}\n")
```

## Configuration Options

**Static HNSW (baseline, no adaptation):**
```python
store = VectorStore(
    dimension=384,
    enable_intent_detection=False  # Just use standard HNSW
)
```

**Layer-Adaptive (experimental):**
```python
store = VectorStore(
    dimension=384,
    enable_intent_detection=True   # Use adaptive multi-path search
)
```

## Current Performance

**Test: 5,000 medical documents, 300 queries**

| Metric | Static HNSW | Layer-Adaptive | Change |
|--------|-------------|----------------|--------|
| Recall | 98.93% | 97.87% | -1.06% |
| Latency | 7.8ms | 27.8ms | +254% |

**Current Issue**: On this corpus, all queries were classified as "easy" because the vector space is uniformly dense. Layer-adaptive defaulted to single-path search (same as static) but incurred overhead from difficulty computation. The adaptive mechanism needs improvement.

**What's working:**
- Local density estimation produces sensible difficulty scores
- No longer misclassifying 80% of queries as "hard" (previous bug)
- Correctly identifies when multi-path is not needed

**What needs work:**
- Better thresholds for different corpus sizes
- Performance benefits only appear on large, heterogeneous corpora
- Difficulty computation adds latency overhead
- Need more sophisticated difficulty metrics

## When to Use This Library

**Recommended for:**
- Experimentation and research on adaptive search
- Learning about HNSW and vector search internals
- Small to medium projects where you control the code

**Not recommended for:**
- Production systems requiring proven reliability
- Latency-critical applications (standard HNSW is faster)
- Very large scale (100K+ documents) without testing first

**Use static HNSW (set `enable_intent_detection=False`) if:**
- You need predictable, fast performance
- Your corpus is small (<5K documents) or uniform
- You are okay with standard HNSW recall rates

## API Reference

### VectorStore

**Constructor:**
```python
VectorStore(
    dimension: int,                  # Vector dimensionality (required)
    max_elements: int = 10000,       # Maximum number of vectors
    M: int = 16,                     # HNSW edges per node (typical: 16-32)
    ef_construction: int = 200,      # Build quality (higher = better graph)
    ef_search: int = 100,            # Search quality (higher = more recall)
    normalize: bool = True,          # Normalize vectors to unit length
    enable_intent_detection: bool = True  # Enable adaptive search
)
```

**Methods:**
```python
# Add vectors to the store
add(vectors, ids=None, metadata=None) -> List[str]

# Search for k nearest neighbors
search(query, k=10, ef_search=None) -> List[dict]
    # Returns: [{"id": str, "distance": float, "vector": array, "metadata": dict}, ...]

# Delete vectors by ID (soft delete)
delete(ids: List[str]) -> None

# Get store statistics
get_statistics() -> dict

# Save/load to disk
save(filepath: str) -> None
VectorStore.load(filepath: str) -> VectorStore
```

## How It Works

### 1. Vector Embeddings

Text or images are converted to high-dimensional vectors (embeddings) where similar items are close together. For example, "headache medicine" and "pain relief medication" would have similar vectors.

### 2. HNSW Index

Vectors are organized into a hierarchical graph structure:
- **Layer 0** (bottom): Contains all vectors, densely connected
- **Higher layers**: Sparse "highway" connections for fast navigation
- **Entry point**: Top-level node where all searches begin

### 3. Query Difficulty Estimation

When a query comes in, the system estimates difficulty:
```python
# Find 50 nearest neighbors to the query
neighbors = find_knn(query, k=50)

# Compute mean distance to these neighbors
distances = [distance(query, neighbor) for neighbor in neighbors]
difficulty = mean(distances)

# Low difficulty (0.2-0.35): Dense region, easy query
# Medium difficulty (0.35-0.45): Use 2 search paths
# High difficulty (0.45+): Sparse region, use 3 search paths
```

### 4. Adaptive Search

Based on difficulty, the search uses different strategies:

**Easy queries (1 path):**
- Single descent through graph hierarchy
- Fast, works well when query is near database center

**Hard queries (3 paths):**
- Maintain 3 parallel paths through hierarchy
- Converge to 3 different entry points at layer 0
- Search from all 3 entry points and merge results
- Better coverage but 3x overhead

## Project Structure

```
dynhnsw/
├── vector_store.py              # Main user API
├── intent_aware_hnsw.py         # Adaptive search logic
├── adaptive_thresholds.py       # Threshold learning (experimental)
├── hnsw/
│   ├── graph.py                 # HNSW graph structure
│   ├── builder.py               # Graph construction
│   ├── distance.py              # Distance metrics (cosine, etc.)
│   └── utils.py                 # Layer assignment, neighbor selection
└── config.py                    # Configuration classes

tests/                           # Unit tests
legacy_experiment_tests/         # Prior approaches (UCB1, epsilon decay)
```

## Testing

```bash
# Run unit tests
pytest tests/

# Test difficulty calibration
python test_local_density_difficulty.py

# Test on medical domain
python test_medical_domain.py
```

## Known Limitations

1. **Performance not proven**: Adaptive search doesn't consistently beat standard HNSW
2. **Latency overhead**: Difficulty computation adds 10-50ms per query
3. **Threshold tuning**: Default thresholds (0.35, 0.45) may not fit all corpora
4. **Small corpus bias**: Works best on large (50K+), diverse datasets
5. **In-memory only**: No built-in persistence (use save/load)
6. **Single-threaded**: No parallel search execution

## Research Goals

This project explores:
- Can we classify query difficulty cheaply and accurately?
- When does multi-path search actually help?
- What difficulty metrics work for normalized embeddings?
- How to learn optimal thresholds automatically?

**Current findings:**
- Local density (mean k-NN distance) is a better metric than distance-to-entry-point
- Uniform corpora don't benefit from adaptive search
- Overhead of difficulty computation must be minimized
- Need better understanding of when multi-path helps

**Areas needing improvement:**
- Faster difficulty computation (currently 50-NN search)
- Better threshold calibration strategies
- Testing on larger, more diverse datasets
- Parallel multi-path execution
- More sophisticated difficulty metrics

## Acknowledgments

Built on HNSW algorithm by Malkov & Yashunin (2016).
Inspired by work on query-adaptive search and vector databases.


