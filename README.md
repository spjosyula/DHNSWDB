# DynHNSW - Dynamic Intent-Aware Vector Database

<<<<<<< HEAD
A research-focused Python library for vector similarity search with **adaptive, intent-aware indexing**. Unlike traditional vector databases, DynHNSW learns from query patterns and feedback to optimize search behavior for different user intents.

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## What Makes This Different?

**Traditional Vector DBs**: Same search strategy for every query
**DynHNSW**: Adapts search based on detected query intent

```
Query Type A (exploratory) → Entry Point A → Optimized for broad recall
Query Type B (precise)     → Entry Point B → Optimized for specific matches
```

The system:
1. **Detects query intent** using clustering on query embeddings
2. **Learns optimal entry points** per intent from user feedback
3. **Adapts search** to start from intent-specific locations in the graph
4. **Maintains stability** even with noisy feedback
=======
Experimenting with **smart, query-aware indexes**. Instead of being static, this index can adjust start-point behavior based on the query and learn from feedback.  

Basically a normal HNSW graph, but that understands a more efficient starting point by learning through user queries.
>>>>>>> dcd13bb93b2394aea4075416cda733c88b904a22

---

## Quick Start

### Installation

**Option 1: Install from GitHub (Recommended)**
```bash
pip install git+https://github.com/spjosyula/DHNSWDB.git
```

**Option 2: Install from Source**
```bash
git clone https://github.com/spjosyula/DHNSWDB.git
cd DHNSWDB
pip install -e .
```

**Option 3: Local Development**
```bash
# Clone the repo
git clone https://github.com/spjosyula/DHNSWDB.git
cd DHNSWDB

# Install with dev dependencies
pip install -e ".[dev]"

# Run tests to verify installation
pytest tests/
```

---

## Usage Example

### Basic Vector Search

```python
import numpy as np
from dynhnsw import VectorStore

# Create store (384 dimensions for all-MiniLM-L6-v2 embeddings)
store = VectorStore(
    dimension=384,
    M=16,                    # Max connections per node
    ef_search=50,            # Search quality parameter
    enable_intent_detection=True  # Enable adaptive learning
)

# Add vectors (pre-embedded)
vectors = [
    np.random.rand(384).astype(np.float32) for _ in range(1000)
]
ids = store.add(vectors)

# Search
query = np.random.rand(384).astype(np.float32)
results = store.search(query, k=10)

for result in results:
    print(f"ID: {result['id']}, Distance: {result['distance']:.4f}")
```

### With Text Embeddings

```python
from sentence_transformers import SentenceTransformer
from dynhnsw import VectorStore

# Initialize embedding model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Create store (dimension must match model output)
store = VectorStore(dimension=384)

# Add documents
documents = [
    "The cat sleeps on the mat",
    "A dog plays in the park",
    "Machine learning is fascinating",
    "Vector databases enable semantic search"
]

# Convert to embeddings and add
embeddings = model.encode(documents, convert_to_numpy=True)
doc_ids = store.add(embeddings, ids=[f"doc_{i}" for i in range(len(documents))])

# Search with a query
query_text = "Where is the cat?"
query_embedding = model.encode([query_text], convert_to_numpy=True)[0]
results = store.search(query_embedding, k=3)

for result in results:
    print(f"{result['id']}: {documents[int(result['id'].split('_')[1])]}")
```

### With Feedback (Adaptive Learning)

```python
# Search
results = store.search(query, k=10)

# User marks results 0, 2, 5 as relevant
relevant_ids = [results[i]['id'] for i in [0, 2, 5]]

# Provide feedback to improve future searches
store.provide_feedback(relevant_ids=relevant_ids)

# Next search will use learned entry points for this query intent
```

---

## Research Context

### Core Innovation

**Problem**: Traditional HNSW always searches from the same entry point, regardless of query type.

**Solution**: Detect query intent and learn optimal entry points per intent through feedback.

### Validated Features

**Intent Detection**: K-means clustering on query vectors (>70% accuracy)
**Entry Point Learning**: Exponential moving average on feedback scores
**Learning Convergence**: Scores stabilize after ~30-50 queries
**Robustness**: Stable with 30% noisy feedback

### Performance Trade-offs

Based on A/B testing with synthetic clustered data:

| Metric | Adaptive | Static | Notes |
|--------|----------|--------|-------|
| Recall@10 | 81% | 90% | -9% trade-off for intent optimization |
| Satisfaction | 100% | 100% | Intent-matching maintained |
| Stability | High | High | Both stable |

**Interpretation**: Adaptive mode trades ~9% global recall for better intent-specific matching. This is acceptable when user intent matters more than absolute retrieval.

---

## API Reference

### VectorStore

**Constructor**
```python
VectorStore(
    dimension: int,                  # Vector dimensionality
    max_elements: int = 10000,       # Max vectors to store
    ef_construction: int = 200,      # Build quality (higher = better, slower)
    M: int = 16,                     # Max connections per node
    ef_search: int = 50,             # Search quality (higher = better recall, slower)
    normalize: bool = True,          # Normalize vectors (recommended for cosine)
    enable_intent_detection: bool = True,  # Enable adaptive learning
    k_intents: int = 5,              # Number of intent clusters
    learning_rate: float = 0.1,      # Entry point learning rate
    min_queries_for_clustering: int = 30  # Queries before clustering starts
)
```

**Methods**

```python
# Add vectors
add(vectors, ids=None, metadata=None) -> List[str]

# Search
search(query, k=10, ef_search=None) -> List[dict]

# Provide feedback
provide_feedback(relevant_ids: Union[List[str], Set[str]]) -> None

# Delete vectors (soft delete)
delete(ids: Union[str, List[str]]) -> None

# Get statistics
get_statistics() -> Dict[str, Any]

# Save/Load
save(filepath: str) -> None
load(filepath: str) -> VectorStore  # Class method
```

**Important Notes**

**This library operates on pre-embedded vectors**. You must convert text to vectors using an embedding model (like sentence-transformers) before adding to the store.

See the "With Text Embeddings" example above for how to integrate with sentence-transformers.

---

## Architecture

```
DynHNSW/
├── dynhnsw/
│   ├── vector_store.py          # Main API
│   ├── intent_aware_hnsw.py     # Adaptive search engine
│   ├── intent_detector.py       # K-means intent clustering
│   ├── entry_point_selector.py  # Entry point learning
│   ├── feedback.py              # Feedback collection
│   └── hnsw/
│       ├── graph.py             # HNSW graph structure
│       ├── builder.py           # Graph construction
│       └── distance.py          # Similarity metrics
└── tests/
    ├── test_vector_store.py     # Core functionality tests
    └── test_intent_learning.py  # Adaptive learning validation
```

**Key Components:**

1. **HNSW Graph** (`hnsw/`): Custom implementation of Hierarchical Navigable Small World graph
2. **Intent Detection** (`intent_detector.py`): K-means clustering on query vectors
3. **Entry Point Selection** (`entry_point_selector.py`): Epsilon-greedy learning of optimal entry points
4. **Feedback Loop** (`feedback.py`): Collect and process user feedback signals

---

## Testing

**Run all tests:**
```bash
pytest tests/ -v
```

**Run with coverage:**
```bash
pytest tests/ --cov=dynhnsw --cov-report=term-missing
```

**Test categories:**
- HNSW validation (recall vs brute force)
- Graph connectivity (bidirectional edges, reachability)
- Intent detection accuracy
- Learning convergence
- A/B testing (adaptive vs static)

**Current Test Status**: 34/34 passing (50% code coverage)

---

## Configuration Tips

### For High Recall (>90%)
```python
store = VectorStore(
    dimension=384,
    M=32,              # More connections
    ef_construction=400,  # Higher quality build
    ef_search=150,     # Thorough search
    enable_intent_detection=False  # Use static HNSW
)
```

### For Fast Search with Intent Adaptation
```python
store = VectorStore(
    dimension=384,
    M=16,
    ef_search=50,
    enable_intent_detection=True,
    k_intents=3,       # Fewer intent clusters
    learning_rate=0.15  # Faster learning
)
```

### For Large Datasets (100k+ vectors)
```python
store = VectorStore(
    dimension=384,
    M=24,
    ef_construction=300,
    ef_search=100,
    min_queries_for_clustering=50  # More data before clustering
)
```

---

## Known Limitations

1. **No text embedding module**: You must handle text→vector conversion externally
2. **In-memory only**: No built-in persistence (use `save()`/`load()` for serialization)
3. **Single-threaded**: No parallelization of search or indexing
4. **Intent detection is vector-based**: Detects query vector clusters, not semantic intent types
5. **Recall trade-off**: Adaptive mode may have ~9% lower recall for intent optimization

---

## License

MIT License - see [LICENSE](LICENSE) file for details

---

## Research References

- **HNSW**: Malkov & Yashunin (2016) - "Efficient and robust approximate nearest neighbor search using Hierarchical Navigable Small World graphs"
- **Intent-Aware Retrieval**: Query intent classification in information retrieval
- **Adaptive Systems**: Online learning and feedback optimization

---

**Built for research. Optimized for intent.** 
