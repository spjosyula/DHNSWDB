# DynHNSW - Dynamic Intent-Aware Vector Database

A research-focused Python library for vector similarity search with **adaptive, query-intent-aware search optimization**. Unlike traditional vector databases with static search parameters, DynHNSW learns optimal search configurations for different query patterns through reinforcement learning.

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## Key Innovation

**Traditional Vector DBs**: Fixed search parameters (ef_search) for all queries

**DynHNSW**: Learns optimal ef_search per query intent using Q-learning

The system:
1. Detects query intent through K-means clustering on query embeddings
2. Learns optimal search parameters (ef_search) per intent using Q-learning
3. Adapts search breadth based on query type (exploratory vs precise)
4. Improves search efficiency through feedback-driven optimization

---

## Performance Improvements

Based on real-world validation with sentence transformer embeddings:

### Efficiency Gains
- **5-8% improvement** in search efficiency (satisfaction per second)
- **Intent differentiation**: Different query types learn different optimal ef_search values
- **Adaptive convergence**: Q-values stabilize after 100-150 queries

### Learned Behavior
- Exploratory queries: ef_search = 130-150 (broader search)
- Precise queries: ef_search = 50-80 (faster, focused search)
- Static baseline: ef_search = 100 (fixed, one-size-fits-all)

### Validation Results
```
A/B Test: Adaptive vs Static (100 real documents, 160 queries)
---------------------------------------------------------
Average Efficiency:    120.98 vs 114.35 sat/sec  (+5.8%)
Intent Differentiation: 3 intents, 3 different ef values
Q-Learning Exploration: All ef candidates (50-250) tested
```

---

## Quick Start

### Installation

**From GitHub (Recommended):**
```bash
pip install git+https://github.com/spjosyula/DHNSWDB.git
```

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
pytest tests/  # Verify installation
```

### Basic Usage

#### 1. With Pre-Embedded Vectors
```python
import numpy as np
from dynhnsw import VectorStore

# Create store
store = VectorStore(
    dimension=384,              # Match your embedding dimension
    M=16,                       # Max connections per node
    ef_search=100,              # Default search parameter
    enable_intent_detection=True  # Enable Q-learning adaptation
)

# Add vectors (must be pre-embedded)
vectors = [np.random.rand(384).astype(np.float32) for _ in range(1000)]
ids = store.add(vectors)

# Search
query = np.random.rand(384).astype(np.float32)
results = store.search(query, k=10)

for result in results:
    print(f"ID: {result['id']}, Distance: {result['distance']:.4f}")
```

#### 2. With Text Embeddings (Sentence Transformers)
```python
from sentence_transformers import SentenceTransformer
from dynhnsw import VectorStore

# Load embedding model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Create store (dimension must match model)
store = VectorStore(dimension=384, enable_intent_detection=True)

# Prepare documents
documents = [
    "Python is a high-level programming language",
    "Machine learning enables data-driven predictions",
    "Vector databases support semantic search"
]

# Embed and add to store
embeddings = model.encode(documents, convert_to_numpy=True)
doc_ids = store.add(embeddings, ids=[f"doc_{i}" for i in range(len(documents))])

# Search with text query
query_text = "What is Python?"
query_embedding = model.encode([query_text], convert_to_numpy=True)[0]
results = store.search(query_embedding, k=3)

for result in results:
    doc_idx = int(result['id'].split('_')[1])
    print(f"{result['id']}: {documents[doc_idx]}")
```

#### 3. With Feedback Learning
```python
# Perform search
results = store.search(query_embedding, k=10)

# User indicates which results were relevant (e.g., clicked on results 0, 2, 5)
relevant_ids = [results[i]['id'] for i in [0, 2, 5]]

# Provide feedback to improve future searches
store.provide_feedback(relevant_ids=relevant_ids)

# Next search will use optimized ef_search for this query intent
next_results = store.search(another_query, k=10)
```

---

## How It Works

### 1. Intent Detection
- Uses K-means clustering on query embeddings
- Activates after minimum queries threshold (default: 30)
- Assigns each query to an intent cluster with confidence score

### 2. Q-Learning Optimization
- **Problem formulation**: Contextual multi-armed bandit
  - Context: Query intent (from K-means)
  - Actions: ef_search values [50, 75, 100, 150, 200, 250]
  - Reward: Efficiency = satisfaction / latency

- **Algorithm**: Action-value Q-learning
  - Q(intent, ef_search) = average efficiency observed
  - Epsilon-greedy exploration (starts at 40%, decays to 5%)
  - Optimistic initialization (unexplored actions get high value)

### 3. Adaptive Search
- Low confidence or cold start: Use default ef_search
- High confidence: Select ef_search with highest Q-value for intent
- Exploration phase: Try random ef_search values
- Exploitation phase: Use learned optimal ef_search

### 4. Feedback Loop
- Collect user feedback (relevant vs irrelevant results)
- Calculate satisfaction score
- Update Q(intent, ef_search) with observed efficiency
- Decay exploration rate over time

---

## Configuration

### For High Recall (Traditional HNSW)
```python
store = VectorStore(
    dimension=384,
    M=32,                    # More connections
    ef_construction=400,     # Higher build quality
    ef_search=150,           # Thorough search
    enable_intent_detection=False  # Disable adaptation
)
```

### For Adaptive Learning (Recommended)
```python
store = VectorStore(
    dimension=384,
    M=16,
    ef_search=100,           # Default (will be adapted)
    enable_intent_detection=True,
    k_intents=3,             # Number of intent clusters
    learning_rate=0.15,      # Faster learning
    min_queries_for_clustering=30
)

# Important: Lower confidence threshold for Q-learning to work
store._searcher.confidence_threshold = 0.1
```

### For Large Datasets (100k+ vectors)
```python
store = VectorStore(
    dimension=384,
    M=24,
    ef_construction=300,
    ef_search=100,
    k_intents=5,             # More intent clusters
    min_queries_for_clustering=50  # More data before clustering
)
```

---

## API Reference

### VectorStore Class

**Constructor Parameters:**
- `dimension` (int): Vector dimensionality
- `max_elements` (int): Maximum vectors to store (default: 10000)
- `ef_construction` (int): Build quality parameter (default: 200)
- `M` (int): Max connections per node (default: 16)
- `ef_search` (int): Default search parameter (default: 50)
- `normalize` (bool): Normalize vectors to unit length (default: True)
- `enable_intent_detection` (bool): Enable Q-learning adaptation (default: True)
- `k_intents` (int): Number of intent clusters (default: 5)
- `learning_rate` (float): Learning rate for adaptation (default: 0.1)
- `min_queries_for_clustering` (int): Queries before clustering starts (default: 30)

**Methods:**
```python
add(vectors, ids=None, metadata=None) -> List[str]
    # Add vectors to the store

search(query, k=10, ef_search=None) -> List[dict]
    # Search for k nearest neighbors

provide_feedback(relevant_ids: Union[List[str], Set[str]]) -> None
    # Provide feedback to improve future searches

delete(ids: Union[str, List[str]]) -> None
    # Soft delete vectors

get_statistics() -> Dict[str, Any]
    # Get store statistics including Q-learning metrics

save(filepath: str) -> None
    # Serialize store to disk

load(filepath: str) -> VectorStore  # Class method
    # Load store from disk
```

---

## Testing & Validation

### Run Tests
```bash
# All tests
pytest tests/ -v

# With coverage
pytest tests/ --cov=dynhnsw --cov-report=term-missing

# Real-world validation
python examples/real_world_ab_test.py
```

### Validation Examples

**Synthetic Data Tests:**
- `examples/adaptive_ef_demo.py` - Basic demonstration
- `examples/large_scale_ef_validation.py` - 10k vectors, convergence tracking
- `examples/adaptive_vs_static_ef_comparison.py` - A/B comparison

**Real-World Tests (Recommended):**
- `examples/real_world_ab_test.py` - Sentence transformers, 100 real documents
- `examples/real_world_debug_confidence.py` - Debug Q-learning behavior

See `examples/REAL_WORLD_VALIDATION_GUIDE.md` for complete documentation.

---

## Architecture

```
dynhnsw/
├── vector_store.py          # Main API
├── intent_aware_hnsw.py     # Adaptive search engine
├── intent_detector.py       # K-means intent clustering
├── ef_search_selector.py    # Q-learning ef_search optimization
├── feedback.py              # Feedback collection
├── performance_monitor.py   # Performance tracking
└── hnsw/
    ├── graph.py             # HNSW graph structure
    ├── builder.py           # Graph construction
    ├── searcher.py          # HNSW search algorithm
    ├── distance.py          # Distance metrics
    └── utils.py             # HNSW utilities
```

**Key Components:**

1. **HNSW Graph**: Custom implementation with bidirectional edges and layer hierarchy
2. **Intent Detection**: K-means clustering on query vectors (activates after min queries)
3. **Q-Learning**: Action-value learning for ef_search selection per intent
4. **Feedback Processing**: Collects relevance signals and updates Q-values
5. **Performance Monitoring**: Tracks recall, precision, latency, and efficiency

---

## What Can Be Built On Top

### 1. Production Enhancements
- **Persistent Q-table storage**: Save/load learned Q-values between sessions
- **Multi-user learning**: Separate Q-tables per user or global shared learning
- **Online learning**: Continuous Q-value updates in production
- **Degradation detection**: Monitor performance and reset if Q-learning degrades

### 2. Advanced Features
- **Hybrid search**: Combine dense vectors with keyword filtering
- **Multi-modal embeddings**: Support text, image, audio embeddings
- **Dynamic corpus**: Handle growing/shrinking document sets
- **Context-aware ranking**: Re-rank results based on user context

### 3. Research Extensions
- **Deep Q-learning**: Neural network for Q-function approximation
- **Multi-objective optimization**: Balance latency, recall, precision
- **Transfer learning**: Apply learned Q-values across similar corpora
- **Explainable AI**: Interpret why certain ef_search values are optimal

### 4. Domain Applications
- **E-commerce**: Product search with purchase intent detection
- **Medical**: Clinical literature search with diagnosis intent
- **Legal**: Case law retrieval with query type classification
- **Customer support**: FAQ retrieval with question type detection

### 5. Scalability Improvements
- **Distributed HNSW**: Multi-node graph sharding
- **GPU acceleration**: CUDA-based distance computation
- **Approximate Q-learning**: Reduce computation for large intent spaces
- **Incremental clustering**: Update intent clusters without full recomputation

---

## Known Limitations

1. **Pre-embedding required**: Library works with vectors only, not raw text
2. **In-memory storage**: No built-in disk persistence (use save/load)
3. **Single-threaded**: No parallel search or indexing
4. **Confidence threshold sensitivity**: Requires tuning (0.1-0.2 recommended)
5. **Cold start period**: Needs 30-50 queries before Q-learning is effective
6. **Intent detection is vector-based**: Clusters query embeddings, not semantic concepts

---

## Research Context

### Theoretical Foundation
- **HNSW Algorithm**: Malkov & Yashunin (2016) - Hierarchical Navigable Small World graphs
- **Contextual Bandits**: Q-learning for action-value estimation with context
- **Intent-Aware Retrieval**: Query classification in information retrieval systems
- **Online Learning**: Adaptive systems with user feedback optimization

### Validation Methodology
- **Synthetic tests**: Gaussian clusters, controlled query patterns
- **Real-world tests**: Sentence transformers, diverse text corpus
- **A/B comparison**: Adaptive vs static baseline
- **Metrics**: Efficiency (satisfaction/latency), recall, precision, convergence

### Key Findings
- Q-learning successfully differentiates query intents
- 5-8% efficiency improvement over static ef_search
- Learned ef_search values align with query characteristics
- Robust to noisy feedback and confidence threshold tuning

---

## License

MIT License - see [LICENSE](LICENSE) file for details

---