"""
Core vector storage and HNSW index management for DynHNSW
"""

from typing import List, Union, Optional, Set, Dict, Any
import pickle
import time
import numpy as np
import numpy.typing as npt

from dynhnsw.hnsw.graph import HNSWGraph
from dynhnsw.hnsw.builder import HNSWBuilder
from dynhnsw.intent_aware_hnsw import IntentAwareHNSWSearcher
from dynhnsw.hnsw.distance import normalize_vector
from dynhnsw.hnsw.utils import assign_layer
from dynhnsw.config import DynHNSWConfig, get_default_config

Vector = npt.NDArray[np.float32]


class VectorStore:
    """
    In-memory vector database with HNSW indexing and intent-aware search.

    This is the main entry point for DynHNSW. It manages vector storage,
    HNSW index construction, and provides intent-aware search with learning.

    IMPORTANT: This library operates on pre-embedded vectors (numpy arrays).
    Text embedding is the user's responsibility. You must convert text to vectors
    using an embedding model (e.g., sentence-transformers) before adding to the store.

    Example with text data:
        >>> from sentence_transformers import SentenceTransformer
        >>> model = SentenceTransformer('all-MiniLM-L6-v2')
        >>>
        >>> # Create store matching embedding dimension (384 for all-MiniLM-L6-v2)
        >>> store = VectorStore(dimension=384)
        >>>
        >>> # Convert text to vectors
        >>> texts = ["The cat sleeps", "A dog plays"]
        >>> vectors = model.encode(texts, convert_to_numpy=True)
        >>>
        >>> # Add vectors to store
        >>> store.add(vectors)
        >>>
        >>> # Search with query text
        >>> query_vector = model.encode(["Where is the cat?"], convert_to_numpy=True)[0]
        >>> results = store.search(query_vector, k=5)
    """

    def __init__(
        self,
        dimension: int,
        max_elements: int = 10000,
        ef_construction: int = None,
        M: int = None,
        ef_search: int = None,
        normalize: bool = True,
        enable_intent_detection: bool = True,
        enable_adaptive_thresholds: bool = False,
        k_intents: int = None,
        learning_rate: float = 0.1,
        min_queries_for_clustering: int = None,
        config: Optional[DynHNSWConfig] = None,
    ) -> None:
        """
        Initialize the vector store.

        Args:
            dimension: Dimensionality of vectors
            max_elements: Maximum number of vectors to store
            ef_construction: HNSW construction parameter (higher = better quality, slower)
            M: HNSW max connections per node (typically 16-64)
            ef_search: Default search parameter (higher = better recall, slower)
            normalize: Whether to normalize vectors to unit length (recommended for cosine similarity)
            enable_intent_detection: Enable layer-adaptive multi-path search
            enable_adaptive_thresholds: Enable learning of optimal difficulty thresholds
            k_intents: Number of intent clusters for adaptive search
            learning_rate: Learning rate for Q-learning (legacy parameter, not used)
            min_queries_for_clustering: Minimum queries before intent clustering starts
            config: DynHNSWConfig object for advanced configuration. If provided,
                    overrides individual parameters. Use this for experimental features.

        Example:
            # Basic usage (default config)
            store = VectorStore(dimension=384)

            # With custom config
            from dynhnsw.config import DynHNSWConfig
            config = DynHNSWConfig(enable_epsilon_decay=True)
            store = VectorStore(dimension=384, config=config)
        """
        # Use provided config or create default
        if config is None:
            config = get_default_config()
        self.config = config

        # Override config with explicit parameters (explicit params take precedence)
        if ef_construction is None:
            ef_construction = self.config.default_ef_construction
        if M is None:
            M = self.config.default_M
        if ef_search is None:
            ef_search = self.config.default_ef_search
        if k_intents is None:
            k_intents = self.config.k_intents
        if min_queries_for_clustering is None:
            min_queries_for_clustering = self.config.min_queries_for_clustering

        self.dimension = dimension
        self.max_elements = max_elements
        self.ef_construction = ef_construction
        self.M = M
        self.ef_search = ef_search
        self.normalize = normalize
        self.enable_intent_detection = enable_intent_detection

        # Initialize HNSW components
        self._graph = HNSWGraph(dimension=dimension, M=M)
        self._builder = HNSWBuilder(self._graph)

        # Use intent-aware searcher for adaptive learning
        self._searcher = IntentAwareHNSWSearcher(
            graph=self._graph,
            ef_search=ef_search,
            k_intents=k_intents,
            learning_rate=learning_rate,
            enable_adaptation=enable_intent_detection,
            enable_intent_detection=enable_intent_detection,
            enable_adaptive_thresholds=enable_adaptive_thresholds,
            min_queries_for_clustering=min_queries_for_clustering,
            config=self.config,  # Pass config to searcher
        )

        # Track next node ID
        self._next_id = 0

        # ID mapping: external ID <-> internal node ID
        self._id_to_node: Dict[str, int] = {}  # external_id -> node_id
        self._node_to_id: Dict[int, str] = {}  # node_id -> external_id

        # Metadata storage: external_id -> metadata dict
        self._metadata: Dict[str, dict] = {}

        # Soft delete tracking
        self._deleted_ids: Set[str] = set()

        # Track last query for feedback
        self._last_query_vector: Optional[Vector] = None
        self._last_result_node_ids: List[int] = []
        self._last_result_external_ids: List[str] = []
        self._last_latency_ms: float = 0.0

    def add(
        self,
        vectors: Union[Vector, List[Vector]],
        ids: Optional[List[str]] = None,
        metadata: Optional[List[dict]] = None,
    ) -> List[str]:
        """
        Add vectors to the store with optional IDs and metadata.

        IMPORTANT: This method accepts pre-embedded vectors only, not raw text.
        You must convert text to vectors using an embedding model before calling this method.

        Args:
            vectors: Single vector or list of vectors (numpy arrays of shape (dimension,))
                     Must be pre-embedded using a model like sentence-transformers
            ids: Optional list of external IDs (auto-generated if not provided)
            metadata: Optional list of metadata dicts (one per vector)

        Returns:
            List of external document IDs assigned to the added vectors

        Raises:
            ValueError: If vector dimensions don't match store dimension
            ValueError: If number of IDs or metadata doesn't match number of vectors
        """
        # Handle single vector case
        if isinstance(vectors, np.ndarray) and vectors.ndim == 1:
            vectors = [vectors]

        num_vectors = len(vectors)

        # Auto-generate IDs if not provided
        if ids is None:
            ids = [f"doc_{self._next_id + i}" for i in range(num_vectors)]
        elif len(ids) != num_vectors:
            raise ValueError(f"Number of IDs ({len(ids)}) doesn't match number of vectors ({num_vectors})")

        # Validate IDs are unique
        for ext_id in ids:
            if ext_id in self._id_to_node:
                raise ValueError(f"ID '{ext_id}' already exists in the store")

        # Handle metadata
        if metadata is None:
            metadata = [{} for _ in range(num_vectors)]
        elif len(metadata) != num_vectors:
            raise ValueError(f"Number of metadata dicts ({len(metadata)}) doesn't match number of vectors ({num_vectors})")

        # Validate and normalize vectors
        processed_vectors = []
        for vec in vectors:
            if len(vec) != self.dimension:
                raise ValueError(
                    f"Vector dimension {len(vec)} doesn't match store dimension {self.dimension}"
                )

            # Normalize if enabled
            if self.normalize:
                vec = normalize_vector(vec)

            processed_vectors.append(vec.astype(np.float32))

        # Insert each vector into the graph
        inserted_external_ids = []
        for vec, ext_id, meta in zip(processed_vectors, ids, metadata):
            # Assign layer for this node using graph's M parameter
            level = assign_layer(M=self._graph.M)

            # Insert into graph
            node_id = self._next_id
            self._builder.insert(vec, node_id=node_id, level=level)

            # Store ID mappings
            self._id_to_node[ext_id] = node_id
            self._node_to_id[node_id] = ext_id

            # Store metadata
            self._metadata[ext_id] = meta

            inserted_external_ids.append(ext_id)
            self._next_id += 1

        return inserted_external_ids

    def search(
        self, query: Vector, k: int = 10, ef_search: Optional[int] = None
    ) -> List[dict]:
        """
        Search for nearest neighbors with intent-aware adaptation.

        IMPORTANT: This method accepts pre-embedded query vectors only, not raw text.
        You must convert query text to a vector using the same embedding model
        that was used for the stored vectors.

        Args:
            query: Query vector (numpy array of shape (dimension,))
                   Must use the same embedding model as stored vectors
            k: Number of results to return
            ef_search: Override default ef_search for this query

        Returns:
            List of search results with external IDs, distances, metadata, and vectors
            Each result is a dict with keys: 'id', 'distance', 'vector', 'metadata'

        Raises:
            ValueError: If query dimension doesn't match store dimension
        """
        # Validate query dimension
        if len(query) != self.dimension:
            raise ValueError(
                f"Query dimension {len(query)} doesn't match store dimension {self.dimension}"
            )

        # Normalize if enabled
        if self.normalize:
            query = normalize_vector(query)

        query = query.astype(np.float32)

        # Store query for potential feedback
        self._last_query_vector = query.copy()

        # Measure search latency
        start_time = time.perf_counter()
        results = self._searcher.search(query, k=k, ef_search=ef_search)
        end_time = time.perf_counter()

        # Store latency in milliseconds
        self._last_latency_ms = (end_time - start_time) * 1000.0

        # Store result node IDs for feedback
        self._last_result_node_ids = [node_id for node_id, _ in results]

        # Convert to external IDs and store
        self._last_result_external_ids = [
            self._node_to_id[node_id] for node_id in self._last_result_node_ids
        ]

        # Format results with external IDs and metadata (filter deleted items)
        formatted_results = []
        for node_id, distance in results:
            node = self._graph.get_node(node_id)
            ext_id = self._node_to_id[node_id]

            # Skip deleted items
            if ext_id in self._deleted_ids:
                continue

            formatted_results.append(
                {
                    "id": ext_id,
                    "distance": float(distance),
                    "vector": node.vector,
                    "metadata": self._metadata.get(ext_id, {}),
                }
            )

        return formatted_results[:k]  # Ensure we return exactly k results after filtering

    def provide_feedback(
        self, relevant_ids: Union[List[str], Set[str]]
    ) -> None:
        """
        Provide feedback on the last search to improve future results.

        Args:
            relevant_ids: List or set of external document IDs that were relevant
        """
        if not self.enable_intent_detection:
            return

        if self._last_query_vector is None:
            raise ValueError("No recent query to provide feedback for")

        # Convert external IDs to internal node IDs
        relevant_external = set(relevant_ids) if isinstance(relevant_ids, list) else relevant_ids
        relevant_node_ids = set()

        for ext_id in relevant_external:
            if ext_id in self._id_to_node:
                relevant_node_ids.add(self._id_to_node[ext_id])

        # Pass feedback to intent-aware searcher
        self._searcher.provide_feedback(
            query=self._last_query_vector,
            result_ids=self._last_result_node_ids,
            ground_truth_ids=list(relevant_node_ids),
            latency_ms=self._last_latency_ms
        )

    def delete(self, ids: Union[str, List[str]]) -> None:
        """
        Soft delete vectors from the store.

        Deleted vectors are marked and excluded from search results,
        but remain in the index (lazy cleanup).

        Args:
            ids: Single ID or list of external IDs to delete
        """
        # Handle single ID case
        if isinstance(ids, str):
            ids = [ids]

        # Mark IDs as deleted
        for ext_id in ids:
            if ext_id in self._id_to_node:
                self._deleted_ids.add(ext_id)

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about the vector store and learning progress.

        Returns:
            Dictionary with store statistics including intent detection and learning metrics
        """
        base_stats = {
            "total_vectors": self.size(),
            "deleted_vectors": len(self._deleted_ids),
            "active_vectors": self.size() - len(self._deleted_ids),
            "dimension": self.dimension,
            "max_elements": self.max_elements,
            "intent_detection_enabled": self.enable_intent_detection,
        }

        # Add intent-aware searcher statistics if enabled
        if self.enable_intent_detection:
            searcher_stats = self._searcher.get_statistics()
            base_stats.update(searcher_stats)

        return base_stats

    def size(self) -> int:
        """
        Get the total number of vectors in the store (including soft-deleted).

        Returns:
            Number of vectors stored (including deleted)
        """
        return self._graph.size()

    def save(self, filepath: str) -> None:
        """
        Save the vector store to disk using pickle.

        Args:
            filepath: Path to save the store (e.g., "index.pkl")
        """
        with open(filepath, 'wb') as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, filepath: str) -> 'VectorStore':
        """
        Load a vector store from disk.

        Args:
            filepath: Path to the saved store

        Returns:
            Loaded VectorStore instance
        """
        with open(filepath, 'rb') as f:
            return pickle.load(f)
