"""
Core vector storage and HNSW index management for DynHNSW
"""

from typing import List, Union, Optional
import numpy as np
import numpy.typing as npt

from dynhnsw.hnsw.graph import HNSWGraph
from dynhnsw.hnsw.builder import HNSWBuilder
from dynhnsw.hnsw.searcher import HNSWSearcher
from dynhnsw.hnsw.distance import normalize_vector
from dynhnsw.hnsw.utils import assign_layer

Vector = npt.NDArray[np.float32]


class VectorStore:
    """
    In-memory vector database with HNSW indexing and intent-aware search.

    This is the main entry point for DynHNSW. It manages vector storage,
    HNSW index construction, and provides search capabilities.
    """

    def __init__(
        self,
        dimension: int,
        max_elements: int = 10000,
        ef_construction: int = 200,
        M: int = 16,
        ef_search: int = 50,
        normalize: bool = True,
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
        """
        self.dimension = dimension
        self.max_elements = max_elements
        self.ef_construction = ef_construction
        self.M = M
        self.ef_search = ef_search
        self.normalize = normalize

        # Initialize HNSW components
        self._graph = HNSWGraph(dimension=dimension, M=M)
        self._builder = HNSWBuilder(self._graph)
        self._searcher = HNSWSearcher(self._graph, ef_search=ef_search)

        # Track next node ID
        self._next_id = 0

    def add(self, vectors: Union[Vector, List[Vector]]) -> List[int]:
        """
        Add vectors to the store.

        Args:
            vectors: Single vector or list of vectors (numpy arrays)

        Returns:
            List of document IDs assigned to the added vectors
        """
        # Handle single vector case
        if isinstance(vectors, np.ndarray) and vectors.ndim == 1:
            vectors = [vectors]

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
        inserted_ids = []
        for vec in processed_vectors:
            # Assign layer for this node
            level = assign_layer()

            # Insert into graph
            node_id = self._next_id
            self._builder.insert(vec, node_id=node_id, level=level)
            inserted_ids.append(node_id)

            self._next_id += 1

        return inserted_ids

    def search(
        self, query: Vector, k: int = 10, ef_search: Optional[int] = None
    ) -> List[dict]:
        """
        Search for nearest neighbors.

        Args:
            query: Query vector (numpy array)
            k: Number of results to return
            ef_search: Override default ef_search for this query

        Returns:
            List of search results with IDs, distances, and vectors
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

        # Perform search
        results = self._searcher.search(query, k=k, ef_search=ef_search)

        # Format results
        formatted_results = []
        for node_id, distance in results:
            node = self._graph.get_node(node_id)
            formatted_results.append(
                {
                    "id": node_id,
                    "distance": float(distance),
                    "vector": node.vector,
                }
            )

        return formatted_results

    def size(self) -> int:
        """
        Get the number of vectors in the store.

        Returns:
            Number of vectors stored
        """
        return self._graph.size()
