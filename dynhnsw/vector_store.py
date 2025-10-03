"""
Core vector storage and HNSW index management for DynHNSW
"""

from typing import List, Union
import numpy as np
import numpy.typing as npt

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
    ) -> None:
        """
        Initialize the vector store.

        Args:
            dimension: Dimensionality of vectors
            max_elements: Maximum number of vectors to store
            ef_construction: HNSW construction parameter (higher = better quality, slower)
            M: HNSW max connections per node (typically 16-64)
        """
        self.dimension = dimension
        self.max_elements = max_elements
        self.ef_construction = ef_construction
        self.M = M

        # Placeholder - will implement in next phase
        self._index = None
        self._count = 0

    def add(self, vectors: Union[List[str], List[Vector]]) -> List[int]:
        """
        Add vectors or text documents to the store.

        Args:
            vectors: Either text strings (will be embedded) or numpy arrays

        Returns:
            List of document IDs assigned to the added vectors
        """
        raise NotImplementedError("Will implement in Phase 1")

    def search(self, query: Union[str, Vector], k: int = 10) -> List[dict]:
        """
        Search for nearest neighbors.

        Args:
            query: Query text or vector
            k: Number of results to return

        Returns:
            List of search results with IDs and distances
        """
        raise NotImplementedError("Will implement in Phase 1")
