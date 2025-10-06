"""
DynHNSW - Dynamic, Intent-Aware Vector Database

A research-focused vector database that adapts search behavior based on query intent
through feedback-driven learning and dynamic HNSW parameter adjustment.
"""

__version__ = "0.1.0"

from dynhnsw.vector_store import VectorStore
from dynhnsw.config import (
    DynHNSWConfig,
    get_default_config,
    get_epsilon_decay_config,
)

__all__ = [
    "VectorStore",
    "DynHNSWConfig",
    "get_default_config",
    "get_epsilon_decay_config",
]
