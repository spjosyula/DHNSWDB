"""
DynHNSW - Dynamic, Intent-Aware Vector Database

A research-focused vector database that adapts search behavior based on query intent
through feedback-driven learning and dynamic HNSW parameter adjustment.
"""

__version__ = "0.1.0"

from dynhnsw.vector_store import VectorStore

__all__ = ["VectorStore"]
