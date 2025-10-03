"""
Pytest configuration and shared fixtures for DynHNSW tests
"""

import pytest
import numpy as np
from typing import List


@pytest.fixture
def sample_vectors() -> np.ndarray:
    """Generate sample vectors for testing."""
    np.random.seed(42)
    return np.random.rand(10, 384).astype(np.float32)


@pytest.fixture
def sample_texts() -> List[str]:
    """Sample text documents for testing."""
    return [
        "Machine learning is a subset of artificial intelligence",
        "Deep learning uses neural networks with multiple layers",
        "Natural language processing helps computers understand human language",
        "Computer vision enables machines to interpret visual information",
        "Reinforcement learning trains agents through rewards and penalties",
    ]


@pytest.fixture
def dimension() -> int:
    """Standard vector dimension for testing."""
    return 384
