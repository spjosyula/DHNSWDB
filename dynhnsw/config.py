"""Configuration system for DynHNSW experimental features.

Usage:
    from dynhnsw import VectorStore, DynHNSWConfig

    # Default config
    store = VectorStore(dimension=384)

    # Custom config
    config = DynHNSWConfig(enable_epsilon_decay=True)
    store = VectorStore(dimension=384, config=config)

    # From file
    config = DynHNSWConfig.from_json("my_config.json")
    store = VectorStore(dimension=384, config=config)
"""

from typing import Dict, Any
import json
from dataclasses import dataclass, asdict


@dataclass
class DynHNSWConfig:
    """Configuration for DynHNSW.

    Feature Flags:
        enable_epsilon_decay: Enable epsilon decay (False recommended, no proven benefit)

    Hyperparameters:
        exploration_rate: Initial epsilon for epsilon-greedy (0.0 to 1.0)
        epsilon_decay_mode: "none", "multiplicative", or "glie"
        min_epsilon: Minimum epsilon value
        k_intents: Number of intent clusters
        min_queries_for_clustering: Queries needed before clustering starts
        confidence_threshold: Minimum confidence for intent-specific ef_search
    """

    # Feature flags
    enable_epsilon_decay: bool = False

    # Hyperparameters
    exploration_rate: float = 0.15
    epsilon_decay_mode: str = "none"
    min_epsilon: float = 0.01
    k_intents: int = 3
    min_queries_for_clustering: int = 30
    confidence_threshold: float = 0.1

    # HNSW defaults
    default_ef_search: int = 100
    default_M: int = 16
    default_ef_construction: int = 200

    # Metadata
    config_name: str = "default"

    def __post_init__(self):
        """Validate configuration."""
        valid_decay_modes = ["none", "multiplicative", "glie"]
        if self.epsilon_decay_mode not in valid_decay_modes:
            raise ValueError(f"epsilon_decay_mode must be one of {valid_decay_modes}")

        if not 0.0 <= self.exploration_rate <= 1.0:
            raise ValueError("exploration_rate must be in [0.0, 1.0]")

        if not 0.0 <= self.min_epsilon <= 1.0:
            raise ValueError("min_epsilon must be in [0.0, 1.0]")

        if self.k_intents < 1:
            raise ValueError("k_intents must be >= 1")

        if not 0.0 <= self.confidence_threshold <= 1.0:
            raise ValueError("confidence_threshold must be in [0.0, 1.0]")

        # Auto-sync epsilon decay settings
        if self.epsilon_decay_mode == "none":
            self.enable_epsilon_decay = False
        if not self.enable_epsilon_decay:
            self.epsilon_decay_mode = "none"

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return asdict(self)

    def to_json(self, filepath: str) -> None:
        """Save configuration to JSON file."""
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'DynHNSWConfig':
        """Load configuration from dictionary."""
        return cls(**config_dict)

    @classmethod
    def from_json(cls, filepath: str) -> 'DynHNSWConfig':
        """Load configuration from JSON file."""
        with open(filepath, 'r') as f:
            config_dict = json.load(f)
        return cls.from_dict(config_dict)

    def __repr__(self) -> str:
        """String representation."""
        features = []
        if self.enable_epsilon_decay:
            features.append(f"epsilon_decay={self.epsilon_decay_mode}")

        features_str = ", ".join(features) if features else "default"

        return (
            f"DynHNSWConfig("
            f"{self.config_name}, "
            f"{features_str}, "
            f"eps={self.exploration_rate})"
        )


# Preset configurations

def get_default_config() -> DynHNSWConfig:
    """Default configuration (recommended)."""
    return DynHNSWConfig(config_name="default")


def get_epsilon_decay_config() -> DynHNSWConfig:
    """Configuration with GLIE epsilon decay enabled.

    Note: A/B testing showed no improvement. Use for research only.
    """
    return DynHNSWConfig(
        config_name="epsilon_decay",
        enable_epsilon_decay=True,
        epsilon_decay_mode="glie",
    )
