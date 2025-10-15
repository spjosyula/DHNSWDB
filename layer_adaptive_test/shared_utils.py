"""Shared utilities for large-scale recall testing with zero-cost difficulty proxy.

This module provides utilities for testing the breakthrough distance-to-entry-point
difficulty proxy that eliminates 350% overhead while maintaining recall quality.

Key Features:
- Large-scale corpus generation (10K-50K+ documents)
- Real sentence-transformers embeddings
- Ground truth pre-computation for accurate recall@k measurement
- Comparison: Static HNSW vs Dynamic HNSW with new proxy
- Comprehensive metrics: recall, latency, overhead
"""

import time
import json
import numpy as np
from typing import List, Dict, Tuple, Any, Optional
from pathlib import Path


class RecallExperimentTracker:
    """Track recall-focused metrics for large-scale experiments."""

    def __init__(self, name: str, compare_baseline: bool = True):
        """Initialize recall tracker.

        Args:
            name: Experiment name
            compare_baseline: Whether to compare against static HNSW baseline
        """
        self.name = name
        self.recalls = []
        self.latencies = []
        self.ef_values = []
        self.intent_ids = []
        self.query_types = []
        self.difficulty_values = []

        # For comparison with baseline
        self.compare_baseline = compare_baseline
        self.baseline_recalls = []
        self.baseline_latencies = []

        # Overhead tracking
        self.difficulty_computation_times = []

    def record_query(
        self,
        recall: float,
        latency_ms: float,
        ef_used: int,
        intent_id: int,
        query_type: str = "unknown",
        difficulty: float = 0.0,
        difficulty_time_ms: float = 0.0,
    ) -> None:
        """Record metrics for a single query.

        Args:
            recall: Recall@k for this query
            latency_ms: Query latency in milliseconds
            ef_used: ef_search value used
            intent_id: Detected intent cluster ID
            query_type: Type of query (exploratory/precise/mixed)
            difficulty: Computed difficulty value
            difficulty_time_ms: Time to compute difficulty (overhead)
        """
        self.recalls.append(recall)
        self.latencies.append(latency_ms)
        self.ef_values.append(ef_used)
        self.intent_ids.append(intent_id)
        self.query_types.append(query_type)
        self.difficulty_values.append(difficulty)
        self.difficulty_computation_times.append(difficulty_time_ms)

    def record_baseline(self, recall: float, latency_ms: float) -> None:
        """Record baseline (static HNSW) metrics.

        Args:
            recall: Baseline recall@k
            latency_ms: Baseline latency
        """
        self.baseline_recalls.append(recall)
        self.baseline_latencies.append(latency_ms)

    def get_metrics(self) -> Dict[str, Any]:
        """Get comprehensive metrics.

        Returns:
            Dictionary with all metrics
        """
        if not self.recalls:
            return {}

        metrics = {
            "num_queries": len(self.recalls),
            "avg_recall": np.mean(self.recalls),
            "median_recall": np.median(self.recalls),
            "min_recall": np.min(self.recalls),
            "max_recall": np.max(self.recalls),
            "avg_latency_ms": np.mean(self.latencies),
            "median_latency_ms": np.median(self.latencies),
            "p95_latency_ms": np.percentile(self.latencies, 95),
            "p99_latency_ms": np.percentile(self.latencies, 99),
            "avg_ef": np.mean(self.ef_values),
            "avg_difficulty": np.mean(self.difficulty_values),
            "avg_difficulty_time_ms": np.mean(self.difficulty_computation_times),
            "total_difficulty_overhead_ms": np.sum(self.difficulty_computation_times),
        }

        # Compute overhead percentage
        total_latency = np.sum(self.latencies)
        overhead_pct = (metrics["total_difficulty_overhead_ms"] / total_latency * 100) if total_latency > 0 else 0
        metrics["difficulty_overhead_percent"] = overhead_pct

        # Baseline comparison
        if self.compare_baseline and self.baseline_recalls:
            metrics["baseline_avg_recall"] = np.mean(self.baseline_recalls)
            metrics["baseline_avg_latency_ms"] = np.mean(self.baseline_latencies)

            # Improvements
            recall_improvement = (metrics["avg_recall"] - metrics["baseline_avg_recall"]) / metrics["baseline_avg_recall"] * 100
            latency_improvement = (metrics["baseline_avg_latency_ms"] - metrics["avg_latency_ms"]) / metrics["baseline_avg_latency_ms"] * 100

            metrics["recall_improvement_percent"] = recall_improvement
            metrics["latency_improvement_percent"] = latency_improvement

        return metrics

    def get_intent_breakdown(self) -> Dict[int, Dict[str, float]]:
        """Get metrics broken down by intent.

        Returns:
            Dictionary mapping intent_id to metrics
        """
        if not self.intent_ids:
            return {}

        breakdown = {}
        unique_intents = set(self.intent_ids)

        for intent in unique_intents:
            if intent < 0:  # Skip cold start
                continue

            mask = [i for i, x in enumerate(self.intent_ids) if x == intent]
            if not mask:
                continue

            intent_recalls = [self.recalls[i] for i in mask]
            intent_latencies = [self.latencies[i] for i in mask]
            intent_efs = [self.ef_values[i] for i in mask]
            intent_difficulties = [self.difficulty_values[i] for i in mask]

            breakdown[intent] = {
                "count": len(mask),
                "avg_recall": np.mean(intent_recalls),
                "avg_latency_ms": np.mean(intent_latencies),
                "avg_ef": np.mean(intent_efs),
                "avg_difficulty": np.mean(intent_difficulties),
            }

        return breakdown

    def save_results(self, filepath: str) -> None:
        """Save results to JSON file.

        Args:
            filepath: Path to save results
        """
        results = {
            "experiment_name": self.name,
            "overall_metrics": self.get_metrics(),
            "intent_breakdown": {str(k): v for k, v in self.get_intent_breakdown().items()},
            "raw_data": {
                "recalls": self.recalls,
                "latencies": self.latencies,
                "ef_values": self.ef_values,
                "intent_ids": self.intent_ids,
                "query_types": self.query_types,
                "difficulty_values": self.difficulty_values,
                "difficulty_times_ms": self.difficulty_computation_times,
                "baseline_recalls": self.baseline_recalls if self.compare_baseline else [],
                "baseline_latencies": self.baseline_latencies if self.compare_baseline else [],
            },
        }

        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2)

        print(f"\n[RESULTS SAVED] {filepath}")


def print_results_summary(tracker: RecallExperimentTracker) -> None:
    """Print formatted results summary with recall focus.

    Args:
        tracker: Experiment tracker with results
    """
    print("\n" + "=" * 100)
    print(f"RECALL EXPERIMENT RESULTS: {tracker.name}")
    print("=" * 100)

    metrics = tracker.get_metrics()
    if not metrics:
        print("No metrics to display.")
        return

    print(f"\n{'Metric':<45} | {'Value':>25}")
    print("-" * 100)
    print(f"{'Total Queries':<45} | {metrics['num_queries']:>25}")
    print(f"{'Average Recall@k':<45} | {metrics['avg_recall']:>24.1%}")
    print(f"{'Median Recall@k':<45} | {metrics['median_recall']:>24.1%}")
    print(f"{'Min Recall@k':<45} | {metrics['min_recall']:>24.1%}")
    print(f"{'Max Recall@k':<45} | {metrics['max_recall']:>24.1%}")
    print(f"{'Average Latency (ms)':<45} | {metrics['avg_latency_ms']:>25.2f}")
    print(f"{'Median Latency (ms)':<45} | {metrics['median_latency_ms']:>25.2f}")
    print(f"{'P95 Latency (ms)':<45} | {metrics['p95_latency_ms']:>25.2f}")
    print(f"{'P99 Latency (ms)':<45} | {metrics['p99_latency_ms']:>25.2f}")
    print(f"{'Average ef_search':<45} | {metrics['avg_ef']:>25.1f}")
    print(f"{'Average Difficulty':<45} | {metrics['avg_difficulty']:>25.4f}")

    print(f"\n{'Overhead Analysis':<45} | {'Value':>25}")
    print("-" * 100)
    print(f"{'Avg Difficulty Computation Time (ms)':<45} | {metrics['avg_difficulty_time_ms']:>25.4f}")
    print(f"{'Total Difficulty Overhead (ms)':<45} | {metrics['total_difficulty_overhead_ms']:>25.2f}")
    print(f"{'Difficulty Overhead (%)':<45} | {metrics['difficulty_overhead_percent']:>24.2f}%")

    if tracker.compare_baseline and 'baseline_avg_recall' in metrics:
        print(f"\n{'Comparison with Static HNSW Baseline':<45} | {'Value':>25}")
        print("-" * 100)
        print(f"{'Static HNSW Recall@k':<45} | {metrics['baseline_avg_recall']:>24.1%}")
        print(f"{'Dynamic HNSW Recall@k':<45} | {metrics['avg_recall']:>24.1%}")
        print(f"{'Recall Improvement (%)':<45} | {metrics['recall_improvement_percent']:>24.2f}%")
        print(f"{'Static HNSW Latency (ms)':<45} | {metrics['baseline_avg_latency_ms']:>25.2f}")
        print(f"{'Dynamic HNSW Latency (ms)':<45} | {metrics['avg_latency_ms']:>25.2f}")
        print(f"{'Latency Improvement (%)':<45} | {metrics['latency_improvement_percent']:>24.2f}%")

    # Intent breakdown
    intent_breakdown = tracker.get_intent_breakdown()
    if intent_breakdown:
        print(f"\n{'Per-Intent Breakdown':<20} | {'Count':>10} | {'Recall':>10} | {'Latency':>12} | {'ef':>8}")
        print("-" * 100)
        for intent_id, stats in sorted(intent_breakdown.items()):
            print(f"{'Intent ' + str(intent_id):<20} | {stats['count']:>10} | {stats['avg_recall']:>9.1%} | "
                  f"{stats['avg_latency_ms']:>10.2f}ms | {stats['avg_ef']:>8.1f}")

    print("=" * 100)


def generate_large_corpus(size: int = 10000, seed: int = 42) -> List[str]:
    """Generate large diverse corpus for scalability testing.

    Args:
        size: Number of documents
        seed: Random seed for reproducibility

    Returns:
        List of document strings
    """
    np.random.seed(seed)
    corpus = []

    # Technical domains (25%)
    tech_templates = [
        "Python {feature} implementation with {library} for {task}",
        "Machine learning {algorithm} optimization using {framework}",
        "{language} design patterns for {architecture} systems",
        "Docker container {operation} with {orchestration} deployment",
        "{database} query optimization for {workload} performance",
        "API design {pattern} with {protocol} authentication",
        "Cloud {service} configuration for {deployment} scaling",
        "Testing {strategy} with {tool} automation framework",
    ]

    features = ["async", "multithreading", "caching", "serialization", "validation", "logging"]
    libraries = ["asyncio", "numpy", "pandas", "tensorflow", "pytorch", "fastapi"]
    tasks = ["data processing", "model training", "web scraping", "API integration"]
    algorithms = ["gradient descent", "random forest", "neural network", "clustering"]
    frameworks = ["scikit-learn", "keras", "xgboost", "lightgbm"]

    for i in range(size // 4):
        template = np.random.choice(tech_templates)
        doc = template.format(
            feature=np.random.choice(features),
            library=np.random.choice(libraries),
            task=np.random.choice(tasks),
            algorithm=np.random.choice(algorithms),
            framework=np.random.choice(frameworks),
            language=np.random.choice(["Python", "Java", "JavaScript", "Go"]),
            architecture=np.random.choice(["microservices", "monolithic", "serverless"]),
            operation=np.random.choice(["build", "deploy", "monitor", "scale"]),
            orchestration=np.random.choice(["Kubernetes", "Docker Swarm", "Nomad"]),
            database=np.random.choice(["PostgreSQL", "MongoDB", "Redis", "Elasticsearch"]),
            workload=np.random.choice(["OLTP", "OLAP", "mixed", "analytics"]),
            pattern=np.random.choice(["RESTful", "GraphQL", "gRPC", "WebSocket"]),
            protocol=np.random.choice(["OAuth2", "JWT", "API Key", "Basic Auth"]),
            service=np.random.choice(["EC2", "Lambda", "S3", "RDS"]),
            deployment=np.random.choice(["blue-green", "canary", "rolling"]),
            strategy=np.random.choice(["unit", "integration", "end-to-end", "performance"]),
            tool=np.random.choice(["pytest", "jest", "selenium", "k6"]),
        )
        corpus.append(doc + f" Document {i}.")

    # Product descriptions (25%)
    for i in range(size // 4, size // 2):
        product = np.random.choice(["laptop", "headphones", "monitor", "keyboard", "mouse", "webcam", "chair"])
        brand = np.random.choice(["Premium", "Professional", "Gaming", "Ergonomic", "Wireless"])
        feature = np.random.choice(["high-performance", "ultra-slim", "noise-canceling", "mechanical", "RGB"])
        corpus.append(f"{brand} {product} with {feature} technology for enhanced productivity. Model {i}.")

    # FAQ/Support (25%)
    for i in range(size // 2, 3 * size // 4):
        topic = np.random.choice(["password", "billing", "shipping", "returns", "support", "account", "privacy"])
        action = np.random.choice(["reset", "update", "manage", "configure", "troubleshoot"])
        corpus.append(f"How to {action} your {topic}? Contact support or visit help center. FAQ {i}.")

    # Scientific/Knowledge (25%)
    for i in range(3 * size // 4, size):
        subject = np.random.choice(["physics", "chemistry", "biology", "mathematics", "computer science"])
        concept = np.random.choice(["theory", "principle", "equation", "algorithm", "process"])
        corpus.append(f"Scientific {concept} in {subject} explains natural phenomena. Knowledge item {i}.")

    return corpus[:size]


def create_diverse_queries(
    exploratory_count: int = 300,
    precise_count: int = 500,
    mixed_count: int = 200,
    seed: int = 43,
) -> Tuple[List[str], List[str]]:
    """Create diverse query set with varying difficulty.

    Args:
        exploratory_count: Number of exploratory queries (need high ef)
        precise_count: Number of precise queries (need low ef)
        mixed_count: Number of mixed queries (medium ef)
        seed: Random seed

    Returns:
        Tuple of (queries, query_types)
    """
    np.random.seed(seed)
    queries = []
    query_types = []

    # Exploratory queries (broad topics)
    exploratory_templates = [
        "overview of {} systems",
        "introduction to {} concepts",
        "guide to {} best practices",
        "what is {} technology",
        "explain {} architecture",
        "comparison of {} approaches",
    ]

    topics = ["machine learning", "cloud computing", "database design", "API development", "testing strategies"]

    for _ in range(exploratory_count):
        template = np.random.choice(exploratory_templates)
        topic = np.random.choice(topics)
        queries.append(template.format(topic))
        query_types.append("exploratory")

    # Precise queries (specific details)
    precise_templates = [
        "{} {} implementation in {}",
        "{} {} optimization technique",
        "{} {} configuration example",
        "{} {} error handling pattern",
    ]

    for _ in range(precise_count):
        template = np.random.choice(precise_templates)
        lang = np.random.choice(["Python", "Java", "JavaScript"])
        feature = np.random.choice(["async", "caching", "serialization", "validation"])
        context = np.random.choice(["web server", "API", "microservice", "batch job"])
        queries.append(template.format(lang, feature, context))
        query_types.append("precise")

    # Mixed queries
    for _ in range(mixed_count):
        topic = np.random.choice(["database", "API", "testing", "deployment"])
        detail = np.random.choice(["performance", "security", "scalability"])
        queries.append(f"{topic} {detail} considerations")
        query_types.append("mixed")

    # Shuffle while maintaining alignment
    indices = list(range(len(queries)))
    np.random.shuffle(indices)
    queries = [queries[i] for i in indices]
    query_types = [query_types[i] for i in indices]

    return queries, query_types


def compute_ground_truth_brute_force(
    query_vectors: np.ndarray,
    database_vectors: np.ndarray,
    k: int = 10,
) -> List[List[int]]:
    """Compute exact k-NN ground truth via brute force (cosine similarity).

    Args:
        query_vectors: Query embeddings (n_queries, dim)
        database_vectors: Database embeddings (n_docs, dim)
        k: Number of neighbors

    Returns:
        List of ground truth neighbor ID lists (one per query)
    """
    print(f"\n[Ground Truth] Computing exact k-NN for {len(query_vectors)} queries...")
    ground_truth = []

    # Normalize vectors for cosine similarity
    query_norms = np.linalg.norm(query_vectors, axis=1, keepdims=True)
    db_norms = np.linalg.norm(database_vectors, axis=1, keepdims=True)

    query_normalized = query_vectors / (query_norms + 1e-10)
    db_normalized = database_vectors / (db_norms + 1e-10)

    for i, query in enumerate(query_normalized):
        # Cosine similarity = dot product of normalized vectors
        similarities = np.dot(db_normalized, query)

        # Get top-k indices (highest similarity = lowest distance)
        top_k_indices = np.argpartition(-similarities, k)[:k]
        top_k_sorted = top_k_indices[np.argsort(-similarities[top_k_indices])]

        ground_truth.append(top_k_sorted.tolist())

        if (i + 1) % 100 == 0:
            print(f"  Progress: {i+1}/{len(query_vectors)} queries")

    print(f"[Ground Truth] Complete!")
    return ground_truth


def compute_recall_at_k(retrieved_ids: List[int], ground_truth_ids: List[int], k: int) -> float:
    """Compute recall@k: fraction of ground truth neighbors retrieved.

    Args:
        retrieved_ids: IDs returned by search
        ground_truth_ids: True k-nearest neighbor IDs
        k: Number of neighbors to consider

    Returns:
        Recall@k value (0.0 to 1.0)
    """
    retrieved_set = set(retrieved_ids[:k])
    ground_truth_set = set(ground_truth_ids[:k])
    correct = len(retrieved_set & ground_truth_set)
    return correct / k if k > 0 else 0.0
