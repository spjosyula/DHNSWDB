"""Shared utilities for adaptive ef_search experiments.

This module provides common functionality used across all test scenarios:
- Corpus generation
- Query creation
- Metrics tracking
- Result comparison

Focus: Comparing static ef_search vs adaptive learning (no epsilon decay)
"""

import time
import json
import numpy as np
from typing import List, Dict, Tuple, Any
from pathlib import Path


class ExperimentTracker:
    """Track metrics for adaptive ef_search experiments."""

    def __init__(self, name: str):
        """Initialize tracker.

        Args:
            name: Experiment name for identification
        """
        self.name = name
        self.latencies = []
        self.satisfactions = []
        self.efficiencies = []
        self.ef_values = []
        self.query_types = []
        self.intent_ids = []
        self.q_value_history = []  # Track Q-value evolution

    def record(
        self,
        latency_ms: float,
        satisfaction: float,
        ef_used: int,
        query_type: str,
        intent_id: int = -1,
    ) -> None:
        """Record metrics for a single query.

        Args:
            latency_ms: Query latency in milliseconds
            satisfaction: Satisfaction score (0-1)
            ef_used: ef_search value used
            query_type: Type of query (exploratory, precise, mixed)
            intent_id: Detected intent cluster ID
        """
        self.latencies.append(latency_ms)
        self.satisfactions.append(satisfaction)
        efficiency = (satisfaction / (latency_ms / 1000.0)) if latency_ms > 0 else 0
        self.efficiencies.append(efficiency)
        self.ef_values.append(ef_used)
        self.query_types.append(query_type)
        self.intent_ids.append(intent_id)

    def record_q_values(self, q_values: Dict[int, Dict[int, float]]) -> None:
        """Record snapshot of Q-table values.

        Args:
            q_values: Dictionary mapping (intent_id, ef) to Q-value
        """
        self.q_value_history.append(q_values.copy())

    def get_phase_metrics(self, start_idx: int, end_idx: int) -> Dict[str, Any]:
        """Get metrics for a specific phase of the experiment.

        Args:
            start_idx: Start query index
            end_idx: End query index (exclusive)

        Returns:
            Dictionary with aggregated metrics
        """
        if end_idx > len(self.latencies):
            end_idx = len(self.latencies)

        if start_idx >= end_idx:
            return {}

        return {
            "avg_latency_ms": np.mean(self.latencies[start_idx:end_idx]),
            "avg_satisfaction": np.mean(self.satisfactions[start_idx:end_idx]),
            "avg_efficiency": np.mean(self.efficiencies[start_idx:end_idx]),
            "avg_ef": np.mean(self.ef_values[start_idx:end_idx]),
            "num_queries": end_idx - start_idx,
            "total_efficiency": np.sum(self.efficiencies[start_idx:end_idx]),
        }

    def get_convergence_metrics(self) -> Dict[str, Any]:
        """Compute convergence metrics (how quickly Q-values stabilize).

        Returns:
            Dictionary with convergence statistics
        """
        if len(self.q_value_history) < 2:
            return {}

        # Measure Q-value stability over time
        q_value_changes = []
        for i in range(1, len(self.q_value_history)):
            prev_q = self.q_value_history[i - 1]
            curr_q = self.q_value_history[i]

            # Compute average absolute change in Q-values
            changes = []
            for intent_id in curr_q:
                if intent_id not in prev_q:
                    continue
                for ef in curr_q[intent_id]:
                    if ef not in prev_q[intent_id]:
                        continue
                    prev_val = prev_q[intent_id].get(ef)
                    curr_val = curr_q[intent_id].get(ef)
                    # Skip if either value is None
                    if prev_val is None or curr_val is None:
                        continue
                    changes.append(abs(curr_val - prev_val))

            if changes:
                q_value_changes.append(np.mean(changes))

        return {
            "avg_q_change": np.mean(q_value_changes) if q_value_changes else 0,
            "final_q_change": q_value_changes[-1] if q_value_changes else 0,
            "q_stability": 1.0 - (np.mean(q_value_changes[-10:]) if len(q_value_changes) >= 10 else 0),
        }

    def save_results(self, filepath: str) -> None:
        """Save results to JSON file.

        Args:
            filepath: Path to save results
        """
        results = {
            "name": self.name,
            "num_queries": len(self.latencies),
            "overall_metrics": self.get_phase_metrics(0, len(self.latencies)),
            "convergence": self.get_convergence_metrics(),
            "raw_data": {
                "latencies": self.latencies,
                "satisfactions": self.satisfactions,
                "efficiencies": self.efficiencies,
                "ef_values": self.ef_values,
                "intent_ids": self.intent_ids,
            },
        }

        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2)


def compare_results(
    static: ExperimentTracker,
    adaptive: ExperimentTracker,
    phases: List[Tuple[str, int, int]] = None,
) -> Dict[str, Any]:
    """Compare static vs adaptive experiment results.

    Args:
        static: Static ef_search tracker
        adaptive: Adaptive learning tracker
        phases: List of (name, start_idx, end_idx) tuples for phase analysis

    Returns:
        Dictionary with comparison results
    """
    if phases is None:
        total = len(static.latencies)
        phases = [
            ("Early (0-25%)", 0, total // 4),
            ("Middle (25%-75%)", total // 4, 3 * total // 4),
            ("Late (75%-100%)", 3 * total // 4, total),
            ("Overall", 0, total),
        ]

    comparison = {
        "static_name": static.name,
        "adaptive_name": adaptive.name,
        "phases": {},
    }

    for phase_name, start, end in phases:
        static_metrics = static.get_phase_metrics(start, end)
        adaptive_metrics = adaptive.get_phase_metrics(start, end)

        if not static_metrics or not adaptive_metrics:
            continue

        # Compute improvements
        eff_improvement = (
            (adaptive_metrics["avg_efficiency"] - static_metrics["avg_efficiency"])
            / static_metrics["avg_efficiency"]
            * 100
            if static_metrics["avg_efficiency"] > 0
            else 0
        )

        sat_improvement = (
            (adaptive_metrics["avg_satisfaction"] - static_metrics["avg_satisfaction"])
            / static_metrics["avg_satisfaction"]
            * 100
            if static_metrics["avg_satisfaction"] > 0
            else 0
        )

        lat_improvement = (
            (static_metrics["avg_latency_ms"] - adaptive_metrics["avg_latency_ms"])
            / static_metrics["avg_latency_ms"]
            * 100
            if static_metrics["avg_latency_ms"] > 0
            else 0
        )

        comparison["phases"][phase_name] = {
            "static": static_metrics,
            "adaptive": adaptive_metrics,
            "improvements": {
                "efficiency_pct": eff_improvement,
                "satisfaction_pct": sat_improvement,
                "latency_pct": lat_improvement,
            },
        }

    return comparison


def print_comparison_table(comparison: Dict[str, Any]) -> None:
    """Print formatted comparison table.

    Args:
        comparison: Comparison results from compare_results()
    """
    print("\n" + "=" * 100)
    print("ADAPTIVE EF_SEARCH EXPERIMENT COMPARISON")
    print("=" * 100)
    print(f"\nStatic: {comparison['static_name']}")
    print(f"Adaptive: {comparison['adaptive_name']}")

    for phase_name, phase_data in comparison["phases"].items():
        print(f"\n{phase_name}")
        print("-" * 100)

        static = phase_data["static"]
        adaptive = phase_data["adaptive"]
        improvements = phase_data["improvements"]

        print(
            f"{'Metric':<30} | {'Static':>20} | {'Adaptive':>20} | {'Improvement':>20}"
        )
        print("-" * 100)

        # Efficiency (primary metric)
        print(
            f"{'Efficiency (sat/sec)':<30} | {static['avg_efficiency']:>20.2f} | "
            f"{adaptive['avg_efficiency']:>20.2f} | {improvements['efficiency_pct']:>19.1f}%"
        )

        # Satisfaction
        print(
            f"{'Satisfaction':<30} | {static['avg_satisfaction']:>19.1%} | "
            f"{adaptive['avg_satisfaction']:>19.1%} | {improvements['satisfaction_pct']:>19.1f}%"
        )

        # Latency
        print(
            f"{'Latency (ms)':<30} | {static['avg_latency_ms']:>20.2f} | "
            f"{adaptive['avg_latency_ms']:>20.2f} | {improvements['latency_pct']:>19.1f}%"
        )

        # ef_search
        print(
            f"{'Average ef_search':<30} | {static['avg_ef']:>20.1f} | "
            f"{adaptive['avg_ef']:>20.1f} | {'--':>20}"
        )

    # Overall verdict
    print("\n" + "=" * 100)
    overall_imp = comparison["phases"]["Overall"]["improvements"]["efficiency_pct"]

    print("VERDICT:")
    if overall_imp > 5.0:
        print(
            f"  Adaptive learning shows {overall_imp:.1f}% efficiency improvement - SIGNIFICANT"
        )
        print("  Adaptive approach strongly recommended.")
    elif overall_imp > 2.0:
        print(
            f"  Adaptive learning shows {overall_imp:.1f}% efficiency improvement - MODERATE"
        )
        print("  Adaptive approach provides measurable benefit.")
    elif overall_imp > 0:
        print(
            f"  Adaptive learning shows {overall_imp:.1f}% efficiency improvement - MARGINAL"
        )
        print("  Improvement exists but may not justify complexity.")
    else:
        print(
            f"  Adaptive learning shows {overall_imp:.1f}% efficiency change - NO IMPROVEMENT"
        )
        print("  Static ef_search performs as well or better.")

    print("=" * 100)


def generate_realistic_corpus(size: int = 1000) -> List[str]:
    """Generate realistic, diverse corpus for testing.

    Args:
        size: Number of documents to generate

    Returns:
        List of document strings
    """
    corpus = []

    # Technical documentation
    tech_topics = [
        "Python programming language features object-oriented functional paradigms",
        "Machine learning classification regression clustering algorithms neural networks",
        "Docker container orchestration microservices deployment isolation virtualization",
        "Kubernetes cluster management pod orchestration service discovery scaling",
        "REST API design principles HTTP methods resource endpoints versioning",
        "GraphQL query language schema types mutations subscriptions resolver functions",
        "PostgreSQL relational database ACID transactions indexing query optimization",
        "MongoDB NoSQL document database collections JSON aggregation pipelines",
        "Git version control distributed branches merging conflict resolution workflows",
        "CI/CD continuous integration deployment pipelines testing automation Jenkins",
        "JavaScript event loop asynchronous callbacks promises async await ES6",
        "TypeScript static typing interfaces generics compilation strict mode",
        "React component lifecycle hooks state management virtual DOM reconciliation",
        "Vue.js reactive data binding directives single file components composition API",
        "Angular framework dependency injection components services RxJS observables",
        "Node.js server-side runtime event-driven non-blocking I/O streams",
        "Express.js web framework middleware routing template engines REST APIs",
        "Django Python framework MVC ORM admin authentication middleware templates",
        "Flask lightweight microframework decorators Jinja2 templates blueprints extensions",
        "FastAPI async Python framework type hints automatic documentation Pydantic validation",
        "TensorFlow machine learning framework tensors computational graphs neural networks",
        "PyTorch deep learning dynamic graphs autograd tensors GPU acceleration",
        "Scikit-learn machine learning estimators pipelines preprocessing cross-validation metrics",
        "Pandas data manipulation dataframes series groupby merge pivot operations",
        "NumPy numerical computing arrays vectorization broadcasting linear algebra",
        "AWS cloud platform EC2 S3 Lambda RDS CloudFormation elastic infrastructure",
        "Azure Microsoft cloud services virtual machines storage databases serverless",
        "Google Cloud Platform compute engine storage BigQuery machine learning AI",
        "Linux operating system kernel shell commands file system processes networking",
        "Redis in-memory cache key-value store data structures pub/sub persistence",
        "RabbitMQ message broker queue AMQP routing exchanges bindings patterns",
        "Kafka distributed streaming platform topics partitions consumer groups replication",
        "Nginx reverse proxy load balancer web server static files caching",
        "Elasticsearch distributed search engine inverted index full-text queries aggregations",
        "Prometheus monitoring metrics time-series alerting Grafana visualization dashboards",
    ]

    # E-commerce product descriptions
    products = [
        "Wireless Bluetooth headphones premium sound quality active noise cancellation comfortable over-ear design",
        "Ergonomic office chair adjustable lumbar support breathable mesh high-density foam armrests",
        "Insulated stainless steel water bottle leak-proof double-wall vacuum 24-hour cold retention",
        "Aluminum laptop stand adjustable height ventilation holes cable management compatible MacBook",
        "Mechanical keyboard RGB backlight Cherry MX switches programmable keys anti-ghosting",
        "HD webcam autofocus microphone 1080p streaming video conferencing low-light correction",
        "Electric standing desk programmable height memory settings sturdy frame cable tray",
        "LED desk lamp adjustable brightness color temperature USB charging port touch control",
        "USB condenser microphone cardioid pattern shock mount pop filter recording streaming",
        "Powered USB hub charging ports data transfer aluminum compact design overcurrent protection",
        "Wireless gaming mouse high DPI programmable buttons ergonomic grip RGB lighting",
        "Dual monitor arm VESA mount gas spring height adjustment cable management rotation",
        "Cable management sleeve neoprene zipper design organize desk wires flexible adjustable",
        "Extended desk mat waterproof non-slip smooth surface keyboard mouse pad stitched edges",
        "Ergonomic footrest adjustable angle massage surface non-slip base circulation support",
        "Wooden headphone stand universal holder stable base cable organizer premium finish",
        "Portable power bank fast charging USB-C high capacity multiple devices LED indicator",
        "Padded laptop sleeve water-resistant zipper pocket fits 13-15 inch lightweight protection",
        "Wireless charging pad Qi-certified fast charge LED indicator non-slip surface compatible",
        "Blue light blocking glasses anti-glare UV protection comfortable frames screen eyestrain",
    ]

    # Customer support FAQs
    faq_templates = [
        ("reset account password", "Navigate to login page, click forgot password, enter email address, follow reset link sent to inbox within 5 minutes."),
        ("track shipping order", "Log into account, go to order history, click tracking number, view real-time delivery status and estimated arrival."),
        ("return defective product", "Items eligible for return within 30 days with original packaging. Contact support for RMA number and prepaid shipping label."),
        ("cancel subscription service", "Visit account billing settings, select active subscription, click cancel, confirm cancellation effective next billing cycle."),
        ("update payment method", "Go to account settings, select payment options, add new card or update existing, set as default payment method."),
        ("apply promotional discount code", "During checkout, enter promo code in designated field before completing purchase, discount automatically applied to total."),
        ("change delivery address", "Log into account, go to shipping addresses, add new address or modify existing, select as default for future orders."),
        ("request invoice receipt", "Access order history, select completed purchase, download PDF invoice or request email copy sent to registered address."),
        ("enable two-factor authentication", "Navigate to security settings, enable 2FA, scan QR code with authenticator app, enter verification code to confirm."),
        ("contact customer support team", "Email support@company.com or use live chat feature Monday-Friday 9AM-6PM, expect response within 24 hours."),
    ]

    # Research paper abstracts
    research_topics = [
        "Neural network architecture attention mechanism transformer models natural language processing semantic understanding",
        "Reinforcement learning policy gradient Q-learning exploration exploitation reward optimization convergence",
        "Computer vision convolutional networks object detection image segmentation feature extraction deep learning",
        "Natural language generation sequence-to-sequence encoder-decoder beam search language models GPT",
        "Graph neural networks node classification link prediction message passing graph convolution relational data",
        "Time series forecasting LSTM recurrent networks sequence modeling temporal dependencies prediction intervals",
        "Anomaly detection outlier analysis statistical methods isolation forest autoencoder unsupervised learning",
        "Recommendation systems collaborative filtering matrix factorization content-based hybrid approaches personalization",
        "Clustering algorithms k-means hierarchical DBSCAN density-based partitioning distance metrics silhouette",
        "Dimensionality reduction PCA t-SNE UMAP manifold learning visualization high-dimensional data compression",
    ]

    # Build corpus by category
    category_sizes = size // 4

    # Add technical docs
    for i in range(category_sizes):
        topic = tech_topics[i % len(tech_topics)]
        corpus.append(f"Documentation: {topic}. Comprehensive guide with examples, best practices, and troubleshooting tips.")

    # Add product descriptions
    for i in range(category_sizes):
        product = products[i % len(products)]
        corpus.append(f"Product {1000 + i}: {product}. Free shipping on orders over $50. 30-day return policy guaranteed.")

    # Add FAQs
    for i in range(category_sizes):
        question, answer = faq_templates[i % len(faq_templates)]
        corpus.append(f"Frequently Asked Question: How to {question}? Answer: {answer}")

    # Add research content
    for i in range(category_sizes):
        topic = research_topics[i % len(research_topics)]
        corpus.append(f"Research paper abstract: Novel approach to {topic}. Experimental results demonstrate significant improvements over baseline methods.")

    # Fill remaining with mixed content
    while len(corpus) < size:
        idx = len(corpus)
        if idx % 4 == 0:
            corpus.append(f"Technical specification document {idx} covering implementation details, API reference, configuration parameters.")
        elif idx % 4 == 1:
            corpus.append(f"User guide tutorial {idx} step-by-step instructions, screenshots, common pitfalls, troubleshooting procedures.")
        elif idx % 4 == 2:
            corpus.append(f"Product review {idx} customer feedback, ratings, pros and cons, usage experience, recommendations.")
        else:
            corpus.append(f"Knowledge base article {idx} detailed explanation, examples, related topics, external references.")

    return corpus[:size]


def create_realistic_queries(
    exploratory_count: int = 200,
    precise_count: int = 400,
    mixed_count: int = 400,
) -> Tuple[List[str], List[str]]:
    """Create realistic query set with diverse patterns.

    Args:
        exploratory_count: Number of exploratory queries (broad, need high ef_search)
        precise_count: Number of precise queries (specific, need low ef_search)
        mixed_count: Number of mixed queries (moderate specificity)

    Returns:
        Tuple of (queries, query_types) lists
    """
    queries = []
    query_types = []

    # Exploratory queries - broad, conceptual, need high recall
    exploratory_base = [
        "tell me about modern web development frameworks and tools",
        "what is cloud computing and how does it work",
        "explain machine learning concepts for beginners",
        "overview of database systems and when to use them",
        "introduction to containerization and orchestration",
        "best practices for API design and implementation",
        "software testing strategies and methodologies",
        "security fundamentals for web applications",
        "what products are available for home office setup",
        "show me ergonomic equipment for desk work",
        "help with account management and settings",
        "general information about shipping and delivery",
        "overview of return policies and procedures",
        "introduction to data science and analytics",
        "explain DevOps culture and practices",
        "what are microservices and their benefits",
        "guide to version control systems",
        "overview of agile development methodologies",
        "introduction to neural networks and deep learning",
        "explain reinforcement learning concepts",
    ]

    # Precise queries - specific, targeted, need low ef_search for speed
    precise_base = [
        "Python list comprehension syntax with nested loops",
        "PostgreSQL CREATE INDEX statement with CONCURRENTLY option",
        "React useState hook with functional component example",
        "Docker compose networking bridge mode configuration",
        "JWT token authentication middleware Express.js implementation",
        "Redis SET command with EX expiration time parameter",
        "wireless bluetooth headphones model 1000 specifications",
        "ergonomic office chair model 1001 price and availability",
        "mechanical keyboard RGB backlight model 1004 reviews",
        "HD webcam 1080p autofocus model 1005 compatibility",
        "how to reset forgotten password for user account",
        "track order using tracking number from confirmation email",
        "cancel monthly subscription before next billing cycle",
        "apply discount promotional code at checkout process",
        "photosynthesis light-dependent reactions chemical equation",
        "DNA replication semiconservative mechanism enzymes involved",
        "Newton second law of motion F equals ma formula",
        "water molecule H2O covalent bonding structure angle",
        "MongoDB aggregate pipeline group stage syntax example",
        "Kubernetes pod yaml manifest resource limits requests",
        "TensorFlow keras sequential model compile fit predict",
        "NumPy array broadcasting rules dimensional alignment",
        "Git rebase interactive squash commits edit history",
        "AWS S3 bucket policy JSON syntax permissions example",
        "Nginx reverse proxy pass configuration location block",
    ]

    # Mixed queries - moderate specificity, benefit from adaptation
    mixed_base = [
        "best JavaScript framework for building single page applications",
        "database comparison SQL versus NoSQL use cases",
        "laptop accessories to improve productivity and ergonomics",
        "affordable office chair under 300 dollars with lumbar support",
        "international shipping options to Canada and delivery time",
        "warranty coverage for electronic products and repair process",
        "renewable energy solar panel efficiency and cost savings",
        "climate change impact carbon emissions reduction strategies",
        "machine learning libraries Python scikit-learn TensorFlow PyTorch",
        "REST API authentication methods JWT OAuth session tokens",
        "Docker versus Kubernetes when to use each technology",
        "wireless headphones noise cancellation battery life comparison",
        "standing desk electric vs manual height adjustment pros cons",
        "monitor arm dual screen VESA mount compatibility",
        "password manager security features encryption cloud sync",
        "backup solution cloud storage versus local drive redundancy",
        "programming language comparison Python Java JavaScript TypeScript",
        "web framework performance benchmarks Django Flask FastAPI",
    ]

    # Generate queries by repeating base patterns
    for _ in range((exploratory_count // len(exploratory_base)) + 1):
        for q in exploratory_base:
            if len([qt for qt in query_types if qt == "exploratory"]) < exploratory_count:
                queries.append(q)
                query_types.append("exploratory")

    for _ in range((precise_count // len(precise_base)) + 1):
        for q in precise_base:
            if len([qt for qt in query_types if qt == "precise"]) < precise_count:
                queries.append(q)
                query_types.append("precise")

    for _ in range((mixed_count // len(mixed_base)) + 1):
        for q in mixed_base:
            if len([qt for qt in query_types if qt == "mixed"]) < mixed_count:
                queries.append(q)
                query_types.append("mixed")

    # Shuffle while keeping alignment
    indices = list(range(len(queries)))
    np.random.shuffle(indices)
    queries = [queries[i] for i in indices]
    query_types = [query_types[i] for i in indices]

    return queries, query_types


def simulate_feedback(
    query_type: str, results: List[Dict], k: int = 10
) -> Tuple[List[str], float]:
    """Simulate realistic user feedback based on query type.

    Args:
        query_type: Type of query (exploratory, precise, mixed)
        results: Search results
        k: Number of results requested

    Returns:
        Tuple of (relevant_ids, satisfaction_score)
    """
    if not results:
        return [], 0.0

    if query_type == "exploratory":
        # Exploratory: many relevant results (broad satisfaction)
        # User wants to see diverse results, high recall is valued
        relevant_ids = [r["id"] for r in results[:8]]
    elif query_type == "precise":
        # Precise: few highly relevant results (precision matters)
        # User wants exact match, speed is valued
        relevant_ids = [r["id"] for r in results[:2]]
    else:
        # Mixed: moderate relevance
        relevant_ids = [r["id"] for r in results[:5]]

    satisfaction = len(relevant_ids) / len(results) if results else 0
    return relevant_ids, satisfaction
