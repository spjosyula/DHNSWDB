"""Shared utilities for epsilon decay experiments.

This module provides common functionality used across all test scenarios:
- Corpus generation
- Query creation
- Metrics tracking
- Result comparison
"""

import time
import json
import numpy as np
from typing import List, Dict, Tuple, Any
from pathlib import Path


class ExperimentTracker:
    """Track metrics for epsilon decay experiments."""

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
        self.epsilon_values = []
        self.intent_ids = []
        self.q_value_history = []  # Track Q-value evolution

    def record(
        self,
        latency_ms: float,
        satisfaction: float,
        ef_used: int,
        query_type: str,
        epsilon: float,
        intent_id: int = -1,
    ) -> None:
        """Record metrics for a single query.

        Args:
            latency_ms: Query latency in milliseconds
            satisfaction: Satisfaction score (0-1)
            ef_used: ef_search value used
            query_type: Type of query (exploratory, precise, mixed)
            epsilon: Current exploration rate
            intent_id: Detected intent cluster ID
        """
        self.latencies.append(latency_ms)
        self.satisfactions.append(satisfaction)
        efficiency = (satisfaction / (latency_ms / 1000.0)) if latency_ms > 0 else 0
        self.efficiencies.append(efficiency)
        self.ef_values.append(ef_used)
        self.query_types.append(query_type)
        self.epsilon_values.append(epsilon)
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
            "avg_epsilon": np.mean(self.epsilon_values[start_idx:end_idx]),
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
                "epsilon_values": self.epsilon_values,
                "intent_ids": self.intent_ids,
            },
        }

        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2)


def compare_results(
    control: ExperimentTracker,
    treatment: ExperimentTracker,
    phases: List[Tuple[str, int, int]] = None,
) -> Dict[str, Any]:
    """Compare control vs treatment experiment results.

    Args:
        control: Control group tracker (e.g., fixed epsilon)
        treatment: Treatment group tracker (e.g., GLIE decay)
        phases: List of (name, start_idx, end_idx) tuples for phase analysis

    Returns:
        Dictionary with comparison results
    """
    if phases is None:
        total = len(control.latencies)
        phases = [
            ("Early (0-25%)", 0, total // 4),
            ("Middle (25%-75%)", total // 4, 3 * total // 4),
            ("Late (75%-100%)", 3 * total // 4, total),
            ("Overall", 0, total),
        ]

    comparison = {
        "control_name": control.name,
        "treatment_name": treatment.name,
        "phases": {},
    }

    for phase_name, start, end in phases:
        control_metrics = control.get_phase_metrics(start, end)
        treatment_metrics = treatment.get_phase_metrics(start, end)

        if not control_metrics or not treatment_metrics:
            continue

        # Compute improvements
        eff_improvement = (
            (treatment_metrics["avg_efficiency"] - control_metrics["avg_efficiency"])
            / control_metrics["avg_efficiency"]
            * 100
            if control_metrics["avg_efficiency"] > 0
            else 0
        )

        sat_improvement = (
            (treatment_metrics["avg_satisfaction"] - control_metrics["avg_satisfaction"])
            / control_metrics["avg_satisfaction"]
            * 100
            if control_metrics["avg_satisfaction"] > 0
            else 0
        )

        lat_improvement = (
            (control_metrics["avg_latency_ms"] - treatment_metrics["avg_latency_ms"])
            / control_metrics["avg_latency_ms"]
            * 100
            if control_metrics["avg_latency_ms"] > 0
            else 0
        )

        comparison["phases"][phase_name] = {
            "control": control_metrics,
            "treatment": treatment_metrics,
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
    print("EPSILON DECAY EXPERIMENT COMPARISON")
    print("=" * 100)
    print(f"\nControl: {comparison['control_name']}")
    print(f"Treatment: {comparison['treatment_name']}")

    for phase_name, phase_data in comparison["phases"].items():
        print(f"\n{phase_name}")
        print("-" * 100)

        control = phase_data["control"]
        treatment = phase_data["treatment"]
        improvements = phase_data["improvements"]

        print(
            f"{'Metric':<30} | {'Control':>20} | {'Treatment':>20} | {'Improvement':>20}"
        )
        print("-" * 100)

        # Epsilon
        print(
            f"{'Average Epsilon':<30} | {control['avg_epsilon']:>20.4f} | "
            f"{treatment['avg_epsilon']:>20.4f} | {'--':>20}"
        )

        # Efficiency (primary metric)
        print(
            f"{'Efficiency (sat/sec)':<30} | {control['avg_efficiency']:>20.2f} | "
            f"{treatment['avg_efficiency']:>20.2f} | {improvements['efficiency_pct']:>19.1f}%"
        )

        # Satisfaction
        print(
            f"{'Satisfaction':<30} | {control['avg_satisfaction']:>19.1%} | "
            f"{treatment['avg_satisfaction']:>19.1%} | {improvements['satisfaction_pct']:>19.1f}%"
        )

        # Latency
        print(
            f"{'Latency (ms)':<30} | {control['avg_latency_ms']:>20.2f} | "
            f"{treatment['avg_latency_ms']:>20.2f} | {improvements['latency_pct']:>19.1f}%"
        )

        # ef_search
        print(
            f"{'Average ef_search':<30} | {control['avg_ef']:>20.1f} | "
            f"{treatment['avg_ef']:>20.1f} | {'--':>20}"
        )

    # Overall verdict
    print("\n" + "=" * 100)
    overall_imp = comparison["phases"]["Overall"]["improvements"]["efficiency_pct"]

    print("VERDICT:")
    if overall_imp > 2.0:
        print(
            f"  Treatment shows {overall_imp:.1f}% efficiency improvement - SIGNIFICANT"
        )
        print("  Theoretical improvement validated. Treatment recommended.")
    elif overall_imp > 0:
        print(
            f"  Treatment shows {overall_imp:.1f}% efficiency improvement - MARGINAL"
        )
        print("  Improvement exists but may not justify added complexity.")
    else:
        print(
            f"  Treatment shows {overall_imp:.1f}% efficiency change - NO IMPROVEMENT"
        )
        print("  Control performs as well or better. Keep current implementation.")

    print("=" * 100)


def generate_large_corpus(size: int = 200) -> List[str]:
    """Generate diverse corpus for testing.

    Args:
        size: Number of documents to generate

    Returns:
        List of document strings
    """
    corpus = []

    # Technical topics
    tech_topics = [
        "Python programming language",
        "Machine learning algorithms",
        "Docker containers",
        "Kubernetes orchestration",
        "REST APIs",
        "GraphQL queries",
        "PostgreSQL database",
        "MongoDB NoSQL",
        "Git version control",
        "CI/CD pipelines",
        "JavaScript runtime",
        "TypeScript compiler",
        "React framework",
        "Vue.js framework",
        "Angular platform",
        "Node.js server",
        "Express.js middleware",
        "Django framework",
        "Flask microframework",
        "FastAPI async",
        "TensorFlow library",
        "PyTorch tensors",
        "Scikit-learn models",
        "Pandas dataframes",
        "NumPy arrays",
        "AWS cloud services",
        "Azure platform",
        "Google Cloud Platform",
        "Linux operating system",
        "Windows OS",
        "MacOS system",
        "Redis cache",
        "RabbitMQ messaging",
        "Kafka streaming",
        "Nginx web server",
        "Apache HTTP server",
        "Elasticsearch search",
        "Grafana monitoring",
        "Prometheus metrics",
        "Terraform infrastructure",
        "Ansible automation",
        "Jenkins CI",
        "GitHub Actions",
        "GitLab CI/CD",
        "Bitbucket pipelines",
        "Jira project tracking",
        "Confluence documentation",
        "Slack communication",
        "Microsoft Teams",
        "Zoom video conferencing",
    ]

    for topic in tech_topics[:size // 4]:
        corpus.append(
            f"{topic} provides powerful features for modern software development and deployment."
        )

    # Product descriptions
    products = [
        "wireless headphones",
        "office chair",
        "water bottle",
        "laptop stand",
        "mechanical keyboard",
        "webcam",
        "standing desk",
        "desk lamp",
        "USB microphone",
        "USB hub",
        "wireless mouse",
        "monitor arm",
        "cable organizer",
        "desk mat",
        "ergonomic footrest",
        "headphone stand",
        "power bank",
        "laptop sleeve",
        "wireless charging pad",
        "blue light glasses",
        "ergonomic keyboard",
        "gaming mouse",
        "mechanical switches",
        "USB-C cable",
        "HDMI adapter",
        "screen protector",
        "phone case",
        "tablet stylus",
        "smart watch",
        "fitness tracker",
        "portable speaker",
        "white noise machine",
        "air purifier",
        "desk organizer",
        "pen holder",
        "notebook planner",
        "sticky notes",
        "highlighters",
        "scissors",
        "stapler",
        "tape dispenser",
        "calculator",
        "ruler",
        "clipboard",
        "file folders",
        "binder clips",
        "document shredder",
        "label maker",
        "battery charger",
        "surge protector",
    ]

    for i, product in enumerate(products[: size // 4]):
        corpus.append(
            f"Premium {product} with advanced features and excellent build quality for daily use. Model {i+1000}."
        )

    # FAQ documents
    faqs = [
        ("password", "Navigate to account settings and click forgot password link."),
        (
            "return",
            "Items can be returned within 30 days of purchase for full refund.",
        ),
        ("track order", "Use the tracking number sent to your confirmation email."),
        ("cancel subscription", "Visit billing settings and select cancel option."),
        ("payment", "We accept credit cards, PayPal, and bank transfers."),
        ("shipping time", "Standard delivery takes 5-7 business days."),
        ("international", "We ship to over 100 countries worldwide."),
        ("customer support", "Email support@company.com or call 24/7 hotline."),
        ("privacy", "We protect your data with encryption and secure storage."),
        ("billing", "Go to account settings to update payment details."),
        ("modify order", "Contact support within 1 hour of placing order."),
        ("gift wrap", "Select gift wrap option during checkout."),
        ("warranty", "All products include 1-year manufacturer warranty."),
        ("discount code", "Enter promotional code at checkout for savings."),
        ("wishlist", "Add items to wishlist from product page."),
        ("mobile app", "Available on iOS and Android app stores."),
        ("unsubscribe", "Click unsubscribe link in email footer."),
        ("hours", "Support available Monday-Friday 9AM-5PM."),
        ("bulk discount", "Contact sales team for enterprise pricing."),
        ("security", "We use SSL encryption and PCI DSS compliance."),
    ]

    for topic, answer in faqs[: size // 4]:
        corpus.append(f"How to {topic}? {answer}")

    # General knowledge
    knowledge = [
        "Great Wall China ancient fortification",
        "Photosynthesis plants sunlight conversion",
        "Water cycle atmosphere ocean evaporation",
        "Black holes spacetime gravity collapse",
        "DNA genetic instructions heredity",
        "Climate change global temperature rise",
        "Antibiotics bacterial infection treatment",
        "Renewable energy solar wind power",
        "Gravity force mass attraction",
        "Evolution natural selection adaptation",
        "Atoms molecules chemical elements",
        "Periodic table chemistry organization",
        "Cellular respiration energy production",
        "Immune system pathogen defense",
        "Nervous system signal transmission",
        "Cardiovascular heart blood circulation",
        "Digestive system nutrient absorption",
        "Skeletal bones structural support",
        "Muscular movement force generation",
        "Endocrine hormones regulation",
        "Mitosis cell division growth",
        "Meiosis genetic variation reproduction",
        "Ecosystems biodiversity interactions",
        "Food chains energy transfer",
        "Carbon cycle atmosphere exchange",
    ]

    for i, topic in enumerate(knowledge[: size // 4]):
        corpus.append(
            f"Scientific fact {i+1}: {topic} explains natural phenomena through empirical observation."
        )

    # Ensure we have exactly the requested size
    while len(corpus) < size:
        corpus.append(
            f"Additional document {len(corpus)} with general information content."
        )

    return corpus[:size]


def create_query_set(
    exploratory_count: int = 100,
    precise_count: int = 170,
    mixed_count: int = 56,
) -> Tuple[List[str], List[str]]:
    """Create diverse query set.

    Args:
        exploratory_count: Number of exploratory queries
        precise_count: Number of precise queries
        mixed_count: Number of mixed queries

    Returns:
        Tuple of (queries, query_types) lists
    """
    queries = []
    query_types = []

    # Exploratory queries (broad, need high ef_search)
    exploratory_base = [
        "tell me about programming",
        "what is cloud computing",
        "explain machine learning",
        "how does kubernetes work",
        "overview of web development",
        "introduction to databases",
        "what are microservices",
        "explain DevOps practices",
        "what is containerization",
        "how APIs work",
        "best practices coding",
        "software architecture patterns",
        "agile methodology overview",
        "testing strategies guide",
        "security basics introduction",
        "what products do you sell",
        "show me office equipment",
        "ergonomic products available",
        "technology accessories list",
        "home office setup guide",
        "productivity tools overview",
        "help with my account",
        "general questions support",
        "shipping information guide",
        "return policies overview",
    ]

    # Precise queries (specific, need low ef_search)
    precise_base = [
        "Python list comprehension syntax",
        "PostgreSQL index optimization technique",
        "React hooks useState example code",
        "Docker compose networking configuration",
        "JWT token authentication implementation",
        "Redis caching TTL configuration",
        "wireless bluetooth headphones model 1000",
        "ergonomic office chair model 1001",
        "mechanical keyboard RGB model 1004",
        "4K webcam autofocus model 1005",
        "how to reset password account",
        "track my order tracking number",
        "cancel subscription immediately now",
        "apply discount code at checkout",
        "photosynthesis chemical equation formula",
        "DNA double helix structure diagram",
        "Newton second law force formula",
        "water H2O molecular structure composition",
    ]

    # Mixed queries (medium specificity)
    mixed_base = [
        "best JavaScript framework 2024",
        "database comparison SQL NoSQL",
        "laptop accessories for productivity",
        "office chair under 300 dollars",
        "international shipping to Canada",
        "warranty coverage electronics products",
        "renewable energy solar panels efficiency",
        "climate change carbon emissions reduction",
    ]

    # Repeat to reach desired counts
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
        relevant_ids = [r["id"] for r in results[:7]]
    elif query_type == "precise":
        # Precise: few highly relevant results
        relevant_ids = [r["id"] for r in results[:2]]
    else:
        # Mixed: moderate relevance
        relevant_ids = [r["id"] for r in results[:4]]

    satisfaction = len(relevant_ids) / len(results) if results else 0
    return relevant_ids, satisfaction
