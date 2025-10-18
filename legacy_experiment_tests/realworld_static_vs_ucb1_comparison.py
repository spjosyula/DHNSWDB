"""Three-Way A/B Test: Static HNSW vs Baseline UCB1 vs Optimized UCB1

Comprehensive comparison of three configurations on real-world data:
1. Static HNSW: Fixed ef_search=100 (traditional approach)
2. Baseline UCB1: c=1.414, no warm start (standard UCB1)
3. Optimized UCB1: c=0.5, warm start with HNSW priors (proposed optimization)

Hypothesis:
  - Static HNSW provides consistent baseline performance
  - Baseline UCB1 may underperform due to exploration overhead
  - Optimized UCB1 should close the gap or exceed static HNSW by reducing overhead
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time
import json
import numpy as np
from typing import List, Dict, Tuple
from dynhnsw import VectorStore
from dynhnsw.config import DynHNSWConfig


def generate_corpus(size: int = 1000) -> List[str]:
    """Generate realistic document corpus."""
    tech_docs = [
        "Docker containers provide isolated environments for application deployment",
        "Kubernetes manages containerized applications across clusters",
        "REST APIs use HTTP methods for CRUD operations",
        "PostgreSQL is a relational database with ACID compliance",
        "MongoDB stores documents in JSON-like format",
        "Redis is an in-memory data store for caching",
        "Git tracks code changes with commits and branches",
        "CI/CD pipelines automate testing and deployment",
        "TensorFlow builds machine learning models",
        "PyTorch provides dynamic computational graphs",
    ]

    faq_docs = [
        "Reset password by clicking 'Forgot Password' on login page",
        "Orders can be returned within 30 days of purchase",
        "Track order using tracking number in confirmation email",
        "Cancel subscription in account settings",
        "We accept Visa, Mastercard, AmEx, and PayPal",
        "Standard shipping takes 5-7 business days",
        "Contact support via email, phone, or live chat",
        "Data is encrypted at rest and in transit",
        "Update billing information in account settings",
        "Warranty covers manufacturing defects for 1 year",
    ]

    product_docs = [
        "Wireless Bluetooth Headphones with 30-hour battery life",
        "Ergonomic Office Chair with lumbar support",
        "Stainless Steel Water Bottle with vacuum insulation",
        "Mechanical Keyboard with Cherry MX switches",
        "4K Webcam with autofocus and built-in microphone",
        "Electric Standing Desk with programmable presets",
        "LED Desk Lamp with adjustable color temperature",
        "USB Condenser Microphone for podcasting",
        "Ergonomic Wireless Mouse with 6 programmable buttons",
        "Dual Monitor Arm with gas spring system",
    ]

    tutorial_docs = [
        "Install Python from python.org and create virtual environment",
        "Clone repository with git clone and create branch",
        "Create Dockerfile and build with docker build command",
        "Define REST API routes with FastAPI decorators",
        "Create PostgreSQL database and define tables",
        "React components use JSX syntax and hooks",
        "CSS flexbox with display flex and justify-content",
        "Unit testing with pytest and assert statements",
        "Deploy application with nginx reverse proxy",
        "API documentation with OpenAPI and Swagger",
    ]

    corpus = []
    per_category = size // 4

    for _ in range(per_category):
        corpus.append(np.random.choice(tech_docs))
        corpus.append(np.random.choice(faq_docs))
        corpus.append(np.random.choice(product_docs))
        corpus.append(np.random.choice(tutorial_docs))

    return corpus[:size]


def generate_queries(num_queries: int = 2000) -> Tuple[List[str], List[str]]:
    """Generate realistic queries."""
    exploratory = [
        "How do I set up Docker containers?",
        "What are best practices for REST API design?",
        "Explain Git branching strategies",
        "How does Kubernetes work?",
        "What are microservices architectures?",
        "How do I optimize database performance?",
        "What products do you have for home office?",
        "Show me ergonomic furniture options",
        "How do I get started with programming?",
        "Explain software testing methodologies",
    ]

    precise = [
        "Docker run command syntax",
        "PostgreSQL create index statement",
        "React useState example",
        "Wireless headphones battery life",
        "How to reset password?",
        "Track order steps",
        "Cancel subscription process",
        "Return policy timeframe",
        "Customer support contact",
        "Warranty coverage details",
    ]

    navigational = [
        "Find Docker documentation",
        "Look up Kubernetes guide",
        "PostgreSQL tutorial",
        "React hooks reference",
        "Git commands cheat sheet",
        "FastAPI documentation",
        "pytest testing guide",
        "nginx configuration",
        "Product catalog",
        "FAQ page",
    ]

    queries = []
    query_types = []

    # 40% exploratory, 40% precise, 20% navigational
    for _ in range(num_queries):
        rand = np.random.random()
        if rand < 0.4:
            queries.append(np.random.choice(exploratory))
            query_types.append("exploratory")
        elif rand < 0.8:
            queries.append(np.random.choice(precise))
            query_types.append("precise")
        else:
            queries.append(np.random.choice(navigational))
            query_types.append("navigational")

    return queries, query_types


def simulate_feedback(query_type: str, results: List[Tuple[int, float]], k: int) -> Tuple[List[int], float]:
    """Simulate user feedback.

    Args:
        query_type: Type of query
        results: List of (doc_id, distance) tuples from search
        k: Number of results requested

    Returns:
        Tuple of (relevant_ids, satisfaction_score)
    """
    if not results:
        return [], 0.0

    if query_type == "exploratory":
        # High satisfaction with many results
        num_relevant = min(int(0.7 * k), len(results))
        satisfaction = 0.6 + 0.3 * (num_relevant / k)
    elif query_type == "precise":
        # High satisfaction with top result
        num_relevant = min(2, len(results))
        satisfaction = 0.8 if len(results) > 0 else 0.0
    else:  # navigational
        # Medium satisfaction
        num_relevant = min(3, len(results))
        satisfaction = 0.5 + 0.3 * (num_relevant / k)

    # results is list of dicts with 'id', 'distance', 'vector', 'metadata' keys
    relevant_ids = [result['id'] for result in results[:num_relevant]]
    return relevant_ids, satisfaction


def run_experiment(variant_name: str, config: DynHNSWConfig, corpus: List[str],
                   queries: List[str], query_types: List[str], model) -> Dict:
    """Run experiment for a single variant."""

    print(f"\n{'='*100}")
    print(f"{variant_name}")
    print(f"{'='*100}")

    # Embed corpus
    embeddings = model.encode(corpus, convert_to_numpy=True, show_progress_bar=False)
    embeddings = embeddings.astype(np.float32)

    # Create store
    store = VectorStore(
        dimension=embeddings.shape[1],
        M=16,
        ef_construction=200,
        ef_search=100,
        enable_intent_detection=config.enable_ucb1,  # Only for UCB1 variants
        k_intents=3,
        config=config,
    )

    store.add(embeddings)
    print(f"  Corpus: {len(corpus)} documents")
    print(f"  Queries: {len(queries)}")
    print(f"  Configuration: {config.config_name}")
    if config.enable_ucb1:
        print(f"    UCB1 c: {config.ucb1_exploration_constant}")
        print(f"    Warm start: {config.enable_ucb1_warm_start}")

    # Run queries
    query_embeddings = model.encode(queries, convert_to_numpy=True, show_progress_bar=False)

    latencies = []
    satisfactions = []
    efficiencies = []
    ef_values = []

    for i, (query_vec, query_type) in enumerate(zip(query_embeddings, query_types)):
        k = 15 if query_type == "exploratory" else (5 if query_type == "precise" else 10)

        start_time = time.perf_counter()
        results = store.search(query_vec, k=k)
        latency_ms = (time.perf_counter() - start_time) * 1000.0

        relevant_ids, satisfaction = simulate_feedback(query_type, results, k)
        if relevant_ids and config.enable_ucb1:
            store.provide_feedback(relevant_ids=relevant_ids)

        latencies.append(latency_ms)
        satisfactions.append(satisfaction)
        efficiency = satisfaction / (latency_ms / 1000.0) if latency_ms > 0 else 0
        efficiencies.append(efficiency)

        if config.enable_ucb1:
            ef_values.append(store._searcher.last_ef_used)
        else:
            ef_values.append(100)  # Static ef_search

        if (i + 1) % 500 == 0:
            print(f"  Progress: {i+1}/{len(queries)} | Efficiency: {np.mean(efficiencies[max(0, i-499):i+1]):.2f} sat/sec")

    # Compute metrics
    avg_efficiency = np.mean(efficiencies)
    avg_satisfaction = np.mean(satisfactions)
    avg_latency = np.mean(latencies)
    avg_ef = np.mean(ef_values)

    print(f"\n  Results:")
    print(f"    Avg Efficiency: {avg_efficiency:.2f} sat/sec")
    print(f"    Avg Satisfaction: {avg_satisfaction:.1%}")
    print(f"    Avg Latency: {avg_latency:.2f} ms")
    print(f"    Avg ef_search: {avg_ef:.1f}")

    return {
        "variant": variant_name,
        "config_name": config.config_name,
        "avg_efficiency": avg_efficiency,
        "avg_satisfaction": avg_satisfaction,
        "avg_latency_ms": avg_latency,
        "avg_ef": avg_ef,
        "num_queries": len(queries),
        "raw_data": {
            "latencies": latencies,
            "satisfactions": satisfactions,
            "efficiencies": efficiencies,
            "ef_values": ef_values,
        }
    }


if __name__ == "__main__":
    print("\n" + "="*100)
    print("REAL-WORLD A/B TEST: Static HNSW vs Baseline UCB1 vs Optimized UCB1")
    print("="*100)

    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        print("ERROR: sentence-transformers not installed!")
        sys.exit(1)

    # Generate data
    print("\nGenerating realistic corpus and queries...")
    corpus = generate_corpus(size=1000)
    queries, query_types = generate_queries(num_queries=2000)

    print(f"  Corpus: {len(corpus)} documents")
    print(f"  Queries: {len(queries)}")
    print(f"  Query types:")
    print(f"    Exploratory: {query_types.count('exploratory')}")
    print(f"    Precise: {query_types.count('precise')}")
    print(f"    Navigational: {query_types.count('navigational')}")

    # Load model
    print("\nLoading sentence transformer model...")
    model = SentenceTransformer('all-MiniLM-L6-v2')

    # Run experiments
    results = {}

    # 1. Static HNSW (baseline)
    static_config = DynHNSWConfig(
        config_name="static_hnsw",
        enable_ucb1=False,
    )
    results["static"] = run_experiment(
        "VARIANT 1: Static HNSW (ef=100)",
        static_config,
        corpus,
        queries,
        query_types,
        model
    )

    # 2. Baseline UCB1
    baseline_ucb1_config = DynHNSWConfig(
        config_name="baseline_ucb1",
        enable_ucb1=True,
        enable_ucb1_warm_start=False,
        ucb1_exploration_constant=1.414,
        k_intents=3,
    )
    results["baseline_ucb1"] = run_experiment(
        "VARIANT 2: Baseline UCB1 (c=1.414, no warm start)",
        baseline_ucb1_config,
        corpus,
        queries,
        query_types,
        model
    )

    # 3. Optimized UCB1
    optimized_ucb1_config = DynHNSWConfig(
        config_name="optimized_ucb1",
        enable_ucb1=True,
        enable_ucb1_warm_start=True,
        ucb1_exploration_constant=0.5,
        k_intents=3,
    )
    results["optimized_ucb1"] = run_experiment(
        "VARIANT 3: Optimized UCB1 (c=0.5, warm start)",
        optimized_ucb1_config,
        corpus,
        queries,
        query_types,
        model
    )

    # Comparison
    print(f"\n\n{'='*100}")
    print("COMPARISON SUMMARY")
    print(f"{'='*100}")

    print(f"\n{'Variant':<40} | {'Efficiency':>15} | {'Latency (ms)':>15} | {'Avg ef':>10}")
    print("-" * 100)

    for key in ["static", "baseline_ucb1", "optimized_ucb1"]:
        r = results[key]
        print(f"{r['variant']:<40} | {r['avg_efficiency']:>15.2f} | {r['avg_latency_ms']:>15.2f} | {r['avg_ef']:>10.1f}")

    # Calculate improvements
    static_eff = results["static"]["avg_efficiency"]
    baseline_eff = results["baseline_ucb1"]["avg_efficiency"]
    optimized_eff = results["optimized_ucb1"]["avg_efficiency"]

    baseline_vs_static = ((baseline_eff - static_eff) / static_eff) * 100
    optimized_vs_static = ((optimized_eff - static_eff) / static_eff) * 100
    optimized_vs_baseline = ((optimized_eff - baseline_eff) / baseline_eff) * 100

    print(f"\n{'Comparison':<40} | {'Change':>15}")
    print("-" * 60)
    print(f"{'Baseline UCB1 vs Static':<40} | {baseline_vs_static:>14.1f}%")
    print(f"{'Optimized UCB1 vs Static':<40} | {optimized_vs_static:>14.1f}%")
    print(f"{'Optimized UCB1 vs Baseline UCB1':<40} | {optimized_vs_baseline:>14.1f}%")

    # Verdict
    print(f"\n{'='*100}")
    print("VERDICT")
    print(f"{'='*100}")

    if optimized_eff > static_eff and optimized_eff > baseline_eff:
        print(f"[BEST] Optimized UCB1 is the top performer")
        print(f"   - {optimized_vs_static:+.1f}% vs Static HNSW")
        print(f"   - {optimized_vs_baseline:+.1f}% vs Baseline UCB1")
    elif static_eff > optimized_eff and static_eff > baseline_eff:
        print(f"[WINNER] Static HNSW outperforms both UCB1 variants")
        print(f"   - Baseline UCB1: {baseline_vs_static:.1f}% (overhead too high)")
        print(f"   - Optimized UCB1: {optimized_vs_static:.1f}% (failed optimization)")
    else:
        print(f"[INFO] Mixed results - workload characteristics matter")

    # Save results
    output_path = "examples/results/realworld_3way_comparison.json"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Remove raw data for cleaner JSON
    summary = {}
    for key, value in results.items():
        summary[key] = {k: v for k, v in value.items() if k != "raw_data"}

    summary["comparison"] = {
        "baseline_ucb1_vs_static_pct": baseline_vs_static,
        "optimized_ucb1_vs_static_pct": optimized_vs_static,
        "optimized_ucb1_vs_baseline_pct": optimized_vs_baseline,
    }

    with open(output_path, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\n\nResults saved to: {output_path}")
    print("="*100)
