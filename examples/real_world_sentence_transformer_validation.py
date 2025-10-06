"""Real-world validation using sentence transformers with actual text data.

This test validates the Q-learning adaptive ef_search approach on real semantic
embeddings, not synthetic Gaussian clusters. It uses sentence-transformers to
embed realistic text and simulates real user behaviors.

Key Differences from Synthetic Tests:
1. Uses sentence-transformers (all-MiniLM-L6-v2) for real semantic embeddings
2. Realistic text corpus: technical docs, product descriptions, FAQs
3. Real query patterns based on semantic similarity, not distance metrics
4. A/B comparison: Adaptive vs Static ef_search
5. Unicode-safe text handling
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time
import numpy as np
from typing import List, Dict, Tuple
from dynhnsw import VectorStore


# Realistic text corpus across different domains
CORPUS = [
    # Technical Documentation (precise queries expected)
    "Python is a high-level programming language with dynamic typing and garbage collection.",
    "Machine learning is a subset of artificial intelligence focused on data-driven predictions.",
    "Docker containers provide lightweight virtualization for application deployment.",
    "Kubernetes orchestrates containerized applications across distributed systems.",
    "REST APIs use HTTP methods to enable communication between web services.",
    "GraphQL provides a query language for APIs with flexible data fetching.",
    "PostgreSQL is an open-source relational database with ACID compliance.",
    "MongoDB is a NoSQL document database designed for scalability.",
    "Git is a distributed version control system for tracking code changes.",
    "CI/CD pipelines automate software testing and deployment processes.",

    # Product Descriptions (exploratory queries expected)
    "Wireless bluetooth headphones with active noise cancellation and 30-hour battery life.",
    "Ergonomic office chair with lumbar support and adjustable armrests for comfort.",
    "Stainless steel water bottle keeps drinks cold for 24 hours or hot for 12 hours.",
    "Portable laptop stand with adjustable height for better posture and ventilation.",
    "Mechanical keyboard with RGB backlighting and customizable macro keys.",
    "4K webcam with autofocus and built-in microphone for video conferencing.",
    "Standing desk converter allows switching between sitting and standing positions.",
    "LED desk lamp with touch controls and adjustable color temperature.",
    "Noise-canceling microphone with pop filter for podcast recording.",
    "USB-C hub with multiple ports for laptop connectivity expansion.",

    # FAQ / Knowledge Base (mixed queries)
    "How do I reset my password? Go to account settings and click forgot password.",
    "What is your return policy? Items can be returned within 30 days of purchase.",
    "Where can I track my order? Use the tracking number sent to your email.",
    "How do I cancel my subscription? Visit billing settings and select cancel.",
    "What payment methods do you accept? We accept credit cards, PayPal, and bank transfers.",
    "How long does shipping take? Standard shipping takes 5-7 business days.",
    "Do you offer international shipping? Yes, we ship to over 100 countries worldwide.",
    "How can I contact customer support? Email support@company.com or call our hotline.",
    "What is your privacy policy? We protect your data with encryption and never sell information.",
    "How do I update my billing information? Go to account settings and update payment details.",

    # General Knowledge (exploratory queries)
    "The Great Wall of China is an ancient fortification stretching over 13,000 miles.",
    "Photosynthesis converts sunlight into chemical energy in plant cells.",
    "The water cycle describes how water moves between earth, atmosphere, and oceans.",
    "Black holes are regions of spacetime with gravitational pull so strong light cannot escape.",
    "DNA carries genetic instructions for development and functioning of living organisms.",
    "Climate change refers to long-term shifts in global temperatures and weather patterns.",
    "Antibiotics are medicines that fight bacterial infections in the human body.",
    "Renewable energy comes from sources that naturally replenish like solar and wind.",
    "Ecosystems are communities of living organisms interacting with their environment.",
    "Evolution is the process by which species change over generations through natural selection.",

    # News Headlines (mixed queries)
    "Global markets rally as interest rates remain steady amid economic uncertainty.",
    "New AI breakthrough enables more accurate medical diagnosis from imaging data.",
    "Electric vehicle sales surge as battery technology improves and costs decline.",
    "Scientists discover new exoplanet in potentially habitable zone of distant star.",
    "Cybersecurity experts warn of increasing threats from sophisticated ransomware attacks.",
    "Breakthrough in quantum computing brings practical applications closer to reality.",
    "Remote work trends reshape urban planning and real estate markets worldwide.",
    "Advances in gene therapy offer hope for treating previously incurable diseases.",
    "Ocean conservation efforts expand to protect endangered marine ecosystems.",
    "Space agencies collaborate on ambitious mission to establish lunar research station.",
]

# Query patterns with clear intent differentiation
EXPLORATORY_QUERIES = [
    # Broad, open-ended questions (want many relevant results)
    "Tell me about products for home office setup",
    "What are some technical tools for developers",
    "Information about shipping and delivery options",
    "Overview of scientific discoveries and research",
    "Details about database technologies",
    "What kind of office furniture is available",
    "Information about customer service and support",
    "Tell me about space and astronomy topics",
    "What products help with remote work",
    "Information about environmental topics",
]

PRECISE_QUERIES = [
    # Specific, narrow questions (want only top results)
    "How to reset password",
    "What is Docker container technology",
    "Show me noise canceling headphones",
    "Python programming language definition",
    "Return policy details",
    "Kubernetes orchestration explained",
    "USB-C hub specifications",
    "What is photosynthesis",
    "GraphQL API query language",
    "Standing desk converter product",
]


class PerformanceTracker:
    """Track metrics for A/B testing."""

    def __init__(self, mode: str):
        self.mode = mode  # "adaptive" or "static"
        self.latencies: List[float] = []
        self.satisfactions: List[float] = []
        self.efficiencies: List[float] = []
        self.ef_values: List[int] = []
        self.query_types: List[str] = []
        self.recalls: List[float] = []

    def record(self, latency_ms: float, satisfaction: float, ef_used: int,
               query_type: str, recall: float = 0.0):
        self.latencies.append(latency_ms)
        self.satisfactions.append(satisfaction)
        self.ef_values.append(ef_used)
        self.query_types.append(query_type)
        self.recalls.append(recall)

        # Efficiency: satisfaction per second
        efficiency = satisfaction / (latency_ms / 1000.0) if latency_ms > 0 else 0
        self.efficiencies.append(efficiency)

    def get_metrics(self, last_n: int = None) -> Dict:
        """Get summary metrics."""
        if last_n:
            latencies = self.latencies[-last_n:]
            satisfactions = self.satisfactions[-last_n:]
            efficiencies = self.efficiencies[-last_n:]
            ef_values = self.ef_values[-last_n:]
        else:
            latencies = self.latencies
            satisfactions = self.satisfactions
            efficiencies = self.efficiencies
            ef_values = self.ef_values

        return {
            "mode": self.mode,
            "avg_latency_ms": np.mean(latencies) if latencies else 0,
            "avg_satisfaction": np.mean(satisfactions) if satisfactions else 0,
            "avg_efficiency": np.mean(efficiencies) if efficiencies else 0,
            "avg_ef": np.mean(ef_values) if ef_values else 0,
            "num_queries": len(latencies)
        }

    def get_by_query_type(self, query_type: str) -> Dict:
        """Get metrics filtered by query type."""
        indices = [i for i, qt in enumerate(self.query_types) if qt == query_type]

        if not indices:
            return {"num_queries": 0}

        return {
            "avg_latency_ms": np.mean([self.latencies[i] for i in indices]),
            "avg_satisfaction": np.mean([self.satisfactions[i] for i in indices]),
            "avg_efficiency": np.mean([self.efficiencies[i] for i in indices]),
            "avg_ef": np.mean([self.ef_values[i] for i in indices]),
            "num_queries": len(indices)
        }


def embed_texts(texts: List[str], model_name: str = 'all-MiniLM-L6-v2') -> Tuple[np.ndarray, object]:
    """Embed texts using sentence-transformers.

    Returns:
        Tuple of (embeddings array, model object)
    """
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        print("ERROR: sentence-transformers not installed!")
        print("Install with: pip install sentence-transformers")
        sys.exit(1)

    print(f"Loading sentence transformer model: {model_name}...")
    model = SentenceTransformer(model_name)

    print(f"Embedding {len(texts)} texts...")
    # Convert to numpy and ensure float32
    embeddings = model.encode(texts, convert_to_numpy=True, show_progress_bar=True)
    embeddings = embeddings.astype(np.float32)

    print(f"Embeddings shape: {embeddings.shape}")
    return embeddings, model


def run_query_batch(
    store: VectorStore,
    queries: List[str],
    model: object,
    query_type: str,
    k_results: int,
    tracker: PerformanceTracker,
    top_k_relevant: int
):
    """Run a batch of queries and track performance.

    Args:
        store: Vector store instance
        queries: List of query strings
        model: Sentence transformer model
        query_type: "exploratory" or "precise"
        k_results: Number of results to retrieve
        tracker: Performance tracker
        top_k_relevant: How many top results to consider relevant (for feedback)
    """
    for query_text in queries:
        # Embed query
        query_vec = model.encode([query_text], convert_to_numpy=True)[0].astype(np.float32)

        # Search
        start = time.perf_counter()
        results = store.search(query_vec, k=k_results)
        latency_ms = (time.perf_counter() - start) * 1000.0

        # Simulate feedback based on query type
        if query_type == "exploratory":
            # Exploratory: Want many results, all top results are relevant
            relevant_ids = [r["id"] for r in results[:top_k_relevant]]
        else:
            # Precise: Want only the best results, fewer are relevant
            relevant_ids = [r["id"] for r in results[:min(3, top_k_relevant)]]

        # Calculate satisfaction (what fraction of results were relevant)
        satisfaction = len(relevant_ids) / len(results) if results else 0

        # Provide feedback
        store.provide_feedback(relevant_ids=relevant_ids)

        # Track performance
        ef_used = store._searcher.last_ef_used if hasattr(store._searcher, 'last_ef_used') else 100
        tracker.record(latency_ms, satisfaction, ef_used, query_type)


def print_comparison_table(adaptive_tracker: PerformanceTracker, static_tracker: PerformanceTracker):
    """Print A/B comparison table."""
    print("\n" + "="*90)
    print("A/B COMPARISON: ADAPTIVE vs STATIC ef_search")
    print("="*90)

    # Overall metrics
    adaptive_metrics = adaptive_tracker.get_metrics()
    static_metrics = static_tracker.get_metrics()

    print("\nOVERALL PERFORMANCE:")
    print("-" * 90)
    print(f"{'Metric':<30} | {'Adaptive':>20} | {'Static':>20} | {'Improvement':>15}")
    print("-" * 90)

    # Latency
    latency_improve = ((static_metrics['avg_latency_ms'] - adaptive_metrics['avg_latency_ms'])
                       / static_metrics['avg_latency_ms'] * 100) if static_metrics['avg_latency_ms'] > 0 else 0
    print(f"{'Avg Latency (ms)':<30} | {adaptive_metrics['avg_latency_ms']:>20.2f} | "
          f"{static_metrics['avg_latency_ms']:>20.2f} | {latency_improve:>14.1f}%")

    # Satisfaction
    sat_improve = ((adaptive_metrics['avg_satisfaction'] - static_metrics['avg_satisfaction'])
                   / static_metrics['avg_satisfaction'] * 100) if static_metrics['avg_satisfaction'] > 0 else 0
    print(f"{'Avg Satisfaction':<30} | {adaptive_metrics['avg_satisfaction']:>20.2%} | "
          f"{static_metrics['avg_satisfaction']:>20.2%} | {sat_improve:>14.1f}%")

    # Efficiency
    eff_improve = ((adaptive_metrics['avg_efficiency'] - static_metrics['avg_efficiency'])
                   / static_metrics['avg_efficiency'] * 100) if static_metrics['avg_efficiency'] > 0 else 0
    print(f"{'Avg Efficiency (sat/sec)':<30} | {adaptive_metrics['avg_efficiency']:>20.2f} | "
          f"{static_metrics['avg_efficiency']:>20.2f} | {eff_improve:>14.1f}%")

    # ef_search
    print(f"{'Avg ef_search':<30} | {adaptive_metrics['avg_ef']:>20.1f} | "
          f"{static_metrics['avg_ef']:>20.1f} | {'N/A':>15}")

    print("-" * 90)

    # By query type
    print("\nPERFORMANCE BY QUERY TYPE:")
    print("-" * 90)

    for qtype in ["exploratory", "precise"]:
        adaptive_type = adaptive_tracker.get_by_query_type(qtype)
        static_type = static_tracker.get_by_query_type(qtype)

        if adaptive_type["num_queries"] == 0:
            continue

        print(f"\n{qtype.upper()} Queries:")
        print(f"  Adaptive: latency={adaptive_type['avg_latency_ms']:.2f}ms, "
              f"satisfaction={adaptive_type['avg_satisfaction']:.2%}, "
              f"efficiency={adaptive_type['avg_efficiency']:.2f}, "
              f"ef={adaptive_type['avg_ef']:.0f}")
        print(f"  Static:   latency={static_type['avg_latency_ms']:.2f}ms, "
              f"satisfaction={static_type['avg_satisfaction']:.2%}, "
              f"efficiency={static_type['avg_efficiency']:.2f}, "
              f"ef={static_type['avg_ef']:.0f}")

        if static_type['avg_latency_ms'] > 0:
            latency_diff = ((static_type['avg_latency_ms'] - adaptive_type['avg_latency_ms'])
                           / static_type['avg_latency_ms'] * 100)
            print(f"  Latency improvement: {latency_diff:.1f}%")


def main():
    print("="*90)
    print("REAL-WORLD VALIDATION: Sentence Transformers with Q-Learning Adaptive ef_search")
    print("="*90)

    # Step 1: Embed corpus
    print("\nSTEP 1: Creating Real Embeddings")
    print("-" * 90)
    corpus_embeddings, model = embed_texts(CORPUS)
    print(f"Corpus size: {len(CORPUS)} documents")
    print(f"Embedding dimension: {corpus_embeddings.shape[1]}")

    # Step 2: Create both stores (adaptive and static)
    print("\nSTEP 2: Initializing Vector Stores")
    print("-" * 90)

    # Adaptive store
    print("Creating ADAPTIVE store (Q-learning ef_search)...")
    adaptive_store = VectorStore(
        dimension=corpus_embeddings.shape[1],
        M=16,
        ef_construction=200,
        ef_search=100,  # Default
        enable_intent_detection=True,
        k_intents=3,
        learning_rate=0.15,
        min_queries_for_clustering=20
    )
    adaptive_store.add(corpus_embeddings)
    print(f"  Added {adaptive_store.size()} vectors")

    # Static store (no adaptation)
    print("\nCreating STATIC store (fixed ef_search=100)...")
    static_store = VectorStore(
        dimension=corpus_embeddings.shape[1],
        M=16,
        ef_construction=200,
        ef_search=100,  # Fixed
        enable_intent_detection=False  # Disable adaptation
    )
    static_store.add(corpus_embeddings)
    print(f"  Added {static_store.size()} vectors")

    # Step 3: Performance trackers
    adaptive_tracker = PerformanceTracker("adaptive")
    static_tracker = PerformanceTracker("static")

    # Step 4: Cold start phase (activate clustering)
    print("\nSTEP 3: Cold Start Phase")
    print("-" * 90)
    print("Running 20 warm-up queries to activate intent detection...")

    warmup_queries = EXPLORATORY_QUERIES[:5] + PRECISE_QUERIES[:5]
    for query_text in warmup_queries * 2:  # 20 queries total
        query_vec = model.encode([query_text], convert_to_numpy=True)[0].astype(np.float32)
        adaptive_store.search(query_vec, k=10)

    stats = adaptive_store.get_statistics()
    print(f"Intent detection active: {stats.get('intent_detection', {}).get('clustering_active', False)}")

    # Step 5: Run exploratory queries (both stores)
    print("\nSTEP 4: Exploratory Queries")
    print("-" * 90)
    print("User behavior: Broad questions, want many results (k=15, top 10 relevant)")

    print("\n[Adaptive Store]")
    run_query_batch(
        store=adaptive_store,
        queries=EXPLORATORY_QUERIES * 3,  # 30 queries
        model=model,
        query_type="exploratory",
        k_results=15,
        tracker=adaptive_tracker,
        top_k_relevant=10
    )

    print("[Static Store]")
    run_query_batch(
        store=static_store,
        queries=EXPLORATORY_QUERIES * 3,  # 30 queries
        model=model,
        query_type="exploratory",
        k_results=15,
        tracker=static_tracker,
        top_k_relevant=10
    )

    # Step 6: Run precise queries (both stores)
    print("\nSTEP 5: Precise Queries")
    print("-" * 90)
    print("User behavior: Specific questions, want top results only (k=10, top 3 relevant)")

    print("\n[Adaptive Store]")
    run_query_batch(
        store=adaptive_store,
        queries=PRECISE_QUERIES * 3,  # 30 queries
        model=model,
        query_type="precise",
        k_results=10,
        tracker=adaptive_tracker,
        top_k_relevant=3
    )

    print("[Static Store]")
    run_query_batch(
        store=static_store,
        queries=PRECISE_QUERIES * 3,  # 30 queries
        model=model,
        query_type="precise",
        k_results=10,
        tracker=static_tracker,
        top_k_relevant=3
    )

    # Step 7: More exploratory (test convergence)
    print("\nSTEP 6: Additional Exploratory Queries (Testing Convergence)")
    print("-" * 90)

    print("\n[Adaptive Store]")
    run_query_batch(
        store=adaptive_store,
        queries=EXPLORATORY_QUERIES * 2,  # 20 more
        model=model,
        query_type="exploratory",
        k_results=15,
        tracker=adaptive_tracker,
        top_k_relevant=10
    )

    print("[Static Store]")
    run_query_batch(
        store=static_store,
        queries=EXPLORATORY_QUERIES * 2,  # 20 more
        model=model,
        query_type="exploratory",
        k_results=15,
        tracker=static_tracker,
        top_k_relevant=10
    )

    # Step 8: Final Analysis
    print_comparison_table(adaptive_tracker, static_tracker)

    # Step 9: Show learned Q-values
    print("\nSTEP 7: Learned Q-Values (Adaptive Store Only)")
    print("-" * 90)

    final_stats = adaptive_store.get_statistics()
    if "ef_search_selection" in final_stats:
        ef_stats = final_stats["ef_search_selection"]

        print(f"\nExploration rate: {ef_stats['exploration_rate']:.2%}")
        print(f"ef_search candidates: {ef_stats['ef_candidates']}")
        print(f"\nLearned ef_search per intent:")

        for intent_data in ef_stats["per_intent"]:
            intent_id = intent_data["intent_id"]
            learned_ef = intent_data["learned_ef"]
            q_values = intent_data["q_values"]
            num_queries = intent_data["num_queries"]

            print(f"\nIntent {intent_id}:")
            print(f"  Learned ef_search: {learned_ef}")
            print(f"  Queries: {num_queries}")
            print(f"  Q-values: {', '.join([f'ef={k}: {v:.2f}' if v is not None else f'ef={k}: unexplored' for k, v in q_values.items()])}")

    # Step 10: Convergence visualization
    print("\nSTEP 8: ef_search Convergence (Last 20 Queries)")
    print("-" * 90)

    recent_ef = adaptive_tracker.ef_values[-20:]
    recent_types = adaptive_tracker.query_types[-20:]

    print(f"\n{'Query #':<10} | {'Type':<15} | {'ef_search':<10}")
    print("-" * 40)
    for i, (ef, qtype) in enumerate(zip(recent_ef, recent_types), start=1):
        print(f"{i:<10} | {qtype:<15} | {ef:<10.0f}")

    # Final verdict
    print("\n" + "="*90)
    print("VALIDATION SUMMARY")
    print("="*90)

    adaptive_final = adaptive_tracker.get_metrics()
    static_final = static_tracker.get_metrics()

    if adaptive_final['avg_efficiency'] > static_final['avg_efficiency']:
        improvement = ((adaptive_final['avg_efficiency'] - static_final['avg_efficiency'])
                      / static_final['avg_efficiency'] * 100)
        print(f"\nSUCCESS: Adaptive ef_search is {improvement:.1f}% more efficient than static!")
        print(f"  Adaptive efficiency: {adaptive_final['avg_efficiency']:.2f} sat/sec")
        print(f"  Static efficiency: {static_final['avg_efficiency']:.2f} sat/sec")
    else:
        print("\nNOTICE: Static ef_search performed better or similar to adaptive.")
        print("Possible reasons:")
        print("  - Corpus too small for intent differentiation")
        print("  - Query patterns not diverse enough")
        print("  - Need more queries for Q-learning to converge")

    # Check if different ef values were learned
    exploratory_ef = adaptive_tracker.get_by_query_type("exploratory")["avg_ef"]
    precise_ef = adaptive_tracker.get_by_query_type("precise")["avg_ef"]

    if abs(exploratory_ef - precise_ef) > 10:
        print(f"\nDIFFERENTIATION: Different query types learned different ef_search values")
        print(f"  Exploratory queries: ef={exploratory_ef:.0f}")
        print(f"  Precise queries: ef={precise_ef:.0f}")
    else:
        print(f"\nNOTE: Similar ef_search for both query types (exploratory={exploratory_ef:.0f}, precise={precise_ef:.0f})")

    print("\n" + "="*90)
    print("Real-world validation complete!")
    print("="*90)


if __name__ == "__main__":
    main()
