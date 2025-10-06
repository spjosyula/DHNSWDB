"""Scenario 4: Intent Diversity Test

Test adaptive ef_search with highly diverse query intents and workload.

Hypothesis:
    Diverse query intents with different optimal ef_search values:
    1. Technical queries (precise, need low ef)
    2. Product search (exploratory, need high ef)
    3. FAQ lookup (precise, need low ef)
    4. Research queries (exploratory, need high ef)
    Adaptive learning identifies distinct intent patterns and optimizes per-intent.
    Should show 10-15% efficiency improvement over static.

Configuration:
    - Corpus size: 4000 documents (balanced across categories)
    - Embedding: sentence-transformers all-MiniLM-L6-v2 (384 dims)
    - Queries: 1500 balanced across intent types
    - k_intents: 8 (capture fine-grained intent diversity)
    - ef_candidates: [20, 35, 50, 70, 90, 110, 140, 180]
    - Static baseline: ef_search = 90
    - Adaptive: Q-learning with feedback

Expected Outcome:
    Adaptive should show 10-15% efficiency improvement by learning distinct
    ef_search preferences for each intent category.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time
import numpy as np
from typing import List, Tuple
from dynhnsw import VectorStore

# Import shared utilities
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from shared_utils import (
    ExperimentTracker,
    compare_results,
    print_comparison_table,
    generate_realistic_corpus,
    simulate_feedback,
)


EF_CANDIDATES = [20, 35, 50, 70, 90, 110, 140, 180]


def create_diverse_intent_queries(total_count: int = 1500) -> Tuple[List[str], List[str]]:
    """Create query set with diverse intent categories.

    Args:
        total_count: Total number of queries

    Returns:
        Tuple of (queries, query_types)
    """
    queries = []
    query_types = []

    per_category = total_count // 4

    # Category 1: Technical precise queries (low ef optimal)
    tech_precise = [
        "Python dict comprehension nested loops syntax error handling",
        "PostgreSQL VACUUM ANALYZE performance tuning parameters",
        "React useEffect cleanup function dependency array rules",
        "Docker multi-stage build optimization layer caching",
        "Git cherry-pick conflict resolution merge strategy",
        "Kubernetes CronJob schedule timezone configuration",
        "Redis pipeline transaction MULTI EXEC atomic operations",
        "MongoDB createIndex background unique sparse options",
        "Nginx location block regex matching priority order",
        "Elasticsearch mapping dynamic template field types",
        "TensorFlow checkpoint restore variable name scope",
        "NumPy einsum notation matrix multiplication broadcast",
        "AWS Lambda environment variable encryption KMS key",
        "Django queryset select_related prefetch_related optimization",
        "TypeScript generic constraints extends keyof typeof",
    ]

    for _ in range((per_category // len(tech_precise)) + 1):
        for q in tech_precise:
            if len([qt for qt in query_types if qt == "precise"]) < per_category:
                queries.append(q)
                query_types.append("precise")

    # Category 2: Product exploratory queries (high ef optimal)
    product_exploratory = [
        "best ergonomic office equipment for remote work setup",
        "show me wireless peripherals for productivity",
        "what desk accessories improve workspace organization",
        "ergonomic furniture for standing desk configuration",
        "USB devices for multi-monitor workstation",
        "cable management solutions for clean desk setup",
        "ambient lighting options for home office",
        "storage organizers for desk accessories",
        "ergonomic keyboard mouse combo recommendations",
        "monitor mount arms for dual screen setup",
        "desk mat extended size for keyboard mouse",
        "wireless charging accessories compatible devices",
        "laptop stands portable adjustable height",
        "headphone stands with USB hub features",
        "footrest adjustable ergonomic under desk",
    ]

    for _ in range((per_category // len(product_exploratory)) + 1):
        for q in product_exploratory:
            if len([qt for qt in query_types if qt == "exploratory"]) < per_category:
                queries.append(q)
                query_types.append("exploratory")

    # Category 3: FAQ precise queries (low ef optimal)
    faq_precise = [
        "how to reset password for locked account immediately",
        "track shipment with tracking number from email",
        "cancel subscription before next billing cycle today",
        "apply discount code at checkout cart page",
        "update payment method credit card expiration",
        "change shipping address for pending order",
        "download invoice receipt for past purchase",
        "enable two factor authentication security settings",
        "contact customer support urgent issue help",
        "return defective product get refund process",
        "unsubscribe marketing emails notification settings",
        "delete account permanently remove data GDPR",
        "recover deleted items from trash bin restore",
        "transfer ownership to another user account",
        "export data CSV JSON format download backup",
    ]

    for _ in range((per_category // len(faq_precise)) + 1):
        for q in faq_precise:
            if len([qt for qt in query_types if qt == "precise"]) < per_category * 2:
                queries.append(q)
                query_types.append("precise")

    # Category 4: Research exploratory queries (high ef optimal)
    research_exploratory = [
        "explain neural network architecture deep learning",
        "overview of reinforcement learning algorithms applications",
        "introduction to computer vision object detection",
        "natural language processing transformer models guide",
        "graph neural networks relational data overview",
        "time series forecasting methods machine learning",
        "anomaly detection techniques unsupervised learning",
        "recommendation systems collaborative filtering approaches",
        "clustering algorithms comparison use cases",
        "dimensionality reduction visualization techniques",
        "transfer learning pre-trained models fine-tuning",
        "ensemble methods boosting bagging stacking",
        "attention mechanisms sequence modeling NLP",
        "generative adversarial networks GAN training",
        "meta-learning few-shot learning paradigms",
    ]

    for _ in range((per_category // len(research_exploratory)) + 1):
        for q in research_exploratory:
            if len([qt for qt in query_types if qt == "exploratory"]) < per_category * 2:
                queries.append(q)
                query_types.append("exploratory")

    # Mixed queries for balance
    mixed_queries = [
        "database SQL NoSQL comparison when to use",
        "JavaScript framework React Vue Angular best choice",
        "cloud provider AWS Azure GCP pricing comparison",
        "office chair budget ergonomic under 400 dollars",
        "monitor 4K 27 inch color accuracy programming",
        "mechanical keyboard quiet switches typing speed",
        "microphone USB XLR audio quality streaming",
        "camera DSLR mirrorless video recording specs",
    ]

    remaining = total_count - len(queries)
    for _ in range((remaining // len(mixed_queries)) + 1):
        for q in mixed_queries:
            if len(queries) < total_count:
                queries.append(q)
                query_types.append("mixed")

    # Shuffle while keeping alignment
    indices = list(range(len(queries)))
    np.random.shuffle(indices)
    queries = [queries[i] for i in indices]
    query_types = [query_types[i] for i in indices]

    return queries[:total_count], query_types[:total_count]


def run_static_baseline(
    corpus_embeddings: np.ndarray,
    corpus_ids: List[str],
    queries: List[str],
    query_types: List[str],
    embedding_model,
    ef_search: int = 90,
) -> ExperimentTracker:
    """Run baseline with static ef_search.

    Args:
        corpus_embeddings: Pre-computed corpus embeddings
        corpus_ids: Document IDs
        queries: Query strings
        query_types: Query type labels
        embedding_model: SentenceTransformer model for encoding queries
        ef_search: Static ef_search value

    Returns:
        ExperimentTracker with results
    """
    print(f"\nCreating static baseline VectorStore (ef_search={ef_search})...")

    store = VectorStore(
        dimension=384,
        M=16,
        ef_construction=200,
        ef_search=ef_search,
        enable_intent_detection=False,  # No adaptation
    )

    print(f"Adding {len(corpus_embeddings)} documents...")
    store.add(corpus_embeddings, ids=corpus_ids)

    tracker = ExperimentTracker(f"static_ef{ef_search}_diverse")

    print(f"Running {len(queries)} queries with static ef_search={ef_search}...")
    for i, (query_text, qtype) in enumerate(zip(queries, query_types)):
        if (i + 1) % 150 == 0:
            print(f"  Progress: {i+1}/{len(queries)} queries")

        # Embed query
        q_vec = embedding_model.encode([query_text], convert_to_numpy=True)[0].astype(np.float32)

        # Search
        start = time.perf_counter()
        results = store.search(q_vec, k=10)
        latency_ms = (time.perf_counter() - start) * 1000.0

        # Simulate feedback
        relevant_ids, satisfaction = simulate_feedback(qtype, results, k=10)

        # Record metrics
        tracker.record(latency_ms, satisfaction, ef_search, qtype)

    print(f"Static baseline complete!")
    return tracker


def run_adaptive_learning(
    corpus_embeddings: np.ndarray,
    corpus_ids: List[str],
    queries: List[str],
    query_types: List[str],
    embedding_model,
    k_intents: int = 8,
) -> ExperimentTracker:
    """Run experiment with adaptive ef_search learning.

    Args:
        corpus_embeddings: Pre-computed corpus embeddings
        corpus_ids: Document IDs
        queries: Query strings
        query_types: Query type labels
        embedding_model: SentenceTransformer model for encoding queries
        k_intents: Number of intent clusters

    Returns:
        ExperimentTracker with results
    """
    print(f"\nCreating adaptive VectorStore (k_intents={k_intents})...")

    store = VectorStore(
        dimension=384,
        M=16,
        ef_construction=200,
        ef_search=100,  # Default, will be adapted
        enable_intent_detection=True,
        k_intents=k_intents,
        min_queries_for_clustering=80,
    )

    # Override ef_candidates
    from dynhnsw.ef_search_selector import EfSearchSelector
    store._searcher.ef_selector = EfSearchSelector(
        k_intents=k_intents,
        default_ef=100,
        exploration_rate=0.15,
        ef_candidates=EF_CANDIDATES,
    )

    # Lower confidence threshold for more aggressive adaptation
    store._searcher.confidence_threshold = 0.1

    print(f"Adding {len(corpus_embeddings)} documents...")
    store.add(corpus_embeddings, ids=corpus_ids)

    tracker = ExperimentTracker(f"adaptive_k{k_intents}_diverse")

    print(f"Running {len(queries)} queries with adaptive learning...")
    q_snapshot_interval = 150

    for i, (query_text, qtype) in enumerate(zip(queries, query_types)):
        if (i + 1) % 150 == 0:
            print(f"  Progress: {i+1}/{len(queries)} queries")

        # Embed query
        q_vec = embedding_model.encode([query_text], convert_to_numpy=True)[0].astype(np.float32)

        # Search
        start = time.perf_counter()
        results = store.search(q_vec, k=10)
        latency_ms = (time.perf_counter() - start) * 1000.0

        # Simulate feedback
        relevant_ids, satisfaction = simulate_feedback(qtype, results, k=10)
        store.provide_feedback(relevant_ids=relevant_ids)

        # Record metrics
        ef_used = store._searcher.last_ef_used
        intent_id = store._searcher.last_intent_id
        tracker.record(latency_ms, satisfaction, ef_used, qtype, intent_id)

        # Snapshot Q-values periodically
        if (i + 1) % q_snapshot_interval == 0:
            stats = store._searcher.ef_selector.get_statistics()
            q_values = {}
            for intent_info in stats["per_intent"]:
                intent_id = intent_info["intent_id"]
                q_values[intent_id] = intent_info["q_values"]
            tracker.record_q_values(q_values)

    print(f"Adaptive learning complete!")

    # Print final statistics
    stats = store._searcher.ef_selector.get_statistics()
    print(f"\nFinal Learning Statistics:")
    print(f"  Total updates: {stats['total_updates']}")
    print(f"  k_intents: {stats['k_intents']}")

    print(f"\nef_search distribution:")
    ef_counts = {}
    for ef in tracker.ef_values:
        ef_counts[ef] = ef_counts.get(ef, 0) + 1
    for ef in sorted(ef_counts.keys()):
        percentage = ef_counts[ef] / len(tracker.ef_values) * 100
        print(f"  ef={ef:>3}: {ef_counts[ef]:>4} queries ({percentage:>5.1f}%)")

    # Print per-intent learned preferences
    print(f"\nLearned ef_search preferences per intent:")
    for intent_info in stats["per_intent"]:
        intent_id = intent_info["intent_id"]
        q_vals = intent_info["q_values"]
        if q_vals:
            best_ef = max(q_vals, key=q_vals.get)
            best_q = q_vals[best_ef]
            print(f"  Intent {intent_id}: prefers ef={best_ef} (Q={best_q:.3f})")

    return tracker


def main():
    """Run Scenario 4: Intent Diversity."""
    print("\n" + "=" * 100)
    print("SCENARIO 4: INTENT DIVERSITY TEST")
    print("=" * 100)
    print("\nHypothesis:")
    print("  Diverse query intents have different optimal ef_search values.")
    print("  Adaptive learning identifies distinct patterns per intent.")
    print("  Should show 10-15% efficiency improvement over static.")
    print("\nConfiguration:")
    print(f"  Corpus size: 4000 documents")
    print(f"  Embedding model: sentence-transformers all-MiniLM-L6-v2 (384 dims)")
    print(f"  Queries: 1500 (diverse intents: tech, product, FAQ, research)")
    print(f"  k_intents: 8 (fine-grained intent detection)")
    print(f"  ef_candidates: {EF_CANDIDATES}")
    print("\nComparison:")
    print("  Static: ef_search = 90 (compromise)")
    print("  Adaptive: Q-learning with intent clustering")

    # Check dependencies
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        print("\nERROR: sentence-transformers not installed!")
        print("Install with: pip install sentence-transformers")
        sys.exit(1)

    # Generate corpus
    print("\n" + "-" * 100)
    print("CORPUS GENERATION")
    print("-" * 100)
    print("Generating realistic corpus...")
    corpus = generate_realistic_corpus(size=4000)
    print(f"  Generated {len(corpus)} documents")

    # Embed corpus
    print("\nEmbedding corpus with sentence-transformers...")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    print("  Model loaded: all-MiniLM-L6-v2")
    embeddings = model.encode(corpus, convert_to_numpy=True, show_progress_bar=True)
    embeddings = embeddings.astype(np.float32)
    corpus_ids = [f"doc_{i}" for i in range(len(embeddings))]
    print(f"  Embeddings shape: {embeddings.shape}")

    # Generate diverse intent queries
    print("\nGenerating diverse intent query set...")
    queries, query_types = create_diverse_intent_queries(total_count=1500)
    print(f"  Total queries: {len(queries)}")
    print(f"  Exploratory: {query_types.count('exploratory')}")
    print(f"  Precise: {query_types.count('precise')}")
    print(f"  Mixed: {query_types.count('mixed')}")

    # Run static baseline
    print("\n" + "-" * 100)
    print("STATIC BASELINE EXPERIMENT")
    print("-" * 100)
    static_tracker = run_static_baseline(
        corpus_embeddings=embeddings,
        corpus_ids=corpus_ids,
        queries=queries,
        query_types=query_types,
        embedding_model=model,
        ef_search=90,
    )

    # Run adaptive learning
    print("\n" + "-" * 100)
    print("ADAPTIVE LEARNING EXPERIMENT")
    print("-" * 100)
    adaptive_tracker = run_adaptive_learning(
        corpus_embeddings=embeddings,
        corpus_ids=corpus_ids,
        queries=queries,
        query_types=query_types,
        embedding_model=model,
        k_intents=8,
    )

    # Compare results
    comparison = compare_results(static_tracker, adaptive_tracker)
    print_comparison_table(comparison)

    # Convergence analysis
    print("\n" + "=" * 100)
    print("CONVERGENCE ANALYSIS")
    print("=" * 100)

    adaptive_conv = adaptive_tracker.get_convergence_metrics()

    print(f"\nAdaptive Learning Convergence:")
    print(f"  Average Q-value change: {adaptive_conv.get('avg_q_change', 0):.4f}")
    print(f"  Final Q-value change: {adaptive_conv.get('final_q_change', 0):.4f}")
    print(f"  Q-stability score: {adaptive_conv.get('q_stability', 0):.4f}")

    # Save results
    print("\n" + "=" * 100)
    print("Saving results...")
    static_tracker.save_results("adaptive_ef_tests/results/scenario_4_static.json")
    adaptive_tracker.save_results("adaptive_ef_tests/results/scenario_4_adaptive.json")
    print("  Saved to: adaptive_ef_tests/results/scenario_4_*.json")
    print("=" * 100)


if __name__ == "__main__":
    main()
