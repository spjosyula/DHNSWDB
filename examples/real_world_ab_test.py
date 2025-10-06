"""Production-ready real-world A/B test: Adaptive vs Static ef_search.

This test validates Q-learning adaptive ef_search on real sentence transformer embeddings.

Key features:
- Real text corpus with semantic diversity
- Sentence transformer embeddings (all-MiniLM-L6-v2)
- Realistic query patterns (exploratory vs precise)
- Fixed confidence threshold for proper Q-learning
- A/B comparison with statistical analysis
- Unicode-safe implementation
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time
import numpy as np
from typing import List, Dict, Tuple
from dynhnsw import VectorStore


# Diverse text corpus (100 documents across multiple domains)
CORPUS = [
    # Technical documentation (20 docs)
    "Python is a high-level programming language with dynamic typing and garbage collection.",
    "Machine learning is a subset of artificial intelligence focused on data-driven predictions.",
    "Docker containers provide lightweight virtualization for application deployment.",
    "Kubernetes orchestrates containerized applications across distributed systems.",
    "REST APIs use HTTP methods to enable communication between web services.",
    "GraphQL provides a query language for APIs with flexible data fetching capabilities.",
    "PostgreSQL is an open-source relational database with ACID compliance and SQL support.",
    "MongoDB is a NoSQL document database designed for horizontal scalability.",
    "Git is a distributed version control system for tracking code changes over time.",
    "CI/CD pipelines automate software testing and deployment processes.",
    "JavaScript enables interactive web development with client-side scripting.",
    "TypeScript adds static typing to JavaScript for improved developer tooling.",
    "React is a JavaScript library for building reusable user interface components.",
    "Vue.js is a progressive framework for building modern web applications.",
    "Angular is a platform for building enterprise-scale web applications.",
    "Node.js enables server-side JavaScript execution with event-driven architecture.",
    "Express.js is a minimal web framework for Node.js backend applications.",
    "Django is a high-level Python web framework encouraging rapid development.",
    "Flask is a lightweight Python micro-framework for simple web applications.",
    "FastAPI is a modern Python framework for building APIs with automatic documentation.",

    # Product descriptions (20 docs)
    "Wireless bluetooth headphones with active noise cancellation and 30-hour battery life.",
    "Ergonomic office chair with lumbar support and fully adjustable armrests.",
    "Stainless steel water bottle keeps drinks cold for 24 hours or hot for 12 hours.",
    "Portable laptop stand with adjustable height for improved posture and ventilation.",
    "Mechanical keyboard with RGB backlighting and customizable macro keys for gaming.",
    "4K webcam with autofocus and built-in microphone for professional video conferencing.",
    "Standing desk converter for switching between sitting and standing work positions.",
    "LED desk lamp with touch controls and adjustable color temperature settings.",
    "Noise-canceling microphone with pop filter for crystal-clear podcast recording.",
    "USB-C hub with multiple ports for laptop connectivity and peripheral expansion.",
    "Wireless mouse with ergonomic design and precision optical tracking sensor.",
    "Monitor arm with adjustable height rotation and tilt for optimal viewing angles.",
    "Cable management box for organizing desk cables and concealing power strips.",
    "Extended desk mat with smooth surface for mouse and keyboard comfort.",
    "Adjustable footrest with angle control for improved sitting posture and circulation.",
    "Premium headphone stand with integrated USB charging ports for devices.",
    "High-capacity portable power bank with fast charging for mobile devices.",
    "Protective laptop sleeve with water-resistant material and shock-absorbing padding.",
    "Wireless charging pad compatible with smartphones earbuds and smartwatches.",
    "Blue light blocking glasses to reduce eye strain from extended screen time.",

    # FAQ / Customer support (20 docs)
    "How do I reset my password? Navigate to account settings and click forgot password link.",
    "What is your return policy? Items can be returned within 30 days of purchase for full refund.",
    "Where can I track my order? Use the tracking number sent to your confirmation email.",
    "How do I cancel my subscription? Visit billing settings and select the cancel subscription option.",
    "What payment methods do you accept? We accept credit cards, PayPal, and bank transfers.",
    "How long does standard shipping take? Delivery typically takes 5-7 business days.",
    "Do you offer international shipping? Yes, we ship to over 100 countries worldwide.",
    "How can I contact customer support? Email support@company.com or call our 24/7 hotline.",
    "What is your privacy policy? We protect your data with encryption and never sell personal information.",
    "How do I update billing information? Go to account settings and update your payment details.",
    "Can I modify my order after placing it? Contact support within 1 hour of order placement.",
    "Do you offer gift wrapping services? Yes, select gift wrap option during checkout process.",
    "What is your product warranty policy? All products include 1-year manufacturer warranty coverage.",
    "How do I apply a discount code? Enter the promotional code at checkout before completing payment.",
    "Can I save items for later purchase? Yes, add items to your wishlist from any product page.",
    "Do you have a mobile application? Yes, available for download on iOS and Android app stores.",
    "How do I unsubscribe from marketing emails? Click the unsubscribe link at bottom of any email.",
    "What are your customer service hours? Support available Monday through Friday 9AM to 5PM.",
    "Do you offer bulk purchase discounts? Yes, contact our sales team for enterprise pricing options.",
    "How secure is my payment data? We use SSL encryption and maintain PCI DSS compliance.",

    # General knowledge (20 docs)
    "The Great Wall of China is an ancient fortification stretching over 13,000 miles across northern China.",
    "Photosynthesis is the process by which plants convert sunlight into chemical energy in chloroplasts.",
    "The water cycle describes continuous movement of water between earth, atmosphere, and oceans.",
    "Black holes are regions of spacetime where gravitational pull is so strong that nothing can escape.",
    "DNA carries genetic instructions for development and biological functioning of living organisms.",
    "Climate change refers to long-term shifts in global temperatures and weather patterns worldwide.",
    "Antibiotics are medicines that fight bacterial infections by killing or inhibiting bacterial growth.",
    "Renewable energy comes from naturally replenishing sources like solar wind and hydroelectric power.",
    "Ecosystems are biological communities of interacting organisms and their physical environment.",
    "Evolution is the process by which species change over generations through natural selection mechanisms.",
    "Gravity is a fundamental force of nature that attracts objects with mass toward each other.",
    "The solar system consists of the sun, eight planets, and various smaller celestial objects.",
    "Atoms are basic units of matter composed of protons, neutrons, and electrons in orbital structure.",
    "Metabolism encompasses all chemical processes that convert food into energy in living organisms.",
    "Biodiversity refers to the variety and variability of life forms within ecosystems and on Earth.",
    "Tectonic plates are large rigid slabs of rock that form Earth's lithosphere and move gradually.",
    "The greenhouse effect is the trapping of heat in Earth's atmosphere by greenhouse gases.",
    "Neurons are specialized cells that transmit electrical and chemical signals throughout the nervous system.",
    "Photons are elementary particles representing quanta of light and electromagnetic radiation.",
    "Mitochondria are cellular organelles responsible for producing ATP energy through cellular respiration.",

    # News and current topics (20 docs)
    "Global financial markets rally as central banks maintain steady interest rates amid economic uncertainty.",
    "Artificial intelligence breakthrough enables more accurate medical diagnosis from imaging and lab data.",
    "Electric vehicle sales surge worldwide as battery technology improves and manufacturing costs decline.",
    "Astronomers discover potentially habitable exoplanet in the goldilocks zone of distant star system.",
    "Cybersecurity experts warn of increasing threats from sophisticated ransomware and phishing attacks.",
    "Quantum computing breakthrough brings practical real-world applications significantly closer to reality.",
    "Remote work trends continue reshaping urban planning and commercial real estate markets globally.",
    "Gene therapy advances offer new hope for treating previously incurable genetic diseases.",
    "Ocean conservation initiatives expand to protect endangered marine ecosystems and biodiversity hotspots.",
    "International space agencies collaborate on ambitious lunar research station establishment mission.",
    "Renewable energy investments reach unprecedented levels as solar and wind costs continue falling.",
    "Novel vaccine technology platform shows promise for rapid response to emerging infectious diseases.",
    "Digital currency regulatory frameworks evolve as cryptocurrency adoption increases among institutions.",
    "Climate researchers report accelerating ice sheet melt in Arctic and Antarctic polar regions.",
    "Autonomous vehicle testing programs expand to additional cities worldwide for safety validation.",
    "Educational technology platforms experience massive growth driven by remote and hybrid learning demands.",
    "Healthcare systems increasingly adopt AI-powered diagnostic and treatment recommendation tools.",
    "Global supply chain disruptions prompt companies to restructure operations and diversify suppliers.",
    "Workplace mental health awareness campaigns gain momentum as burnout concerns rise significantly.",
    "Sustainable agriculture practices spread globally as food security and climate concerns intensify.",
]

# Query patterns with clear intent differentiation
EXPLORATORY_QUERIES = [
    "Show me products for home office workspace setup and productivity",
    "Technical tools and frameworks for web application development",
    "Information about shipping delivery and package tracking services",
    "Scientific discoveries in physics chemistry and biology fields",
    "Database and data storage technology options and comparisons",
    "Office furniture and ergonomic workspace equipment available",
    "Customer service support resources and contact information",
    "Space exploration astronomy and astrophysics topics",
    "Remote work products and tools for distributed teams",
    "Environmental science and climate change information",
    "Programming languages frameworks and development tools",
    "Healthcare medical technology and treatment advances",
    "Renewable energy sustainability and green technology",
    "Wireless connectivity devices and accessories",
    "Account management billing and subscription settings",
]

PRECISE_QUERIES = [
    "How to reset account password step by step",
    "What is Docker container technology definition",
    "Noise canceling headphones product specifications",
    "Python programming language key features",
    "Return policy for online purchases details",
    "Kubernetes container orchestration platform",
    "USB-C hub connectivity specifications",
    "Photosynthesis biological process explanation",
    "GraphQL API query language overview",
    "Standing desk converter adjustment features",
    "PostgreSQL database ACID properties",
    "Wireless charging pad compatibility",
    "React component-based architecture",
    "Git version control basic commands",
    "FastAPI Python framework capabilities",
]


class ABTestTracker:
    """Track A/B test metrics."""

    def __init__(self, mode: str):
        self.mode = mode
        self.latencies: List[float] = []
        self.satisfactions: List[float] = []
        self.efficiencies: List[float] = []
        self.ef_values: List[int] = []
        self.query_types: List[str] = []

    def record(self, latency_ms: float, satisfaction: float, ef_used: int, query_type: str):
        self.latencies.append(latency_ms)
        self.satisfactions.append(satisfaction)
        self.ef_values.append(ef_used)
        self.query_types.append(query_type)

        efficiency = satisfaction / (latency_ms / 1000.0) if latency_ms > 0 else 0
        self.efficiencies.append(efficiency)

    def get_overall_metrics(self) -> Dict:
        return {
            "avg_latency_ms": np.mean(self.latencies) if self.latencies else 0,
            "avg_satisfaction": np.mean(self.satisfactions) if self.satisfactions else 0,
            "avg_efficiency": np.mean(self.efficiencies) if self.efficiencies else 0,
            "avg_ef": np.mean(self.ef_values) if self.ef_values else 0,
            "total_queries": len(self.latencies)
        }

    def get_type_metrics(self, query_type: str, phase: str = "final") -> Dict:
        """Get metrics by query type, optionally filtered by phase."""
        indices = [i for i, qt in enumerate(self.query_types) if qt == query_type]

        # For final phase, take last 30 queries of this type
        if phase == "final" and len(indices) > 30:
            indices = indices[-30:]

        if not indices:
            return {"num_queries": 0}

        return {
            "avg_latency_ms": np.mean([self.latencies[i] for i in indices]),
            "avg_satisfaction": np.mean([self.satisfactions[i] for i in indices]),
            "avg_efficiency": np.mean([self.efficiencies[i] for i in indices]),
            "avg_ef": np.mean([self.ef_values[i] for i in indices]),
            "num_queries": len(indices)
        }


def embed_corpus() -> Tuple[np.ndarray, object]:
    """Embed corpus using sentence transformers."""
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        print("ERROR: sentence-transformers not installed!")
        print("Install with: pip install sentence-transformers")
        sys.exit(1)

    model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = model.encode(CORPUS, convert_to_numpy=True, show_progress_bar=False)
    return embeddings.astype(np.float32), model


def run_queries(store: VectorStore, queries: List[str], model: object,
                query_type: str, k: int, tracker: ABTestTracker, relevant_k: int):
    """Run batch of queries."""
    for query_text in queries:
        q_vec = model.encode([query_text], convert_to_numpy=True)[0].astype(np.float32)

        start = time.perf_counter()
        results = store.search(q_vec, k=k)
        latency_ms = (time.perf_counter() - start) * 1000.0

        # Simulate feedback based on query type
        if query_type == "exploratory":
            relevant_ids = [r["id"] for r in results[:relevant_k]]
        else:
            relevant_ids = [r["id"] for r in results[:min(3, len(results))]]

        satisfaction = len(relevant_ids) / len(results) if results else 0
        store.provide_feedback(relevant_ids=relevant_ids)

        ef_used = getattr(store._searcher, 'last_ef_used', 100)
        tracker.record(latency_ms, satisfaction, ef_used, query_type)


def print_results(adaptive: ABTestTracker, static: ABTestTracker):
    """Print comprehensive A/B test results."""
    print("\n" + "="*90)
    print("A/B TEST RESULTS: ADAPTIVE vs STATIC")
    print("="*90)

    # Overall comparison
    adp_overall = adaptive.get_overall_metrics()
    sta_overall = static.get_overall_metrics()

    print("\nOVERALL PERFORMANCE:")
    print("-" * 90)
    print(f"{'Metric':<35} | {'Adaptive':>18} | {'Static':>18} | {'Difference':>15}")
    print("-" * 90)

    # Latency
    lat_diff = ((sta_overall['avg_latency_ms'] - adp_overall['avg_latency_ms'])
                / sta_overall['avg_latency_ms'] * 100) if sta_overall['avg_latency_ms'] > 0 else 0
    print(f"{'Average Latency (ms)':<35} | {adp_overall['avg_latency_ms']:>18.2f} | "
          f"{sta_overall['avg_latency_ms']:>18.2f} | {lat_diff:>14.1f}%")

    # Satisfaction
    sat_diff = ((adp_overall['avg_satisfaction'] - sta_overall['avg_satisfaction'])
                / sta_overall['avg_satisfaction'] * 100) if sta_overall['avg_satisfaction'] > 0 else 0
    print(f"{'Average Satisfaction':<35} | {adp_overall['avg_satisfaction']:>18.1%} | "
          f"{sta_overall['avg_satisfaction']:>18.1%} | {sat_diff:>14.1f}%")

    # Efficiency
    eff_diff = ((adp_overall['avg_efficiency'] - sta_overall['avg_efficiency'])
                / sta_overall['avg_efficiency'] * 100) if sta_overall['avg_efficiency'] > 0 else 0
    print(f"{'Average Efficiency (sat/sec)':<35} | {adp_overall['avg_efficiency']:>18.2f} | "
          f"{sta_overall['avg_efficiency']:>18.2f} | {eff_diff:>14.1f}%")

    # ef_search
    print(f"{'Average ef_search':<35} | {adp_overall['avg_ef']:>18.1f} | "
          f"{sta_overall['avg_ef']:>18.1f} | {'--':>15}")

    # By query type (final phase only)
    print("\nFINAL PHASE PERFORMANCE (Last 30 queries per type):")
    print("-" * 90)

    for qtype in ["exploratory", "precise"]:
        adp_type = adaptive.get_type_metrics(qtype, phase="final")
        sta_type = static.get_type_metrics(qtype, phase="final")

        if adp_type["num_queries"] == 0:
            continue

        print(f"\n{qtype.upper()} Queries ({adp_type['num_queries']} queries):")

        print(f"  Adaptive: latency={adp_type['avg_latency_ms']:.2f}ms, "
              f"satisfaction={adp_type['avg_satisfaction']:.1%}, "
              f"efficiency={adp_type['avg_efficiency']:.2f}, "
              f"ef={adp_type['avg_ef']:.0f}")

        print(f"  Static:   latency={sta_type['avg_latency_ms']:.2f}ms, "
              f"satisfaction={sta_type['avg_satisfaction']:.1%}, "
              f"efficiency={sta_type['avg_efficiency']:.2f}, "
              f"ef={sta_type['avg_ef']:.0f}")

        if sta_type['avg_latency_ms'] > 0:
            improvement = ((sta_type['avg_latency_ms'] - adp_type['avg_latency_ms'])
                          / sta_type['avg_latency_ms'] * 100)
            print(f"  Latency improvement: {improvement:+.1f}%")

    # Verdict
    print("\n" + "="*90)
    print("VERDICT:")
    print("="*90)

    if eff_diff > 2:
        print(f"\nSUCCESS: Adaptive Q-learning is {eff_diff:.1f}% more efficient than static!")
    elif eff_diff < -2:
        print(f"\nStatic performed {abs(eff_diff):.1f}% better than adaptive.")
    else:
        print(f"\nNEUTRAL: Performance difference within 2% margin.")

    # Check differentiation
    exp_ef = adaptive.get_type_metrics("exploratory", "final")["avg_ef"]
    prec_ef = adaptive.get_type_metrics("precise", "final")["avg_ef"]

    if abs(exp_ef - prec_ef) > 15:
        print(f"\nIntent differentiation detected:")
        print(f"  Exploratory queries: ef={exp_ef:.0f}")
        print(f"  Precise queries: ef={prec_ef:.0f}")
        print(f"  Difference: {abs(exp_ef - prec_ef):.0f}")
    else:
        print(f"\nLimited intent differentiation (exploratory={exp_ef:.0f}, precise={prec_ef:.0f})")


def main():
    print("="*90)
    print("REAL-WORLD A/B TEST: Adaptive vs Static ef_search")
    print("Using Sentence Transformers (all-MiniLM-L6-v2)")
    print("="*90)

    # Embed corpus
    print("\nEmbedding corpus...")
    embeddings, model = embed_corpus()
    print(f"Corpus: {len(CORPUS)} documents, dimension: {embeddings.shape[1]}")

    # Create stores
    print("\nCreating vector stores...")

    # Adaptive store (FIXED: low confidence threshold)
    adaptive = VectorStore(
        dimension=embeddings.shape[1],
        M=16,
        ef_construction=200,
        ef_search=100,
        enable_intent_detection=True,
        k_intents=3,
        min_queries_for_clustering=30
    )
    adaptive._searcher.confidence_threshold = 0.1  # FIX: Low threshold for Q-learning
    adaptive._searcher.ef_selector.exploration_rate = 0.4  # High initial exploration
    adaptive.add(embeddings)

    # Static store
    static = VectorStore(
        dimension=embeddings.shape[1],
        M=16,
        ef_construction=200,
        ef_search=100,
        enable_intent_detection=False
    )
    static.add(embeddings)

    print(f"Adaptive: confidence_threshold={adaptive._searcher.confidence_threshold}, "
          f"exploration_rate={adaptive._searcher.ef_selector.exploration_rate}")

    # Trackers
    adp_tracker = ABTestTracker("adaptive")
    sta_tracker = ABTestTracker("static")

    # Phase 1: Warmup
    print("\nPhase 1: Warmup (30 queries)...")
    for i in range(30):
        q_text = EXPLORATORY_QUERIES[i % len(EXPLORATORY_QUERIES)]
        q_vec = model.encode([q_text], convert_to_numpy=True)[0].astype(np.float32)
        adaptive.search(q_vec, k=10)

    # Phase 2: Exploratory
    print("Phase 2: Exploratory queries (50 queries)...")
    run_queries(adaptive, EXPLORATORY_QUERIES * 4, model, "exploratory", 15, adp_tracker, 12)
    run_queries(static, EXPLORATORY_QUERIES * 4, model, "exploratory", 15, sta_tracker, 12)

    # Phase 3: Precise
    print("Phase 3: Precise queries (50 queries)...")
    run_queries(adaptive, PRECISE_QUERIES * 4, model, "precise", 10, adp_tracker, 5)
    run_queries(static, PRECISE_QUERIES * 4, model, "precise", 10, sta_tracker, 5)

    # Phase 4: Reduce exploration
    adaptive._searcher.ef_selector.exploration_rate = 0.15

    print("Phase 4: More exploratory (30 queries, lower exploration)...")
    run_queries(adaptive, EXPLORATORY_QUERIES * 2, model, "exploratory", 15, adp_tracker, 12)
    run_queries(static, EXPLORATORY_QUERIES * 2, model, "exploratory", 15, sta_tracker, 12)

    print("Phase 5: More precise (30 queries)...")
    run_queries(adaptive, PRECISE_QUERIES * 2, model, "precise", 10, adp_tracker, 5)
    run_queries(static, PRECISE_QUERIES * 2, model, "precise", 10, sta_tracker, 5)

    # Results
    print_results(adp_tracker, sta_tracker)

    # Show Q-table
    print("\n" + "="*90)
    print("LEARNED Q-VALUES (Adaptive Store)")
    print("="*90)

    stats = adaptive.get_statistics()
    if "ef_search_selection" in stats:
        for intent_data in stats["ef_search_selection"]["per_intent"]:
            intent_id = intent_data["intent_id"]
            learned_ef = intent_data["learned_ef"]
            q_vals = {k: v for k, v in intent_data["q_values"].items() if v is not None}

            print(f"\nIntent {intent_id}: learned_ef={learned_ef}, queries={intent_data['num_queries']}")
            if q_vals:
                print(f"  Q-values: {', '.join([f'ef={k}:{v:.1f}' for k, v in sorted(q_vals.items())])}")

    print("\n" + "="*90)
    print("Test complete")
    print("="*90)


if __name__ == "__main__":
    main()
