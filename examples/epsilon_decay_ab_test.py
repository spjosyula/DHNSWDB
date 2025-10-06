"""A/B Test: GLIE Epsilon Decay vs Fixed Epsilon for Q-Learning.

This test validates the theoretical improvement of epsilon decay by comparing:
- Control: Fixed epsilon = 0.15 (current implementation)
- Treatment: GLIE epsilon decay starting at 0.4, decaying as ε(t) = ε₀/(1 + t/100)

Expected Impact:
- Better exploration early (higher initial epsilon)
- More exploitation late (lower epsilon after convergence)
- Faster convergence to optimal Q-values
- Higher overall efficiency

Theoretical Foundation:
GLIE (Greedy in the Limit with Infinite Exploration) guarantees convergence
to optimal policy in reinforcement learning by ensuring:
1. All state-action pairs visited infinitely often
2. Policy becomes greedy in the limit

Test Design:
- Large real-world corpus (200+ documents)
- Sentence transformer embeddings
- 300+ queries (sufficient for decay to show effect)
- Diverse query types
- Realistic feedback patterns
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time
import numpy as np
from typing import List, Dict, Tuple
from dynhnsw import VectorStore


# Large diverse corpus (200 documents)
def generate_large_corpus() -> List[str]:
    """Generate diverse corpus of 200+ documents."""
    corpus = []

    # Technical topics (50 docs)
    tech_topics = [
        "Python programming language", "Machine learning algorithms",
        "Docker containers", "Kubernetes orchestration", "REST APIs",
        "GraphQL queries", "PostgreSQL database", "MongoDB NoSQL",
        "Git version control", "CI/CD pipelines", "JavaScript",
        "TypeScript", "React framework", "Vue.js", "Angular",
        "Node.js runtime", "Express.js", "Django framework", "Flask",
        "FastAPI", "TensorFlow", "PyTorch", "Scikit-learn",
        "Pandas", "NumPy", "AWS cloud", "Azure platform",
        "Google Cloud", "Linux server", "Windows OS", "MacOS system",
        "Redis cache", "RabbitMQ", "Kafka streaming", "Nginx",
        "Apache server", "Elasticsearch", "Grafana monitoring",
        "Prometheus metrics", "Terraform infrastructure", "Ansible automation",
        "Jenkins CI", "GitHub Actions", "GitLab pipelines", "Bitbucket",
        "Jira tracking", "Confluence wiki", "Slack messaging", "Teams collaboration",
        "Zoom meetings", "Visual Studio Code"
    ]

    for topic in tech_topics:
        corpus.append(f"{topic} provides powerful features for modern software development and deployment.")

    # Product descriptions (50 docs)
    products = [
        "wireless headphones", "office chair", "water bottle", "laptop stand",
        "keyboard", "webcam", "standing desk", "desk lamp", "microphone",
        "USB hub", "wireless mouse", "monitor arm", "cable box", "desk mat",
        "footrest", "headphone stand", "power bank", "laptop sleeve",
        "charging pad", "blue light glasses", "ergonomic keyboard",
        "gaming mouse", "mechanical switches", "USB cable", "HDMI adapter",
        "screen protector", "phone case", "tablet stylus", "smart watch",
        "fitness tracker", "portable speaker", "noise machine", "air purifier",
        "desk organizer", "pen holder", "notebook", "planner", "sticky notes",
        "markers", "highlighters", "scissors", "stapler", "tape dispenser",
        "calculator", "ruler", "clipboard", "file folders", "binder clips",
        "document shredder", "label maker"
    ]

    for i, product in enumerate(products):
        corpus.append(f"Premium {product} with advanced features and excellent build quality for daily use. Model {i+1000}.")

    # FAQ documents (50 docs)
    faqs = [
        ("password", "Navigate to account settings and click forgot password link."),
        ("return", "Items can be returned within 30 days of purchase for full refund."),
        ("track order", "Use the tracking number sent to your confirmation email."),
        ("cancel subscription", "Visit billing settings and select cancel option."),
        ("payment", "We accept credit cards, PayPal, and bank transfers."),
        ("shipping time", "Standard delivery takes 5-7 business days."),
        ("international", "We ship to over 100 countries worldwide."),
        ("customer support", "Email support@company.com or call 24/7 hotline."),
        ("privacy", "We protect your data with encryption."),
        ("billing", "Go to account settings to update payment details."),
        ("modify order", "Contact support within 1 hour of placing order."),
        ("gift wrap", "Select gift wrap option during checkout."),
        ("warranty", "All products include 1-year manufacturer warranty."),
        ("discount code", "Enter promotional code at checkout."),
        ("wishlist", "Add items to wishlist from product page."),
        ("mobile app", "Available on iOS and Android app stores."),
        ("unsubscribe", "Click unsubscribe link in email footer."),
        ("hours", "Support available Monday-Friday 9AM-5PM."),
        ("bulk discount", "Contact sales team for enterprise pricing."),
        ("security", "We use SSL encryption and PCI DSS compliance."),
        ("shipping cost", "Free shipping on orders over $50."),
        ("gift card", "Purchase gift cards in any denomination."),
        ("referral", "Refer a friend and get 10% off next purchase."),
        ("account delete", "Contact support to permanently delete account."),
        ("data export", "Request data export from privacy settings."),
        ("newsletter", "Subscribe to weekly newsletter for deals."),
        ("sale events", "Major sales happen quarterly throughout year."),
        ("student discount", "Students get 15% off with valid ID."),
        ("price match", "We match competitor prices on identical items."),
        ("backorder", "Backordered items ship within 2-3 weeks."),
        ("damaged item", "Report damaged items within 48 hours."),
        ("exchange", "Exchanges processed within 5 business days."),
        ("size guide", "Check size charts on product pages."),
        ("product care", "Follow care instructions on product labels."),
        ("assembly", "Assembly instructions included with furniture."),
        ("installation", "Professional installation available for fee."),
        ("compatibility", "Check compatibility before purchasing accessories."),
        ("upgrade", "Upgrade subscription anytime from account page."),
        ("downgrade", "Downgrade will apply at next billing cycle."),
        ("pause subscription", "Pause subscription for up to 3 months."),
        ("auto-renew", "Subscriptions auto-renew unless canceled."),
        ("free trial", "New users get 14-day free trial period."),
        ("demo", "Request live demo from sales team."),
        ("documentation", "Full documentation available in help center."),
        ("API access", "API access available on premium plans."),
        ("webhooks", "Configure webhooks in developer settings."),
        ("rate limits", "API rate limits vary by subscription tier."),
        ("SLA", "Enterprise plans include 99.9% uptime SLA."),
        ("support ticket", "Create support ticket from dashboard."),
        ("live chat", "Live chat available during business hours.")
    ]

    for topic, answer in faqs:
        corpus.append(f"How to {topic}? {answer}")

    # General knowledge (50 docs)
    knowledge = [
        "Great Wall China ancient fortification", "Photosynthesis plants sunlight",
        "Water cycle atmosphere oceans", "Black holes spacetime gravity",
        "DNA genetic instructions", "Climate change global temperatures",
        "Antibiotics bacterial infections", "Renewable energy solar wind",
        "Gravity force attraction", "Evolution natural selection",
        "Atoms molecules elements", "Periodic table chemistry",
        "Cellular respiration energy", "Immune system defense",
        "Nervous system signals", "Cardiovascular heart blood",
        "Digestive system nutrients", "Skeletal bones structure",
        "Muscular movement contraction", "Endocrine hormones regulation",
        "Mitosis cell division", "Meiosis genetic variation",
        "Photosynthesis oxygen glucose", "Ecosystems biodiversity",
        "Food chains energy transfer", "Carbon cycle atmosphere",
        "Nitrogen cycle bacteria", "Oxygen cycle photosynthesis",
        "Water properties life", "pH acids bases",
        "Chemical reactions bonds", "States matter solid liquid gas",
        "Energy conservation thermodynamics", "Newton laws motion",
        "Electromagnetic spectrum waves", "Sound waves frequency",
        "Light reflection refraction", "Electricity current voltage",
        "Magnetism poles fields", "Nuclear reactions fission fusion",
        "Quantum mechanics particles", "Relativity spacetime Einstein",
        "Big Bang universe origin", "Solar system planets sun",
        "Moon phases tides", "Eclipses shadows alignment",
        "Seasons Earth tilt orbit", "Weather patterns atmosphere",
        "Hurricanes tropical storms", "Tornados wind vortex",
        "Earthquakes tectonic plates", "Volcanoes magma eruption"
    ]

    for i, topic in enumerate(knowledge):
        corpus.append(f"Scientific fact {i+1}: {topic} explains natural phenomena through empirical observation.")

    return corpus


def create_query_set() -> Tuple[List[str], List[str]]:
    """Create diverse query set with 300+ queries."""
    queries = []
    query_types = []

    # Exploratory queries (broad, need high ef_search)
    exploratory = [
        "tell me about programming", "what is cloud computing",
        "explain machine learning", "how does kubernetes work",
        "overview of web development", "introduction to databases",
        "what are microservices", "explain DevOps practices",
        "what is containerization", "how APIs work",
        "best practices coding", "software architecture patterns",
        "agile methodology", "testing strategies", "security basics",
        "what products do you sell", "show me office equipment",
        "ergonomic products", "technology accessories",
        "home office setup", "productivity tools",
        "help with my account", "general questions",
        "shipping information", "return policies",
        "science topics", "how nature works",
        "environmental concepts", "physics principles"
    ] * 4  # 100 exploratory queries

    # Precise queries (specific, need low ef_search)
    precise = [
        "Python list comprehension syntax", "PostgreSQL index optimization",
        "React hooks useState example", "Docker compose networking",
        "JWT token authentication", "Redis caching TTL",
        "wireless bluetooth headphones model 1000", "ergonomic office chair model 1001",
        "mechanical keyboard RGB model 1004", "4K webcam autofocus model 1005",
        "how to reset password", "track my order number",
        "cancel subscription immediately", "apply discount code checkout",
        "photosynthesis chemical equation", "DNA double helix structure",
        "Newton second law formula", "water H2O molecular structure"
    ] * 10  # 170 precise queries

    # Mixed queries (medium specificity, need medium ef_search)
    mixed = [
        "best JavaScript framework 2024", "database comparison SQL NoSQL",
        "laptop accessories for productivity", "office chair under 300 dollars",
        "international shipping to Canada", "warranty coverage electronics",
        "renewable energy solar panels", "climate change carbon emissions"
    ] * 7  # 56 mixed queries

    for q in exploratory:
        queries.append(q)
        query_types.append("exploratory")

    for q in precise:
        queries.append(q)
        query_types.append("precise")

    for q in mixed:
        queries.append(q)
        query_types.append("mixed")

    # Shuffle while keeping type alignment
    indices = list(range(len(queries)))
    np.random.shuffle(indices)
    queries = [queries[i] for i in indices]
    query_types = [query_types[i] for i in indices]

    return queries, query_types


class ExperimentTracker:
    """Track metrics for epsilon decay experiment."""

    def __init__(self, name: str):
        self.name = name
        self.latencies = []
        self.satisfactions = []
        self.efficiencies = []
        self.ef_values = []
        self.query_types = []
        self.epsilon_values = []  # Track epsilon over time

    def record(self, latency_ms: float, satisfaction: float, ef_used: int,
               query_type: str, epsilon: float):
        """Record query metrics."""
        self.latencies.append(latency_ms)
        self.satisfactions.append(satisfaction)
        efficiency = (satisfaction / (latency_ms / 1000.0)) if latency_ms > 0 else 0
        self.efficiencies.append(efficiency)
        self.ef_values.append(ef_used)
        self.query_types.append(query_type)
        self.epsilon_values.append(epsilon)

    def get_phase_metrics(self, start_idx: int, end_idx: int) -> Dict:
        """Get metrics for a specific phase."""
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
            "num_queries": end_idx - start_idx
        }


def run_experiment(epsilon_decay_mode: str, initial_epsilon: float = 0.15) -> ExperimentTracker:
    """Run one experiment with specified epsilon decay mode."""

    # Import sentence transformers
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        print("ERROR: sentence-transformers not installed!")
        print("Install with: pip install sentence-transformers")
        sys.exit(1)

    # Generate data
    print(f"\nGenerating corpus and queries...")
    corpus = generate_large_corpus()
    queries, query_types = create_query_set()

    print(f"  Corpus: {len(corpus)} documents")
    print(f"  Queries: {len(queries)} queries")
    print(f"  Exploratory: {query_types.count('exploratory')}")
    print(f"  Precise: {query_types.count('precise')}")
    print(f"  Mixed: {query_types.count('mixed')}")

    # Embed corpus
    print(f"\nEmbedding corpus with sentence transformers...")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = model.encode(corpus, convert_to_numpy=True, show_progress_bar=False)
    embeddings = embeddings.astype(np.float32)

    # Create store with epsilon decay mode
    print(f"\nCreating VectorStore (epsilon_decay_mode={epsilon_decay_mode})...")
    store = VectorStore(
        dimension=384,
        M=16,
        ef_search=100,
        enable_intent_detection=True,
        k_intents=3,
        min_queries_for_clustering=30,
    )

    # Override ef_selector to use specified decay mode
    from dynhnsw.ef_search_selector import EfSearchSelector
    store._searcher.ef_selector = EfSearchSelector(
        k_intents=3,
        default_ef=100,
        exploration_rate=initial_epsilon,
        epsilon_decay_mode=epsilon_decay_mode,
        min_epsilon=0.01
    )

    # Lower confidence threshold for Q-learning to work
    store._searcher.confidence_threshold = 0.1

    # Add documents
    print(f"\nAdding {len(embeddings)} documents to store...")
    ids = [f"doc_{i}" for i in range(len(embeddings))]
    store.add(embeddings, ids=ids)

    # Run queries
    tracker = ExperimentTracker(f"{epsilon_decay_mode}_eps{initial_epsilon}")

    print(f"\nRunning {len(queries)} queries...")
    for i, (query_text, qtype) in enumerate(zip(queries, query_types)):
        if (i + 1) % 50 == 0:
            epsilon = store._searcher.ef_selector.exploration_rate
            print(f"  Progress: {i+1}/{len(queries)} queries, epsilon={epsilon:.4f}")

        # Embed query
        q_vec = model.encode([query_text], convert_to_numpy=True)[0].astype(np.float32)

        # Search
        start = time.perf_counter()
        results = store.search(q_vec, k=10)
        latency_ms = (time.perf_counter() - start) * 1000.0

        # Simulate realistic feedback
        if qtype == "exploratory":
            # Exploratory: many relevant results (broad satisfaction)
            relevant_ids = [r["id"] for r in results[:7]]
        elif qtype == "precise":
            # Precise: few highly relevant results
            relevant_ids = [r["id"] for r in results[:2]]
        else:
            # Mixed: moderate relevance
            relevant_ids = [r["id"] for r in results[:4]]

        satisfaction = len(relevant_ids) / len(results) if results else 0
        store.provide_feedback(relevant_ids=relevant_ids)

        # Record metrics
        ef_used = store._searcher.last_ef_used
        epsilon = store._searcher.ef_selector.exploration_rate
        tracker.record(latency_ms, satisfaction, ef_used, qtype, epsilon)

    print(f"\nExperiment complete!")
    return tracker


def compare_results(fixed: ExperimentTracker, glie: ExperimentTracker):
    """Compare fixed epsilon vs GLIE epsilon decay."""

    print("\n" + "="*100)
    print("EPSILON DECAY A/B TEST RESULTS")
    print("="*100)

    # Define phases
    phases = [
        ("Early (0-100)", 0, 100),
        ("Middle (100-200)", 100, 200),
        ("Late (200-326)", 200, 326),
        ("Overall (0-326)", 0, 326)
    ]

    for phase_name, start, end in phases:
        print(f"\n{phase_name}")
        print("-"*100)

        fixed_metrics = fixed.get_phase_metrics(start, end)
        glie_metrics = glie.get_phase_metrics(start, end)

        if not fixed_metrics or not glie_metrics:
            continue

        print(f"{'Metric':<30} | {'Fixed eps=0.15':>20} | {'GLIE eps(t)':>20} | {'Improvement':>20}")
        print("-"*100)

        # Epsilon
        print(f"{'Average Epsilon':<30} | {fixed_metrics['avg_epsilon']:>20.4f} | "
              f"{glie_metrics['avg_epsilon']:>20.4f} | {'--':>20}")

        # Efficiency (main metric)
        eff_imp = ((glie_metrics['avg_efficiency'] - fixed_metrics['avg_efficiency'])
                   / fixed_metrics['avg_efficiency'] * 100) if fixed_metrics['avg_efficiency'] > 0 else 0
        print(f"{'Efficiency (sat/sec)':<30} | {fixed_metrics['avg_efficiency']:>20.2f} | "
              f"{glie_metrics['avg_efficiency']:>20.2f} | {eff_imp:>19.1f}%")

        # Satisfaction
        sat_imp = ((glie_metrics['avg_satisfaction'] - fixed_metrics['avg_satisfaction'])
                   / fixed_metrics['avg_satisfaction'] * 100) if fixed_metrics['avg_satisfaction'] > 0 else 0
        print(f"{'Satisfaction':<30} | {fixed_metrics['avg_satisfaction']:>19.1%} | "
              f"{glie_metrics['avg_satisfaction']:>19.1%} | {sat_imp:>19.1f}%")

        # Latency
        lat_imp = ((fixed_metrics['avg_latency_ms'] - glie_metrics['avg_latency_ms'])
                   / fixed_metrics['avg_latency_ms'] * 100) if fixed_metrics['avg_latency_ms'] > 0 else 0
        print(f"{'Latency (ms)':<30} | {fixed_metrics['avg_latency_ms']:>20.2f} | "
              f"{glie_metrics['avg_latency_ms']:>20.2f} | {lat_imp:>19.1f}%")

        # ef_search
        print(f"{'Average ef_search':<30} | {fixed_metrics['avg_ef']:>20.1f} | "
              f"{glie_metrics['avg_ef']:>20.1f} | {'--':>20}")

    # Verdict
    print("\n" + "="*100)
    overall_fixed = fixed.get_phase_metrics(0, len(fixed.latencies))
    overall_glie = glie.get_phase_metrics(0, len(glie.latencies))

    eff_improvement = ((overall_glie['avg_efficiency'] - overall_fixed['avg_efficiency'])
                       / overall_fixed['avg_efficiency'] * 100)

    print("VERDICT:")
    if eff_improvement > 2.0:
        print(f"  GLIE epsilon decay shows {eff_improvement:.1f}% efficiency improvement - SIGNIFICANT!")
        print("  Theoretical improvement validated. GLIE decay recommended.")
    elif eff_improvement > 0:
        print(f"  GLIE epsilon decay shows {eff_improvement:.1f}% efficiency improvement - MARGINAL")
        print("  Improvement exists but may not justify added complexity.")
    else:
        print(f"  GLIE epsilon decay shows {eff_improvement:.1f}% efficiency change - NO IMPROVEMENT")
        print("  Fixed epsilon performs as well or better. Keep current implementation.")

    print("="*100)


if __name__ == "__main__":
    print("\nEPSILON DECAY A/B TEST")
    print("="*100)
    print("\nComparison:")
    print("  Control: Fixed epsilon = 0.15 (NO decay)")
    print("  Treatment: GLIE epsilon decay eps(t) = 0.15 / (1 + t/100)")
    print("\nExpected:")
    print("  GLIE should show better efficiency through optimal exploration-exploitation trade-off")
    print("  Early: More exploration (higher epsilon) for better Q-value estimation")
    print("  Late: More exploitation (lower epsilon) for optimal ef_search selection")

    # Run experiments
    print("\n" + "-"*100)
    print("EXPERIMENT 1: Fixed Epsilon = 0.15 (NO DECAY)")
    print("-"*100)
    fixed_tracker = run_experiment(epsilon_decay_mode="none", initial_epsilon=0.15)

    print("\n" + "-"*100)
    print("EXPERIMENT 2: GLIE Epsilon Decay (starts at 0.15)")
    print("-"*100)
    glie_tracker = run_experiment(epsilon_decay_mode="glie", initial_epsilon=0.15)

    # Compare results
    compare_results(fixed_tracker, glie_tracker)
