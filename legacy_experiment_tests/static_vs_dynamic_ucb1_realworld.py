"""Real-World Comparison: Static HNSW vs Dynamic HNSW with UCB1

Large-scale A/B test comparing traditional static HNSW with dynamic HNSW
using UCB1 exploration on real-world sentence embedding data.

Scenario:
    Simulate a production RAG (Retrieval-Augmented Generation) system with:
    - Large document corpus (1000+ documents)
    - Diverse query patterns (questions, statements, commands)
    - Real sentence embeddings (all-MiniLM-L6-v2)
    - 2000 queries to test convergence

Hypothesis:
    Dynamic HNSW with UCB1 should outperform static HNSW by:
    1. Learning optimal ef_search per query type (exploratory vs precise)
    2. Reducing latency for precise queries (lower ef_search)
    3. Maintaining recall for exploratory queries (higher ef_search)
    4. Overall efficiency improvement: 5-10%
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time
import numpy as np
from typing import List, Dict, Tuple
from dynhnsw import VectorStore
from dynhnsw.config import DynHNSWConfig


def generate_realistic_corpus(size: int = 1000) -> List[str]:
    """Generate realistic document corpus for RAG system.

    Categories:
    - Technical documentation
    - FAQ answers
    - Product descriptions
    - Tutorial content
    - API references
    """
    corpus = []

    # Technical documentation (25%)
    tech_docs = [
        "Docker containers provide isolated environments for application deployment. Use docker run to start containers and docker-compose for multi-container applications.",
        "Kubernetes manages containerized applications across clusters. Pods are the smallest deployable units, containing one or more containers.",
        "REST APIs use HTTP methods (GET, POST, PUT, DELETE) for CRUD operations. Authentication typically uses JWT tokens or API keys.",
        "GraphQL allows clients to request exactly the data they need. Define schema with types, queries, and mutations for flexible data fetching.",
        "PostgreSQL is a relational database with ACID compliance. Use indexes, foreign keys, and transactions for data integrity.",
        "MongoDB stores documents in JSON-like format. Collections hold documents, and queries use JavaScript-like syntax.",
        "Redis is an in-memory data store supporting strings, hashes, lists, sets. Use for caching, session storage, and pub/sub messaging.",
        "Git tracks code changes with commits. Use branches for parallel development, merge or rebase to integrate changes.",
        "CI/CD pipelines automate testing and deployment. Jenkins, GitHub Actions, and GitLab CI are popular tools.",
        "TensorFlow builds machine learning models using computational graphs. Define layers, compile with optimizer, train on data.",
        "PyTorch provides dynamic computational graphs for deep learning. Use torch.nn for layers, autograd for backpropagation.",
        "FastAPI builds async APIs with automatic OpenAPI docs. Type hints enable validation and serialization.",
        "React components render UI using JSX syntax. State and props manage data, hooks enable stateful functional components.",
        "Vue.js uses reactive data binding and component composition. Single-file components combine template, script, and style.",
        "Angular is a full framework with TypeScript and RxJS. Modules organize code, services handle business logic.",
        "Node.js runs JavaScript server-side using V8 engine. Event loop handles async operations without blocking.",
        "Express.js middleware processes requests. Route handlers respond based on HTTP method and path.",
        "Django follows MTV pattern with models, templates, views. ORM maps Python classes to database tables.",
        "Flask is a micro-framework for Python web apps. Routes map URLs to functions, Jinja2 templates render HTML.",
        "Nginx reverse proxies and load balances traffic. Configures virtual hosts, SSL certificates, and caching.",
    ]

    # FAQ answers (25%)
    faq_docs = [
        "To reset your password, click 'Forgot Password' on the login page. Enter your email address, and we'll send you a reset link valid for 24 hours.",
        "Orders can be returned within 30 days of purchase. Items must be unused with original packaging. Refunds processed within 5-7 business days.",
        "Track your order using the tracking number in your confirmation email. Enter it on our tracking page to see real-time delivery status.",
        "Cancel subscription by visiting account settings and clicking 'Cancel Plan'. You'll retain access until the current billing period ends.",
        "We accept Visa, Mastercard, American Express, PayPal, and bank transfers. All transactions are encrypted with SSL security.",
        "Standard shipping takes 5-7 business days. Express shipping (2-3 days) and overnight delivery available for additional cost.",
        "International shipping available to over 150 countries. Delivery times vary by destination, customs clearance may add 3-5 days.",
        "Contact customer support via email, phone, or live chat. Support hours are Monday-Friday 9AM-6PM EST, emergency line 24/7.",
        "Your data is encrypted at rest and in transit. We follow GDPR and CCPA compliance, never sell personal information to third parties.",
        "Update billing information in account settings. Add or remove payment methods, set default payment for subscriptions.",
        "Modify order within 1 hour of placement by contacting support. After processing begins, changes may not be possible.",
        "Gift wrapping available at checkout for $5. Include personalized message up to 250 characters for no additional charge.",
        "Warranty covers manufacturing defects for 1 year from purchase date. Extended warranties available for electronics up to 3 years.",
        "Apply discount code at checkout. Enter code in promo field before payment. Codes cannot be combined with other offers.",
        "Create wishlist by clicking heart icon on product pages. Share with friends or save for later purchase.",
        "Mobile app available on iOS (12+) and Android (8+). Download from App Store or Google Play for faster checkout experience.",
        "Unsubscribe from emails using link in footer. Takes 48 hours to process. Transactional emails (receipts) cannot be disabled.",
        "Business hours: Monday-Friday 9AM-5PM EST. Support available via email 24/7, response within 24 hours guaranteed.",
        "Bulk discounts for orders of 10+ items. Contact sales team for enterprise pricing and volume agreements.",
        "Security: We use AES-256 encryption, regular security audits, and PCI DSS compliance for payment processing.",
    ]

    # Product descriptions (25%)
    product_docs = [
        "Wireless Bluetooth Headphones with Active Noise Cancellation. 30-hour battery life, comfortable over-ear design, premium sound quality. Model WH-1000XM5.",
        "Ergonomic Office Chair with lumbar support and adjustable armrests. Breathable mesh back, 360-degree swivel, supports up to 300 lbs. Model EC-2000.",
        "Stainless Steel Water Bottle with double-wall vacuum insulation. Keeps drinks cold 24 hours, hot 12 hours. BPA-free, 32 oz capacity.",
        "Adjustable Laptop Stand with aluminum construction. Elevates screen 2-10 inches, improves posture, compatible with 10-17 inch laptops.",
        "Mechanical Keyboard with Cherry MX switches and RGB backlight. Programmable keys, aluminum frame, detachable USB-C cable. Model MK-750.",
        "4K Webcam with autofocus and built-in microphone. 30 FPS video, wide-angle lens, USB plug-and-play, compatible with all platforms.",
        "Electric Standing Desk with programmable height presets. Dual motors, lifts 220 lbs, anti-collision system, 48x30 inch surface.",
        "LED Desk Lamp with adjustable color temperature. Touch controls, USB charging port, flicker-free, energy-efficient, modern design.",
        "USB Condenser Microphone with cardioid pickup pattern. Pop filter included, plug-and-play, great for podcasting and streaming.",
        "10-Port USB Hub with individual power switches. 7 USB-A, 3 USB-C ports, 100W power delivery, aluminum housing.",
        "Ergonomic Wireless Mouse with 6 programmable buttons. 4000 DPI, rechargeable battery, works on any surface, comfortable grip.",
        "Dual Monitor Arm with gas spring system. Holds two 27-inch monitors, VESA compatible, cable management, full articulation.",
        "Cable Management Box with 6 outlets and surge protection. Hides power strips and cables, flame-retardant, multiple size options.",
        "Large Desk Mat with stitched edges and non-slip rubber base. Waterproof surface, 36x18 inches, multiple color options.",
        "Ergonomic Footrest with adjustable angle and height. Massage surface, improves circulation, anti-slip, supports good posture.",
        "Premium Headphone Stand with USB hub. Aluminum construction, 3 USB-A ports, cable organizer, supports all headphone sizes.",
        "20000mAh Power Bank with fast charging. Dual USB output, USB-C input/output, LED battery indicator, airline-safe.",
        "Laptop Sleeve with water-resistant fabric. Soft interior lining, external pocket, handle strap, fits 13-15 inch laptops.",
        "Wireless Charging Pad with Qi certification. 15W fast charging, LED indicator, non-slip surface, works with cases up to 5mm.",
        "Blue Light Blocking Glasses with anti-reflective coating. Reduces eye strain from screens, comfortable lightweight frame.",
    ]

    # Tutorial content (25%)
    tutorial_docs = [
        "Step 1: Install Python 3.9+ from python.org. Step 2: Create virtual environment with 'python -m venv venv'. Step 3: Activate with 'source venv/bin/activate'.",
        "Git basics: Clone repository with 'git clone URL'. Create branch with 'git checkout -b feature'. Stage changes with 'git add .', commit with 'git commit -m'.",
        "Docker tutorial: Create Dockerfile defining base image and dependencies. Build with 'docker build -t name .'. Run with 'docker run -p 8000:8000 name'.",
        "REST API tutorial: Define routes with @app.get() decorator. Parse request body with Pydantic models. Return JSON responses with FastAPI.",
        "Database setup: Install PostgreSQL, create database with 'CREATE DATABASE mydb'. Define tables with CREATE TABLE. Insert data with INSERT INTO.",
        "React tutorial: Create component with function and JSX. Manage state with useState hook. Handle events with onClick handlers. Update UI when state changes.",
        "CSS flexbox: Set display: flex on container. Use justify-content for horizontal alignment. Use align-items for vertical alignment. flex-wrap for multiline.",
        "Authentication flow: User submits credentials. Server validates, generates JWT token. Token sent in Authorization header. Server verifies token on protected routes.",
        "Unit testing: Write test function with test_ prefix. Use assert statements to verify behavior. Mock external dependencies. Run with pytest command.",
        "Deployment guide: Build production bundle with 'npm run build'. Upload to server via FTP or git. Configure Nginx reverse proxy. Set up SSL certificate.",
        "API documentation: Use OpenAPI/Swagger for REST APIs. Define schemas, endpoints, and examples. Generate interactive docs automatically with FastAPI.",
        "Error handling: Use try-except blocks for exceptions. Log errors with logging module. Return appropriate HTTP status codes. Show user-friendly error messages.",
        "Performance optimization: Use database indexes for faster queries. Cache frequent requests with Redis. Compress assets with gzip. Lazy load images.",
        "Security best practices: Sanitize user input to prevent SQL injection. Use parameterized queries. Hash passwords with bcrypt. Implement rate limiting.",
        "Monitoring setup: Install Prometheus for metrics collection. Configure Grafana dashboards for visualization. Set up alerts for critical events.",
        "CI/CD setup: Create .github/workflows/main.yml file. Define jobs for test and deploy. Use actions for checkout, setup, and deployment.",
        "Debugging techniques: Use print statements or debugger breakpoints. Check logs for error messages. Use browser DevTools for frontend issues.",
        "Code review checklist: Verify functionality, test edge cases, check code style, review security, ensure documentation, run automated tests.",
        "Refactoring guide: Extract repeated code into functions. Split large functions into smaller ones. Use meaningful variable names. Add type hints.",
        "Scaling strategies: Horizontal scaling with load balancers. Database replication for read-heavy loads. Caching layers for frequently accessed data.",
    ]

    # Build corpus
    docs_per_category = size // 4
    corpus.extend(tech_docs[:docs_per_category])
    corpus.extend(faq_docs[:docs_per_category])
    corpus.extend(product_docs[:docs_per_category])
    corpus.extend(tutorial_docs[:docs_per_category])

    # Fill remaining with technical docs
    while len(corpus) < size:
        corpus.append(f"Technical documentation entry {len(corpus)}: Additional content covering software development, deployment, and best practices.")

    return corpus[:size]


def generate_realistic_queries(num_queries: int = 2000) -> Tuple[List[str], List[str]]:
    """Generate realistic queries simulating RAG system usage.

    Query Types:
    - Exploratory (40%): Broad questions needing multiple results
    - Precise (40%): Specific questions needing 1-2 results
    - Navigational (20%): Looking for specific resource
    """
    queries = []
    query_types = []

    # Exploratory queries (broad, need high ef_search)
    exploratory = [
        "How do I set up Docker containers?",
        "What are best practices for REST API design?",
        "Explain Git branching strategies",
        "How does Kubernetes orchestration work?",
        "What is the difference between SQL and NoSQL databases?",
        "How do I optimize database performance?",
        "What are microservices architectures?",
        "Explain CI/CD pipeline concepts",
        "How do I secure web applications?",
        "What are React hooks and how to use them?",
        "Tell me about machine learning frameworks",
        "How does authentication and authorization work?",
        "What are different caching strategies?",
        "Explain cloud computing services",
        "How do I monitor application performance?",
        "What products do you have for home office?",
        "Show me ergonomic furniture options",
        "What are your shipping and return policies?",
        "How do I get started with programming?",
        "Explain software testing methodologies",
    ]

    # Precise queries (specific, need low ef_search)
    precise = [
        "Docker run command syntax",
        "PostgreSQL create index statement",
        "React useState example",
        "JWT token authentication flow",
        "Wireless headphones model WH-1000XM5 specs",
        "How to reset password?",
        "Track order with tracking number",
        "Cancel subscription steps",
        "Return policy timeframe",
        "Customer support phone number",
        "Bulk discount pricing",
        "International shipping countries",
        "Gift wrap cost",
        "Warranty coverage duration",
        "Mobile app download link",
        "Python virtual environment activation command",
        "Git commit message format",
        "Nginx reverse proxy configuration",
        "Redis cache TTL setting",
        "FastAPI route decorator syntax",
    ]

    # Navigational queries (looking for specific doc)
    navigational = [
        "Docker documentation",
        "API reference guide",
        "Installation instructions",
        "Product manual",
        "FAQ page",
        "Pricing information",
        "Contact support",
        "Privacy policy",
        "Terms of service",
        "Security documentation",
    ]

    # Generate query distribution
    exploratory_count = int(num_queries * 0.4)
    precise_count = int(num_queries * 0.4)
    navigational_count = num_queries - exploratory_count - precise_count

    # Repeat queries to reach desired count
    for _ in range((exploratory_count // len(exploratory)) + 1):
        for q in exploratory:
            if len([t for t in query_types if t == "exploratory"]) < exploratory_count:
                queries.append(q)
                query_types.append("exploratory")

    for _ in range((precise_count // len(precise)) + 1):
        for q in precise:
            if len([t for t in query_types if t == "precise"]) < precise_count:
                queries.append(q)
                query_types.append("precise")

    for _ in range((navigational_count // len(navigational)) + 1):
        for q in navigational:
            if len([t for t in query_types if t == "navigational"]) < navigational_count:
                queries.append(q)
                query_types.append("navigational")

    # Shuffle
    indices = list(range(len(queries)))
    np.random.seed(42)
    np.random.shuffle(indices)
    queries = [queries[i] for i in indices]
    query_types = [query_types[i] for i in indices]

    return queries, query_types


def simulate_user_feedback(query_type: str, results: List[Dict], k: int) -> Tuple[List[str], float]:
    """Simulate realistic user feedback based on query type."""
    if not results:
        return [], 0.0

    if query_type == "exploratory":
        # Wants many results, high satisfaction if many returned
        relevant_count = min(int(k * 0.7), len(results))
        relevant_ids = [r["id"] for r in results[:relevant_count]]
    elif query_type == "precise":
        # Wants top 1-2, low satisfaction if too many
        relevant_count = min(2, len(results))
        relevant_ids = [r["id"] for r in results[:relevant_count]]
    else:  # navigational
        # Wants specific result, happy if top-3 contains it
        relevant_count = min(3, len(results))
        relevant_ids = [r["id"] for r in results[:relevant_count]]

    satisfaction = len(relevant_ids) / len(results) if results else 0
    return relevant_ids, satisfaction


class ExperimentTracker:
    """Track metrics for A/B comparison."""

    def __init__(self, name: str):
        self.name = name
        self.latencies = []
        self.satisfactions = []
        self.efficiencies = []
        self.ef_values = []

    def record(self, latency_ms: float, satisfaction: float, ef_used: int):
        self.latencies.append(latency_ms)
        self.satisfactions.append(satisfaction)
        efficiency = (satisfaction / (latency_ms / 1000.0)) if latency_ms > 0 else 0
        self.efficiencies.append(efficiency)
        self.ef_values.append(ef_used)

    def get_summary(self) -> Dict:
        return {
            "avg_latency_ms": np.mean(self.latencies),
            "avg_satisfaction": np.mean(self.satisfactions),
            "avg_efficiency": np.mean(self.efficiencies),
            "avg_ef": np.mean(self.ef_values),
            "total_queries": len(self.latencies),
        }


def main():
    print("="*100)
    print("REAL-WORLD A/B TEST: Static HNSW vs Dynamic HNSW with UCB1")
    print("="*100)

    # Import sentence transformers
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        print("\nERROR: sentence-transformers not installed!")
        print("Install with: pip install sentence-transformers")
        sys.exit(1)

    # Generate realistic data
    print("\n[1] Generating realistic corpus (RAG system simulation)...")
    corpus = generate_realistic_corpus(size=1000)
    print(f"    Created {len(corpus)} documents")
    print(f"    Categories: Technical docs, FAQ, Products, Tutorials (25% each)")

    print("\n[2] Generating realistic queries...")
    queries, query_types = generate_realistic_queries(num_queries=2000)
    print(f"    Created {len(queries)} queries")
    print(f"    Exploratory: {query_types.count('exploratory')} (broad questions)")
    print(f"    Precise: {query_types.count('precise')} (specific questions)")
    print(f"    Navigational: {query_types.count('navigational')} (find specific doc)")

    # Embed corpus
    print("\n[3] Embedding corpus with sentence-transformers (all-MiniLM-L6-v2)...")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    print("    This may take a minute for 1000 documents...")
    embeddings = model.encode(corpus, convert_to_numpy=True, show_progress_bar=True)
    embeddings = embeddings.astype(np.float32)
    print(f"    Embedded {len(embeddings)} documents ({embeddings.shape[1]} dimensions)")

    # Create STATIC store
    print("\n[4] Creating STATIC HNSW (baseline)...")
    print("    ef_search=100 (fixed)")
    print("    No intent detection, no adaptation")

    static_store = VectorStore(
        dimension=embeddings.shape[1],
        M=16,
        ef_construction=200,
        ef_search=100,  # FIXED
        enable_intent_detection=False,  # STATIC MODE
    )
    static_store.add(embeddings)

    # Create DYNAMIC store with UCB1
    print("\n[5] Creating DYNAMIC HNSW with UCB1...")
    print("    UCB1 exploration (c=1.414)")
    print("    k_intents=3, adaptive ef_search")

    config = DynHNSWConfig(
        config_name="ucb1_realworld",
        enable_ucb1=True,
        ucb1_exploration_constant=1.414,
        k_intents=3,
        min_queries_for_clustering=30,
    )

    dynamic_store = VectorStore(
        dimension=embeddings.shape[1],
        M=16,
        ef_construction=200,
        ef_search=100,  # Default, will adapt
        enable_intent_detection=True,  # DYNAMIC MODE
        k_intents=3,
        config=config,
    )
    dynamic_store.add(embeddings)

    # Run A/B test
    print("\n" + "="*100)
    print("RUNNING A/B TEST (2000 queries)")
    print("="*100)

    static_tracker = ExperimentTracker("Static HNSW")
    dynamic_tracker = ExperimentTracker("Dynamic HNSW (UCB1)")

    # Embed queries
    print("\n[6] Embedding queries...")
    query_embeddings = model.encode(queries, convert_to_numpy=True, show_progress_bar=True)

    print("\n[7] Running queries on BOTH stores (Static and Dynamic)...")

    for i, (query_vec, query_type) in enumerate(zip(query_embeddings, query_types)):
        # Determine k based on query type
        if query_type == "exploratory":
            k = 15
        elif query_type == "precise":
            k = 5
        else:  # navigational
            k = 10

        # STATIC store
        start_time = time.perf_counter()
        static_results = static_store.search(query_vec, k=k)
        static_latency = (time.perf_counter() - start_time) * 1000.0

        static_relevant, static_satisfaction = simulate_user_feedback(query_type, static_results, k)
        static_tracker.record(static_latency, static_satisfaction, 100)  # Always ef=100

        # DYNAMIC store
        start_time = time.perf_counter()
        dynamic_results = dynamic_store.search(query_vec, k=k)
        dynamic_latency = (time.perf_counter() - start_time) * 1000.0

        dynamic_relevant, dynamic_satisfaction = simulate_user_feedback(query_type, dynamic_results, k)
        if dynamic_relevant:
            dynamic_store.provide_feedback(relevant_ids=dynamic_relevant)

        dynamic_ef = dynamic_store._searcher.last_ef_used
        dynamic_tracker.record(dynamic_latency, dynamic_satisfaction, dynamic_ef)

        # Progress update
        if (i + 1) % 200 == 0:
            print(f"    Progress: {i+1}/{len(queries)}")

    print(f"\n[COMPLETE] Processed {len(queries)} queries")

    # Results
    print("\n" + "="*100)
    print("RESULTS")
    print("="*100)

    static_summary = static_tracker.get_summary()
    dynamic_summary = dynamic_tracker.get_summary()

    print(f"\n{'Metric':<30} | {'Static HNSW':>20} | {'Dynamic HNSW (UCB1)':>20} | {'Improvement':>15}")
    print("-" * 100)

    # Efficiency
    eff_improvement = ((dynamic_summary['avg_efficiency'] - static_summary['avg_efficiency']) /
                       static_summary['avg_efficiency']) * 100
    print(f"{'Efficiency (sat/sec)':<30} | {static_summary['avg_efficiency']:>20.2f} | "
          f"{dynamic_summary['avg_efficiency']:>20.2f} | {eff_improvement:>14.1f}%")

    # Satisfaction
    sat_improvement = ((dynamic_summary['avg_satisfaction'] - static_summary['avg_satisfaction']) /
                       static_summary['avg_satisfaction']) * 100
    print(f"{'Satisfaction':<30} | {static_summary['avg_satisfaction']:>19.1%} | "
          f"{dynamic_summary['avg_satisfaction']:>19.1%} | {sat_improvement:>14.1f}%")

    # Latency
    lat_improvement = ((static_summary['avg_latency_ms'] - dynamic_summary['avg_latency_ms']) /
                       static_summary['avg_latency_ms']) * 100
    print(f"{'Latency (ms)':<30} | {static_summary['avg_latency_ms']:>20.2f} | "
          f"{dynamic_summary['avg_latency_ms']:>20.2f} | {lat_improvement:>14.1f}%")

    # ef_search
    print(f"{'Average ef_search':<30} | {static_summary['avg_ef']:>20.0f} | "
          f"{dynamic_summary['avg_ef']:>20.0f} | {'--':>15}")

    # Summary
    print("\n" + "="*100)
    print("SUMMARY")
    print("="*100)

    print(f"\nEfficiency:")
    if eff_improvement > 5:
        print(f"  [SUCCESS] Dynamic HNSW with UCB1 is {eff_improvement:.1f}% MORE EFFICIENT")
        print(f"  Achieved through adaptive ef_search selection per query type")
    elif eff_improvement > 0:
        print(f"  [POSITIVE] Dynamic HNSW shows {eff_improvement:.1f}% improvement")
    else:
        print(f"  [NEUTRAL] Similar performance ({abs(eff_improvement):.1f}% difference)")

    print(f"\nLatency:")
    if lat_improvement > 5:
        print(f"  [SUCCESS] Dynamic HNSW is {lat_improvement:.1f}% FASTER")
    elif lat_improvement > 0:
        print(f"  [POSITIVE] Dynamic HNSW shows {lat_improvement:.1f}% latency reduction")
    else:
        print(f"  [NEUTRAL] Similar latency ({abs(lat_improvement):.1f}% difference)")

    print(f"\nAdaptive Learning:")
    print(f"  Static ef_search: {static_summary['avg_ef']:.0f} (fixed)")
    print(f"  Dynamic ef_search: {dynamic_summary['avg_ef']:.0f} (learned)")

    # Show learned values
    stats = dynamic_store.get_statistics()
    if "ef_search_selection" in stats:
        print(f"\n  Learned ef_search per intent:")
        for intent_data in stats["ef_search_selection"]["per_intent"]:
            if intent_data["num_queries"] > 50:  # Only show active intents
                print(f"    Intent {intent_data['intent_id']}: ef={intent_data['learned_ef']} "
                      f"({intent_data['num_queries']} queries)")

    # Verdict
    print("\n" + "="*100)
    print("VERDICT")
    print("="*100)

    if eff_improvement > 5 and lat_improvement > 0:
        print("\n  [RECOMMEND] Dynamic HNSW with UCB1 for production")
        print("  Significant efficiency gains with adaptive ef_search selection")
    elif eff_improvement > 2:
        print("\n  [CONSIDER] Dynamic HNSW shows promise")
        print("  May benefit from longer training period or more intents")
    else:
        print("\n  [STATIC SUFFICIENT] Static HNSW performs adequately")
        print("  Dynamic adaptation may not justify added complexity for this workload")

    print("\n" + "="*100)
    print("Test Complete!")
    print("="*100)


if __name__ == "__main__":
    main()
