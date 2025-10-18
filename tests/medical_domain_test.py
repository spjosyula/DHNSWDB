"""Medical Domain Validation Test - Realistic Medicine Recommendation Scenario

This test simulates a real-world medical information retrieval system where:
- Corpus: Medical documents about medicines, their uses, side effects, and dosages
- Queries: User questions like "what's the best medicine for headache?"
- Critical metric: Recall (missing relevant medicines can be dangerous)
- Secondary metric: Latency (users expect fast responses)

Compares Static HNSW vs Layer-Adaptive HNSW in hyper-realistic conditions.
"""

import numpy as np
from sentence_transformers import SentenceTransformer
from dynhnsw import VectorStore
from typing import List, Dict, Tuple
import time
from collections import defaultdict


# ============================================================================
# MEDICAL CORPUS GENERATION
# ============================================================================

def generate_medical_corpus(size: int = 10000) -> Tuple[List[str], List[Dict]]:
    """Generate realistic medical corpus with medicine information.

    Returns:
        corpus: List of medical document texts
        metadata: List of metadata dicts with medicine info
    """
    print(f"[Corpus] Generating {size} medical documents...")

    # Common symptoms and conditions
    symptoms = [
        "headache", "migraine", "fever", "cold", "cough", "sore throat",
        "muscle pain", "joint pain", "back pain", "stomach pain", "nausea",
        "diarrhea", "constipation", "allergies", "skin rash", "insomnia",
        "anxiety", "depression", "high blood pressure", "diabetes",
        "asthma", "arthritis", "infection", "inflammation", "acid reflux"
    ]

    # Medicine categories with realistic names
    medicine_categories = {
        "pain_relief": [
            ("Ibuprofen", "NSAID for general pain relief"),
            ("Acetaminophen", "Analgesic for mild to moderate pain"),
            ("Aspirin", "NSAID for pain and inflammation"),
            ("Naproxen", "Long-acting NSAID for chronic pain"),
            ("Diclofenac", "Potent NSAID for severe pain")
        ],
        "antibiotics": [
            ("Amoxicillin", "Broad-spectrum antibiotic"),
            ("Azithromycin", "Macrolide antibiotic"),
            ("Ciprofloxacin", "Fluoroquinolone antibiotic"),
            ("Doxycycline", "Tetracycline antibiotic"),
            ("Cephalexin", "Cephalosporin antibiotic")
        ],
        "antihistamines": [
            ("Cetirizine", "Non-drowsy antihistamine"),
            ("Loratadine", "Long-acting antihistamine"),
            ("Diphenhydramine", "First-generation antihistamine"),
            ("Fexofenadine", "Non-sedating antihistamine")
        ],
        "gastrointestinal": [
            ("Omeprazole", "Proton pump inhibitor for acid reflux"),
            ("Ranitidine", "H2 blocker for heartburn"),
            ("Loperamide", "Anti-diarrheal medication"),
            ("Metoclopramide", "Prokinetic for nausea"),
            ("Simethicone", "Anti-gas medication")
        ],
        "respiratory": [
            ("Albuterol", "Bronchodilator for asthma"),
            ("Montelukast", "Leukotriene modifier for asthma"),
            ("Dextromethorphan", "Cough suppressant"),
            ("Guaifenesin", "Expectorant for mucus"),
            ("Pseudoephedrine", "Decongestant")
        ],
        "cardiovascular": [
            ("Lisinopril", "ACE inhibitor for hypertension"),
            ("Amlodipine", "Calcium channel blocker"),
            ("Metoprolol", "Beta blocker for heart conditions"),
            ("Atorvastatin", "Statin for cholesterol"),
            ("Warfarin", "Anticoagulant")
        ]
    }

    # Side effects pool
    side_effects = [
        "drowsiness", "dizziness", "nausea", "dry mouth", "headache",
        "stomach upset", "constipation", "diarrhea", "fatigue",
        "insomnia", "increased appetite", "weight gain", "rash"
    ]

    # Dosage forms
    dosage_forms = ["tablet", "capsule", "syrup", "injection", "cream", "inhaler"]

    corpus = []
    metadata_list = []

    doc_id = 0

    # Generate documents for each medicine
    for category, medicines in medicine_categories.items():
        for medicine_name, description in medicines:
            # Multiple documents per medicine (different aspects)

            # Document 1: General information
            symptoms_treated = np.random.choice(symptoms, size=np.random.randint(2, 5), replace=False)
            doc = (
                f"{medicine_name} is a {description}. "
                f"It is commonly prescribed for {', '.join(symptoms_treated[:-1])} and {symptoms_treated[-1]}. "
                f"Available as {np.random.choice(dosage_forms)}. "
                f"Effective relief within 30-60 minutes of administration."
            )
            corpus.append(doc)
            metadata_list.append({
                "medicine": medicine_name,
                "category": category,
                "symptoms": list(symptoms_treated),
                "doc_type": "general_info"
            })
            doc_id += 1

            # Document 2: Dosage and administration
            doc = (
                f"{medicine_name} dosage recommendations: "
                f"Adults: {np.random.randint(200, 800)}mg every {np.random.choice([4, 6, 8, 12])} hours. "
                f"Maximum daily dose: {np.random.randint(1600, 3200)}mg. "
                f"Take with food to minimize stomach upset. Consult physician for long-term use."
            )
            corpus.append(doc)
            metadata_list.append({
                "medicine": medicine_name,
                "category": category,
                "doc_type": "dosage"
            })
            doc_id += 1

            # Document 3: Side effects and precautions
            medicine_side_effects = np.random.choice(side_effects, size=np.random.randint(3, 6), replace=False)
            doc = (
                f"{medicine_name} side effects: Common side effects include {', '.join(medicine_side_effects)}. "
                f"Serious reactions rare but may include allergic reactions. "
                f"Do not use if allergic to {category.replace('_', ' ')} medications. "
                f"Consult doctor if symptoms persist beyond 7 days."
            )
            corpus.append(doc)
            metadata_list.append({
                "medicine": medicine_name,
                "category": category,
                "side_effects": list(medicine_side_effects),
                "doc_type": "side_effects"
            })
            doc_id += 1

            # Document 4: Interactions and contraindications
            doc = (
                f"{medicine_name} drug interactions: May interact with other {category.replace('_', ' ')} medications. "
                f"Inform doctor about all current medications including supplements. "
                f"Avoid alcohol consumption while taking this medication. "
                f"Not recommended during pregnancy without medical supervision."
            )
            corpus.append(doc)
            metadata_list.append({
                "medicine": medicine_name,
                "category": category,
                "doc_type": "interactions"
            })
            doc_id += 1

    # Add filler documents to reach target size (general medical information)
    general_topics = [
        "healthy diet", "exercise benefits", "sleep hygiene", "stress management",
        "preventive care", "vaccination importance", "mental health awareness",
        "chronic disease management", "medication adherence", "patient education"
    ]

    while len(corpus) < size:
        topic = np.random.choice(general_topics)
        doc = (
            f"Medical information about {topic}: "
            f"Maintaining good health involves regular checkups and lifestyle modifications. "
            f"Consult healthcare providers for personalized advice. "
            f"Early detection and prevention are key to long-term wellness."
        )
        corpus.append(doc)
        metadata_list.append({
            "topic": topic,
            "doc_type": "general_medical_info"
        })

    print(f"[Corpus] Generated {len(corpus)} documents")
    print(f"  Medicine-specific docs: {doc_id}")
    print(f"  General medical info: {len(corpus) - doc_id}")

    return corpus, metadata_list


def generate_realistic_queries(n_queries: int = 500) -> Tuple[List[str], List[List[str]]]:
    """Generate realistic user queries about medicines.

    Returns:
        queries: List of query strings
        relevant_medicines: List of lists containing expected relevant medicines
    """
    print(f"\n[Queries] Generating {n_queries} realistic user queries...")

    query_templates = [
        # Direct symptom queries
        ("What is the best medicine for {symptom}?", "specific"),
        ("Which medication helps with {symptom}?", "specific"),
        ("What can I take for {symptom}?", "specific"),
        ("Recommend medicine for {symptom}", "specific"),

        # Comparative queries
        ("Is {medicine} good for {symptom}?", "specific_medicine"),
        ("Compare {medicine} and alternatives for {symptom}", "specific_medicine"),

        # Side effect queries
        ("What are the side effects of {medicine}?", "side_effects"),
        ("Is {medicine} safe to use?", "safety"),

        # Dosage queries
        ("How much {medicine} should I take?", "dosage"),
        ("What is the recommended dosage of {medicine}?", "dosage"),

        # Complex queries
        ("What medicine for {symptom} with minimal side effects?", "specific_quality"),
        ("Best over-the-counter medication for {symptom}", "specific"),
        ("Which {category} medicine is most effective?", "category")
    ]

    symptoms_medicine_map = {
        "headache": ["Ibuprofen", "Acetaminophen", "Aspirin"],
        "migraine": ["Ibuprofen", "Naproxen", "Aspirin"],
        "fever": ["Acetaminophen", "Ibuprofen", "Aspirin"],
        "cold": ["Dextromethorphan", "Pseudoephedrine", "Guaifenesin"],
        "cough": ["Dextromethorphan", "Guaifenesin"],
        "allergies": ["Cetirizine", "Loratadine", "Fexofenadine"],
        "pain": ["Ibuprofen", "Acetaminophen", "Naproxen", "Aspirin"],
        "stomach pain": ["Omeprazole", "Ranitidine", "Simethicone"],
        "nausea": ["Metoclopramide", "Diphenhydramine"],
        "diarrhea": ["Loperamide"],
        "acid reflux": ["Omeprazole", "Ranitidine"],
        "asthma": ["Albuterol", "Montelukast"],
        "high blood pressure": ["Lisinopril", "Amlodipine", "Metoprolol"]
    }

    medicines = list(set([med for meds in symptoms_medicine_map.values() for med in meds]))
    categories = ["pain relief", "antibiotic", "antihistamine", "heart", "respiratory"]

    queries = []
    relevant_medicines_list = []

    for _ in range(n_queries):
        idx = np.random.randint(0, len(query_templates))
        template, query_type = query_templates[idx]

        if "{symptom}" in template:
            symptom = np.random.choice(list(symptoms_medicine_map.keys()))
            query = template.replace("{symptom}", symptom)
            relevant_meds = symptoms_medicine_map[symptom]
        elif "{medicine}" in template:
            medicine = np.random.choice(medicines)
            symptom = np.random.choice([s for s, meds in symptoms_medicine_map.items() if medicine in meds])
            query = template.replace("{medicine}", medicine).replace("{symptom}", symptom)
            relevant_meds = [medicine]
        elif "{category}" in template:
            category = np.random.choice(categories)
            query = template.replace("{category}", category)
            relevant_meds = []  # Category queries are harder to map
        else:
            symptom = np.random.choice(list(symptoms_medicine_map.keys()))
            query = template.replace("{symptom}", symptom)
            relevant_meds = symptoms_medicine_map.get(symptom, [])

        queries.append(query)
        relevant_medicines_list.append(relevant_meds)

    print(f"[Queries] Generated {len(queries)} queries")
    query_types = defaultdict(int)
    for q in queries:
        if "best medicine for" in q or "helps with" in q:
            query_types["symptom_search"] += 1
        elif "side effects" in q:
            query_types["side_effects"] += 1
        elif "dosage" in q or "how much" in q:
            query_types["dosage"] += 1
        else:
            query_types["other"] += 1

    print(f"  Query distribution:")
    for qtype, count in query_types.items():
        print(f"    {qtype}: {count} ({count/len(queries)*100:.1f}%)")

    return queries, relevant_medicines_list


# ============================================================================
# EVALUATION FUNCTIONS
# ============================================================================

def compute_ground_truth(query_emb, corpus_embeddings, k=10):
    """Brute force k-NN for ground truth."""
    query_norm = query_emb / (np.linalg.norm(query_emb) + 1e-10)
    corpus_norm = corpus_embeddings / (np.linalg.norm(corpus_embeddings, axis=1, keepdims=True) + 1e-10)
    similarities = corpus_norm @ query_norm
    top_k = np.argsort(similarities)[::-1][:k]
    return top_k.tolist()


def compute_recall(retrieved_ids, ground_truth_ids):
    """Compute recall@k."""
    retrieved_set = set(retrieved_ids)
    ground_truth_set = set(ground_truth_ids)
    if len(ground_truth_set) == 0:
        return 0.0
    return len(retrieved_set & ground_truth_set) / len(ground_truth_set)


def compute_precision(retrieved_ids, ground_truth_ids):
    """Compute precision@k."""
    retrieved_set = set(retrieved_ids)
    ground_truth_set = set(ground_truth_ids)
    if len(retrieved_set) == 0:
        return 0.0
    return len(retrieved_set & ground_truth_set) / len(retrieved_set)


def compute_mrr(retrieved_ids, ground_truth_ids):
    """Compute Mean Reciprocal Rank."""
    ground_truth_set = set(ground_truth_ids)
    for rank, doc_id in enumerate(retrieved_ids, start=1):
        if doc_id in ground_truth_set:
            return 1.0 / rank
    return 0.0


# ============================================================================
# MAIN TEST
# ============================================================================

def main():
    print("=" * 80)
    print("MEDICAL DOMAIN VALIDATION TEST")
    print("Realistic Medicine Recommendation Scenario")
    print("=" * 80)

    # Configuration
    corpus_size = 5000
    n_queries = 300
    k = 10  # Top-10 results (critical for medical - need comprehensive results)

    print(f"\nConfiguration:")
    print(f"  Corpus size: {corpus_size} medical documents")
    print(f"  Queries: {n_queries} realistic user questions")
    print(f"  k: {k} (retrieving top-{k} most relevant documents)")
    print(f"  Domain: Medical/Pharmaceutical")
    print(f"  Embedding model: all-MiniLM-L6-v2 (384-dim)")

    # Generate corpus
    corpus, metadata = generate_medical_corpus(corpus_size)

    # Load embedding model
    print(f"\n[Setup] Loading embedding model...")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    print(f"  Model: all-MiniLM-L6-v2 (384 dimensions)")

    # Embed corpus
    print(f"\n[Setup] Embedding corpus...")
    corpus_embeddings = model.encode(corpus, convert_to_numpy=True, show_progress_bar=True, batch_size=32)
    corpus_embeddings = corpus_embeddings.astype(np.float32)
    print(f"  Embedded {len(corpus_embeddings)} documents")

    # Generate queries
    queries, relevant_medicines = generate_realistic_queries(n_queries)

    # Embed queries
    print(f"\n[Setup] Embedding queries...")
    query_embeddings = model.encode(queries, convert_to_numpy=True, show_progress_bar=False, batch_size=32)
    query_embeddings = query_embeddings.astype(np.float32)

    # Compute ground truth
    print(f"\n[Setup] Computing ground truth (brute force k-NN)...")
    ground_truth = []
    for i, query_emb in enumerate(query_embeddings):
        if (i + 1) % 100 == 0:
            print(f"  {i+1}/{len(query_embeddings)}")
        gt = compute_ground_truth(query_emb, corpus_embeddings, k=k)
        ground_truth.append(gt)

    # ========================================================================
    # TEST 1: STATIC HNSW (Baseline)
    # ========================================================================

    print(f"\n" + "=" * 80)
    print("TEST 1: STATIC HNSW (Pure HNSW, No Adaptation)")
    print("=" * 80)

    store_static = VectorStore(
        dimension=384,
        M=16,
        ef_construction=200,
        ef_search=100,
        enable_intent_detection=False,  # STATIC HNSW
        normalize=False
    )

    print(f"\nBuilding static HNSW index...")
    start_build = time.time()
    store_static.add(corpus_embeddings, metadata=metadata)
    build_time_static = time.time() - start_build

    print(f"  Build time: {build_time_static:.2f}s")
    print(f"  Graph max level: {store_static._graph.get_max_level()}")

    print(f"\nRunning {n_queries} queries on static HNSW...")
    recalls_static = []
    precisions_static = []
    mrrs_static = []
    latencies_static = []

    for i, query_emb in enumerate(query_embeddings):
        if (i + 1) % 100 == 0:
            print(f"  {i+1}/{n_queries}")

        start = time.perf_counter()
        results = store_static.search(query_emb, k=k)
        latency = (time.perf_counter() - start) * 1000  # ms

        retrieved_ids = [int(r['id'].split('_')[1]) for r in results]
        gt = ground_truth[i]

        recall = compute_recall(retrieved_ids, gt)
        precision = compute_precision(retrieved_ids, gt)
        mrr = compute_mrr(retrieved_ids, gt)

        recalls_static.append(recall)
        precisions_static.append(precision)
        mrrs_static.append(mrr)
        latencies_static.append(latency)

    # ========================================================================
    # TEST 2: LAYER-ADAPTIVE HNSW (Dynamic)
    # ========================================================================

    print(f"\n" + "=" * 80)
    print("TEST 2: LAYER-ADAPTIVE HNSW (Dynamic Multi-Path)")
    print("=" * 80)

    store_adaptive = VectorStore(
        dimension=384,
        M=16,
        ef_construction=200,
        ef_search=100,
        enable_intent_detection=True,  # LAYER-ADAPTIVE
        normalize=False
    )

    print(f"\nBuilding layer-adaptive HNSW index...")
    start_build = time.time()
    store_adaptive.add(corpus_embeddings, metadata=metadata)
    build_time_adaptive = time.time() - start_build

    print(f"  Build time: {build_time_adaptive:.2f}s")
    print(f"  Graph max level: {store_adaptive._graph.get_max_level()}")

    print(f"\nRunning {n_queries} queries on layer-adaptive HNSW...")
    recalls_adaptive = []
    precisions_adaptive = []
    mrrs_adaptive = []
    latencies_adaptive = []
    difficulties = []
    path_counts = []

    for i, query_emb in enumerate(query_embeddings):
        if (i + 1) % 100 == 0:
            print(f"  {i+1}/{n_queries}")

        start = time.perf_counter()
        results = store_adaptive.search(query_emb, k=k)
        latency = (time.perf_counter() - start) * 1000  # ms

        retrieved_ids = [int(r['id'].split('_')[1]) for r in results]
        gt = ground_truth[i]

        recall = compute_recall(retrieved_ids, gt)
        precision = compute_precision(retrieved_ids, gt)
        mrr = compute_mrr(retrieved_ids, gt)

        recalls_adaptive.append(recall)
        precisions_adaptive.append(precision)
        mrrs_adaptive.append(mrr)
        latencies_adaptive.append(latency)

        # Track difficulty and path count
        difficulty = store_adaptive._searcher.last_difficulty
        num_paths = store_adaptive._searcher.last_num_paths
        difficulties.append(difficulty)
        path_counts.append(num_paths)

    # ========================================================================
    # RESULTS ANALYSIS
    # ========================================================================

    print(f"\n" + "=" * 80)
    print("RESULTS: STATIC vs LAYER-ADAPTIVE HNSW")
    print("=" * 80)

    # Overall metrics
    avg_recall_static = np.mean(recalls_static) * 100
    avg_recall_adaptive = np.mean(recalls_adaptive) * 100
    avg_precision_static = np.mean(precisions_static) * 100
    avg_precision_adaptive = np.mean(precisions_adaptive) * 100
    avg_mrr_static = np.mean(mrrs_static)
    avg_mrr_adaptive = np.mean(mrrs_adaptive)
    avg_latency_static = np.mean(latencies_static)
    avg_latency_adaptive = np.mean(latencies_adaptive)

    print(f"\n1. RECALL (Critical for Medical Domain)")
    print(f"   {'Metric':<25s} {'Static':>12s} {'Adaptive':>12s} {'Improvement':>15s}")
    print(f"   {'-'*70}")
    print(f"   {'Average Recall@10':<25s} {avg_recall_static:>11.2f}% {avg_recall_adaptive:>11.2f}% {(avg_recall_adaptive-avg_recall_static):>14.2f}pp")
    print(f"   {'Median Recall@10':<25s} {np.median(recalls_static)*100:>11.2f}% {np.median(recalls_adaptive)*100:>11.2f}% {(np.median(recalls_adaptive)-np.median(recalls_static))*100:>14.2f}pp")
    print(f"   {'Min Recall@10':<25s} {np.min(recalls_static)*100:>11.2f}% {np.min(recalls_adaptive)*100:>11.2f}% {(np.min(recalls_adaptive)-np.min(recalls_static))*100:>14.2f}pp")
    print(f"   {'Max Recall@10':<25s} {np.max(recalls_static)*100:>11.2f}% {np.max(recalls_adaptive)*100:>11.2f}% {(np.max(recalls_adaptive)-np.max(recalls_static))*100:>14.2f}pp")

    # Zero recall queries (critical failures)
    zero_recall_static = sum(1 for r in recalls_static if r == 0)
    zero_recall_adaptive = sum(1 for r in recalls_adaptive if r == 0)
    print(f"\n   {'Zero-Recall Queries':<25s} {zero_recall_static:>8d} ({zero_recall_static/len(recalls_static)*100:.1f}%) {zero_recall_adaptive:>8d} ({zero_recall_adaptive/len(recalls_adaptive)*100:.1f}%) {zero_recall_static-zero_recall_adaptive:>10d} fewer")

    print(f"\n2. LATENCY (Speed Performance)")
    print(f"   {'Metric':<25s} {'Static':>12s} {'Adaptive':>12s} {'Overhead':>15s}")
    print(f"   {'-'*70}")
    print(f"   {'Average Latency':<25s} {avg_latency_static:>10.3f}ms {avg_latency_adaptive:>10.3f}ms {(avg_latency_adaptive/avg_latency_static-1)*100:>13.1f}%")
    print(f"   {'Median Latency':<25s} {np.median(latencies_static):>10.3f}ms {np.median(latencies_adaptive):>10.3f}ms {(np.median(latencies_adaptive)/np.median(latencies_static)-1)*100:>13.1f}%")
    print(f"   {'P95 Latency':<25s} {np.percentile(latencies_static, 95):>10.3f}ms {np.percentile(latencies_adaptive, 95):>10.3f}ms {(np.percentile(latencies_adaptive, 95)/np.percentile(latencies_static, 95)-1)*100:>13.1f}%")
    print(f"   {'P99 Latency':<25s} {np.percentile(latencies_static, 99):>10.3f}ms {np.percentile(latencies_adaptive, 99):>10.3f}ms {(np.percentile(latencies_adaptive, 99)/np.percentile(latencies_static, 99)-1)*100:>13.1f}%")

    print(f"\n3. OTHER METRICS")
    print(f"   {'Metric':<25s} {'Static':>12s} {'Adaptive':>12s} {'Difference':>15s}")
    print(f"   {'-'*70}")
    print(f"   {'Average Precision@10':<25s} {avg_precision_static:>11.2f}% {avg_precision_adaptive:>11.2f}% {avg_precision_adaptive-avg_precision_static:>14.2f}pp")
    print(f"   {'Mean Reciprocal Rank':<25s} {avg_mrr_static:>12.4f} {avg_mrr_adaptive:>12.4f} {avg_mrr_adaptive-avg_mrr_static:>+15.4f}")
    print(f"   {'Build Time':<25s} {build_time_static:>10.2f}s {build_time_adaptive:>10.2f}s {build_time_adaptive-build_time_static:>13.2f}s")

    # Difficulty analysis
    print(f"\n4. QUERY DIFFICULTY DISTRIBUTION (Adaptive)")
    print(f"   {'-'*70}")
    easy_queries = sum(1 for d in difficulties if d < 0.8)
    medium_queries = sum(1 for d in difficulties if 0.8 <= d < 0.9)
    hard_queries = sum(1 for d in difficulties if d >= 0.9)

    print(f"   Easy queries (<0.8):     {easy_queries:4d} ({easy_queries/len(difficulties)*100:5.1f}%)")
    print(f"   Medium queries (0.8-0.9): {medium_queries:4d} ({medium_queries/len(difficulties)*100:5.1f}%)")
    print(f"   Hard queries (>=0.9):     {hard_queries:4d} ({hard_queries/len(difficulties)*100:5.1f}%)")

    print(f"\n   Average difficulty: {np.mean(difficulties):.3f}")
    print(f"   Median difficulty:  {np.median(difficulties):.3f}")

    # Path usage
    path_1 = sum(1 for p in path_counts if p == 1)
    path_2 = sum(1 for p in path_counts if p == 2)
    path_3 = sum(1 for p in path_counts if p == 3)

    print(f"\n   Path usage:")
    print(f"     1 path: {path_1:4d} ({path_1/len(path_counts)*100:5.1f}%)")
    print(f"     2 paths: {path_2:4d} ({path_2/len(path_counts)*100:5.1f}%)")
    print(f"     3 paths: {path_3:4d} ({path_3/len(path_counts)*100:5.1f}%)")

    # Recall by difficulty
    print(f"\n5. RECALL BY QUERY DIFFICULTY")
    print(f"   {'Category':<20s} {'Count':>8s} {'Static Recall':>15s} {'Adaptive Recall':>17s} {'Improvement':>15s}")
    print(f"   {'-'*80}")

    for diff_label, diff_range in [("Easy (<0.8)", (0, 0.8)),
                                     ("Medium (0.8-0.9)", (0.8, 0.9)),
                                     ("Hard (>=0.9)", (0.9, 999))]:
        mask = [(diff_range[0] <= d < diff_range[1]) for d in difficulties]
        if sum(mask) > 0:
            recalls_s = np.array([recalls_static[i] for i, m in enumerate(mask) if m])
            recalls_a = np.array([recalls_adaptive[i] for i, m in enumerate(mask) if m])

            avg_s = np.mean(recalls_s) * 100
            avg_a = np.mean(recalls_a) * 100
            improvement = avg_a - avg_s

            print(f"   {diff_label:<20s} {sum(mask):>8d} {avg_s:>14.2f}% {avg_a:>16.2f}% {improvement:>14.2f}pp")

    # Summary
    print(f"\n" + "=" * 80)
    print("SUMMARY FOR MEDICAL DOMAIN")
    print("=" * 80)

    recall_improvement = avg_recall_adaptive - avg_recall_static
    latency_overhead = (avg_latency_adaptive / avg_latency_static - 1) * 100

    print(f"\nRecall Performance (Critical for Medical Safety):")
    print(f"  Static HNSW:          {avg_recall_static:.2f}%")
    print(f"  Layer-Adaptive HNSW:  {avg_recall_adaptive:.2f}%")
    print(f"  Improvement:          {recall_improvement:+.2f} percentage points ({recall_improvement/avg_recall_static*100:+.1f}% relative)")

    print(f"\nLatency Performance (User Experience):")
    print(f"  Static HNSW:          {avg_latency_static:.3f}ms")
    print(f"  Layer-Adaptive HNSW:  {avg_latency_adaptive:.3f}ms")
    print(f"  Overhead:             {latency_overhead:+.1f}%")

    print(f"\nCritical Failures:")
    print(f"  Zero-recall queries (Static):    {zero_recall_static} ({zero_recall_static/n_queries*100:.1f}%)")
    print(f"  Zero-recall queries (Adaptive):  {zero_recall_adaptive} ({zero_recall_adaptive/n_queries*100:.1f}%)")
    print(f"  Reduction:                       {zero_recall_static-zero_recall_adaptive} fewer failures")

    print(f"\nRecommendation for Medical Domain:")
    if recall_improvement > 2:
        print(f"  STRONGLY RECOMMEND Layer-Adaptive HNSW")
        print(f"  Reason: {recall_improvement:.1f}pp recall improvement critical for medical safety")
        print(f"          {latency_overhead:.1f}% latency overhead is acceptable tradeoff")
    elif recall_improvement > 0.5:
        print(f"  RECOMMEND Layer-Adaptive HNSW")
        print(f"  Reason: Modest recall improvement with manageable latency overhead")
    else:
        print(f"  EVALUATE: Marginal improvement may not justify overhead")

    print(f"\n" + "=" * 80)


if __name__ == "__main__":
    np.random.seed(42)
    main()
