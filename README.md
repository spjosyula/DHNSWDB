# Dynamic HNSW Vector Database

Experimenting with **smart, query-aware indexes**. Instead of being static, this index can adjust search behavior based on the query and even learn from feedback.  

Basically a normal HNSW graph, but with a small brain that tweaks edges and improves over time.

---

## What It Does

- **Query-aware search:** Adjusts how the index searches depending on the query.  
- **Feedback-driven edges:** Frequently used nodes get temporarily boosted.  
- **Dynamic insertion:** Add new embeddings without rebuilding everything. (over-optimistic goal to try)

---

## How It Works
This section gonna come soon
---
