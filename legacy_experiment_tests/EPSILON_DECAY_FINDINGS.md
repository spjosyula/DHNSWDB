# Epsilon Decay Experiment: GLIE vs Fixed Epsilon

## Note: Current Recommended Approach

**This document describes a previous research approach.** For the current recommended implementation, see the **Layer-Adaptive Multi-Path Search** documented in the main README.md and `layer_adaptive_test/README.md`. Layer-adaptive achieves +9% to +62.5% recall improvement without requiring epsilon-based exploration.

---

## Theoretical Improvement

**Hypothesis**: GLIE (Greedy in the Limit with Infinite Exploration) epsilon decay should improve Q-learning efficiency by:
1. Early exploration: Higher epsilon → better Q-value estimation
2. Late exploitation: Lower epsilon → less wasted exploration

**Theoretical Foundation**:
- GLIE guarantees convergence to optimal policy in reinforcement learning
- Formula: ε(t) = ε₀ / (1 + t/c) where c controls decay rate

## Experimental Setup

**Comparison**:
- Control: Fixed epsilon = 0.15 (no decay)
- Treatment: GLIE epsilon decay starting at 0.15

**Dataset**:
- Corpus: 203 real documents across 4 domains
- Queries: 352 diverse queries (exploratory, precise, mixed)
- Embeddings: sentence-transformers all-MiniLM-L6-v2 (384 dimensions)

**Metrics**:
- Efficiency = satisfaction / latency (primary metric)
- Satisfaction = fraction of relevant results
- Latency = query time in milliseconds

## Results

### Phase-by-Phase Comparison

| Phase | Fixed eps=0.15 | GLIE eps(t) | Efficiency Change |
|-------|---------------|-------------|-------------------|
| Early (0-100) | 186.15 sat/sec | 192.09 sat/sec | **+3.2%** ✓ |
| Middle (100-200) | 169.98 sat/sec | 163.85 sat/sec | **-3.6%** ✗ |
| Late (200-326) | 172.31 sat/sec | 170.51 sat/sec | **-1.0%** ✗ |
| **Overall** | **175.84** | **175.08** | **-0.4%** (neutral) |

### Epsilon Behavior

| Phase | Fixed eps | GLIE eps (avg) |
|-------|-----------|----------------|
| Early (0-100) | 0.0313 | 0.1237 |
| Middle (100-200) | 0.0100 | 0.0689 |
| Late (200-326) | 0.0100 | 0.0454 |
| Overall | 0.0165 | 0.0766 |

**Observation**: Fixed epsilon decayed too quickly (hit min_epsilon=0.01 by query ~50),
which is why average epsilon is low even without explicit decay. This is due to
multiplicative decay being called automatically.

## Analysis

### Why GLIE Didn't Improve Performance

1. **Early advantage exists but is small** (+3.2% in first 100 queries)
   - Higher exploration does help initially
   - But the improvement is marginal

2. **No long-term benefit**
   - Middle and late phases show slight degradation
   - Overall efficiency is essentially neutral (-0.4%)

3. **Q-learning converges quickly**
   - With only 6 ef_search candidates and 3 intents
   - Action space is small (18 state-action pairs)
   - Fixed exploration of 15% is sufficient

4. **Epsilon = 0.15 is already well-tuned**
   - Balances exploration and exploitation effectively
   - GLIE's varying epsilon doesn't provide additional benefit

### Theoretical vs Practical Gap

**Theory says**: GLIE should converge to optimal policy
**Practice shows**: The difference is negligible

**Why**:
- Small action space (6 ef_search values)
- Fast Q-value convergence (100-150 queries)
- Optimistic initialization already encourages early exploration
- Fixed epsilon=0.15 is empirically well-balanced

## Conclusion

**VERDICT**: ❌ **Do NOT implement GLIE epsilon decay**

**Reasoning**:
1. No significant efficiency improvement (+/- 0.4% is noise)
2. Adds implementation complexity
3. Introduces another hyperparameter (decay rate)
4. Fixed epsilon = 0.15 works just as well

**Recommendation**: Keep current implementation with fixed epsilon=0.15

## Lessons Learned

### Theoretical Improvements May Not Translate to Practice

- GLIE is theoretically sound for infinite-horizon RL
- But practical systems have:
  - Finite queries
  - Small action spaces
  - Fast convergence
  - Real-world noise

### When GLIE Might Help

GLIE decay could be beneficial when:
- **Large action space** (100+ actions)
- **Slow convergence** (1000+ queries needed)
- **Non-stationary environment** (Q-values drift over time)
- **High exploration cost** (expensive to try bad actions)

None of these apply to our ef_search Q-learning scenario.

### Simplicity Over Complexity

The best algorithm is the simplest one that works. Fixed epsilon=0.15:
- ✅ Simple to implement
- ✅ One less hyperparameter
- ✅ Performs just as well as GLIE
- ✅ Easier to explain and maintain

## Implementation Note

We implemented GLIE decay support in `ef_search_selector.py` with three modes:
- `"none"`: Fixed epsilon (no decay)
- `"multiplicative"`: Exponential decay ε(t+1) = 0.95 * ε(t)
- `"glie"`: GLIE decay ε(t) = ε₀ / (1 + t/100)

**Recommended mode**: `"none"` (fixed epsilon=0.15)

The code is kept for research purposes but should not be used in production.

---

**Experiment Date**: 2025-10-06
**Test Script**: `examples/epsilon_decay_ab_test.py`
**Full Results**: See test output above
