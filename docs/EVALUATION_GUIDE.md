# HealthBot Evaluation Guide

This guide explains how to evaluate the HealthBot RAG system using the appropriate evaluation scripts for your specific needs.

## Quick Decision Tree

**Are you evaluating retrieval quality?**  
→ Use [`eval_retrieval_metrics.py`](#tier-1-eval_retrieval_metricspy) (Tier 1)

**Are you evaluating answer generation quality?**  
→ Use [`simple_ragas.py`](#tier-1-simple_ragaspy) (Tier 1)

**Are you comparing multiple retrieval strategies?**  
→ Use [`experiments.py`](#tier-2-experimentspy) (Tier 2)

**Need a quick smoke test?**  
→ Use [`simple_eval.py`](#tier-3-simple_evalpy) (Tier 3)

---

## Evaluation Tiers

### Tier 1: Primary Evaluation Scripts

These are the **authoritative evaluation methods** you should use for systematic assessment.

#### `eval_retrieval_metrics.py`

**Purpose:** Evaluate retrieval quality using proper Information Retrieval (IR) metrics

**When to Use:**
- Testing changes to retrieval algorithms (hybrid weights, BM25 tuning, embedding models)
- Evaluating reranker impact
- Establishing retrieval quality baselines
- Comparing different retrieval configurations

**Metrics:**
- **Recall@K**: What percentage of relevant documents were retrieved in top-K results?
- **Precision@K**: What percentage of retrieved documents are relevant?
- **MRR (Mean Reciprocal Rank)**: How highly ranked is the first relevant document?
- **nDCG@K**: Normalized discounted cumulative gain (accounts for ranking order)
- **Hit Rate@K**: What percentage of queries retrieved at least one relevant document?

**Runtime:** ~2-3 minutes (50 test cases with caching)

**Usage:**
```bash
# Evaluate on full test suite
python -m healthbot.evaluation.eval_retrieval_metrics

# Evaluate on sample
python -m healthbot.evaluation.eval_retrieval_metrics --sample-size 20

# Specify k value
python -m healthbot.evaluation.eval_retrieval_metrics --k 5
```

**Output:** `retrieval_metrics_results.json` with metrics grouped by medical condition

---

#### `simple_ragas.py`

**Purpose:** Evaluate answer generation quality using RAGAS-style metrics without external dependencies

**When to Use:**
- Testing changes to LLM prompts
- Evaluating generation quality improvements
- Measuring faithfulness (hallucination prevention)
- Assessing answer relevancy

**Metrics:**
- **Faithfulness (0-1)**: Is the answer grounded in retrieved context?
- **Relevancy (0-1)**: Does the answer address the question asked?

**Runtime:** ~5-7 minutes (depends on LLM speed, uses existing LLM as judge)

**Usage:**
```bash
# Evaluate on full test suite
python -m healthbot.evaluation.simple_ragas

# Evaluate on sample
python -m healthbot.evaluation.simple_ragas --sample-size 20
```

**Output:** `simple_ragas_results.json` with faithfulness and relevancy scores

**Note:** This is a lightweight alternative to the full RAGAS library (`ragas_eval.py`) that avoids dependency issues.

---

### Tier 2: Specialized Analysis Scripts

Use these for **comparative analysis** and **ablation studies**.

#### `experiments.py`

**Purpose:** Compare 4 different retrieval strategies side-by-side

**When to Use:**
- Running ablation studies (which components matter most?)
- Comparing retrieval approaches for research
- Testing the impact of adding reranking
- Demonstrating retrieval strategy tradeoffs

**Strategies Compared:**
1. **Dense-only**: Semantic search via Pinecone/ChromaDB embeddings
2. **BM25-only**: Keyword-based lexical search
3. **Hybrid**: RRF (Reciprocal Rank Fusion) of dense + BM25
4. **Hybrid + Reranker**: Hybrid with cross-encoder reranking

**Metrics:** All Tier 1 retrieval metrics (Recall@K, MRR, nDCG, Hit Rate, Precision@K) plus latency

**Runtime:** ~8-10 minutes (runs all 4 strategies on each query)

**Usage:**
```bash
# Run comparison on sample
python -m healthbot.evaluation.experiments

# When prompted, enter sample size (e.g., 10)
```

**Output:** `experiment_results.json` with side-by-side comparison table

---

### Tier 3: Legacy/Deprecated Scripts

These scripts are **superseded by Tier 1 scripts** but kept for backward compatibility.

#### `simple_eval.py`

**Purpose:** Basic evaluation with simple success rate and latency metrics

**Status:** ⚠️ Superseded by `eval_retrieval_metrics.py`

**When to Use:** Quick smoke tests only (not recommended for systematic evaluation)

**Limitations:**
- No proper IR metrics (just "100% success rate")
- No ground truth comparison
- No distinction between retrieval quality levels

**Usage:**
```bash
python -m healthbot.evaluation.simple_eval
```

---

#### `ragas_eval.py`

**Purpose:** Full RAGAS framework evaluation (faithfulness, answer_relevancy, context_recall, context_precision)

**Status:** ⚠️ Has dependency compatibility issues, superseded by `simple_ragas.py`

**When to Use:** Only if you need the full RAGAS metrics and have resolved dependency conflicts

**Known Issues:**
- LangChain version conflicts
- External RAGAS library installation issues
- Slower than `simple_ragas.py` alternative

---

## When to Use Which Script

| Scenario | Primary Script | Secondary Script |
|----------|---------------|------------------|
| Improving retrieval algorithm | `eval_retrieval_metrics.py` | `experiments.py` |
| Tuning LLM prompts | `simple_ragas.py` | - |
| Testing reranker impact | `experiments.py` | `eval_retrieval_metrics.py` |
| End-to-end system evaluation | `simple_ragas.py` | `eval_retrieval_metrics.py` |
| Ablation study (which components matter?) | `experiments.py` | - |
| Quick sanity check | `simple_eval.py` | - |
| Establishing baselines | `eval_retrieval_metrics.py` + `simple_ragas.py` | - |

---

## Common Evaluation Workflows

### Workflow 1: Test Retrieval Algorithm Change

```bash
# 1. Establish baseline
python -m healthbot.evaluation.eval_retrieval_metrics
mv retrieval_metrics_results.json baseline_retrieval.json

# 2. Make change (e.g., enable reranker)
# Edit config.env: USE_RERANKER=True

# 3. Re-evaluate
python -m healthbot.evaluation.eval_retrieval_metrics
mv retrieval_metrics_results.json reranker_retrieval.json

# 4. Compare (manual diff or custom script)
# Expected: Recall@5 improvement, +40ms latency
```

---

### Workflow 2: Test Prompt Engineering Change

```bash
# 1. Establish baseline
python -m healthbot.evaluation.simple_ragas --sample-size 20
mv simple_ragas_results.json baseline_ragas.json

# 2. Edit prompts in healthbot/prompts.py

# 3. Re-evaluate
python -m healthbot.evaluation.simple_ragas --sample-size 20
mv simple_ragas_results.json new_prompt_ragas.json

# 4. Compare faithfulness and relevancy scores
# Expected: Improved faithfulness without sacrificing relevancy
```

---

### Workflow 3: Full System Evaluation (Before Deployment)

```bash
# 1. Retrieval quality check
python -m healthbot.evaluation.eval_retrieval_metrics

# 2. Answer quality check
python -m healthbot.evaluation.simple_ragas --sample-size 30

# 3. Review results
# - Recall@5 should be ≥ 0.75
# - Faithfulness should be ≥ 0.85
# - Relevancy should be ≥ 0.80

# 4. If metrics acceptable, proceed to deployment
```

---

### Workflow 4: Ablation Study (Research)

```bash
# Run comprehensive comparison of all retrieval strategies
python -m healthbot.evaluation.experiments

# When prompted, use sample size of 50 for statistical significance

# Analyze experiment_results.json to understand:
# - Dense vs BM25 tradeoffs
# - RRF fusion benefit over single method
# - Reranker precision improvement vs latency cost
```

---

## Metric Definitions

### Retrieval Metrics (IR)

**Recall@K**
- Formula: `(# relevant docs in top-K) / (total # relevant docs)`
- Range: 0.0 to 1.0 (higher is better)
- Example: Recall@5 = 0.78 means we found 78% of relevant documents in top 5 results

**Precision@K**
- Formula: `(# relevant docs in top-K) / K`
- Range: 0.0 to 1.0 (higher is better)
- Example: Precision@5 = 0.60 means 3 out of 5 retrieved docs are relevant

**MRR (Mean Reciprocal Rank)**
- Formula: Average of `1 / rank_of_first_relevant_doc` across all queries
- Range: 0.0 to 1.0 (higher is better)
- Example: MRR = 0.65 means first relevant doc typically at rank 1-2

**nDCG@K (Normalized Discounted Cumulative Gain)**
- Measures ranking quality (rewards relevant docs at higher positions)
- Range: 0.0 to 1.0 (1.0 = perfect ranking)
- Example: nDCG@5 = 0.71 indicates good but not perfect ranking

**Hit Rate@K**
- Formula: `(# queries with ≥1 relevant doc in top-K) / (total queries)`
- Range: 0.0 to 1.0 (higher is better)
- Example: Hit Rate@5 = 0.95 means 95% of queries got at least one relevant document

### Generation Metrics (RAGAS-style)

**Faithfulness**
- Measures: Is the answer grounded in retrieved context?
- Range: 0.0 to 1.0 (higher is better)
- Method: LLM-as-judge compares answer claims against retrieved sources
- Example: 0.90 means 90% of claims are supported by context

**Relevancy**
- Measures: Does the answer address the question asked?
- Range: 0.0 to 1.0 (higher is better)
- Method: LLM-as-judge evaluates answer alignment with question
- Example: 0.85 means answer is highly relevant to the question

---

## Expected Baseline Results

These are approximate baselines for the HealthBot system (50-case test suite):

| Metric | Expected Range | Notes |
|--------|----------------|-------|
| Recall@5 | 0.75 - 0.82 | With hybrid retrieval + reranker |
| MRR | 0.62 - 0.68 | First relevant doc at rank 1-2 |
| nDCG@5 | 0.68 - 0.74 | Good ranking quality |
| Hit Rate@5 | 0.92 - 0.98 | Most queries get relevant docs |
| Faithfulness | 0.82 - 0.92 | Answer grounded in context |
| Relevancy | 0.85 - 0.92 | Answer addresses question |
| Latency (retrieval) | 280-360ms | With/without reranker |

**Note:** Results vary based on:
- USE_RERANKER setting (adds ~40ms, improves Recall by ~5%)
- LLM model used (affects faithfulness/relevancy)
- Test case difficulty distribution

---

## Troubleshooting

### Issue: `eval_retrieval_metrics.py` is slow on first run

**Cause:** Ground truth generation loads 2,578 document chunks from knowledge base

**Solution:** Script automatically caches ground truth in `evaluation_cache.pkl`. First run takes ~3 minutes, subsequent runs take ~30 seconds.

---

### Issue: `simple_ragas.py` gives inconsistent scores

**Cause:** LLM-as-judge scoring has inherent variance

**Solution:** 
- Use larger sample sizes (30+ cases) for more stable averages
- Run evaluation multiple times and average results
- Set LLM temperature to 0.0 in config for determinism

---

### Issue: `ragas_eval.py` fails with dependency errors

**Cause:** LangChain version conflicts with RAGAS library

**Solution:** Use `simple_ragas.py` instead - provides same core metrics (faithfulness, relevancy) without external dependencies

---

### Issue: All metrics show 0.0 or NaN

**Cause:** Vector database not initialized or empty

**Solution:**
```bash
# Check if knowledge base exists
python -c "from healthbot.retrieval.retriever import HybridRetriever; r = HybridRetriever(); print('DB loaded successfully')"

# If fails, rebuild knowledge base
python scripts/build_vector_db.py
```

---

## Adding New Evaluation Metrics

If you want to add new evaluation metrics:

1. **For retrieval metrics:** Add to `healthbot/evaluation/metrics.py`
   - Implement metric function (e.g., `mean_average_precision()`)
   - Add to `evaluate_retrieval_batch()` function
   - Update `eval_retrieval_metrics.py` to report the metric

2. **For generation metrics:** Add to `simple_ragas.py`
   - Implement LLM-as-judge function (e.g., `evaluate_coherence()`)
   - Add to `run_single_case()` evaluation
   - Update summary statistics

3. **Update this guide:** Document the new metric's purpose, formula, and interpretation

---

## References

- **Reciprocal Rank Fusion (RRF):** Cormack et al., 2009
- **RAGAS Framework:** Es et al., 2023 - [GitHub](https://github.com/explodinggradients/ragas)
- **nDCG Metric:** Järvelin & Kekäläinen, 2002
- **Cross-Encoder Reranking:** Reimers & Gurevych, 2019

---

## Need Help?

- **Which script to use?** See the [Quick Decision Tree](#quick-decision-tree) above
- **Understanding metrics?** See [Metric Definitions](#metric-definitions)
- **Metrics seem wrong?** See [Troubleshooting](#troubleshooting)
- **Want to compare strategies?** Use [`experiments.py`](#tier-2-experimentspy)

**Last Updated:** 2026-08-20 (Phase 2A)
