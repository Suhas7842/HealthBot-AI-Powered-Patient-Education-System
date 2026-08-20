"""
Experiment comparison of retrieval strategies.

**Tier 2: Specialized Analysis Script**

**Purpose:** Compare 4 different retrieval strategies side-by-side

**When to Use:**
- Running ablation studies (which components matter most?)
- Comparing retrieval approaches for research
- Testing the impact of adding reranking
- Demonstrating retrieval strategy tradeoffs

**Strategies Compared:**
1. Dense-only (semantic search via Pinecone)
2. BM25-only (keyword search)
3. Hybrid (RRF fusion)
4. Hybrid + Reranker (with cross-encoder)

**Metrics:** Recall@K, MRR, nDCG, Hit Rate, Latency

For complete evaluation guidance, see docs/EVALUATION_GUIDE.md
"""

import json
import time
from collections import defaultdict

from healthbot.evaluation.metrics import evaluate_retrieval_batch
from healthbot.evaluation.test_suite import enrich_test_cases_with_ground_truth
from healthbot.logger import logger
from healthbot.retrieval.retriever import HybridRetriever


class RetrievalExperiment:
    """Run retrieval strategy experiments."""

    def __init__(self, k: int = 5):
        """
        Initialize experiment runner.

        Args:
            k: Number of top results to evaluate
        """
        self.k = k
        self.retriever = HybridRetriever(use_reranker=False)

    def run_dense_only(self, query: str) -> tuple[list[dict], float]:
        """Run dense (semantic) search only."""
        start = time.time()
        results = self.retriever.semantic_search(query, k=self.k)
        latency = time.time() - start
        return results, latency

    def run_bm25_only(self, query: str) -> tuple[list[dict], float]:
        """Run BM25 keyword search only."""
        start = time.time()
        results = self.retriever.keyword_search(query, k=self.k)
        # Pad if fewer than k results
        while len(results) < self.k:
            results.append(
                {"text": "", "metadata": {}, "score": 0.0, "method": "bm25"}
            )
        latency = time.time() - start
        return results[:self.k], latency

    def run_hybrid(self, query: str) -> tuple[list[dict], float]:
        """Run hybrid retrieval (RRF fusion)."""
        start = time.time()
        # Temporarily disable reranker
        original_flag = self.retriever.use_reranker
        self.retriever.use_reranker = False
        results = self.retriever.retrieve(query, k=self.k)
        self.retriever.use_reranker = original_flag
        latency = time.time() - start
        return results, latency

    def run_hybrid_with_reranker(self, query: str) -> tuple[list[dict], float]:
        """Run hybrid retrieval with cross-encoder reranking."""
        # Need to create new retriever with reranker
        reranking_retriever = HybridRetriever(use_reranker=True)
        start = time.time()
        results = reranking_retriever.retrieve(query, k=self.k)
        latency = time.time() - start
        return results, latency


def run_experiments(sample_size: int = 10, k: int = 5) -> dict:
    """
    Run all retrieval experiments.

    Args:
        sample_size: Number of test cases to evaluate
        k: Top-K for evaluation

    Returns:
        Dictionary with experiment results
    """
    logger.info(f"Starting retrieval experiments (k={k}, sample={sample_size})")

    # Load test cases with ground truth
    logger.info("Loading test cases with ground truth...")
    import pickle
    from pathlib import Path

    cache_file = Path("evaluation_cache.pkl")
    if cache_file.exists():
        with open(cache_file, "rb") as f:
            enriched_cases = pickle.load(f)
    else:
        enriched_cases = enrich_test_cases_with_ground_truth()
        with open(cache_file, "wb") as f:
            pickle.dump(enriched_cases, f)

    # Sample
    if sample_size < len(enriched_cases):
        import random
        enriched_cases = random.sample(enriched_cases, sample_size)

    logger.info(f"Running experiments on {len(enriched_cases)} cases")

    # Initialize experiment runner
    exp = RetrievalExperiment(k=k)

    # Store results for each strategy
    strategies = {
        "dense_only": [],
        "bm25_only": [],
        "hybrid": [],
        "hybrid_reranker": [],
    }

    latencies = defaultdict(list)

    # Run experiments
    for i, case in enumerate(enriched_cases, 1):
        question = case["question"]
        relevant_ids = case["relevant_doc_ids"]

        logger.info(f"[{i}/{len(enriched_cases)}] {question[:50]}...")

        # Experiment 1: Dense only
        docs, lat = exp.run_dense_only(question)
        retrieved_ids = [str(d.get("metadata", {}).get("chunk_id", "")) for d in docs]
        strategies["dense_only"].append({
            "query": question,
            "retrieved_ids": retrieved_ids,
            "relevant_ids": relevant_ids,
        })
        latencies["dense_only"].append(lat)

        # Experiment 2: BM25 only
        docs, lat = exp.run_bm25_only(question)
        retrieved_ids = [str(d.get("metadata", {}).get("chunk_id", "")) for d in docs]
        strategies["bm25_only"].append({
            "query": question,
            "retrieved_ids": retrieved_ids,
            "relevant_ids": relevant_ids,
        })
        latencies["bm25_only"].append(lat)

        # Experiment 3: Hybrid (RRF)
        docs, lat = exp.run_hybrid(question)
        retrieved_ids = [str(d.get("metadata", {}).get("chunk_id", "")) for d in docs]
        strategies["hybrid"].append({
            "query": question,
            "retrieved_ids": retrieved_ids,
            "relevant_ids": relevant_ids,
        })
        latencies["hybrid"].append(lat)

        # Experiment 4: Hybrid + Reranker
        try:
            docs, lat = exp.run_hybrid_with_reranker(question)
            retrieved_ids = [str(d.get("metadata", {}).get("chunk_id", "")) for d in docs]
            strategies["hybrid_reranker"].append({
                "query": question,
                "retrieved_ids": retrieved_ids,
                "relevant_ids": relevant_ids,
            })
            latencies["hybrid_reranker"].append(lat)
        except Exception as e:
            logger.warning(f"Reranker experiment failed: {e}")
            # Use hybrid results as fallback
            strategies["hybrid_reranker"].append(strategies["hybrid"][-1])
            latencies["hybrid_reranker"].append(latencies["hybrid"][-1])

    # Calculate metrics for each strategy
    results = {}

    for strategy_name, strategy_results in strategies.items():
        logger.info(f"Calculating metrics for {strategy_name}...")

        metrics = evaluate_retrieval_batch(strategy_results, k=k)

        results[strategy_name] = {
            "recall_at_k": metrics["recall_at_k"],
            "precision_at_k": metrics["precision_at_k"],
            "mrr": metrics["mrr"],
            "ndcg_at_k": metrics["ndcg_at_k"],
            "hit_rate": metrics["hit_rate"],
            "avg_latency": sum(latencies[strategy_name]) / len(latencies[strategy_name])
            if latencies[strategy_name]
            else 0,
            "num_queries": metrics["num_queries"],
        }

    return {
        "experiment_config": {"k": k, "sample_size": sample_size},
        "results": results,
    }


def print_comparison_table(experiment_results: dict):
    """Print formatted comparison table."""
    results = experiment_results["results"]
    config = experiment_results["experiment_config"]

    print("\n" + "=" * 100)
    print("RETRIEVAL STRATEGY COMPARISON")
    print("=" * 100)

    print(f"\nConfiguration: k={config['k']}, sample_size={config['sample_size']}")

    print("\n" + "-" * 100)
    print(f"{'Strategy':<20} {'Recall@5':<12} {'Precision@5':<12} {'MRR':<10} {'nDCG@5':<10} {'Hit Rate':<10} {'Latency (ms)':<15}")
    print("-" * 100)

    strategy_order = ["dense_only", "bm25_only", "hybrid", "hybrid_reranker"]
    strategy_names = {
        "dense_only": "Dense Only",
        "bm25_only": "BM25 Only",
        "hybrid": "Hybrid (RRF)",
        "hybrid_reranker": "Hybrid + Reranker",
    }

    for strategy in strategy_order:
        if strategy in results:
            r = results[strategy]
            print(
                f"{strategy_names[strategy]:<20} "
                f"{r['recall_at_k']:<12.3f} "
                f"{r['precision_at_k']:<12.3f} "
                f"{r['mrr']:<10.3f} "
                f"{r['ndcg_at_k']:<10.3f} "
                f"{r['hit_rate']:<10.3f} "
                f"{r['avg_latency']*1000:<15.1f}"
            )

    print("-" * 100)

    # Highlight best in each category
    print("\n📊 Best Performers:")
    best_recall = max(results.items(), key=lambda x: x[1]["recall_at_k"])
    best_mrr = max(results.items(), key=lambda x: x[1]["mrr"])
    best_ndcg = max(results.items(), key=lambda x: x[1]["ndcg_at_k"])
    best_latency = min(results.items(), key=lambda x: x[1]["avg_latency"])

    print(f"  • Recall@5: {strategy_names[best_recall[0]]} ({best_recall[1]['recall_at_k']:.3f})")
    print(f"  • MRR: {strategy_names[best_mrr[0]]} ({best_mrr[1]['mrr']:.3f})")
    print(f"  • nDCG@5: {strategy_names[best_ndcg[0]]} ({best_ndcg[1]['ndcg_at_k']:.3f})")
    print(f"  • Latency: {strategy_names[best_latency[0]]} ({best_latency[1]['avg_latency']*1000:.1f}ms)")

    print("\n" + "=" * 100)


def main():
    """Run experiment comparison."""
    print("=" * 100)
    print("RETRIEVAL STRATEGY EXPERIMENT")
    print("=" * 100)
    print("\nThis compares 4 retrieval strategies:")
    print("  1. Dense-only (semantic search)")
    print("  2. BM25-only (keyword search)")
    print("  3. Hybrid (RRF fusion)")
    print("  4. Hybrid + Reranker (cross-encoder)")
    print("\n" + "=" * 100)

    try:
        sample_input = input("\nSample size? (Enter for 10): ").strip()
        sample_size = int(sample_input) if sample_input else 10
    except (ValueError, EOFError):
        sample_size = 10

    print(f"\nRunning experiments on {sample_size} queries...")
    print("This may take several minutes...\n")

    try:
        results = run_experiments(sample_size=sample_size, k=5)

        # Print table
        print_comparison_table(results)

        # Save results
        output_path = "experiment_results.json"
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)

        print(f"\nResults saved to: {output_path}")

    except Exception as e:
        print(f"\nExperiment failed: {e}")
        logger.error(f"Experiment error: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main()
