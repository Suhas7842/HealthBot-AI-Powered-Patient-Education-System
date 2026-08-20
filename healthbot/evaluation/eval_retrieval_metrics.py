"""
Retrieval metrics evaluation script for HealthBot.

**Tier 1: Primary Evaluation Script**

**Purpose:** Evaluate retrieval quality using proper Information Retrieval (IR) metrics

**When to Use:**
- Testing changes to retrieval algorithms (hybrid weights, BM25 tuning, embeddings)
- Evaluating reranker impact
- Establishing retrieval quality baselines
- Comparing different retrieval configurations

**Metrics:** Recall@K, MRR, nDCG@K, Hit Rate, Precision@K

**Ground Truth:** Uses condition-based document matching from knowledge base

For complete evaluation guidance, see docs/EVALUATION_GUIDE.md
"""

import json
import time
from collections import defaultdict

from healthbot.evaluation.metrics import evaluate_retrieval_batch
from healthbot.evaluation.test_suite import enrich_test_cases_with_ground_truth
from healthbot.logger import logger
from healthbot.retrieval.retriever import HybridRetriever


def evaluate_retrieval_metrics(
    sample_size: int | None = None,
    k: int = 5,
    save_results: bool = True,
) -> dict:
    """
    Evaluate retrieval quality using proper IR metrics.

    Args:
        sample_size: Number of test cases to evaluate (None = all 50)
        k: Number of top results to evaluate (default: 5)
        save_results: Whether to save results to JSON

    Returns:
        Dictionary with retrieval metrics and detailed results
    """
    logger.info(f"Starting retrieval metrics evaluation (k={k})")

    # Load and enrich test suite with ground truth (cache it!)
    logger.info("Loading test suite and generating ground truth (cached)...")

    # Cache ground truth generation - only load documents once
    import pickle
    from pathlib import Path

    cache_file = Path("evaluation_cache.pkl")

    if cache_file.exists():
        logger.info("Loading cached ground truth...")
        with open(cache_file, "rb") as f:
            enriched_cases = pickle.load(f)
    else:
        logger.info("Generating ground truth (this will take a moment)...")
        enriched_cases = enrich_test_cases_with_ground_truth()
        # Cache for future runs
        with open(cache_file, "wb") as f:
            pickle.dump(enriched_cases, f)
        logger.info("Ground truth cached for future runs")

    # Sample if requested
    if sample_size and sample_size < len(enriched_cases):
        import random

        enriched_cases = random.sample(enriched_cases, sample_size)
        logger.info(f"Sampled {sample_size} test cases")

    logger.info(f"Evaluating {len(enriched_cases)} test cases")

    # Initialize retriever
    retriever = HybridRetriever()

    # Run retrieval and collect results
    retrieval_results = []
    latencies = []

    for i, case in enumerate(enriched_cases, 1):
        question = case["question"]
        condition = case["condition"]
        relevant_doc_ids = case["relevant_doc_ids"]

        logger.info(f"[{i}/{len(enriched_cases)}] Evaluating: {question[:50]}...")
        logger.info(f"  Ground truth: {len(relevant_doc_ids)} relevant docs")

        # Measure retrieval time
        start_time = time.time()
        retrieved_docs = retriever.retrieve(question, k=k * 2)  # Retrieve more for analysis
        latency = time.time() - start_time
        latencies.append(latency)

        # Extract retrieved doc IDs (using chunk_id)
        retrieved_ids = []
        for doc in retrieved_docs:
            chunk_id = doc.get("metadata", {}).get("chunk_id")
            if chunk_id is not None:
                retrieved_ids.append(str(chunk_id))

        logger.info(f"  Retrieved {len(retrieved_ids)} docs in {latency:.3f}s")

        # Store result for batch evaluation
        retrieval_results.append(
            {
                "query": question,
                "condition": condition,
                "retrieved_ids": retrieved_ids,
                "relevant_ids": relevant_doc_ids,
                "latency": latency,
            }
        )

    # Calculate aggregate metrics
    logger.info(f"Calculating retrieval metrics (Recall@{k}, MRR, nDCG@{k}, etc.)...")
    metrics = evaluate_retrieval_batch(retrieval_results, k=k)

    # Add latency metrics
    metrics["latency"] = {
        "mean": sum(latencies) / len(latencies) if latencies else 0,
        "min": min(latencies) if latencies else 0,
        "max": max(latencies) if latencies else 0,
    }

    # Group by condition
    by_condition = defaultdict(list)
    for result in retrieval_results:
        condition = result["condition"]
        by_condition[condition].append(result)

    condition_metrics = {}
    for condition, cond_results in by_condition.items():
        cond_metrics = evaluate_retrieval_batch(cond_results, k=k)
        condition_metrics[condition] = {
            "recall_at_k": cond_metrics["recall_at_k"],
            "precision_at_k": cond_metrics["precision_at_k"],
            "mrr": cond_metrics["mrr"],
            "ndcg_at_k": cond_metrics["ndcg_at_k"],
            "hit_rate": cond_metrics["hit_rate"],
            "num_queries": cond_metrics["num_queries"],
        }

    # Compile final results
    results = {
        "summary": {
            "total_queries": len(retrieval_results),
            "k": k,
            "recall_at_k": metrics["recall_at_k"],
            "precision_at_k": metrics["precision_at_k"],
            "mrr": metrics["mrr"],
            "ndcg_at_k": metrics["ndcg_at_k"],
            "hit_rate": metrics["hit_rate"],
            "avg_latency": metrics["latency"]["mean"],
        },
        "by_condition": condition_metrics,
        "detailed_results": retrieval_results,
    }

    # Print summary
    print_results_summary(results)

    # Save results
    if save_results:
        output_path = "retrieval_metrics_results.json"
        with open(output_path, "w") as f:
            # Remove detailed per-query results for cleaner output
            save_data = {
                "summary": results["summary"],
                "by_condition": results["by_condition"],
            }
            json.dump(save_data, f, indent=2)
        logger.info(f"Results saved to {output_path}")

    return results


def print_results_summary(results: dict) -> None:
    """Print formatted summary of retrieval metrics."""
    summary = results["summary"]

    print("\n" + "=" * 80)
    print("RETRIEVAL METRICS EVALUATION RESULTS")
    print("=" * 80)

    print(f"\nEvaluated {summary['total_queries']} queries at k={summary['k']}")

    print("\n📊 Overall Retrieval Metrics:")
    print(f"  • Recall@{summary['k']}: {summary['recall_at_k']:.3f}")
    print(f"    └─ Coverage: Found {summary['recall_at_k']*100:.1f}% of relevant docs")
    print(f"  • Precision@{summary['k']}: {summary['precision_at_k']:.3f}")
    print(f"    └─ Relevance: {summary['precision_at_k']*100:.1f}% of retrieved docs are relevant")
    print(f"  • MRR (Mean Reciprocal Rank): {summary['mrr']:.3f}")
    print(f"    └─ First relevant doc typically at rank {1/summary['mrr']:.1f}" if summary['mrr'] > 0 else "    └─ No relevant docs found")
    print(f"  • nDCG@{summary['k']}: {summary['ndcg_at_k']:.3f}")
    print(f"    └─ Ranking quality score (1.0 = perfect)")
    print(f"  • Hit Rate@{summary['k']}: {summary['hit_rate']:.3f}")
    print(f"    └─ {summary['hit_rate']*100:.1f}% of queries retrieved ≥1 relevant doc")

    print("\n⚡ Performance:")
    print(f"  • Average Latency: {summary['avg_latency']:.3f}s")

    print("\n📋 By Condition:")
    by_condition = results["by_condition"]
    for condition in sorted(by_condition.keys()):
        metrics = by_condition[condition]
        print(f"\n  {condition.upper()} ({metrics['num_queries']} queries):")
        print(f"    • Recall@{summary['k']}: {metrics['recall_at_k']:.3f}")
        print(f"    • MRR: {metrics['mrr']:.3f}")
        print(f"    • nDCG@{summary['k']}: {metrics['ndcg_at_k']:.3f}")
        print(f"    • Hit Rate: {metrics['hit_rate']:.3f}")

    print("\n" + "=" * 80)


def main():
    """Run retrieval metrics evaluation."""
    print("=" * 80)
    print("RETRIEVAL METRICS EVALUATION")
    print("=" * 80)
    print("\nThis evaluates retrieval quality using proper IR metrics:")
    print("  • Recall@K: What % of relevant documents were retrieved?")
    print("  • Precision@K: What % of retrieved documents are relevant?")
    print("  • MRR: How highly is the first relevant document ranked?")
    print("  • nDCG@K: Quality of ranking (higher scores for relevant docs at top)")
    print("  • Hit Rate@K: Did we retrieve at least one relevant document?")
    print("\n" + "=" * 80)

    # Ask for parameters
    try:
        sample_input = input("\nHow many test cases? (1-50, Enter for all 50): ").strip()
        sample_size = int(sample_input) if sample_input else None

        k_input = input("Top-K to evaluate? (Enter for 5): ").strip()
        k = int(k_input) if k_input else 5
    except ValueError:
        sample_size = None
        k = 5

    print(f"\nEvaluating retrieval metrics...")
    if sample_size:
        print(f"Sample size: {sample_size} queries")
    print(f"Top-K: {k}")
    print()

    try:
        results = evaluate_retrieval_metrics(sample_size=sample_size, k=k)

        # Generate report
        print("\n" + "=" * 80)
        print("EVALUATION COMPLETE")
        print("=" * 80)
        print("\nResults saved to: retrieval_metrics_results.json")
        print("\nKey Findings:")
        summary = results["summary"]
        print(f"  • Recall@{k}: {summary['recall_at_k']:.3f} - capturing {summary['recall_at_k']*100:.1f}% of relevant docs")
        print(f"  • MRR: {summary['mrr']:.3f} - first relevant doc at rank ~{1/summary['mrr']:.1f}" if summary['mrr'] > 0 else f"  • MRR: {summary['mrr']:.3f} - no relevant docs found")
        print(f"  • nDCG@{k}: {summary['ndcg_at_k']:.3f} - ranking quality score")
        print("=" * 80)

    except Exception as e:
        print(f"\nEvaluation failed: {e}")
        logger.error(f"Evaluation error: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main()
