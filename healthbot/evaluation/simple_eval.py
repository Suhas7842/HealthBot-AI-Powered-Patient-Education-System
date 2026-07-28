"""
Simple evaluation script for HealthBot RAG system.
Measures retrieval quality, response latency, and generation quality.
"""

import json
import time
from collections import defaultdict

from healthbot.evaluation.test_suite import MEDICAL_TEST_CASES
from healthbot.logger import logger
from healthbot.retrieval.retriever import HybridRetriever


def evaluate_retrieval(sample_size: int = 10) -> dict:
    """
    Evaluate RAG retrieval performance.

    Args:
        sample_size: Number of test cases to evaluate

    Returns:
        Dictionary with evaluation metrics
    """
    logger.info(f"Starting evaluation on {sample_size} test cases")

    # Initialize retriever
    retriever = HybridRetriever()

    # Sample test cases
    import random

    test_cases = random.sample(MEDICAL_TEST_CASES, min(sample_size, len(MEDICAL_TEST_CASES)))

    # Evaluation results
    results = []
    latencies = []
    retrieval_scores = []
    method_distribution = defaultdict(int)

    for i, case in enumerate(test_cases, 1):
        question = case["question"]
        condition = case["condition"]

        logger.info(f"[{i}/{sample_size}] Evaluating: {question[:50]}...")

        # Measure retrieval time
        start_time = time.time()
        retrieved_docs = retriever.retrieve(question, k=5)
        latency = time.time() - start_time

        latencies.append(latency)

        # Analyze results
        if retrieved_docs:
            avg_rrf_score = sum(doc.get("rrf_score", 0) for doc in retrieved_docs) / len(retrieved_docs)
            retrieval_scores.append(avg_rrf_score)

            # Track method distribution
            for doc in retrieved_docs:
                methods = doc.get("methods", [])
                method_key = "+".join(sorted(methods))
                method_distribution[method_key] += 1
        else:
            retrieval_scores.append(0.0)

        # Store result
        results.append(
            {
                "question": question,
                "condition": condition,
                "num_retrieved": len(retrieved_docs),
                "latency_seconds": latency,
                "avg_rrf_score": avg_rrf_score if retrieved_docs else 0.0,
            }
        )

    # Calculate summary statistics
    summary = {
        "total_cases": sample_size,
        "avg_latency_seconds": sum(latencies) / len(latencies) if latencies else 0,
        "min_latency_seconds": min(latencies) if latencies else 0,
        "max_latency_seconds": max(latencies) if latencies else 0,
        "avg_rrf_score": sum(retrieval_scores) / len(retrieval_scores) if retrieval_scores else 0,
        "retrieval_success_rate": sum(1 for r in results if r["num_retrieved"] > 0) / len(results),
        "method_distribution": dict(method_distribution),
    }

    # Group by condition
    by_condition = defaultdict(list)
    for result in results:
        by_condition[result["condition"]].append(result)

    condition_stats = {}
    for condition, cond_results in by_condition.items():
        condition_stats[condition] = {
            "count": len(cond_results),
            "avg_latency": sum(r["latency_seconds"] for r in cond_results) / len(cond_results),
            "avg_rrf_score": sum(r["avg_rrf_score"] for r in cond_results) / len(cond_results),
        }

    return {
        "summary": summary,
        "by_condition": condition_stats,
        "detailed_results": results,
    }


def main():
    """Run simple evaluation."""
    print("=" * 80)
    print("HEALTHBOT RAG EVALUATION")
    print("=" * 80)
    print("\nThis evaluates retrieval quality and performance metrics.")
    print("\nMetrics measured:")
    print("  • Retrieval latency (response time)")
    print("  • RRF scores (hybrid ranking quality)")
    print("  • Method distribution (semantic vs BM25)")
    print("  • Success rate (retrieval coverage)")
    print("\n" + "=" * 80)

    # Ask for sample size
    try:
        sample_input = input("\nHow many test cases? (1-50, Enter for 10): ").strip()
        sample_size = int(sample_input) if sample_input else 10
        sample_size = min(max(1, sample_size), 50)
    except ValueError:
        sample_size = 10

    print(f"\nEvaluating {sample_size} test cases...\n")

    try:
        results = evaluate_retrieval(sample_size=sample_size)

        # Print summary
        print("\n" + "=" * 80)
        print("EVALUATION RESULTS")
        print("=" * 80)

        summary = results["summary"]
        print(f"\nTest Cases: {summary['total_cases']}")
        print(f"Success Rate: {summary['retrieval_success_rate']:.1%}")

        print("\nPerformance:")
        print(f"  • Avg Latency: {summary['avg_latency_seconds']:.3f}s")
        print(f"  • Min Latency: {summary['min_latency_seconds']:.3f}s")
        print(f"  • Max Latency: {summary['max_latency_seconds']:.3f}s")

        print("\nRetrieval Quality:")
        print(f"  • Avg RRF Score: {summary['avg_rrf_score']:.4f}")

        print("\nMethod Distribution:")
        for method, count in sorted(summary["method_distribution"].items(), key=lambda x: -x[1]):
            pct = (count / (summary['total_cases'] * 5)) * 100
            print(f"  • {method}: {count} docs ({pct:.1f}%)")

        print("\nBy Condition:")
        for condition, stats in sorted(results["by_condition"].items()):
            print(f"  • {condition.title()}: {stats['count']} cases, "
                  f"{stats['avg_latency']:.3f}s avg latency, "
                  f"{stats['avg_rrf_score']:.4f} avg score")

        # Save results
        output_path = "evaluation_results.json"
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)

        print("\n" + "=" * 80)
        print(f"Results saved to: {output_path}")
        print("=" * 80)

    except Exception as e:
        print(f"\nEvaluation failed: {e}")
        logger.error(f"Evaluation error: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main()
