"""
Custom performance metrics for HealthBot system.
Tracks latency, retrieval quality, cost, and user experience metrics.
"""

import json
from datetime import datetime
from pathlib import Path

import numpy as np


# ============================================================================
# RETRIEVAL EVALUATION METRICS
# ============================================================================


def recall_at_k(retrieved_ids: list[str], relevant_ids: list[str], k: int) -> float:
    """
    Calculate Recall@K - what percentage of relevant documents were retrieved?

    Args:
        retrieved_ids: List of retrieved document IDs (in ranked order)
        relevant_ids: List of ground truth relevant document IDs
        k: Number of top results to consider

    Returns:
        Recall score between 0 and 1

    Example:
        >>> recall_at_k(['doc1', 'doc2', 'doc3'], ['doc1', 'doc4', 'doc5'], k=3)
        0.333  # Found 1 out of 3 relevant docs
    """
    if not relevant_ids:
        return 0.0

    top_k_retrieved = set(retrieved_ids[:k])
    relevant_set = set(relevant_ids)
    retrieved_relevant = top_k_retrieved.intersection(relevant_set)

    return len(retrieved_relevant) / len(relevant_set)


def mean_reciprocal_rank(retrieved_ids: list[str], relevant_ids: list[str]) -> float:
    """
    Calculate Mean Reciprocal Rank (MRR) - how highly is the first relevant doc ranked?

    MRR = 1 / rank_of_first_relevant_doc

    Args:
        retrieved_ids: List of retrieved document IDs (in ranked order)
        relevant_ids: List of ground truth relevant document IDs

    Returns:
        MRR score between 0 and 1

    Example:
        >>> mean_reciprocal_rank(['doc1', 'doc2', 'doc3'], ['doc2'])
        0.5  # First relevant doc is at rank 2, so 1/2 = 0.5
    """
    relevant_set = set(relevant_ids)

    for rank, doc_id in enumerate(retrieved_ids, start=1):
        if doc_id in relevant_set:
            return 1.0 / rank

    return 0.0  # No relevant docs retrieved


def hit_rate(retrieved_ids: list[str], relevant_ids: list[str], k: int) -> float:
    """
    Calculate Hit Rate@K - did we retrieve at least one relevant document?

    Args:
        retrieved_ids: List of retrieved document IDs (in ranked order)
        relevant_ids: List of ground truth relevant document IDs
        k: Number of top results to consider

    Returns:
        1.0 if at least one relevant doc in top-K, else 0.0

    Example:
        >>> hit_rate(['doc1', 'doc2', 'doc3'], ['doc2'], k=3)
        1.0  # Hit! doc2 is in top 3
    """
    top_k_retrieved = set(retrieved_ids[:k])
    relevant_set = set(relevant_ids)

    return 1.0 if top_k_retrieved.intersection(relevant_set) else 0.0


def ndcg_at_k(retrieved_ids: list[str], relevant_ids: list[str], k: int) -> float:
    """
    Calculate Normalized Discounted Cumulative Gain (nDCG@K).

    Measures ranking quality - relevant docs should appear higher in results.
    Score is between 0 and 1, where 1 = perfect ranking.

    DCG = sum(relevance_i / log2(rank_i + 1))
    nDCG = DCG / IDCG (ideal DCG)

    Args:
        retrieved_ids: List of retrieved document IDs (in ranked order)
        relevant_ids: List of ground truth relevant document IDs
        k: Number of top results to consider

    Returns:
        nDCG score between 0 and 1

    Example:
        >>> ndcg_at_k(['doc1', 'doc2', 'doc3'], ['doc1', 'doc2'], k=3)
        1.0  # Perfect ranking - both relevant docs at top
    """
    if not relevant_ids:
        return 0.0

    relevant_set = set(relevant_ids)

    # Calculate DCG
    dcg = 0.0
    for rank, doc_id in enumerate(retrieved_ids[:k], start=1):
        if doc_id in relevant_set:
            relevance = 1.0  # Binary relevance (relevant = 1, not relevant = 0)
            dcg += relevance / np.log2(rank + 1)

    # Calculate IDCG (ideal DCG - all relevant docs at top)
    idcg = 0.0
    for rank in range(1, min(len(relevant_ids), k) + 1):
        idcg += 1.0 / np.log2(rank + 1)

    return dcg / idcg if idcg > 0 else 0.0


def precision_at_k(retrieved_ids: list[str], relevant_ids: list[str], k: int) -> float:
    """
    Calculate Precision@K - what percentage of retrieved docs are relevant?

    Args:
        retrieved_ids: List of retrieved document IDs (in ranked order)
        relevant_ids: List of ground truth relevant document IDs
        k: Number of top results to consider

    Returns:
        Precision score between 0 and 1

    Example:
        >>> precision_at_k(['doc1', 'doc2', 'doc3'], ['doc1', 'doc4'], k=3)
        0.333  # 1 out of 3 retrieved docs is relevant
    """
    if k == 0:
        return 0.0

    top_k_retrieved = retrieved_ids[:k]
    relevant_set = set(relevant_ids)
    relevant_retrieved = sum(1 for doc_id in top_k_retrieved if doc_id in relevant_set)

    return relevant_retrieved / k


def evaluate_retrieval_batch(
    results: list[dict],
    k: int = 5,
) -> dict:
    """
    Evaluate retrieval quality across multiple queries.

    Args:
        results: List of dicts with keys:
            - retrieved_ids: List[str] - retrieved document IDs
            - relevant_ids: List[str] - ground truth relevant IDs
            - query: str (optional) - the query text
        k: Number of top results to evaluate

    Returns:
        Dictionary with average metrics:
        - recall_at_k
        - precision_at_k
        - mrr
        - ndcg_at_k
        - hit_rate
        - per_query_results (detailed breakdown)

    Example:
        >>> results = [
        ...     {'retrieved_ids': ['doc1', 'doc2'], 'relevant_ids': ['doc1']},
        ...     {'retrieved_ids': ['doc3', 'doc4'], 'relevant_ids': ['doc3', 'doc5']},
        ... ]
        >>> evaluate_retrieval_batch(results, k=5)
        {'recall_at_k': 0.75, 'mrr': 1.0, ...}
    """
    if not results:
        return {"error": "No results to evaluate"}

    per_query_metrics = []

    for result in results:
        retrieved = result.get("retrieved_ids", [])
        relevant = result.get("relevant_ids", [])

        if not relevant:  # Skip queries with no ground truth
            continue

        metrics = {
            "query": result.get("query", ""),
            "recall_at_k": recall_at_k(retrieved, relevant, k),
            "precision_at_k": precision_at_k(retrieved, relevant, k),
            "mrr": mean_reciprocal_rank(retrieved, relevant),
            "ndcg_at_k": ndcg_at_k(retrieved, relevant, k),
            "hit_rate": hit_rate(retrieved, relevant, k),
            "num_relevant": len(relevant),
            "num_retrieved": len(retrieved[:k]),
        }

        per_query_metrics.append(metrics)

    # Calculate averages
    avg_metrics = {
        "recall_at_k": float(np.mean([m["recall_at_k"] for m in per_query_metrics])),
        "precision_at_k": float(
            np.mean([m["precision_at_k"] for m in per_query_metrics])
        ),
        "mrr": float(np.mean([m["mrr"] for m in per_query_metrics])),
        "ndcg_at_k": float(np.mean([m["ndcg_at_k"] for m in per_query_metrics])),
        "hit_rate": float(np.mean([m["hit_rate"] for m in per_query_metrics])),
        "k": k,
        "num_queries": len(per_query_metrics),
        "per_query_results": per_query_metrics,
    }

    return avg_metrics


# ============================================================================
# EXISTING METRICS CLASS
# ============================================================================


class HealthBotMetrics:
    """
    Tracks and computes performance metrics for HealthBot system.

    Metrics tracked:
    - Latency (total, per-node, percentiles)
    - Retrieval quality (scores, hit rate)
    - Cost (tokens, estimated API cost)
    - System usage (tool calls, emergency rate)
    """

    def __init__(self, log_file: str = "metrics_log.jsonl"):
        """
        Initialize metrics tracker.

        Args:
            log_file: Path to save metrics log
        """
        self.log_file = log_file
        self.run_history: list[dict] = []
        self._load_history()

    def _load_history(self) -> None:
        """Load existing run history from log file."""
        if Path(self.log_file).exists():
            try:
                with open(self.log_file, "r") as f:
                    self.run_history = [json.loads(line) for line in f]
            except Exception:
                self.run_history = []

    def log_run(self, run_data: dict) -> None:
        """
        Log a single workflow run.

        Args:
            run_data: Dictionary containing run metrics
        """
        # Add timestamp
        run_data["timestamp"] = datetime.now().isoformat()

        # Append to history
        self.run_history.append(run_data)

        # Append to log file
        with open(self.log_file, "a") as f:
            f.write(json.dumps(run_data) + "\n")

    def extract_run_metrics(self, state: dict) -> dict:
        """
        Extract metrics from workflow state.

        Args:
            state: Final workflow state

        Returns:
            Dictionary of metrics
        """
        node_latencies = state.get("node_latencies", {})
        total_latency = sum(node_latencies.values()) if node_latencies else 0

        retrieval_scores = state.get("retrieval_scores", [])
        avg_retrieval_score = np.mean(retrieval_scores) if retrieval_scores else 0

        token_usage = state.get("token_usage", {})
        total_tokens = sum(token_usage.values()) if token_usage else 0

        return {
            "topic": state.get("topic", ""),
            "condition": state.get("topic", "").lower(),  # Simplified
            "total_latency": total_latency,
            "node_latencies": node_latencies,
            "retrieval_score": avg_retrieval_score,
            "num_retrieved_docs": len(state.get("retrieved_docs", [])),
            "tool_calls": state.get("tool_calls", 0),
            "total_tokens": total_tokens,
            "confidence_score": state.get("confidence_score", 0),
            "emergency_detected": state.get("emergency_detected", False),
            "used_rag": len(state.get("retrieved_docs", [])) > 0,
        }

    def calculate_metrics(self, recent_n: int | None = None) -> dict:
        """
        Calculate aggregate metrics from run history.

        Args:
            recent_n: Only use last N runs (None = all)

        Returns:
            Dictionary of computed metrics
        """
        if not self.run_history:
            return {"error": "No run history available"}

        # Use recent runs if specified
        runs = self.run_history[-recent_n:] if recent_n else self.run_history

        # Extract values
        latencies = [r.get("total_latency", 0) for r in runs if "total_latency" in r]
        retrieval_scores = [
            r.get("retrieval_score", 0) for r in runs if "retrieval_score" in r
        ]
        tokens = [r.get("total_tokens", 0) for r in runs if "total_tokens" in r]
        tool_calls = [r.get("tool_calls", 0) for r in runs]
        rag_used = [r.get("used_rag", False) for r in runs]
        emergencies = [r.get("emergency_detected", False) for r in runs]

        # Compute metrics
        metrics = {
            # Latency metrics
            "latency": {
                "mean": float(np.mean(latencies)) if latencies else 0,
                "median": float(np.median(latencies)) if latencies else 0,
                "p95": float(np.percentile(latencies, 95)) if latencies else 0,
                "p99": float(np.percentile(latencies, 99)) if latencies else 0,
                "min": float(np.min(latencies)) if latencies else 0,
                "max": float(np.max(latencies)) if latencies else 0,
            },
            # Retrieval metrics
            "retrieval": {
                "mean_score": float(np.mean(retrieval_scores))
                if retrieval_scores
                else 0,
                "median_score": float(np.median(retrieval_scores))
                if retrieval_scores
                else 0,
                "rag_hit_rate": sum(rag_used) / len(runs) if runs else 0,
            },
            # Cost metrics (rough estimates)
            "cost": {
                "mean_tokens": float(np.mean(tokens)) if tokens else 0,
                "total_tokens": sum(tokens),
                # Rough cost estimate: GPT-4o-mini at $0.15/$0.60 per 1M tokens (input/output)
                "estimated_cost_usd": sum(tokens) * 0.0000004
                if tokens
                else 0,  # Avg rate
            },
            # Usage metrics
            "usage": {
                "total_runs": len(runs),
                "mean_tool_calls": float(np.mean(tool_calls)) if tool_calls else 0,
                "emergency_rate": sum(emergencies) / len(runs) if runs else 0,
            },
            # Time range
            "period": {
                "start": runs[0].get("timestamp", "unknown") if runs else None,
                "end": runs[-1].get("timestamp", "unknown") if runs else None,
            },
        }

        return metrics

    def get_metrics_by_condition(self) -> dict:
        """
        Calculate metrics grouped by medical condition.

        Returns:
            Dictionary mapping conditions to their metrics
        """
        from collections import defaultdict

        by_condition = defaultdict(list)
        for run in self.run_history:
            condition = run.get("condition", "unknown")
            by_condition[condition].append(run)

        condition_metrics = {}
        for condition, runs in by_condition.items():
            latencies = [r["total_latency"] for r in runs if "total_latency" in r]
            scores = [r["retrieval_score"] for r in runs if "retrieval_score" in r]

            condition_metrics[condition] = {
                "count": len(runs),
                "avg_latency": float(np.mean(latencies)) if latencies else 0,
                "avg_retrieval_score": float(np.mean(scores)) if scores else 0,
            }

        return condition_metrics

    def print_summary(self, recent_n: int | None = None) -> None:
        """
        Print formatted metrics summary.

        Args:
            recent_n: Only show last N runs (None = all)
        """
        metrics = self.calculate_metrics(recent_n)

        if "error" in metrics:
            print(f"Error: {metrics['error']}")
            return

        print("=" * 80)
        print("HEALTHBOT PERFORMANCE METRICS")
        print("=" * 80)

        # Usage stats
        usage = metrics["usage"]
        print("\n📊 Usage Statistics")
        print(f"  Total Runs: {usage['total_runs']}")
        print(f"  Avg Tool Calls per Run: {usage['mean_tool_calls']:.2f}")
        print(f"  Emergency Detection Rate: {usage['emergency_rate'] * 100:.1f}%")

        # Latency stats
        latency = metrics["latency"]
        print("\n⚡ Latency")
        print(f"  Mean: {latency['mean']:.2f}s")
        print(f"  Median: {latency['median']:.2f}s")
        print(f"  P95: {latency['p95']:.2f}s")
        print(f"  P99: {latency['p99']:.2f}s")
        print(f"  Range: {latency['min']:.2f}s - {latency['max']:.2f}s")

        # Retrieval stats
        retrieval = metrics["retrieval"]
        print("\n🔍 Retrieval Quality")
        print(f"  Mean Relevance Score: {retrieval['mean_score']:.3f}")
        print(f"  Median Score: {retrieval['median_score']:.3f}")
        print(f"  RAG Hit Rate: {retrieval['rag_hit_rate'] * 100:.1f}%")

        # Cost stats
        cost = metrics["cost"]
        print("\n💰 Cost Estimates")
        print(f"  Mean Tokens per Run: {cost['mean_tokens']:.0f}")
        print(f"  Total Tokens: {cost['total_tokens']:,}")
        print(f"  Estimated Total Cost: ${cost['estimated_cost_usd']:.4f}")

        # Time period
        period = metrics["period"]
        if period["start"] and period["end"]:
            print("\n📅 Period")
            print(f"  From: {period['start'][:19]}")  # Trim microseconds
            print(f"  To: {period['end'][:19]}")

        print("=" * 80)

    def export_report(self, output_path: str = "metrics_report.json") -> None:
        """
        Export comprehensive metrics report.

        Args:
            output_path: Path to save report
        """
        report = {
            "overall_metrics": self.calculate_metrics(),
            "by_condition": self.get_metrics_by_condition(),
            "recent_10": self.calculate_metrics(recent_n=10),
            "total_runs": len(self.run_history),
        }

        with open(output_path, "w") as f:
            json.dump(report, f, indent=2)

        print(f"Metrics report exported to: {output_path}")


def main():
    """Display current metrics."""
    metrics = HealthBotMetrics()

    if not metrics.run_history:
        print("No metrics data available yet.")
        print("Run HealthBot workflow first to generate metrics.")
        return

    # Print summary
    metrics.print_summary()

    # Show per-condition breakdown
    print("\n" + "=" * 80)
    print("METRICS BY CONDITION")
    print("=" * 80)

    by_condition = metrics.get_metrics_by_condition()
    for condition, stats in sorted(by_condition.items()):
        print(f"\n{condition.upper()}")
        print(f"  Runs: {stats['count']}")
        print(f"  Avg Latency: {stats['avg_latency']:.2f}s")
        print(f"  Avg Retrieval Score: {stats['avg_retrieval_score']:.3f}")

    print("=" * 80)

    # Export report
    metrics.export_report()


if __name__ == "__main__":
    main()
