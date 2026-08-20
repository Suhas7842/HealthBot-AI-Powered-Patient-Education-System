"""
Simplified RAGAS-style evaluation without external dependencies.

**Tier 1: Primary Evaluation Script**

**Purpose:** Evaluate answer generation quality using RAGAS-style metrics

**When to Use:**
- Testing changes to LLM prompts
- Evaluating generation quality improvements
- Measuring faithfulness (hallucination prevention)
- Assessing answer relevancy

**Metrics:**
- Faithfulness (0-1): Is the answer grounded in retrieved context?
- Answer Relevancy (0-1): Does the answer address the question?

**Method:** Uses existing LLM as judge (no external RAGAS dependencies)

This avoids RAGAS library dependency issues while providing core metrics.
For complete evaluation guidance, see docs/EVALUATION_GUIDE.md
"""

import json
import time
from typing import Optional

from healthbot.evaluation.test_suite import MEDICAL_TEST_CASES
from healthbot.logger import logger
from healthbot.models import LLMWrapper
from healthbot.tools import ToolSelector


def evaluate_faithfulness(
    question: str, answer: str, contexts: list[str], llm: LLMWrapper
) -> float:
    """
    Evaluate if answer is grounded in retrieved contexts.

    Score 0-1: Higher means better grounding in context.
    """
    context_str = "\n\n".join([f"[Context {i+1}] {ctx}" for i, ctx in enumerate(contexts)])

    prompt = f"""You are evaluating the faithfulness of an AI-generated answer.

Question: {question}

Retrieved Contexts:
{context_str}

Generated Answer:
{answer}

Task: Determine if the answer is fully grounded in the provided contexts.
- Score 1.0: All claims in the answer are supported by the contexts
- Score 0.5: Some claims are supported, some are not
- Score 0.0: Answer contains claims not in contexts or contradicts them

Respond with ONLY a number between 0.0 and 1.0 (e.g., "0.8")
"""

    try:
        from langchain_core.messages import HumanMessage

        response = llm.invoke([HumanMessage(content=prompt)])
        score_str = response.strip()
        score = float(score_str)
        return max(0.0, min(1.0, score))
    except Exception as e:
        logger.error(f"Faithfulness evaluation failed: {e}")
        return 0.5  # Default to neutral


def evaluate_relevancy(question: str, answer: str, llm: LLMWrapper) -> float:
    """
    Evaluate if answer addresses the question.

    Score 0-1: Higher means more relevant to the question.
    """
    prompt = f"""You are evaluating the relevancy of an AI-generated answer.

Question: {question}

Generated Answer:
{answer}

Task: Determine if the answer directly addresses the question asked.
- Score 1.0: Answer fully addresses the question
- Score 0.5: Answer partially addresses the question
- Score 0.0: Answer is off-topic or doesn't address the question

Respond with ONLY a number between 0.0 and 1.0 (e.g., "0.9")
"""

    try:
        from langchain_core.messages import HumanMessage

        response = llm.invoke([HumanMessage(content=prompt)])
        score_str = response.strip()
        score = float(score_str)
        return max(0.0, min(1.0, score))
    except Exception as e:
        logger.error(f"Relevancy evaluation failed: {e}")
        return 0.5  # Default to neutral


def run_single_case(
    question: str,
    ground_truth: str,
    condition: str,
    tool_selector: ToolSelector,
    llm: LLMWrapper,
) -> dict:
    """
    Run single test case through RAG pipeline and evaluate.

    Returns:
        Dict with question, answer, contexts, and scores
    """
    logger.info(f"Evaluating: {question[:60]}...")

    # Retrieve context
    results = tool_selector.select_and_search(question, k=5)

    if not results["success"] or not results["documents"]:
        logger.warning(f"No results for: {question}")
        return {
            "question": question,
            "answer": "Unable to retrieve information",
            "contexts": [],
            "ground_truth": ground_truth,
            "condition": condition,
            "faithfulness": 0.0,
            "relevancy": 0.0,
            "retrieval_success": False,
        }

    # Extract contexts
    contexts = [doc["text"] for doc in results["documents"]]

    # Generate answer
    context_str = "\n\n".join([f"[Source {i + 1}] {ctx}" for i, ctx in enumerate(contexts)])

    prompt = f"""Based on the following medical sources, answer this question: {question}

Medical Sources:
{context_str}

Provide a clear, accurate answer based only on the sources above."""

    from langchain_core.messages import HumanMessage, SystemMessage

    messages = [
        SystemMessage(
            content="You are a medical education assistant. Answer based only on provided sources."
        ),
        HumanMessage(content=prompt),
    ]

    answer = llm.invoke(messages)

    # Evaluate metrics
    faithfulness = evaluate_faithfulness(question, answer, contexts, llm)
    relevancy = evaluate_relevancy(question, answer, llm)

    return {
        "question": question,
        "answer": answer,
        "contexts": contexts,
        "ground_truth": ground_truth,
        "condition": condition,
        "faithfulness": faithfulness,
        "relevancy": relevancy,
        "retrieval_success": True,
        "num_contexts": len(contexts),
    }


def evaluate_test_suite(
    sample_size: Optional[int] = None, save_results: bool = True
) -> dict:
    """
    Evaluate RAG system on test suite.

    Args:
        sample_size: Number of cases to evaluate (None = all)
        save_results: Whether to save results to JSON

    Returns:
        Dictionary with evaluation results
    """
    test_cases = MEDICAL_TEST_CASES

    # Sample if needed
    if sample_size and sample_size < len(test_cases):
        import random

        test_cases = random.sample(test_cases, sample_size)
        logger.info(f"Sampling {sample_size} test cases")

    logger.info(f"Evaluating {len(test_cases)} test cases")

    # Initialize components
    tool_selector = ToolSelector()
    llm = LLMWrapper()

    # Run evaluation
    results = []
    start_time = time.time()

    for i, case in enumerate(test_cases, 1):
        logger.info(f"[{i}/{len(test_cases)}] Processing case...")
        result = run_single_case(
            question=case["question"],
            ground_truth=case["ground_truth"],
            condition=case["condition"],
            tool_selector=tool_selector,
            llm=llm,
        )
        results.append(result)

        # Progress update
        if i % 5 == 0:
            avg_faith = sum(r["faithfulness"] for r in results) / len(results)
            avg_rel = sum(r["relevancy"] for r in results) / len(results)
            logger.info(f"Progress: {i}/{len(test_cases)} | Faithfulness: {avg_faith:.3f} | Relevancy: {avg_rel:.3f}")

    total_time = time.time() - start_time

    # Calculate aggregate metrics
    successful_results = [r for r in results if r["retrieval_success"]]

    summary = {
        "total_cases": len(test_cases),
        "successful_retrievals": len(successful_results),
        "avg_faithfulness": sum(r["faithfulness"] for r in successful_results) / len(successful_results)
        if successful_results
        else 0,
        "avg_relevancy": sum(r["relevancy"] for r in successful_results) / len(successful_results)
        if successful_results
        else 0,
        "total_time_seconds": total_time,
        "avg_time_per_case": total_time / len(test_cases),
    }

    # Group by condition
    from collections import defaultdict

    by_condition = defaultdict(list)
    for result in successful_results:
        by_condition[result["condition"]].append(result)

    condition_metrics = {}
    for condition, cond_results in by_condition.items():
        condition_metrics[condition] = {
            "count": len(cond_results),
            "avg_faithfulness": sum(r["faithfulness"] for r in cond_results) / len(cond_results),
            "avg_relevancy": sum(r["relevancy"] for r in cond_results) / len(cond_results),
        }

    eval_results = {
        "summary": summary,
        "by_condition": condition_metrics,
        "detailed_results": results,
    }

    # Save results
    if save_results:
        output_path = "simple_ragas_results.json"
        with open(output_path, "w") as f:
            json.dump(eval_results, f, indent=2)
        logger.info(f"Results saved to {output_path}")

    return eval_results


def print_results(results: dict):
    """Print formatted evaluation results."""
    summary = results["summary"]

    print("\n" + "=" * 80)
    print("RAG QUALITY EVALUATION RESULTS")
    print("=" * 80)

    print(f"\nTotal Cases: {summary['total_cases']}")
    print(f"Successful Retrievals: {summary['successful_retrievals']}")

    print("\n📊 Answer Quality Metrics:")
    print(f"  • Faithfulness: {summary['avg_faithfulness']:.3f}")
    print("    └─ How well answers are grounded in retrieved context")
    print(f"  • Relevancy: {summary['avg_relevancy']:.3f}")
    print("    └─ How well answers address the question")

    print(f"\n⚡ Performance:")
    print(f"  • Total Time: {summary['total_time_seconds']:.1f}s")
    print(f"  • Avg Time/Case: {summary['avg_time_per_case']:.1f}s")

    if "by_condition" in results:
        print("\n📋 By Condition:")
        for condition in sorted(results["by_condition"].keys()):
            metrics = results["by_condition"][condition]
            print(f"\n  {condition.upper()} ({metrics['count']} cases):")
            print(f"    • Faithfulness: {metrics['avg_faithfulness']:.3f}")
            print(f"    • Relevancy: {metrics['avg_relevancy']:.3f}")

    print("\n" + "=" * 80)


def main():
    """Run simple RAGAS-style evaluation."""
    print("=" * 80)
    print("SIMPLE RAGAS-STYLE EVALUATION")
    print("=" * 80)
    print("\nThis evaluates answer quality without external RAGAS dependencies:")
    print("  • Faithfulness: Is the answer grounded in retrieved context?")
    print("  • Relevancy: Does the answer address the question?")
    print("\n" + "=" * 80)

    # Ask for sample size
    try:
        sample_input = input("\nHow many test cases? (1-50, Enter for all 50): ").strip()
        sample_size = int(sample_input) if sample_input else None
    except (ValueError, EOFError):
        sample_size = 10  # Default to 10 for safety

    print(f"\nEvaluating {sample_size or 'all 50'} test cases...")
    print("This will take several minutes due to LLM calls...\n")

    try:
        results = evaluate_test_suite(sample_size=sample_size)
        print_results(results)

        print("\nResults saved to: simple_ragas_results.json")

    except Exception as e:
        print(f"\nEvaluation failed: {e}")
        logger.error(f"Evaluation error: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main()
