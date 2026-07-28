"""
RAGAS (Retrieval-Augmented Generation Assessment) evaluation for HealthBot.
Measures RAG quality: faithfulness, answer relevancy, context precision/recall.
"""

import json
from typing import List, Dict
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_recall,
    context_precision
)
from healthbot.tools import ToolSelector
from healthbot.models import LLMWrapper
from healthbot.evaluation.test_suite import MEDICAL_TEST_CASES
from healthbot.logger import logger
from langchain_core.messages import SystemMessage, HumanMessage


class RAGASEvaluator:
    """Evaluates RAG system using RAGAS metrics."""

    def __init__(self):
        """Initialize evaluator with tools and LLM."""
        self.tool_selector = ToolSelector()
        self.llm_wrapper = LLMWrapper()

    def run_single_query(self, question: str, ground_truth: str) -> Dict:
        """
        Run a single query through the RAG pipeline.

        Args:
            question: User question
            ground_truth: Expected answer

        Returns:
            Dictionary with question, answer, contexts, and ground truth
        """
        logger.info(f"Running query: {question}")

        # Retrieve context
        results = self.tool_selector.select_and_search(question, k=5)

        if not results["success"] or not results["documents"]:
            logger.warning(f"No results for: {question}")
            return {
                "question": question,
                "answer": "Unable to retrieve information",
                "contexts": [],
                "ground_truth": ground_truth
            }

        # Extract contexts
        contexts = [doc["text"] for doc in results["documents"]]

        # Generate answer
        context_str = "\n\n".join([
            f"[Source {i+1}] {ctx}"
            for i, ctx in enumerate(contexts)
        ])

        prompt = f"""Based on the following medical sources, answer this question: {question}

Medical Sources:
{context_str}

Provide a clear, accurate answer based only on the sources above."""

        messages = [
            SystemMessage(content="You are a medical education assistant. Answer based only on provided sources."),
            HumanMessage(content=prompt)
        ]

        answer = self.llm_wrapper.invoke(messages)

        return {
            "question": question,
            "answer": answer,
            "contexts": contexts,
            "ground_truth": ground_truth
        }

    def evaluate_test_suite(
        self,
        test_cases: List[Dict] = None,
        sample_size: int = None,
        save_results: bool = True
    ) -> Dict:
        """
        Evaluate RAG system on test suite using RAGAS metrics.

        Args:
            test_cases: List of test cases (defaults to MEDICAL_TEST_CASES)
            sample_size: Number of cases to evaluate (None = all)
            save_results: Whether to save results to JSON

        Returns:
            Dictionary with RAGAS scores and detailed results
        """
        if test_cases is None:
            test_cases = MEDICAL_TEST_CASES

        # Sample if needed
        if sample_size and sample_size < len(test_cases):
            import random
            test_cases = random.sample(test_cases, sample_size)
            logger.info(f"Sampling {sample_size} test cases")

        logger.info(f"Evaluating {len(test_cases)} test cases")

        # Run queries
        eval_data = []
        for i, case in enumerate(test_cases, 1):
            logger.info(f"Processing case {i}/{len(test_cases)}")
            result = self.run_single_query(
                case["question"],
                case["ground_truth"]
            )
            result["condition"] = case.get("condition", "unknown")
            eval_data.append(result)

        # Convert to Dataset format for RAGAS
        dataset = Dataset.from_dict({
            "question": [d["question"] for d in eval_data],
            "answer": [d["answer"] for d in eval_data],
            "contexts": [d["contexts"] for d in eval_data],
            "ground_truth": [d["ground_truth"] for d in eval_data]
        })

        logger.info("Running RAGAS evaluation...")

        try:
            # Evaluate with RAGAS
            ragas_result = evaluate(
                dataset,
                metrics=[
                    faithfulness,
                    answer_relevancy,
                    context_recall,
                    context_precision
                ]
            )

            # Extract scores
            scores = {
                "faithfulness": ragas_result["faithfulness"],
                "answer_relevancy": ragas_result["answer_relevancy"],
                "context_recall": ragas_result["context_recall"],
                "context_precision": ragas_result["context_precision"]
            }

            # Compile results
            results = {
                "summary": {
                    "total_cases": len(test_cases),
                    "metrics": scores,
                    "average_score": sum(scores.values()) / len(scores)
                },
                "detailed_results": eval_data,
                "ragas_scores": ragas_result.to_pandas().to_dict('records')
            }

            logger.info(f"RAGAS Evaluation Complete!")
            logger.info(f"  Faithfulness: {scores['faithfulness']:.3f}")
            logger.info(f"  Answer Relevancy: {scores['answer_relevancy']:.3f}")
            logger.info(f"  Context Recall: {scores['context_recall']:.3f}")
            logger.info(f"  Context Precision: {scores['context_precision']:.3f}")

            # Save results
            if save_results:
                output_path = "evaluation_results.json"
                with open(output_path, "w") as f:
                    json.dump(results, f, indent=2)
                logger.info(f"Results saved to {output_path}")

            return results

        except Exception as e:
            logger.error(f"RAGAS evaluation failed: {e}")
            return {
                "error": str(e),
                "summary": {"total_cases": len(test_cases)},
                "detailed_results": eval_data
            }

    def evaluate_by_condition(self, test_cases: List[Dict] = None) -> Dict:
        """
        Evaluate and group results by medical condition.

        Args:
            test_cases: List of test cases

        Returns:
            Dictionary with scores per condition
        """
        if test_cases is None:
            test_cases = MEDICAL_TEST_CASES

        # Group by condition
        from collections import defaultdict
        by_condition = defaultdict(list)

        for case in test_cases:
            condition = case.get("condition", "unknown")
            by_condition[condition].append(case)

        # Evaluate each condition
        condition_scores = {}
        for condition, cases in by_condition.items():
            logger.info(f"Evaluating condition: {condition}")
            result = self.evaluate_test_suite(cases, save_results=False)

            if "error" not in result:
                condition_scores[condition] = result["summary"]["metrics"]

        return condition_scores


def main():
    """Run RAGAS evaluation on test suite."""
    print("="*80)
    print("RAGAS EVALUATION")
    print("="*80)
    print("\nThis will evaluate HealthBot's RAG system using RAGAS metrics.")
    print("Metrics measured:")
    print("  • Faithfulness: Is the answer grounded in retrieved context?")
    print("  • Answer Relevancy: Does the answer address the question?")
    print("  • Context Recall: Did retrieval find all relevant information?")
    print("  • Context Precision: Are retrieved contexts relevant?")
    print("\n" + "="*80)

    # Ask for sample size
    try:
        sample_input = input("\nHow many test cases to evaluate? (1-50, Enter for all 50): ").strip()
        sample_size = int(sample_input) if sample_input else None
    except ValueError:
        sample_size = None

    evaluator = RAGASEvaluator()

    try:
        results = evaluator.evaluate_test_suite(sample_size=sample_size)

        # Print summary
        print("\n" + "="*80)
        print("EVALUATION RESULTS")
        print("="*80)

        if "error" in results:
            print(f"Error: {results['error']}")
        else:
            summary = results["summary"]
            print(f"Test cases evaluated: {summary['total_cases']}")
            print(f"\nRAGAS Scores:")
            for metric, score in summary["metrics"].items():
                print(f"  • {metric.replace('_', ' ').title()}: {score:.3f}")
            print(f"\n  Average Score: {summary['average_score']:.3f}")

        print("="*80)
        print("\nDetailed results saved to: evaluation_results.json")
        print("="*80)

    except Exception as e:
        print(f"\nEvaluation failed: {e}")
        logger.error(f"Evaluation error: {e}", exc_info=True)


if __name__ == "__main__":
    main()
