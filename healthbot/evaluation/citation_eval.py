"""
Citation verification evaluation for HealthBot (Phase 2C).

Evaluates citation quality by measuring:
- Citation coverage: % of claims with citations
- Citation accuracy: % of claims supported by cited sources
- Attribution precision: How well sources support their claims

**Tier 2: Specialized Analysis Script**

**Purpose:** Evaluate citation quality and explainability

**When to Use:**
- Testing citation generation after schema changes
- Measuring hallucination via citation accuracy
- Validating source attribution quality

For complete evaluation guidance, see docs/EVALUATION_GUIDE.md
"""

import json
import time
from collections import defaultdict

from healthbot.citation_verification import CitationVerifier
from healthbot.evaluation.test_suite import MEDICAL_TEST_CASES
from healthbot.logger import logger
from healthbot.models import LLMWrapper
from healthbot.schemas import CitedClaim, CitedMedicalSummary
from healthbot.tools import ToolSelector


def generate_cited_summary(
    question: str, tool_selector: ToolSelector, llm: LLMWrapper
) -> CitedMedicalSummary | None:
    """
    Generate a summary with citations for evaluation.

    Args:
        question: Medical question
        tool_selector: Tool for retrieval
        llm: LLM wrapper for generation

    Returns:
        CitedMedicalSummary with citations, or None if generation fails
    """
    # Retrieve context
    results = tool_selector.select_and_search(question, k=5)

    if not results["success"] or not results["documents"]:
        logger.warning(f"No results for: {question}")
        return None

    # Format sources for prompt
    sources = []
    context_parts = []

    for i, doc in enumerate(results["documents"], 1):
        sources.append(
            {
                "text": doc["text"],
                "metadata": doc.get("metadata", {}),
                "score": doc.get("score", 0.0),
            }
        )
        context_parts.append(f"[Source {i}]\n{doc['text']}")

    context_str = "\n\n".join(context_parts)

    # Build prompt with citation requirements
    prompt = f"""Based on the following medical sources, create a summary with explicit citations.

**Topic**: {question}

**Medical Sources**:
{context_str}

**CRITICAL CITATION REQUIREMENTS**:
1. For EACH claim (cause, symptom, treatment), specify which source(s) support it
2. Use source numbers as they appear above (1, 2, 3, 4, 5)
3. If multiple sources support a claim, list all relevant sources
4. Do NOT make claims without source attribution
5. Only cite sources that actually support the claim

**Output Format** (use this exact JSON structure):
{{
    "title": "condition name",
    "condition": "brief description",
    "cited_causes": [
        {{
            "claim_text": "specific cause statement",
            "citation_ids": [1, 2],
            "confidence": 1.0
        }}
    ],
    "cited_symptoms": [
        {{
            "claim_text": "specific symptom statement",
            "citation_ids": [3],
            "confidence": 1.0
        }}
    ],
    "cited_treatments": [
        {{
            "claim_text": "specific treatment statement",
            "citation_ids": [2, 4],
            "confidence": 1.0
        }}
    ],
    "sources": {json.dumps(sources)},
    "warning": "This information is for educational purposes only. Always consult a qualified healthcare professional for medical advice."
}}

**Return ONLY valid JSON matching CitedMedicalSummary schema. No explanations or extra text.**"""

    try:
        from langchain_core.messages import HumanMessage, SystemMessage

        messages = [
            SystemMessage(
                content="You are a medical education assistant. Provide accurate information with proper source citations."
            ),
            HumanMessage(content=prompt),
        ]

        # Generate cited summary
        cited_summary = llm.invoke_structured(messages, CitedMedicalSummary)
        return cited_summary

    except Exception as e:
        logger.error(f"Failed to generate cited summary: {e}")
        return None


def evaluate_citation_quality(
    sample_size: int | None = None, save_results: bool = True
) -> dict:
    """
    Evaluate citation quality across test cases.

    Args:
        sample_size: Number of test cases (None = all 50)
        save_results: Whether to save results to JSON

    Returns:
        Dictionary with citation quality metrics
    """
    test_cases = MEDICAL_TEST_CASES

    # Sample if requested
    if sample_size and sample_size < len(test_cases):
        import random

        test_cases = random.sample(test_cases, sample_size)
        logger.info(f"Sampled {sample_size} test cases")

    logger.info(f"Evaluating citation quality on {len(test_cases)} cases")

    # Initialize components
    tool_selector = ToolSelector()
    llm = LLMWrapper()
    verifier = CitationVerifier()

    # Run evaluation
    results = []
    start_time = time.time()

    for i, case in enumerate(test_cases, 1):
        logger.info(f"[{i}/{len(test_cases)}] Processing: {case['question'][:60]}...")

        # Generate cited summary
        cited_summary = generate_cited_summary(case["question"], tool_selector, llm)

        if not cited_summary:
            results.append(
                {
                    "question": case["question"],
                    "condition": case["condition"],
                    "generation_success": False,
                    "error": "Failed to generate cited summary",
                }
            )
            continue

        # Verify citations
        verification = verifier.verify_summary(cited_summary)

        results.append(
            {
                "question": case["question"],
                "condition": case["condition"],
                "generation_success": True,
                "total_claims": verification["total_claims"],
                "supported_claims": verification["supported_claims"],
                "partially_supported_claims": verification["partially_supported_claims"],
                "unsupported_claims": verification["unsupported_claims"],
                "verification_score": verification["verification_score"],
                "by_category": verification["by_category"],
                "details": verification["details"],
            }
        )

        # Progress update
        if i % 5 == 0:
            avg_score = sum(r.get("verification_score", 0) for r in results if r.get("generation_success")) / max(
                1, sum(1 for r in results if r.get("generation_success"))
            )
            logger.info(
                f"Progress: {i}/{len(test_cases)} | Avg Verification Score: {avg_score:.2%}"
            )

    total_time = time.time() - start_time

    # Calculate aggregate metrics
    successful_results = [r for r in results if r.get("generation_success")]

    if not successful_results:
        logger.error("No successful results to analyze")
        return {"error": "All generations failed"}

    summary = {
        "total_cases": len(test_cases),
        "successful_generations": len(successful_results),
        "failed_generations": len(test_cases) - len(successful_results),
        "avg_verification_score": sum(r["verification_score"] for r in successful_results)
        / len(successful_results),
        "avg_claims_per_case": sum(r["total_claims"] for r in successful_results)
        / len(successful_results),
        "total_claims_verified": sum(r["total_claims"] for r in successful_results),
        "total_supported": sum(r["supported_claims"] for r in successful_results),
        "total_partially_supported": sum(
            r["partially_supported_claims"] for r in successful_results
        ),
        "total_unsupported": sum(r["unsupported_claims"] for r in successful_results),
        "total_time_seconds": total_time,
        "avg_time_per_case": total_time / len(test_cases),
    }

    # Group by condition
    by_condition = defaultdict(list)
    for result in successful_results:
        by_condition[result["condition"]].append(result)

    condition_metrics = {}
    for condition, cond_results in by_condition.items():
        condition_metrics[condition] = {
            "count": len(cond_results),
            "avg_verification_score": sum(r["verification_score"] for r in cond_results)
            / len(cond_results),
            "avg_claims": sum(r["total_claims"] for r in cond_results)
            / len(cond_results),
        }

    eval_results = {
        "summary": summary,
        "by_condition": condition_metrics,
        "detailed_results": results,
    }

    # Save results
    if save_results:
        output_path = "citation_eval_results.json"
        with open(output_path, "w") as f:
            json.dump(eval_results, f, indent=2)
        logger.info(f"Results saved to {output_path}")

    return eval_results


def print_results(results: dict):
    """Print formatted citation evaluation results."""
    if "error" in results:
        print(f"\n❌ Error: {results['error']}")
        return

    summary = results["summary"]

    print("\n" + "=" * 80)
    print("CITATION QUALITY EVALUATION RESULTS")
    print("=" * 80)

    print(f"\nTotal Cases: {summary['total_cases']}")
    print(
        f"Successful Generations: {summary['successful_generations']}/{summary['total_cases']}"
    )

    print("\n📊 Citation Quality Metrics:")
    print(f"  • Avg Verification Score: {summary['avg_verification_score']:.2%}")
    print("    └─ % of claims supported by their cited sources")
    print(f"  • Avg Claims per Case: {summary['avg_claims_per_case']:.1f}")
    print(f"  • Total Claims Verified: {summary['total_claims_verified']}")
    print(f"  • Supported: {summary['total_supported']}")
    print(f"  • Partially Supported: {summary['total_partially_supported']}")
    print(f"  • Not Supported: {summary['total_unsupported']}")

    print(f"\n⚡ Performance:")
    print(f"  • Total Time: {summary['total_time_seconds']:.1f}s")
    print(f"  • Avg Time/Case: {summary['avg_time_per_case']:.1f}s")

    print("\n📋 By Condition:")
    by_condition = results["by_condition"]
    for condition in sorted(by_condition.keys()):
        metrics = by_condition[condition]
        print(f"\n  {condition.upper()} ({metrics['count']} cases):")
        print(f"    • Verification Score: {metrics['avg_verification_score']:.2%}")
        print(f"    • Avg Claims: {metrics['avg_claims']:.1f}")

    print("\n" + "=" * 80)


def main():
    """Run citation quality evaluation."""
    print("=" * 80)
    print("CITATION QUALITY EVALUATION")
    print("=" * 80)
    print("\nThis evaluates claim-level citation accuracy:")
    print("  • Citation Coverage: Do all claims have source citations?")
    print("  • Citation Accuracy: Are claims supported by cited sources?")
    print("  • Attribution Quality: How well do sources support claims?")
    print("\n" + "=" * 80)

    # Ask for sample size
    try:
        sample_input = input("\nHow many test cases? (1-50, Enter for 10): ").strip()
        sample_size = int(sample_input) if sample_input else 10
    except (ValueError, EOFError):
        sample_size = 10

    print(f"\nEvaluating citation quality on {sample_size} cases...")
    print("This will take several minutes (LLM generation + verification)...\n")

    try:
        results = evaluate_citation_quality(sample_size=sample_size)
        print_results(results)

        print("\n" + "=" * 80)
        print("EVALUATION COMPLETE")
        print("=" * 80)
        print("\nResults saved to: citation_eval_results.json")
        print("\nKey Findings:")
        if "summary" in results:
            summary = results["summary"]
            print(
                f"  • Verification Score: {summary['avg_verification_score']:.2%} - "
                f"{summary['total_supported']}/{summary['total_claims_verified']} claims supported"
            )
            print(f"  • Avg Claims per Case: {summary['avg_claims_per_case']:.1f}")
        print("=" * 80)

    except Exception as e:
        print(f"\n❌ Evaluation failed: {e}")
        logger.error(f"Evaluation error: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main()
