"""
Agent Evaluation for HealthBot Phase 4.

Evaluates agent tool selection accuracy, multi-tool usage, and reasoning quality.

This demonstrates that the agent makes intelligent tool choices, not random selection.
"""

from typing import Dict, Any, List, Tuple
import json
from pathlib import Path


# Test cases for agent evaluation
AGENT_TEST_CASES = [
    {
        "query": "What is BMI and what's mine if I'm 70kg and 1.75m?",
        "expected_tools": {
            "required": ["medical_calculator"],
            "optional": ["medical_rag_search"],
            "inappropriate": []
        },
        "reason": "Calculator required for BMI. RAG optional for explanation (agent could explain inline).",
        "category": "multi_tool",
    },
    {
        "query": "Calculate my medication dosage: 70kg patient, 5mg/kg",
        "expected_tools": ["medical_calculator"],
        "reason": "Pure calculation, no medical context needed",
        "category": "single_tool",
    },
    {
        "query": "What causes Type 2 diabetes?",
        "expected_tools": ["medical_rag_search"],
        "reason": "Common condition in local knowledge base",
        "category": "single_tool",
    },
    {
        "query": "Compare recent studies on diabetes treatment efficacy",
        "expected_tools": ["pubmed_api_search"],
        "reason": "Research comparison needs PubMed, not local KB",
        "category": "single_tool",
    },
    {
        "query": "What are the symptoms of hypertension?",
        "expected_tools": ["medical_rag_search"],
        "reason": "Common condition, symptoms available in local KB",
        "category": "single_tool",
    },
    {
        "query": "Calculate creatinine clearance: 65 year old male, 70kg, serum cr 1.2",
        "expected_tools": ["medical_calculator"],
        "reason": "Kidney function calculation",
        "category": "single_tool",
    },
    {
        "query": "What's my BMI if I weigh 85kg and I'm 1.80m tall? Is that healthy?",
        "expected_tools": {
            "required": ["medical_calculator"],
            "optional": ["medical_rag_search"],
            "inappropriate": []
        },
        "reason": "Calculator required. 'Is that healthy?' could be answered with basic ranges (optional RAG).",
        "category": "multi_tool",
    },
    {
        "query": "Recent COVID-19 treatment updates",
        "expected_tools": {
            "required": ["web_search", "pubmed_api_search"],  # Either is valid
            "optional": ["medical_rag_search"],  # Background info optional
            "inappropriate": []
        },
        "reason": "web_search for news OR pubmed_api_search for recent research both valid. Requires at least one.",
        "category": "single_tool",
    },
    {
        "query": "What are the risk factors for cardiovascular disease?",
        "expected_tools": ["medical_rag_search"],
        "reason": "Medical education, available in local KB",
        "category": "single_tool",
    },
    {
        "query": "Latest research on cancer immunotherapy",
        "expected_tools": ["pubmed_api_search"],
        "reason": "Recent research, broad topic",
        "category": "single_tool",
    },
    {
        "query": "What is asthma and how is it treated?",
        "expected_tools": ["medical_rag_search"],
        "reason": "Common condition with treatment info in local KB",
        "category": "single_tool",
    },
    {
        "query": "Calculate BMI for 60kg, 1.65m and explain what it means",
        "expected_tools": {
            "required": ["medical_calculator"],
            "optional": ["medical_rag_search"],
            "inappropriate": []
        },
        "reason": "Calculator required. 'Explain what it means' ambiguous - could be inline or RAG.",
        "category": "multi_tool",
    },
    {
        "query": "Compare Type 1 vs Type 2 diabetes symptoms",
        "expected_tools": ["medical_rag_search"],
        "reason": "Comparative medical question, both in local KB",
        "category": "single_tool",
    },
    {
        "query": "What's the normal BMI range?",
        "expected_tools": ["medical_rag_search"],
        "reason": "Medical information, no calculation needed",
        "category": "single_tool",
    },
    {
        "query": "Recent meta-analysis on statins for cholesterol",
        "expected_tools": ["pubmed_api_search"],
        "reason": "Specific research query, needs PubMed",
        "category": "single_tool",
    },
    {
        "query": "What is COPD?",
        "expected_tools": ["medical_rag_search"],
        "reason": "Common condition in local knowledge base",
        "category": "single_tool",
    },
    {
        "query": "Calculate medication dose: 50kg patient at 3mg/kg",
        "expected_tools": ["medical_calculator"],
        "reason": "Pure dosage calculation",
        "category": "single_tool",
    },
    {
        "query": "What are modifiable risk factors for heart disease?",
        "expected_tools": ["medical_rag_search"],
        "reason": "Medical education, preventive care",
        "category": "single_tool",
    },
    {
        "query": "New WHO guidelines on obesity management",
        "expected_tools": {
            "required": ["web_search", "pubmed_api_search"],  # Either is valid
            "optional": [],
            "inappropriate": []
        },
        "reason": "web_search for official guidelines OR pubmed_api_search for research both valid.",
        "category": "single_tool",
    },
    {
        "query": "What's my kidney function if creatinine clearance is 45 ml/min?",
        "expected_tools": ["medical_rag_search"],
        "reason": "Interpretation of lab value, no calculation",
        "category": "single_tool",
    },
]


def evaluate_tool_selection(
    actual_tools: List[str],
    expected_tools: List[str] | Dict[str, List[str]]
) -> Dict[str, Any]:
    """
    Evaluate if agent selected appropriate tools.

    Supports flexible expectations to avoid penalizing valid alternative tool combinations.

    Args:
        actual_tools: Tools actually called by agent
        expected_tools: Either:
            - List[str]: Simple list (backward compatible)
            - Dict with keys:
                - 'required': Must use at least one
                - 'optional': May use (no penalty if missing)
                - 'inappropriate': Should NOT use

    Returns:
        Dictionary with:
            - exact_match: bool (perfect tool selection)
            - partial_match: bool (some correct tools)
            - precision: float (% of actual tools that were appropriate)
            - recall: float (% of required tools that were called)
            - score: float (F1 score)
            - has_required_tool: bool (used at least one required tool)
            - used_inappropriate_tool: bool (used tool marked inappropriate)
    """
    if not actual_tools:
        return {
            "exact_match": False,
            "partial_match": False,
            "precision": 0.0,
            "recall": 0.0,
            "score": 0.0,
            "has_required_tool": False,
            "used_inappropriate_tool": False,
        }

    actual_set = set(actual_tools)

    # Handle backward compatibility (simple list format)
    if isinstance(expected_tools, list):
        expected_set = set(expected_tools)

        # Exact match
        exact_match = actual_set == expected_set

        # Partial match (at least one correct tool)
        correct_tools = actual_set & expected_set
        partial_match = len(correct_tools) > 0

        # Precision: % of actual tools that were expected
        precision = len(correct_tools) / len(actual_set) if actual_set else 0.0

        # Recall: % of expected tools that were called
        recall = len(correct_tools) / len(expected_set) if expected_set else 0.0

        # F1 score
        if precision + recall > 0:
            f1_score = 2 * (precision * recall) / (precision + recall)
        else:
            f1_score = 0.0

        return {
            "exact_match": exact_match,
            "partial_match": partial_match,
            "precision": precision,
            "recall": recall,
            "score": f1_score,
            "has_required_tool": partial_match,  # Any match counts as "required"
            "used_inappropriate_tool": False,  # No concept in simple format
        }

    # Handle flexible format (dict with required/optional/inappropriate)
    required = set(expected_tools.get('required', []))
    optional = set(expected_tools.get('optional', []))
    inappropriate = set(expected_tools.get('inappropriate', []))

    # Check if agent used at least one required tool
    has_required = len(actual_set & required) > 0 if required else True

    # Check if agent used any inappropriate tools
    used_inappropriate = len(actual_set & inappropriate) > 0

    # Appropriate tools = required + optional
    appropriate = required | optional
    correct_tools = actual_set & appropriate

    # Exact match: uses required tools and optionally some optional tools, no inappropriate
    exact_match = (
        has_required and
        not used_inappropriate and
        actual_set <= appropriate  # All actual tools are in appropriate set
    )

    # Partial match: at least one correct tool
    partial_match = len(correct_tools) > 0

    # Precision: % of actual tools that were appropriate (not inappropriate)
    # Penalize inappropriate tools heavily
    inappropriate_count = len(actual_set & inappropriate)
    precision = (len(correct_tools) - inappropriate_count) / len(actual_set) if actual_set else 0.0
    precision = max(0.0, precision)  # Don't go negative

    # Recall: % of required tools that were called
    # Optional tools don't affect recall (no penalty for missing)
    recall = len(actual_set & required) / len(required) if required else 1.0

    # F1 score
    if precision + recall > 0:
        f1_score = 2 * (precision * recall) / (precision + recall)
    else:
        f1_score = 0.0

    return {
        "exact_match": exact_match,
        "partial_match": partial_match,
        "precision": precision,
        "recall": recall,
        "score": f1_score,
        "has_required_tool": has_required,
        "used_inappropriate_tool": used_inappropriate,
    }


def evaluate_agent_performance(
    results: List[Dict[str, Any]],
    save_path: Path | None = None
) -> Dict[str, Any]:
    """
    Evaluate overall agent performance across test cases.

    Args:
        results: List of agent execution results with tools_called
        save_path: Optional path to save results

    Returns:
        Dictionary with evaluation metrics
    """
    if len(results) != len(AGENT_TEST_CASES):
        raise ValueError(
            f"Results count ({len(results)}) doesn't match test cases ({len(AGENT_TEST_CASES)})"
        )

    # Evaluate each test case
    evaluations = []
    for i, (test_case, result) in enumerate(zip(AGENT_TEST_CASES, results)):
        actual_tools = result.get("tools_called", [])
        expected_tools = test_case["expected_tools"]

        eval_result = evaluate_tool_selection(actual_tools, expected_tools)
        eval_result.update({
            "test_id": i,
            "query": test_case["query"],
            "expected_tools": expected_tools,
            "actual_tools": actual_tools,
            "category": test_case["category"],
            "reason": test_case["reason"],
        })
        evaluations.append(eval_result)

    # Aggregate metrics
    total_cases = len(evaluations)
    exact_matches = sum(1 for e in evaluations if e["exact_match"])
    partial_matches = sum(1 for e in evaluations if e["partial_match"])

    avg_precision = sum(e["precision"] for e in evaluations) / total_cases
    avg_recall = sum(e["recall"] for e in evaluations) / total_cases
    avg_f1 = sum(e["score"] for e in evaluations) / total_cases

    # Multi-tool usage (how often agent uses multiple tools when beneficial)
    multi_tool_cases = [e for e in evaluations if e["category"] == "multi_tool"]
    multi_tool_used = sum(1 for e in multi_tool_cases if len(e["actual_tools"]) > 1)
    multi_tool_rate = multi_tool_used / len(multi_tool_cases) if multi_tool_cases else 0.0

    # Single-tool cases (should use exactly 1 tool)
    single_tool_cases = [e for e in evaluations if e["category"] == "single_tool"]
    single_tool_correct = sum(1 for e in single_tool_cases if e["exact_match"])
    single_tool_accuracy = single_tool_correct / len(single_tool_cases) if single_tool_cases else 0.0

    summary = {
        "total_cases": total_cases,
        "exact_match_count": exact_matches,
        "exact_match_rate": exact_matches / total_cases,
        "partial_match_count": partial_matches,
        "partial_match_rate": partial_matches / total_cases,
        "avg_precision": avg_precision,
        "avg_recall": avg_recall,
        "avg_f1_score": avg_f1,
        "multi_tool_rate": multi_tool_rate,
        "single_tool_accuracy": single_tool_accuracy,
        "evaluations": evaluations,
    }

    # Save results if path provided
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, 'w') as f:
            json.dump(summary, f, indent=2)

    return summary


def print_evaluation_summary(summary: Dict[str, Any]):
    """
    Print human-readable evaluation summary.

    Args:
        summary: Evaluation summary from evaluate_agent_performance
    """
    print("\n" + "="*70)
    print("AGENT EVALUATION SUMMARY")
    print("="*70)

    print(f"\nTotal Test Cases: {summary['total_cases']}")
    print(f"\nTool Selection Accuracy:")
    print(f"  Exact Match: {summary['exact_match_count']}/{summary['total_cases']} ({summary['exact_match_rate']:.1%})")
    print(f"  Partial Match: {summary['partial_match_count']}/{summary['total_cases']} ({summary['partial_match_rate']:.1%})")

    print(f"\nMetrics:")
    print(f"  Precision: {summary['avg_precision']:.3f}")
    print(f"  Recall: {summary['avg_recall']:.3f}")
    print(f"  F1 Score: {summary['avg_f1_score']:.3f}")

    print(f"\nTool Usage Patterns:")
    print(f"  Multi-Tool Usage Rate: {summary['multi_tool_rate']:.1%}")
    print(f"  Single-Tool Accuracy: {summary['single_tool_accuracy']:.1%}")

    # Show failed cases
    failures = [e for e in summary['evaluations'] if not e['partial_match']]
    if failures:
        print(f"\nFailed Cases ({len(failures)}):")
        for fail in failures:
            print(f"  - Query: {fail['query']}")
            print(f"    Expected: {fail['expected_tools']}")
            print(f"    Actual: {fail['actual_tools']}")
            print()

    print("="*70)


def get_test_cases() -> List[Dict[str, Any]]:
    """Get all test cases for agent evaluation."""
    return AGENT_TEST_CASES


if __name__ == "__main__":
    # Example: Load results and evaluate
    # This would be called after running agent on all test cases
    print("Agent Evaluation Module")
    print(f"Total test cases: {len(AGENT_TEST_CASES)}")
    print("\nCategories:")
    categories = {}
    for case in AGENT_TEST_CASES:
        cat = case["category"]
        categories[cat] = categories.get(cat, 0) + 1
    for cat, count in categories.items():
        print(f"  {cat}: {count}")
