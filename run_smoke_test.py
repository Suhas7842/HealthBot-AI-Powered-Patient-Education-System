"""
Smoke Test for Agent - Quick Validation

Runs 5 representative queries to verify agent tool orchestration works.

Purpose:
- Fast validation (~2 minutes, 5 LLM calls max)
- Covers key capabilities (calculator, RAG, multi-tool, research, web)
- Can run frequently without exhausting free-tier quota

This is cheaper than full 20-case evaluation but still proves real LLM works.
"""

import sys
from pathlib import Path
from datetime import datetime

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from healthbot.evaluation.agent_executor import execute_with_cache, ExecutionResult
from healthbot.evaluation.agent_eval import evaluate_tool_selection
from healthbot.config import settings


# Smoke test cases - cover different tool types
SMOKE_TEST_CASES = [
    {
        "name": "Calculator (single tool)",
        "query": "What's my BMI if I'm 70kg and 1.75m tall?",
        "expected_tools": {
            "required": ["medical_calculator"],
            "optional": [],
            "inappropriate": []
        },
    },
    {
        "name": "RAG (single tool)",
        "query": "What causes Type 2 diabetes?",
        "expected_tools": {
            "required": ["medical_rag_search"],
            "optional": [],
            "inappropriate": []
        },
    },
    {
        "name": "Multi-tool (calculator + RAG)",
        "query": "Calculate my BMI for 70kg and 1.75m, and explain if that's healthy.",
        "expected_tools": {
            "required": ["medical_calculator"],
            "optional": ["medical_rag_search"],
            "inappropriate": []
        },
    },
    {
        "name": "Research (PubMed)",
        "query": "What are recent studies on diabetes treatment?",
        "expected_tools": {
            "required": ["pubmed_api_search", "medical_rag_search"],  # Either valid
            "optional": [],
            "inappropriate": []
        },
    },
    {
        "name": "Web search",
        "query": "Latest COVID-19 vaccine recommendations",
        "expected_tools": {
            "required": ["web_search", "pubmed_api_search"],  # Either valid
            "optional": ["medical_rag_search"],
            "inappropriate": []
        },
    },
]


def run_smoke_test(use_cache: bool = True):
    """Run smoke test with 5 representative queries."""
    print("\n" + "="*70)
    print("AGENT SMOKE TEST - Quick Validation")
    print("="*70)
    print(f"\nStarted at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Get model
    if settings.LLM_PROVIDER == "gemini":
        model = settings.GEMINI_MODEL
    else:
        model = settings.OPENAI_MODEL

    print(f"Model: {model}")
    print(f"Use cache: {use_cache}")
    print(f"\nTest cases: {len(SMOKE_TEST_CASES)}")
    print("Expected time: 2-3 minutes (first run), <10s (cached)")
    print("-"*70)

    results = []
    passed = 0
    failed = 0

    for i, test_case in enumerate(SMOKE_TEST_CASES, 1):
        name = test_case["name"]
        query = test_case["query"]
        expected_tools = test_case["expected_tools"]

        print(f"\n[{i}/{len(SMOKE_TEST_CASES)}] {name}")
        print(f"Query: {query}")

        # Execute
        exec_result = execute_with_cache(query, model, use_cache=use_cache)

        if not exec_result.is_evaluated():
            print(f"Status: {exec_result.status}")
            if exec_result.error:
                print(f"Error: {exec_result.error}")
            failed += 1
            results.append({
                "name": name,
                "status": exec_result.status,
                "passed": False,
            })
            continue

        # Evaluate tool selection
        tools_called = exec_result.get_tools_called()
        print(f"Status: {exec_result.status}")
        print(f"Tools called: {tools_called}")

        eval_result = evaluate_tool_selection(tools_called, expected_tools)

        # Check pass/fail
        if eval_result["has_required_tool"] and not eval_result["used_inappropriate_tool"]:
            print(f"Result: PASS")
            passed += 1
            results.append({
                "name": name,
                "status": exec_result.status,
                "tools_called": tools_called,
                "passed": True,
                "precision": eval_result["precision"],
                "recall": eval_result["recall"],
            })
        else:
            print(f"Result: FAIL")
            print(f"  Has required: {eval_result['has_required_tool']}")
            print(f"  Used inappropriate: {eval_result['used_inappropriate_tool']}")
            failed += 1
            results.append({
                "name": name,
                "status": exec_result.status,
                "tools_called": tools_called,
                "passed": False,
                "precision": eval_result["precision"],
                "recall": eval_result["recall"],
            })

    # Summary
    print("\n" + "="*70)
    print("SMOKE TEST SUMMARY")
    print("="*70)
    print(f"\nPassed: {passed}/{len(SMOKE_TEST_CASES)}")
    print(f"Failed: {failed}/{len(SMOKE_TEST_CASES)}")

    if passed == len(SMOKE_TEST_CASES):
        print("\nSUCCESS: All smoke tests passed!")
        print("Agent tool orchestration is working correctly.")
        return True
    else:
        print(f"\nWARNING: {failed} test(s) failed.")
        print("Review failed cases above.")
        return False


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run agent smoke test")
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Disable cache (force live LLM calls)",
    )

    args = parser.parse_args()

    try:
        success = run_smoke_test(use_cache=not args.no_cache)
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\nSmoke test interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\n\nSmoke test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
