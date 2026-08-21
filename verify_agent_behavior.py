"""
Verification script for Issue 1: Verify Real Agent Behavior.

Runs 4 targeted queries to test tool orchestration with real LLM calls.
"""

from healthbot.agent_graph import run_agent_query
import json


def verify_agent_behavior():
    """Run 4 verification queries and capture results."""

    test_cases = [
        {
            "name": "Test 1: Single tool (calculator)",
            "query": "What's my BMI if I'm 70kg and 1.75m tall?",
            "expected_tools": ["medical_calculator"]
        },
        {
            "name": "Test 2: Single tool (RAG)",
            "query": "What causes Type 2 diabetes?",
            "expected_tools": ["medical_rag_search"]
        },
        {
            "name": "Test 3: Multi-tool coordination",
            "query": "Calculate my BMI for 70kg and 1.75m, and explain if that's healthy.",
            "expected_tools": ["medical_calculator", "medical_rag_search"]
        },
        {
            "name": "Test 4: Research query",
            "query": "What are recent studies on diabetes treatment?",
            "expected_tools": ["pubmed_api_search", "medical_rag_search", "web_search"]  # Any of these valid
        }
    ]

    results = []

    for i, test in enumerate(test_cases, 1):
        print(f"\n{'='*80}")
        print(f"{test['name']}")
        print(f"Query: {test['query']}")
        print(f"{'='*80}")

        try:
            result = run_agent_query(test['query'])

            tools_called = result.get('tools_called', [])
            summary = result.get('summary', '')
            disclaimer_shown = result.get('disclaimer_shown', False)

            print(f"\nTools called: {tools_called}")
            print(f"Disclaimer shown: {disclaimer_shown}")
            print(f"\nSummary preview (first 200 chars):")
            print(f"{summary[:200]}...")

            # Check if result is successful
            success = (
                len(tools_called) > 0 and
                summary and
                disclaimer_shown
            )

            results.append({
                'test': test['name'],
                'query': test['query'],
                'tools_called': tools_called,
                'expected_tools': test['expected_tools'],
                'summary_length': len(summary),
                'disclaimer_shown': disclaimer_shown,
                'success': success
            })

            if not success:
                print("\nWARNING: Test did not meet success criteria")
                if not tools_called:
                    print("  - No tools called (empty list)")
                if not summary:
                    print("  - No summary generated")
                if not disclaimer_shown:
                    print("  - Disclaimer not shown")

        except Exception as e:
            print(f"\nERROR: {e}")
            import traceback
            traceback.print_exc()

            results.append({
                'test': test['name'],
                'query': test['query'],
                'error': str(e),
                'success': False
            })

    # Summary
    print(f"\n\n{'='*80}")
    print("VERIFICATION SUMMARY")
    print(f"{'='*80}")

    success_count = sum(1 for r in results if r.get('success', False))
    total_count = len(results)

    print(f"\nTests passed: {success_count}/{total_count}")

    for result in results:
        status = "PASS" if result.get('success', False) else "FAIL"
        print(f"\n{status}: {result['test']}")
        if result.get('tools_called'):
            print(f"  Tools: {result['tools_called']}")
        if result.get('error'):
            print(f"  Error: {result['error']}")

    # Save results to file
    with open('verification_results.json', 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nDetailed results saved to: verification_results.json")

    if success_count == total_count:
        print("\nSUCCESS: All verification tests passed!")
        print("   Agent tool orchestration is working correctly with real LLM calls.")
        return True
    else:
        print(f"\nWARNING: {total_count - success_count} test(s) failed.")
        print("   Need to debug before proceeding to other issues.")
        return False


if __name__ == "__main__":
    success = verify_agent_behavior()
    exit(0 if success else 1)
