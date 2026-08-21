"""
Run Agent Evaluation - Phase 4 Empirical Validation

Runs the agent on all 20 test cases and measures:
- Tool selection accuracy
- Multi-tool usage rate
- Precision/Recall metrics

This provides empirical evidence for agent performance (like Phase 3 did for RAG).
"""

import sys
import json
from pathlib import Path
from datetime import datetime

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from healthbot.agent_graph import run_agent_query
from healthbot.evaluation.agent_eval import (
    get_test_cases,
    evaluate_agent_performance,
    print_evaluation_summary,
)


def run_evaluation():
    """Run agent evaluation on all test cases."""
    print("\n" + "="*70)
    print("AGENT EVALUATION - Phase 4 Empirical Validation")
    print("="*70)
    print(f"\nStarted at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    test_cases = get_test_cases()
    print(f"\nTotal test cases: {len(test_cases)}")
    print("\nRunning agent on each test case...")
    print("This will take ~10-15 minutes (20 LLM calls)")
    print("-"*70)

    # Run agent on each test case
    results = []
    for i, test_case in enumerate(test_cases, 1):
        query = test_case["query"]
        expected_tools = test_case["expected_tools"]
        category = test_case["category"]

        print(f"\n[{i}/{len(test_cases)}] {category}")
        print(f"Query: {query[:60]}..." if len(query) > 60 else f"Query: {query}")
        print(f"Expected: {expected_tools}")

        try:
            # Run agent
            result = run_agent_query(query)

            # Extract tools called
            tools_called = result.get("tools_called", [])

            print(f"Called: {tools_called if tools_called else 'None'}")

            # Check if successful
            if tools_called:
                if set(tools_called) == set(expected_tools):
                    print("[OK] Exact match")
                elif any(t in expected_tools for t in tools_called):
                    print("[~] Partial match")
                else:
                    print("[X] No match")
            else:
                print("[X] No tools called")

            results.append({
                "query": query,
                "expected_tools": expected_tools,
                "tools_called": tools_called,
                "category": category,
                "success": True,
            })

        except Exception as e:
            print(f"[X] Error: {str(e)[:100]}")
            results.append({
                "query": query,
                "expected_tools": expected_tools,
                "tools_called": [],
                "category": category,
                "success": False,
                "error": str(e),
            })

    print("\n" + "="*70)
    print("EVALUATION COMPLETE")
    print("="*70)

    # Calculate metrics
    print("\nCalculating metrics...")
    summary = evaluate_agent_performance(results)

    # Print summary
    print_evaluation_summary(summary)

    # Save results
    output_dir = Path("evaluation_results/phase4")
    output_dir.mkdir(parents=True, exist_ok=True)

    results_file = output_dir / "agent_evaluation_results.json"
    with open(results_file, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\n[OK] Results saved to: {results_file}")

    # Create summary for README
    readme_summary = f"""
## Agent Evaluation Results (Phase 4)

**Test Cases**: {summary['total_cases']} diverse queries

**Tool Selection Accuracy**:
- Exact Match: {summary['exact_match_count']}/{summary['total_cases']} ({summary['exact_match_rate']:.1%})
- Partial Match: {summary['partial_match_count']}/{summary['total_cases']} ({summary['partial_match_rate']:.1%})

**Metrics**:
- Precision: {summary['avg_precision']:.3f}
- Recall: {summary['avg_recall']:.3f}
- F1 Score: {summary['avg_f1_score']:.3f}

**Tool Usage Patterns**:
- Multi-Tool Usage Rate: {summary['multi_tool_rate']:.1%}
- Single-Tool Accuracy: {summary['single_tool_accuracy']:.1%}

**Evaluation Date**: {datetime.now().strftime('%Y-%m-%d')}
"""

    readme_file = output_dir / "README_summary.txt"
    with open(readme_file, 'w') as f:
        f.write(readme_summary)

    print(f"[OK] README summary saved to: {readme_file}")

    print("\n" + "="*70)
    print("NEXT STEPS")
    print("="*70)
    print("\n1. Review results above")
    print("2. Update README.md with actual metrics")
    print("3. Commit evaluation results to repository")
    print("\nAgent evaluation complete!")

    return summary


if __name__ == "__main__":
    try:
        summary = run_evaluation()
        sys.exit(0)
    except KeyboardInterrupt:
        print("\n\nEvaluation interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\n\nEvaluation failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
